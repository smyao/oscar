"""Archive #13-20/#86/#126/#148/#164/#170: offline OptR numerical safeguards.

These tests prove CPU fitting properties, not NPU/model/performance acceptance.
"""
import copy

import pytest
import torch

from oscar_ascend.optr_calibration import (
    _improves, evaluate_head, fit_head, project_output, quantize_dequantize,
    replay_sample,
)
from tools.bf16_reference import attention_tiles, merge_parts
from tools.official_runtime_oracle import request_samples


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trace(seed=0, request_id="request", *, length=48, dim=8, queries=4, group=2):
    rng = torch.Generator().manual_seed(seed)
    k = torch.randn(length, dim, generator=rng).bfloat16()
    v = torch.randn(length, dim, generator=rng).bfloat16()
    q = torch.randn(queries, group, dim, generator=rng).bfloat16()
    positions = torch.arange(length - queries, length)
    contexts = torch.full((queries,), length - queries)
    scores = torch.einsum("mgd,ld->mgl", q.float(), k.float()) / dim ** .5
    scores.masked_fill_(torch.arange(length)[None, None] > positions[:, None, None], -torch.inf)
    teacher = torch.einsum("mgl,ld->mgd", scores.softmax(-1), v.float()).bfloat16()
    return {"request_id": request_id, "k": k, "v": v, "q": q,
            "gate": torch.sigmoid(torch.randn(queries, group, dim, generator=rng)).bfloat16(),
            "teacher": teacher, "positions": positions, "contexts": contexts,
            "current_ends": torch.full((queries,), length, dtype=torch.int64),
            "wo": torch.randn(group, 12, dim, generator=rng)}


def codec_reference(value, ratio):
    # Independent writer formula; no imported optimizer helper or gradients.
    index = min(int(float(torch.tensor(ratio, dtype=torch.float32)) * value.shape[-1]), value.shape[-1] - 1)
    threshold = value.abs().sort(-1).values[:, index:index + 1]
    x = value.clamp(-threshold, threshold)
    low = x.min(-1, keepdim=True).values
    scale = (x.max(-1, keepdim=True).values - low).clamp_min(1e-8) / 3
    zero = -low / scale
    code = (x / scale + zero + .5).floor().clamp(0, 3)
    return ((code - zero) * scale).bfloat16().float()


def test_codec_fp32_metadata_and_ste_gradient():
    x = torch.ones((2, 256))
    x[0, 128:] = 1.0078125
    x[1] = torch.linspace(-3, 3, 256).bfloat16().float()
    x.requires_grad_()
    actual = quantize_dequantize(x, .96)
    assert torch.equal(actual.detach(), codec_reference(x.detach(), .96))
    assert torch.equal(actual[0], x[0])  # mode1 would produce 1.015625.
    (actual.square().sum()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_source_window_replay_matches_independent_online_tile_oracle():
    sample = trace(length=541, dim=8, queries=3)
    sample["positions"] = torch.tensor([515, 517, 540])
    sample["contexts"] = torch.tensor([515, 515, 515])
    # A nontrivial orthogonal transform checks orientation and the sole inverse.
    rng = torch.Generator().manual_seed(77)
    rk = torch.linalg.qr(torch.randn(8, 8, generator=rng)).Q.contiguous()
    rv = torch.linalg.qr(torch.randn(8, 8, generator=rng)).Q.contiguous()
    mean = torch.linspace(-.7, .7, 8)
    policy = {"sink_tokens": 2, "recent_tokens": 4, "query_batch_size": 3}
    qr = (sample["q"].float() @ rk.bfloat16().float()).bfloat16().float()
    kc = (sample["k"].float() - mean.bfloat16().float()).bfloat16().float()
    kr = (kc @ rk.bfloat16().float()).bfloat16().float()
    vr = (sample["v"].float() @ rv.bfloat16().float()).bfloat16().float()
    parts = []
    history = torch.arange(2, 515)  # Partition before applying the cached cut.
    window = torch.tensor([0, 1, 512, 513, 514])
    current = torch.arange(515, 541)
    for kind, (positions, k, v) in enumerate(((history, codec_reference(kr, .96), codec_reference(vr, .92)),
                            (window, kr, vr), (current, kr, vr))):
        visible = positions[None] <= sample["positions"][:, None]
        if kind == 0:
            visible = visible & (positions[None] < 512)
        parts.append(attention_tiles(qr, k[positions, None], v[positions, None],
                     visible=visible))
    merged = merge_parts(parts).output
    expected = (merged.bfloat16().float() @ rv.bfloat16().float().T).bfloat16().float()
    actual = replay_sample(rk, rv, mean, sample, policy)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Current q540 must retain the same cached cut as q515, not move it to 537.
    assert not torch.equal(actual, replay_sample(rk, rv, mean,
        {**sample, "contexts": sample["positions"].clone()}, policy))


def test_causal_current_and_microbatch_invariance():
    sample = trace(length=15, queries=3)
    sample["contexts"] = torch.zeros(3, dtype=torch.int64)
    sample["positions"] = torch.tensor([2, 5, 8])
    rk = rv = torch.eye(8)
    mean = torch.zeros(8)
    before = replay_sample(rk, rv, mean, sample, {"query_batch_size": 3})
    torch.testing.assert_close(before, replay_sample(rk, rv, mean, sample,
                                {"query_batch_size": 1}), atol=0, rtol=0)
    changed = copy.deepcopy(sample)
    changed["k"][9:] = 100
    changed["v"][9:] = -100
    assert torch.equal(before, replay_sample(rk, rv, mean, changed, {}))


@pytest.mark.parametrize("splits", [1, 20])
def test_captured_frame_splits_match_independent_runtime_oracle(splits):
    sample = trace(length=541, dim=8, queries=3, group=6)
    context = 515
    local = [0, 2, 25]
    sample["positions"] = torch.tensor([context + value for value in local])
    sample["contexts"] = torch.full((3,), context)
    sample["history_splits"] = torch.full((3,), splits)
    rk = rv = torch.eye(8)
    mean = torch.zeros(8)
    k, v = sample["k"].float(), sample["v"].float()
    kq, vq = codec_reference(k, .96), codec_reference(v, .92)
    q_full = torch.zeros((541 - context, 6, 8))
    q_full[local] = sample["q"].float()
    expected = request_samples(q_full, q_full, k[context:, None], v[context:, None],
        context, rv, lambda positions: (kq[positions, None], vq[positions, None]),
        lambda positions: (k[positions, None], v[positions, None]), local,
        cached_quant_end=512, sink=2, recent=4, splits=splits, dequant_mode=0)
    policy = {"sink_tokens": 2, "recent_tokens": 4, "replay_partition": "captured_runtime_frames"}
    actual = replay_sample(rk, rv, mean, sample, policy)
    golden = torch.stack([expected[index][0] for index in local])
    torch.testing.assert_close(actual, golden, atol=0, rtol=0)
    torch.testing.assert_close(actual, replay_sample(rk, rv, mean, sample,
        {**policy, "query_batch_size": 1}), atol=0, rtol=0)


def test_captured_partition_requires_complete_frame_metadata():
    sample = trace()
    with pytest.raises(ValueError, match="requires history_splits"):
        replay_sample(torch.eye(8), torch.eye(8), torch.zeros(8), sample,
                      {"replay_partition": "captured_runtime_frames"})


def test_gate_rounding_and_cross_head_terms_are_in_objective():
    sample = trace(dim=8, queries=1)
    sample["gate"].fill_(.3)
    sample["teacher"].zero_()
    sample["wo"][0].fill_(1.)
    sample["wo"][1] = -sample["wo"][0]
    output = torch.ones_like(sample["q"]).float() * 1.0078125
    projected = project_output(output, sample)
    assert torch.equal(projected, torch.zeros_like(projected))
    candidate = output.clone() * .5
    candidate[:, 1].zero_()
    assert candidate.square().sum() < output.square().sum()
    assert project_output(candidate, sample).square().sum() > projected.square().sum()
    # An independent-head MSE proxy cannot see cancellation, or its removal
    # by an unequal gate. The deployed objective must see both effects.
    sample["gate"][:, 1].zero_()
    projected = project_output(output, sample)
    expected = (output[:, 0] * sample["gate"][:, 0].float()).bfloat16().float() @ sample["wo"][0].T
    torch.testing.assert_close(projected, expected, atol=0, rtol=0)
    assert projected.square().sum() > 0


def test_request_nonregression_gate_rejects_lower_average_bad_candidate():
    baseline = {"normalized_error": 1., "requests": [
        {"request_id": "a", "normalized_error": 1.}, {"request_id": "b", "normalized_error": 1.}]}
    bad = {"normalized_error": .8, "requests": [
        {"request_id": "a", "normalized_error": .5}, {"request_id": "b", "normalized_error": 1.1}]}
    assert not _improves(bad, baseline)
    assert not _improves(baseline, baseline)


def test_actual_orthogonal_fitting_improves_disjoint_validation_output():
    training = trace(0, "train")
    validation = trace(10, "validation")
    base = torch.eye(8)
    mean = torch.zeros(8)
    rk, rv, decision = fit_head(base, base, mean, [training], [validation], policy={
        "sink_tokens": 2, "recent_tokens": 4, "steps": 8, "eval_every": 2, "lr": .005})
    assert decision["accepted"]
    assert decision["selected"]["normalized_error"] < decision["baseline"]["normalized_error"] * .9
    for rotation in (rk, rv):
        torch.testing.assert_close(rotation.T @ rotation, torch.eye(8), atol=1e-5, rtol=0)
    verified = evaluate_head(rk, rv, mean, [validation], decision["policy"])
    assert verified == decision["selected"]
    assert decision["model_quality"] == decision["npu_equivalence"] == "not_measured"
    assert torch.equal(base, torch.eye(8))  # caller baseline is immutable.


def test_no_step_keeps_exact_baseline_and_does_not_claim_improvement():
    base = torch.eye(8)
    rk, rv, decision = fit_head(base, base, torch.zeros(8), [trace(0, "train")],
                                [trace(10, "validation")], policy={"steps": 0})
    assert torch.equal(rk, base) and torch.equal(rv, base)
    assert not decision["accepted"] and decision["baseline"] == decision["selected"]


@pytest.mark.parametrize("malformation", ["nan", "future_context", "missing_gate", "nonorthogonal", "overlap"])
def test_bad_inputs_fail_explicitly(malformation):
    training, validation = trace(0, "train"), trace(1, "validation")
    rk, rv = torch.eye(8), torch.eye(8)
    if malformation == "nan":
        validation["v"][0, 0] = float("nan")
    elif malformation == "future_context":
        validation["contexts"][0] = 999
    elif malformation == "missing_gate":
        del validation["gate"]
    elif malformation == "nonorthogonal":
        rk[0, 0] = 2
    else:
        validation["request_id"] = "train"
    with pytest.raises(ValueError):
        fit_head(rk, rv, torch.zeros(8), [training], [validation], policy={"steps": 0})
