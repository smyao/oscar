"""Archive #170/#174/#175: exact-prefill proxy blindness, CPU diagnostics.

These tests use synthetic BF16 QKV with the real 80+80-step fitter/replay;
they do not identify the cause of every rejected candidate in the user's
native run. No production loss, gate, policy or performance path is changed.
"""
import copy

import pytest
import torch

from oscar_ascend.optr_artifacts import nonworse
from oscar_ascend.optr_calibration import project_output, replay_sample
from oscar_ascend.optr_policy import optr_policy
from oscar_ascend.optr_recipe3_fit import fit_perhead
from oscar_ascend.optr_sequential import _key_probabilities, fit_head, phase_loss, prepare_samples
from tools.prepare_optr_rotations import _metrics
from test_optr_calibration import trace
from test_optr_recipe3_fit import recall_case


STRICT = dict(require_frames=True, require_terminal=True, require_queries=True, require_projection=True)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _frame_rows(metric):
    views = {"fp32": metric, **metric["projection_checks"]}
    return {(view, request["request_id"], frame["frame_id"]): frame
            for view, values in views.items() for request in values["requests"] for frame in request["frames"]}


def first_prefill_diagnostic():
    """Restore a nonzero first-frame sigmoid in the previous fixed fixture."""
    baseline, samples, policy = recall_case()
    policy = {**optr_policy({"optr_baseline": "recipe3"}), "sink_tokens": 2, "recent_tokens": 4}
    for split in samples.values():
        for head in split:
            for sample in head:
                sample["gate"][0].fill_(.5)
    result = fit_perhead(baseline["Rk"], baseline["Rv"], baseline["k_mean"],
                        samples["train"], samples["validation"], policy=policy)
    before = _metrics(baseline, samples["validation"], policy)
    before_frames = _frame_rows(before)
    candidates = []
    for row in result["candidates"]:
        constants = {**baseline, "Rk": row["Rk"], "Rv": row["Rv"]}
        after = _metrics(constants, samples["validation"], policy)
        frames = _frame_rows(after)
        regressions = [{"view": key[0], "request_id": key[1], "frame_id": key[2],
                        "before_squared_error": before_frames[key]["squared_error"],
                        "after_squared_error": frame["squared_error"]}
                       for key, frame in frames.items()
                       if frame["squared_error"] > before_frames[key]["squared_error"]]
        first_partition = after["requests"][0]["frames"]
        candidates.append({"id": row["id"], "validation_squared_error": after["squared_error"],
            "strict_gate_passed": nonworse(before, after, **STRICT) and after["squared_error"] < before["squared_error"],
            "first_prefill_squared_error": first_partition[0]["squared_error"],
            "decode_squared_error": first_partition[1]["squared_error"], "regressed_frames": regressions})
    base_frames = before["requests"][0]["frames"]
    return {"fixture_seed": 12, "synthetic": True, "head_dim": 8, "kv_heads": 2, "gqa": 2,
            "history_tokens": 81, "sampled_prefill_queries": 1, "sampled_decode_queries": 1,
            "first_prefill_sigmoid_gate": .5, "steps_per_phase": 80, "learning_rate": .02,
            "validation_squared_error": before["squared_error"],
            "first_prefill_squared_error": base_frames[0]["squared_error"],
            "decode_squared_error": base_frames[1]["squared_error"], "candidates": candidates}


def d256_exact_prefill_diagnostic():
    """Eight sampled prefill queries, including the first and terminal token."""
    dim, count, length = 256, 8, 337
    sample = trace(20261103, "prefill", length=length, dim=dim, queries=count, group=6)
    sample["positions"] = torch.tensor([index * (length - 1) // (count - 1) for index in range(count)])
    sample["contexts"].zero_()
    sample["gate"].fill_(.5)
    logits = torch.einsum("mgd,ld->mgl", sample["q"].float(), sample["k"].float()) * dim ** -.5
    logits.masked_fill_(torch.arange(length)[None, None] > sample["positions"][:, None, None], -torch.inf)
    sample["teacher"] = torch.einsum("mgl,ld->mgd", logits.softmax(-1), sample["v"].float()).bfloat16()
    generator = torch.Generator().manual_seed(20261103)
    rk, rv = [torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q.contiguous() for _ in range(2)]
    mean = torch.linspace(-.2, .3, dim)
    policy = optr_policy({"optr_baseline": "recipe3"})
    row = prepare_samples([sample], mean, policy, "cpu")[0]
    key = rk.clone().requires_grad_()
    key_loss = phase_loss(key, row, "k", policy)
    key_loss.backward()
    row["selected_k_quant_p"] = _key_probabilities(rk, row, policy).masked_fill(row["exact"], 0).detach()
    value = rv.clone().requires_grad_()
    value_loss = phase_loss(value, row, "v", policy)
    value_loss.backward()
    predicted = project_output(replay_sample(rk, rv, mean, sample, policy), sample)
    target = project_output(sample["teacher"], sample)
    errors = (predicted - target).double().square().sum(-1)
    return {"synthetic": True, "head_dim": dim, "gqa": 6, "tokens": length,
            "query_positions": sample["positions"].tolist(), "sigmoid_gate": .5,
            "k_proxy_loss": float(key_loss.detach()), "v_proxy_loss": float(value_loss.detach()),
            "k_gradient_max_abs": float(key.grad.abs().max()), "v_gradient_max_abs": float(value.grad.abs().max()),
            "bf16_projected_squared_errors": errors.tolist(), "bf16_projected_squared_error": float(errors.sum())}


def first_step_diagnostic():
    rows = []
    for dim in (8, 64, 256):
        train = trace(153, "train", length=541, dim=dim, queries=3, group=6)
        held = trace(159, "held", length=541, dim=dim, queries=3, group=6)
        held["wo"] = train["wo"]
        rotation = torch.linalg.qr(torch.randn(dim, dim, generator=torch.Generator().manual_seed(10))).Q.contiguous()
        checkpoints = []
        policy = {**optr_policy({"optr_baseline": "recipe3"}), "steps": 1}
        fit_head(rotation, rotation, torch.linspace(-.2, .3, dim), [train], [held],
                 policy=policy, global_kv_heads=4, candidate_callback=lambda **row: checkpoints.append(row))
        first = next(row["rotation"] for row in checkpoints if row["phase"] == "k" and row["step"] == 1)
        rows.append({"head_dim": dim,
            "relative_frobenius": float(torch.linalg.vector_norm(first - rotation) / torch.linalg.vector_norm(rotation)),
            "bf16_changed_fraction": float((first.bfloat16() != rotation.bfloat16()).float().mean())})
    return {"synthetic": True, "train_seed": 153, "validation_seed": 159, "rotation_seed": 10,
            "learning_rate": .02, "tokens": 541, "gqa": 6, "queries": 3,
            "global_heads_loss_divisor": 4, "first_key_step": rows}


def test_nonzero_first_prefill_rejects_every_existing_candidate_despite_decode_improvement():
    evidence = first_prefill_diagnostic()
    assert not any(row["strict_gate_passed"] for row in evidence["candidates"])
    candidate = next(row for row in evidence["candidates"] if row["id"] == "k_step_1")
    assert candidate["validation_squared_error"] < evidence["validation_squared_error"] * .6
    assert candidate["decode_squared_error"] < evidence["decode_squared_error"] * .6
    assert candidate["first_prefill_squared_error"] > evidence["first_prefill_squared_error"] * 2
    assert candidate["validation_squared_error"] == pytest.approx(.16963770368685505, rel=1e-4)
    assert evidence["first_prefill_squared_error"] == pytest.approx(7.26331864111529e-5, rel=1e-4)
    assert candidate["first_prefill_squared_error"] == pytest.approx(.00016790324624160746, rel=1e-4)


def test_d256_exact_prefill_has_zero_upstream_loss_gradient_but_nonzero_bf16_error():
    evidence = d256_exact_prefill_diagnostic()
    assert all(evidence[key] == 0 for key in ("k_proxy_loss", "v_proxy_loss", "k_gradient_max_abs", "v_gradient_max_abs"))
    assert len(evidence["bf16_projected_squared_errors"]) == 8
    assert all(error > 0 for error in evidence["bf16_projected_squared_errors"])
    assert evidence["query_positions"][0] == 0 and evidence["query_positions"][-1] == 336


def test_fixed_coordinatewise_adam_rate_is_a_large_d256_rotation_step():
    evidence = first_step_diagnostic()["first_key_step"]
    assert evidence[0]["relative_frobenius"] < .2
    assert evidence[2]["relative_frobenius"] > .5
    assert evidence[2]["relative_frobenius"] > evidence[0]["relative_frobenius"] * 5
    assert evidence[2]["bf16_changed_fraction"] > .99


def test_changing_prefill_gate_does_not_change_upstream_key_value_proxy():
    sample = trace(89, "mixed", length=81, queries=2, group=2)
    sample["positions"] = torch.tensor([79, 80])
    sample["contexts"] = torch.tensor([0, 80])
    original = copy.deepcopy(sample)
    original["gate"][0].zero_()
    sample["gate"][0].fill_(.5)
    policy = {**optr_policy({"optr_baseline": "recipe3"}), "sink_tokens": 2, "recent_tokens": 4}
    values = []
    for current in (original, sample):
        row = prepare_samples([current], torch.linspace(-.2, .3, 8), policy, "cpu")[0]
        rotation = torch.eye(8)
        row["selected_k_quant_p"] = _key_probabilities(rotation, row, policy).masked_fill(row["exact"], 0)
        values.append([float(phase_loss(rotation, row, phase, policy)) for phase in ("k", "v")])
    assert values[0] == values[1]
