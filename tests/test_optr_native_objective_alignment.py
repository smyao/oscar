"""Archive #170/#174/#175/#176/#177: offline objective-alignment evidence.

These fixed CPU synthetic examples distinguish the historical OptR proxy from
the declared deployment arithmetic. They are not native-device equivalence,
the cause/share of a reported model regression, or model-quality acceptance.
The seed-3 example intentionally retains its failed independent test: neither
its seed nor its one training step is reselected after observing that failure.
"""
from __future__ import annotations

import pytest
import torch

from oscar_ascend.optr_calibration import _bf16, project_output, replay_sample
from oscar_ascend.optr_sequential import _key_probabilities, phase_loss, prepare_samples
from oscar_ascend.rotation_calibration import (
    OFFICIAL_COMMIT, OfflineLayerAccumulator, fit_checkpoint_pair,
)


POLICY = {
    "sink_tokens": 1, "recent_tokens": 2, "k_clip": .96, "v_clip": .92,
    "query_batch_size": 8, "kv_tile_size": 256, "history_splits": 1,
    "replay_partition": "single_split", "lambda_output": 1.,
}


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _native_teacher(q, k, v, positions):
    """Independent dense, causal BF16-visible teacher, not NPU FIA evidence."""
    score = torch.einsum("mgd,ld->mgl", q.float(), k.float()) / q.shape[-1] ** .5
    score.masked_fill_(torch.arange(len(k))[None, None] > positions[:, None, None], -torch.inf)
    return torch.einsum("mgl,ld->mgd", score.softmax(-1), v.float()).bfloat16()


def proxy_blindspot_case():
    """Freeze the observed seed, without searching seeds or candidate radii."""
    rng = torch.Generator().manual_seed(11)
    q = torch.randn(4, 2, 8, generator=rng).bfloat16()
    k = torch.randn(13, 8, generator=rng).bfloat16()
    v = torch.randn(13, 8, generator=rng).bfloat16()
    positions = torch.tensor([2, 5, 8, 12])
    sample = {
        "request_id": "synthetic_proxy_blindspot", "q": q, "k": k, "v": v,
        "teacher": _native_teacher(q, k, v, positions),
        "gate": torch.sigmoid(torch.randn(4, 2, 8, generator=rng)).bfloat16(),
        "wo": torch.randn(2, 5, 8, generator=rng), "positions": positions,
        "contexts": torch.tensor([0, 0, 6, 12]),
        "current_ends": torch.tensor([6, 6, 12, 13]),
    }
    rk = torch.linalg.qr(torch.randn(8, 8, generator=rng)).Q
    rv = torch.linalg.qr(torch.randn(8, 8, generator=rng)).Q
    return rk, rv, torch.zeros(8), sample


def _old_value_row(rk, mean, sample):
    row = prepare_samples([sample], mean, POLICY, "cpu")[0]
    row["selected_k_quant_p"] = _key_probabilities(rk, row, POLICY).detach().masked_fill(row["exact"], 0)
    return row


def _gradient_pair(rk, rv, mean, sample):
    row = _old_value_row(rk, mean, sample)
    correction = torch.zeros_like(rv, requires_grad=True)
    rotation = rv @ torch.matrix_exp(correction - correction.T)
    proxy = phase_loss(rotation, row, "v", POLICY)
    proxy_gradient = torch.autograd.grad(proxy, correction)[0]
    rotation = rv @ torch.matrix_exp(correction - correction.T)
    actual = project_output(replay_sample(rk, rotation, mean, sample, POLICY), sample)
    teacher = project_output(sample["teacher"], sample)
    errors = (actual - teacher).square().sum(-1)
    native_gradient = torch.autograd.grad(errors.sum(), correction)[0]
    return proxy.detach(), proxy_gradient, errors.detach(), native_gradient


def proxy_counterexample_evidence():
    rk, rv, mean, sample = proxy_blindspot_case()
    proxy, proxy_gradient, before, native_gradient = _gradient_pair(rk, rv, mean, sample)
    # A fixed small training direction, not a radius selected by deployment.
    correction = -.0001 * proxy_gradient / proxy_gradient.norm()
    candidate = rv @ torch.matrix_exp(correction - correction.T)
    with torch.no_grad():
        proxy_after = phase_loss(candidate, _old_value_row(rk, mean, sample), "v", POLICY)
        actual = project_output(replay_sample(rk, candidate, mean, sample, POLICY), sample)
        teacher = project_output(sample["teacher"], sample)
        after = (actual - teacher).square().sum(-1)
    return {
        "synthetic": True, "seed": 11, "step": .0001,
        "positions": sample["positions"].tolist(), "contexts": sample["contexts"].tolist(),
        "proxy_before": float(proxy), "proxy_after": float(proxy_after),
        "deployed_query_sse_before": before.tolist(), "deployed_query_sse_after": after.tolist(),
        "deployed_total_before": float(before.sum()), "deployed_total_after": float(after.sum()),
        "proxy_native_gradient_dot": float((proxy_gradient * native_gradient).sum()),
        "all_gates_nonzero": bool((sample["gate"] > 0).all()),
    }


def test_old_proxy_descent_can_increase_actual_deployment_error():
    evidence = proxy_counterexample_evidence()
    assert evidence["all_gates_nonzero"]
    assert evidence["proxy_after"] < evidence["proxy_before"]
    assert evidence["proxy_native_gradient_dot"] < 0
    assert evidence["deployed_total_after"] > evidence["deployed_total_before"]
    assert evidence["proxy_before"] == pytest.approx(.02650032937526703, rel=1e-5)
    assert evidence["deployed_total_before"] == pytest.approx(.3814195692539215, rel=1e-5)


def gate_blindspot_evidence():
    rk, rv, mean, sample = proxy_blindspot_case()
    changed = {**sample, "gate": (1 - sample["gate"].float()).bfloat16()}
    before, after = [_gradient_pair(rk, rv, mean, value) for value in (sample, changed)]
    return {
        "synthetic": True, "seed": 11, "counterfactual": "gate_replaced_with_one_minus_gate",
        "qkv_unchanged": all(torch.equal(sample[key], changed[key]) for key in ("q", "k", "v")),
        "teacher_unchanged": torch.equal(sample["teacher"], changed["teacher"]),
        "gate_range_valid": bool(((changed["gate"] > 0) & (changed["gate"] < 1)).all()),
        "proxy_loss_identical": torch.equal(before[0], after[0]),
        "proxy_gradient_identical": torch.equal(before[1], after[1]),
        "deployed_sse_before": float(before[2].sum()), "deployed_sse_after": float(after[2].sum()),
        "deployed_gradient_change_l2": float((before[3] - after[3]).norm()),
    }


def test_legal_gate_counterfactual_is_invisible_to_old_proxy_but_changes_native_objective():
    evidence = gate_blindspot_evidence()
    assert all(evidence[key] for key in (
        "qkv_unchanged", "teacher_unchanged", "gate_range_valid",
        "proxy_loss_identical", "proxy_gradient_identical"))
    assert evidence["deployed_sse_before"] != evidence["deployed_sse_after"]
    assert evidence["deployed_gradient_change_l2"] > 1.


def test_deployment_objective_reads_captured_teacher_and_old_proxy_does_not():
    rk, rv, mean, sample = proxy_blindspot_case()
    # Interface sensitivity only: this deliberately perturbed target is not
    # represented as another native capture or as the physical model's output.
    counterfactual = {**sample, "teacher": (sample["teacher"].float() + .25).bfloat16()}
    before, after = [_gradient_pair(rk, rv, mean, value) for value in (sample, counterfactual)]
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
    assert not torch.equal(before[2], after[2])
    assert not torch.equal(before[3], after[3])


def test_joint_head_residual_retains_cross_terms_that_independent_proxy_discards():
    # The unchanged second head's residual reverses the correct direction.
    variable = torch.tensor(1., requires_grad=True)
    other_head = torch.tensor(-1.1)
    independent = variable.square() + other_head.square()
    joint = (variable + other_head).square()
    independent_gradient = torch.autograd.grad(independent, variable)[0]
    joint_gradient = torch.autograd.grad(joint, variable)[0]
    assert independent_gradient > 0 and joint_gradient < 0
    candidate = variable.detach() - .1 * independent_gradient
    assert candidate.square() + other_head.square() < independent.detach()
    assert (candidate + other_head).square() > joint.detach()


def recipe3_native_case():
    """Real recipe3 fitter on independent synthetic calibration observations.

    These are selected-query numerical samples, not a production capture
    manifest. No TP/capture authentication or NPU equivalence is implied.
    """
    rng = torch.Generator().manual_seed(3)
    heads, group, dim, length = 2, 2, 8, 13
    weights = torch.randn(heads, group, 4, dim, generator=rng).bfloat16().float()
    accumulator = OfflineLayerAccumulator(head_dim=dim, local_kv_heads=heads,
        local_q_heads=heads * group, token_budget=37, testing=True)
    accumulator.observe(torch.randn(37, heads * group, dim, generator=rng).bfloat16(),
        torch.randn(37, heads, dim, generator=rng).bfloat16(),
        torch.randn(37, heads, dim, generator=rng).bfloat16())
    ranks = [{"rank": 0, "tp_size": 1, "global_q_heads": heads * group,
        "global_kv_heads": heads, "prompt_sha256": "synthetic_base", "model_fingerprint": "model",
        "layers": {3: accumulator.export_payload()}}]
    profile = {"official_commit": OFFICIAL_COMMIT, "model_fingerprint": "model",
        "recipe_version": 3, "max_tokens": 37, "rotation_variant": "perhead_center",
        "model_execution": {"dtype": "bfloat16", "mamba_cache_dtype": "bfloat16",
                            "mamba_ssm_cache_dtype": "bfloat16"}}
    key, value = fit_checkpoint_pair(ranks, layer_names=["model.layers.3.self_attn.attn"],
        head_dim=dim, model_path="/synthetic-model", model_fingerprint="model",
        global_q_heads=heads * group, global_kv_heads=heads, tp_size=1,
        prompt_sha256="synthetic_base", max_tokens=37, calibration_profile=profile, testing=True)
    baseline = {"Rk": key["layers"][3]["rotation"], "Rv": value["layers"][3]["rotation"],
                "k_mean": key["layers"][3]["k_mean"]}
    splits = {}
    for label in ("train", "validation", "test"):
        samples = []
        for head in range(heads):
            q = torch.randn(2, group, dim, generator=rng).bfloat16()
            k = torch.randn(length, dim, generator=rng).bfloat16()
            v = torch.randn(length, dim, generator=rng).bfloat16()
            positions = torch.tensor([2, 12])
            samples.append({"request_id": label, "q": q, "k": k, "v": v,
                "teacher": _native_teacher(q, k, v, positions),
                "gate": torch.sigmoid(torch.randn(2, group, dim, generator=rng)).bfloat16(),
                "wo": weights[head], "positions": positions, "contexts": torch.tensor([0, 12]),
                "current_ends": torch.tensor([6, 13])})
        splits[label] = samples
    return baseline, splits


def native_view_errors(constants, samples):
    """Independent differentiable full-head three-view residual construction."""
    actual = [project_output(replay_sample(constants["Rk"][head], constants["Rv"][head],
              constants["k_mean"][head], sample, POLICY), sample)
              for head, sample in enumerate(samples)]
    teachers = [project_output(sample["teacher"], sample) for sample in samples]
    summed = torch.stack(actual).double().sum(0)
    target = torch.stack(teachers).double().sum(0)
    ranks = torch.stack([_bf16(value) for value in actual]).double().sum(0)
    target_ranks = torch.stack([_bf16(value) for value in teachers]).double().sum(0)
    return torch.stack([(summed - target).square().sum(-1),
        (_bf16(summed) - _bf16(target)).square().sum(-1),
        (_bf16(ranks) - _bf16(target_ranks)).square().sum(-1)])


def recipe3_native_evidence():
    baseline, splits = recipe3_native_case()
    original = {key: value.clone() for key, value in baseline.items()}
    correction = torch.zeros(8, 8, requires_grad=True)
    trial = {**baseline, "Rv": baseline["Rv"].clone()}
    trial["Rv"][0] = baseline["Rv"][0] @ torch.matrix_exp(correction - correction.T)
    # One fixed training-only update. Validation/test are not passed to this
    # calculation, and a failing test cannot cause another candidate attempt.
    gradient = torch.autograd.grad(native_view_errors(trial, splits["train"]).sum(), correction)[0]
    correction = -.01 * gradient / gradient.norm()
    candidate = {**baseline, "Rv": baseline["Rv"].clone()}
    candidate["Rv"][0] = baseline["Rv"][0] @ torch.matrix_exp(correction - correction.T)
    evidence = {"synthetic": True, "seed": 3, "step": .01, "training_updates": 1,
        "baseline": "actual_perhead_center_recipe3_fitter", "test_used_for_selection": False,
        "all_gates_nonzero": all(bool((sample["gate"] > 0).all())
                                 for samples in splits.values() for sample in samples),
        "all_means_nonzero": bool((baseline["k_mean"] != 0).all()),
        "baseline_unchanged": all(torch.equal(original[key], baseline[key]) for key in original)}
    with torch.no_grad():
        for label in ("train", "validation", "test"):
            before, after = (native_view_errors(constants, splits[label]) for constants in (baseline, candidate))
            evidence[label] = {"before": before.tolist(), "after": after.tolist(),
                "each_query_three_view_nonworse": bool((after <= before).all()),
                "fp32_total_before": float(before[0].sum()), "fp32_total_after": float(after[0].sum())}
    return evidence


def test_true_recipe3_native_residual_step_can_improve_validation_but_must_fail_independent_test():
    evidence = recipe3_native_evidence()
    assert evidence["baseline_unchanged"] and evidence["all_gates_nonzero"] and evidence["all_means_nonzero"]
    assert evidence["training_updates"] == 1 and evidence["test_used_for_selection"] is False
    for label in ("train", "validation"):
        assert evidence[label]["each_query_three_view_nonworse"]
        assert evidence[label]["fp32_total_after"] < evidence[label]["fp32_total_before"]
        assert evidence[label]["after"][0][0] < evidence[label]["before"][0][0]  # Nonzero first prefill.
    assert not evidence["test"]["each_query_three_view_nonworse"]
    assert evidence["test"]["fp32_total_after"] > evidence["test"]["fp32_total_before"]
    assert evidence["validation"]["fp32_total_before"] == pytest.approx(.2578070764550624, rel=1e-5)
    assert evidence["validation"]["fp32_total_after"] == pytest.approx(.1379146913813152, rel=1e-5)


def source_audit_evidence():
    """Portable receipt generator; importing this module performs no fitting."""
    return {
        "evidence_kind": "fixed_synthetic_cpu_functional_examples",
        "archives_read": [170, 172, 174, 175, 176, 177],
        "device": "cpu", "model_quality": "not_measured", "performance": "not_measured",
        "native_npu_equivalence": "not_measured", "field_failure_contribution": "not_established",
        "proxy_counterexample": proxy_counterexample_evidence(),
        "gate_counterfactual": gate_blindspot_evidence(),
        "real_recipe3_direct_residual": recipe3_native_evidence(),
        "interpretation": {
            "positive_claim": "Native joint residuals can supply a different useful training direction around actual recipe3.",
            "limit": "STE is not an exact derivative of discrete BF16/INT2; finite deployment and independent-test gates remain necessary.",
            "failed_test_retained": True,
            "no_guarantee": "No unconditional strict-improvement existence or model-accuracy theorem follows from these examples.",
        },
    }
