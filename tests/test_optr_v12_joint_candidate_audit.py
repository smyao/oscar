"""Archive #170/#175-177: v12 joint objective versus single-side proposals.

Fixed synthetic CPU counterexample, not a production trace or NPU/quality
certificate. The actual recipe3 fitter and native dense attention teachers
are used; one fixed W_O per head is shared by independent train/val/test.
All original fine-grained gates remain intact. The independent test failure
is preserved and never selects another proposal. The v13 proposal extension is exercised without mocking gradients or metrics.
"""
import hashlib

import pytest
import torch

from oscar_ascend.optr_native_residual import NATIVE_RESIDUAL_POLICY, _passes, select_native_residual
from oscar_ascend.optr_policy import optr_policy
from oscar_ascend.rotation_calibration import OFFICIAL_COMMIT, OfflineLayerAccumulator, fit_checkpoint_pair
from tools.prepare_optr_rotations import _metrics
from test_optr_calibration import trace


def joint_candidate_case():
    seed, heads, group, dim = 114, 2, 2, 8
    rng = torch.Generator().manual_seed(seed)
    weights = torch.randn(heads, group, 12, dim, generator=rng).bfloat16().float()
    accumulator = OfflineLayerAccumulator(head_dim=dim, local_kv_heads=heads,
        local_q_heads=heads * group, token_budget=37, testing=True)
    accumulator.observe(torch.randn(37, heads * group, dim, generator=rng).bfloat16(),
        torch.randn(37, heads, dim, generator=rng).bfloat16(),
        torch.randn(37, heads, dim, generator=rng).bfloat16())
    profile = {"official_commit": OFFICIAL_COMMIT, "model_fingerprint": "model",
        "recipe_version": 3, "max_tokens": 37, "rotation_variant": "perhead_center",
        "model_execution": {"dtype": "bfloat16", "mamba_cache_dtype": "bfloat16",
                            "mamba_ssm_cache_dtype": "bfloat16"}}
    rank = {"rank": 0, "tp_size": 1, "global_q_heads": heads * group,
        "global_kv_heads": heads, "prompt_sha256": "calib", "model_fingerprint": "model",
        "layers": {3: accumulator.export_payload()}}
    key, value = fit_checkpoint_pair([rank], layer_names=["model.layers.3.self_attn.attn"],
        head_dim=dim, model_path="/synthetic", model_fingerprint="model",
        global_q_heads=heads * group, global_kv_heads=heads, tp_size=1,
        prompt_sha256="calib", max_tokens=37, calibration_profile=profile, testing=True)
    baseline = {"layer_id": 3, "Rk": key["layers"][3]["rotation"],
                "Rv": value["layers"][3]["rotation"], "k_mean": key["layers"][3]["k_mean"]}
    policy = optr_policy({"optr_baseline": "recipe3", "optr_recipe3_optimizer": "joint_native_residual_v1",
                          "sink_tokens": 2, "recent_tokens": 4})
    samples = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        samples[split] = []
        for head in range(heads):
            sample = trace(seed * 13 + split_index * 2 + head + 10, split,
                           length=13, dim=dim, queries=2, group=group)
            sample.update(positions=torch.tensor([8, 12]), contexts=torch.tensor([0, 9]),
                current_ends=torch.tensor([9, 13]), frame_ids=torch.tensor([0, 1]),
                history_splits=torch.tensor([1, 2]), token_ids=torch.arange(13), split=split)
            scores = torch.einsum("mgd,ld->mgl", sample["q"].float(), sample["k"].float()) / dim ** .5
            scores.masked_fill_(torch.arange(13)[None, None] > sample["positions"][:, None, None], -torch.inf)
            sample["teacher"] = torch.einsum("mgl,ld->mgd", scores.softmax(-1), sample["v"].float()).bfloat16()
            sample["wo"] = weights[head]
            digest = hashlib.sha256()
            for name in ("q", "k", "v", "teacher", "gate", "wo", "positions", "contexts"):
                digest.update(name.encode())
                digest.update(sample[name].contiguous().view(torch.uint8).numpy().tobytes())
            sample["sample_sha256"] = digest.hexdigest()
            samples[split].append([sample])
    return baseline, samples, policy


def joint_candidate_evidence():
    baseline, samples, policy = joint_candidate_case()
    selected, before, after, decision, proposal = select_native_residual(
        "layer", baseline, samples["train"], samples["validation"], policy)
    single = [row for row in decision["evaluated_train_candidates"] if row["scope"] == "single"]
    joint = [row for row in decision["evaluated_train_candidates"] if row["scope"] == "joint"]
    assert len(single) == 8 and all(not row["accepted"] for row in single)
    assert len(joint) == 2 and sum(row["accepted"] for row in joint) == 1
    assert decision["accepted"] and decision["selected_candidate"] == "native_joint_r1"
    assert decision["validation_candidate_count"] == 1
    winner = next(row for row in joint if row["accepted"])
    radius, train_after = winner["radius"], winner["train"]
    assert after == _metrics(selected, samples["validation"], policy)
    val_after, validation_passed = after, _passes(before, after)
    test_before = _metrics(baseline, samples["test"], policy)
    test_after = _metrics(selected, samples["test"], policy)
    return {"seed": 114, "radius": radius, "original_single_candidates": len(single),
        "original_accepted": any(row["accepted"] for row in single), "joint_train_candidates": 2,
        "extra_gradient_evaluations": 0, "test_used_for_selection": False,
        "train": [decision["train_baseline"]["squared_error"], train_after["squared_error"]],
        "validation": [before["squared_error"], val_after["squared_error"]],
        "test": [test_before["squared_error"], test_after["squared_error"]],
        "train_strict_passed": True, "validation_strict_passed": validation_passed,
        "test_strict_passed": _passes(test_before, test_after), "publication_allowed": False,
        "nonzero_means": bool((baseline["k_mean"] != 0).all()),
        "nonzero_gates": all(bool((sample["gate"] > 0).all()) for split in samples.values()
                             for head in split for sample in head)}


def test_joint_train_directions_reach_candidate_missing_from_single_side_space_but_test_rejects():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        evidence = joint_candidate_evidence()
    finally:
        torch.set_num_threads(previous)
    assert evidence["original_single_candidates"] == 8 and not evidence["original_accepted"]
    assert evidence["radius"] == .0025 and evidence["extra_gradient_evaluations"] == 0
    assert evidence["train"] == pytest.approx([1.6427114499367725, 1.6099393500356545])
    assert evidence["validation"] == pytest.approx([.3293378992000381, .3156968867139547])
    assert evidence["test"] == pytest.approx([.2725189767048608, .3612117186824202])
    assert evidence["train_strict_passed"] and evidence["validation_strict_passed"]
    assert not evidence["test_strict_passed"] and not evidence["publication_allowed"]
    assert evidence["nonzero_gates"] and evidence["nonzero_means"]


def test_joint_native_teacher_candidate_is_locked_before_independent_test_and_failure_cannot_reselect(tmp_path, monkeypatch):
    from oscar_ascend import optr_native_residual as native
    from oscar_ascend.optr_checkpoint import FitCheckpoints
    from tools.prepare_optr_native_residual import refine_layer
    from tools.prepare_optr_rotations import IndependentTestRegression
    baseline, samples, policy = joint_candidate_case()
    checkpoints = FitCheckpoints(tmp_path, {"test_only": True, "case": "native_joint_rejected_test"})
    events = []
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.raises(IndependentTestRegression, match="not published"):
            refine_layer("layer", baseline, samples, policy, checkpoints=checkpoints,
                         progress=lambda **row: events.append(row))
        saved = checkpoints.load("layer-3-shared")
        assert saved["decision"]["selected_candidate"] == "native_joint_r1"
        assert saved["decision"]["test_used_for_selection"] is False
        phases = [row.get("phase") for row in events]
        assert phases.index("native_residual_locked_before_test") < phases.index("joint_output_check")
        pointer = checkpoints.root / "layer-3-shared.json"
        locked = pointer.read_bytes()
        def forbidden(*args, **kwargs):
            pytest.fail("independent test failure must never trigger reselection")
        monkeypatch.setattr(native, "select_native_residual", forbidden)
        monkeypatch.setattr(native, "_gradient_direction", forbidden)
        with pytest.raises(IndependentTestRegression, match="not published"):
            refine_layer("layer", baseline, samples, policy, checkpoints=checkpoints, progress=lambda **row: None)
        assert pointer.read_bytes() == locked
    finally:
        torch.set_num_threads(previous)
