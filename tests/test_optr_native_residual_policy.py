"""Archive #170/#172/#175/#176/#177: offline policy isolation and source reuse.

Policy/host evidence only; no claim of NPU/model quality or serving throughput.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import types

import pytest

from oscar_ascend.optr_policy import (
    RECIPE3_NATIVE_POLICY_VERSION, compatible_capture_profiles, optr_policy,
    validate_refinement_config,
)
from oscar_ascend.rotations import calibration_cache_identity, rotation_artifact_path
from tools.prepare_optr_recipe3_base import base_config
from tools.serving_variants import runtime_feature_summary, variant_config, variant_features
from tools.target_cli import serve_argv, target_env


ROOT = Path(__file__).resolve().parents[1]
BASE = "a599e8a7489dac78edc1b22cd31e338fe1f2d09f"
OPTIMIZER = "joint_native_residual_v1"


def target():
    return json.loads((ROOT / "configs/target.json").read_text())


def historical_module(path, name):
    source = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
    module = types.ModuleType(name)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def test_fixed_candidate_changes_only_offline_policy_and_preserves_launch():
    previous_module = historical_module("tools/serving_variants.py", "old_serving_variants")
    config = target()
    previous = previous_module.variant_config(config, "candidate")
    current = variant_config(config, "candidate")
    expected = {key: value for key, value in previous.items() if key != "optr_recipe3_repair"}
    assert current == {**expected, "optr_recipe3_optimizer": OPTIMIZER}
    assert variant_config(previous, "candidate") == current
    assert variant_features(current) == previous_module.variant_features(previous)
    assert serve_argv(current) == serve_argv(previous)
    assert target_env(current, base={}) == target_env(previous, base={})
    assert config == target()
    assert variant_config(previous, None) == previous  # Explicit v11 remains v11.
    summary = runtime_feature_summary(current, "candidate")
    assert summary["optr_fitting_policy"] == RECIPE3_NATIVE_POLICY_VERSION
    assert summary["optr_steps_per_phase"] == 1
    assert summary["optr_max_deployment_candidates"] == 8
    assert summary["optr_fitting_execution"] == "cpu_native_residual_constrained_selection"
    assert summary["independent_test_blocks_publication"] is True


def test_native_policy_preserves_baseline_geometry_and_all_acceptance_views():
    from oscar_ascend.optr_native_residual import NATIVE_RESIDUAL_POLICY
    config = variant_config(target(), "candidate")
    previous = {key: value for key, value in config.items() if key != "optr_recipe3_optimizer"}
    old = optr_policy(previous)
    policy = optr_policy(config)
    changes = {
        "policy_version": RECIPE3_NATIVE_POLICY_VERSION,
        "native_residual": NATIVE_RESIDUAL_POLICY,
        "fitting_objective": "joint_native_bf16_postgate_query_residual",
        "execution": "cpu_native_residual_constrained_selection",
        "deployment_candidate_selection": "recipe3_native_residual_train_then_validation_v1",
        "max_deployment_candidates": 8, "steps": 1, "eval_every": 1,
    }
    assert policy == {**old, **changes}
    policy["native_residual"]["test_mutation"] = True
    assert "test_mutation" not in NATIVE_RESIDUAL_POLICY
    assert "test_mutation" not in optr_policy(config)["native_residual"]


def test_historical_policies_are_byte_equivalent_and_native_capture_isolated(tmp_path):
    old_module = historical_module("oscar_ascend/optr_policy.py", "oscar_ascend._old_native_policy")
    current = variant_config(target(), "candidate")
    v10 = {key: value for key, value in current.items() if key != "optr_recipe3_optimizer"}
    v11 = {**v10, "optr_recipe3_repair": "single_head_trust_region_v1"}
    for config in (v10, v11):
        assert optr_policy(config) == old_module.optr_policy(config)
        assert optr_policy(config)["steps"] == 80
    fingerprint = "native-residual-policy-model"
    profile, old10, old11 = [calibration_cache_identity(config, fingerprint)
                             for config in (current, v10, v11)]
    assert compatible_capture_profiles(profile, current) == [old10, old11]
    assert len({rotation_artifact_path(config, fingerprint, root=tmp_path)
                for config in (current, v10, v11)}) == 3
    assert base_config(current, fingerprint) == base_config(v10, fingerprint) == base_config(v11, fingerprint)
    assert "optr_recipe3_optimizer" not in base_config(current, fingerprint)
    changed = {**current, "optr_calibration": {"capture_lengths": {
        split: [4096, 32768] for split in ("train", "validation", "test")}}}
    changed_profile = calibration_cache_identity(changed, fingerprint)
    compatible = compatible_capture_profiles(changed_profile, changed)
    assert old10 not in compatible and old11 not in compatible
    assert all(row["refinement_policy"]["capture_lengths"] == changed["optr_calibration"]["capture_lengths"]
               for row in compatible)


@pytest.mark.parametrize("changes", [
    {"optr_recipe3_optimizer": True}, {"optr_recipe3_optimizer": "unknown"},
    {"optr_baseline": None}, {"rotation_refinement": None},
    {"optr_recipe3_repair": "single_head_trust_region_v1"},
])
def test_invalid_native_strategy_cannot_be_silently_normalized(changes):
    current = {**variant_config(target(), "candidate"), **changes}
    with pytest.raises(ValueError):
        validate_refinement_config(current)
    if changes.get("rotation_refinement", "optr") == "optr":
        with pytest.raises(ValueError):
            variant_config(current, "candidate")


@pytest.mark.parametrize("override", [
    {"steps": 80}, {"steps": True}, {"eval_every": 20}, {"lr": .02},
    {"max_deployment_candidates": 64}, {"native_residual": {}},
    {"validation_projection_views": ["fp32"]},
])
def test_native_fixed_budget_and_acceptance_cannot_be_overridden(override):
    config = variant_config(target(), "candidate")
    with pytest.raises(ValueError, match="frozen"):
        optr_policy({**config, "optr_calibration": override})
    same = optr_policy({**config, "optr_calibration": {"steps": 1, "eval_every": 1}})
    assert same == optr_policy(config)


@pytest.mark.parametrize("variant", ["native", "baseline"])
def test_control_variants_remove_native_strategy_without_changing_the_bundle(variant):
    current = variant_config(target(), "candidate")
    previous = {key: value for key, value in current.items() if key != "optr_recipe3_optimizer"}
    assert variant_config(current, variant) == variant_config(previous, variant)
    assert "optr_recipe3_optimizer" not in variant_config(current, variant)


def test_fitting_identity_covers_new_code_but_capture_identity_does_not(tmp_path, monkeypatch):
    from oscar_ascend import optr_checkpoint
    tree = ast.parse((ROOT / "oscar_ascend/optr_checkpoint.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "fitting_implementation_digest")
    names = next(ast.literal_eval(node.value) for node in function.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "names" for target in node.targets))
    required = {"oscar_ascend/optr_native_residual.py", "oscar_ascend/optr_native_residual_certificate.py",
                "tools/prepare_optr_native_residual.py"}
    assert required <= set(names)
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    assert optr_checkpoint.fitting_implementation_digest() == digest.hexdigest()
    config = variant_config(target(), "candidate")
    profile = calibration_cache_identity(config, "model")
    identity = {"profile": profile, "fitting_implementation_sha256": digest.hexdigest()}
    prior = optr_checkpoint.FitCheckpoints(tmp_path, identity)
    prior.save("layer-3-shared", {"old_work": True})
    changed = {**identity, "fitting_implementation_sha256": "f" * 64}
    assert optr_checkpoint.FitCheckpoints(tmp_path, changed).load("layer-3-shared") is None
    assert calibration_cache_identity(config, "model") == profile


def test_native_worker_does_not_initialize_npu(monkeypatch, tmp_path):
    import torch
    from oscar_ascend import optr_artifacts, optr_sequential, optr_traces
    from tools import optr_fit_worker, prepare_optr_recipe3_base

    def forbidden(*args, **kwargs):
        pytest.fail("native residual preparation must not initialize an NPU")

    config = variant_config(target(), "candidate")
    monkeypatch.setattr(optr_sequential, "fitting_device", forbidden)
    monkeypatch.setattr(optr_artifacts, "file_sha256", lambda _: "base")
    monkeypatch.setattr(prepare_optr_recipe3_base, "load_base", lambda *args, **kwargs: {"test_only": True})
    monkeypatch.setattr(optr_traces, "load_trace_manifest", lambda *args, **kwargs: {"_manifest_sha256": "trace"})
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "1")

    class Pipe:
        events = []
        closed = False
        def send(self, event):
            self.events.append(event)
        def recv(self):
            return None
        def close(self):
            self.closed = True

    pipe = Pipe()
    spec = {"config": config, "testing": True, "base_config": {"rotation_artifact": "base.pt"},
            "certificate": {"base_artifact_sha256": "base", "trace_manifest_sha256": "trace"},
            "trace_manifest": "trace.json", "names": [],
            "template": {"head_dim": 256, "model_fingerprint": "model"},
            "checkpoint_directory": str(tmp_path), "checkpoint_identity": {"new": True}}
    optr_fit_worker.fit_worker(spec, 0, 1, pipe)
    ready = next(row for row in pipe.events if row.get("fields", {}).get("phase") == "fit_worker_ready")
    assert ready["fields"]["fitting_device"] == "cpu"
    assert pipe.closed
