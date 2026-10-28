"""Archive #86/#97/#148: offline artifacts and candidate startup boundaries."""
import copy
import json
from pathlib import Path

import pytest
import torch

from oscar_ascend.optr_artifacts import (matrix_digest, nonworse, publish_artifact,
    validate_refinement_metadata)
from oscar_ascend.optr_policy import OBJECTIVE, optr_policy
from oscar_ascend.rotation_calibration import OFFICIAL_COMMIT
from oscar_ascend.rotations import (calibration_cache_identity, checkpoint_identity,
    load_active_artifact, rotation_artifact_path, validate_artifact, prepare_device_rotations)
from tools.serving_variants import variant_config, runtime_feature_summary
from tools.target_cli import serve_argv

NAME = "model.layers.3.self_attn.attn"


def configuration(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 4,
        "full_attention_interval": 4, "head_dim": 64, "hidden_size": 256,
        "num_attention_heads": 4, "num_key_value_heads": 2}))
    selected = variant_config({"model": str(model), "tensor_parallel_size": 2,
        "mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "bfloat16",
        "max_num_batched_tokens": 16384, "use_native_token_budget": False}, "candidate")
    return selected


def metric(split, error=1.):
    row = {"request_id": split, "sample_sha256": ("a" if split == "validation" else "b") * 64,
           "squared_error": error, "teacher_energy": 4., "normalized_error": error / 4.}
    return {key: row[key] for key in ("squared_error", "teacher_energy", "normalized_error")} | {"requests": [row]}


def artifact(config):
    names, dim, fingerprint = checkpoint_identity(config["model"])
    q = torch.linalg.qr(torch.randn(dim, dim, generator=torch.Generator().manual_seed(31))).Q
    entry = {"layer_id": 3, "Rk": q.repeat(2, 1, 1), "Rv": q.T.repeat(2, 1, 1),
             "k_mean": torch.zeros(2, dim)}
    profile = calibration_cache_identity(config, fingerprint)
    constants_hash = matrix_digest({NAME: entry})
    record = {"accepted": False, "base_constants_sha256": constants_hash,
        "selected_constants_sha256": constants_hash,
        **{split: {"baseline": metric(split), "selected": metric(split)} for split in ("validation", "test")}}
    return {"format_version": 2, "source_grouping": "layer_kv_head", "global_kv_heads": 2,
        "rotation_variant": "perhead_center", "head_dim": dim, "model_fingerprint": fingerprint,
        "pr_fingerprint": OFFICIAL_COMMIT, "objective": OBJECTIVE, "calibrated": True,
        "accumulation_dtype": "float32", "test_only": True, "calibration_profile": profile,
        "layers": {NAME: entry}, "refinement_certificate": {"schema_version": 1,
            "policy": profile["refinement_policy"], "metric": "all_tp_heads_bf16_gate_fp32_wo_output",
            "arithmetic_evidence": "cpu_deployment_reference_not_npu_bitwise",
            "model_quality": "not_measured", "performance": "not_measured",
            "base_artifact_sha256": "c" * 64, "trace_manifest_sha256": "d" * 64,
            "matrices_sha256": constants_hash, "layers": {NAME: record}, "accepted_layers": []}}


def test_candidate_enables_optr_without_changing_online_bundle_or_extra_flags():
    config = json.loads(Path("configs/target.json").read_text())
    selected = variant_config(config, "candidate")
    seed = {k: v for k, v in selected.items() if k != "rotation_refinement"}
    assert serve_argv(selected) == serve_argv(seed)
    assert "rotation_refinement" not in config
    assert selected["rotation_refinement"] == "optr"
    assert runtime_feature_summary(selected, "candidate")["calibration_recipe_version"] == 5
    for variant in ("baseline", "native"):
        assert variant_config(selected, variant).get("rotation_refinement") is None
    calibrated = {**selected, "optr_calibration": {"steps": 8}}
    assert variant_config(calibrated, "candidate") == calibrated
    assert variant_config({**config, "rotation_refinement": None}, "candidate") == selected
    assert variant_config(selected, "candidate") == selected
    for override in ({"rotation_method": "hadamard"}, {"quantization": "ascend"},
                     {"use_native_token_budget": True}):
        explicit = variant_config({**config, **override}, "candidate")
        assert explicit.get("rotation_refinement") is None
        with pytest.raises(ValueError, match="OptR requires|OptR native"):
            variant_config({**selected, **override}, "candidate")


def test_refinement_cache_binds_online_math_and_keeps_recipe3_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("OSCAR_ROTATION_ARTIFACT", raising=False)
    config = configuration(tmp_path)
    base = {k: v for k, v in config.items() if k != "rotation_refinement"}
    profile = calibration_cache_identity(config, "model")
    old = calibration_cache_identity(base, "model")
    assert profile["recipe_version"] == 5 and profile["accumulator_dtype"] == "float32"
    assert profile["base_calibration_profile"] == old and old["recipe_version"] == 3
    assert rotation_artifact_path(config, "model", root=tmp_path) != rotation_artifact_path(base, "model", root=tmp_path)
    for key, value in (("sink_tokens", 32), ("recent_tokens", 512), ("k_clip_ratio", .97),
                       ("v_clip_ratio", .93), ("attention_splits", 2), ("cube_cores", 16),
                       ("max_num_batched_tokens", 8192), ("tensor_parallel_size", 4)):
        assert calibration_cache_identity({**config, key: value}, "model") != profile
    with pytest.raises(ValueError, match="frozen"):
        optr_policy({**config, "optr_calibration": {"selection": "allow_regression"}})


def test_published_refinement_loads_normal_tp_constants_and_cannot_be_relabelled(tmp_path):
    config = configuration(tmp_path)
    value = artifact(config)
    path = tmp_path / "refined" / "refined.pt"
    publish_artifact(value, path)
    loaded = load_active_artifact(config, path=path, device="cpu", testing=True)
    for rank in (0, 1):
        prepared = prepare_device_rotations(loaded, device="cpu", testing=True, tp_rank=rank, tp_size=2)[NAME]
        assert torch.equal(prepared.key_transposed.T, value["layers"][NAME]["Rk"][rank])
        assert torch.equal(prepared.inverse_value_transposed, value["layers"][NAME]["Rv"][rank])
    with pytest.raises(ValueError, match="profile"):
        load_active_artifact({**config, "recent_tokens": 128}, path=path, device="cpu", testing=True)
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        load_active_artifact(config, path=path, device="cpu", testing=True)


@pytest.mark.parametrize("fault", ["metric", "regression", "overlap", "constants", "rejected_changed", "recipe", "test_quality"])
def test_certificate_rejects_false_acceptance(tmp_path, fault):
    config = configuration(tmp_path)
    value = artifact(config)
    cert = value["refinement_certificate"]
    row = cert["layers"][NAME]
    if fault == "metric": row["validation"]["selected"]["squared_error"] = 0
    if fault == "regression": row["test"]["selected"] = metric("test", 2)
    if fault == "overlap":
        for key in ("baseline", "selected"):
            row["test"][key]["requests"][0]["sample_sha256"] = "a" * 64
    if fault == "constants": value["layers"][NAME]["Rk"][0, 0, 0] += .1
    if fault == "rejected_changed": row["base_constants_sha256"] = "e" * 64
    if fault == "recipe": value["calibration_profile"]["recipe_version"] = 3
    if fault == "test_quality": cert["model_quality"] = "passed"
    with pytest.raises(ValueError): validate_refinement_metadata(value)


def test_refinement_still_enforces_matrix_orthogonality(tmp_path):
    config = configuration(tmp_path)
    value = artifact(config)
    value["layers"][NAME]["Rk"][0, 0, 0] += 1
    digest = matrix_digest(value["layers"])
    cert = value["refinement_certificate"]
    cert["matrices_sha256"] = digest
    cert["layers"][NAME].update(base_constants_sha256=digest, selected_constants_sha256=digest)
    with pytest.raises(ValueError, match="non-orthogonal"):
        validate_artifact(value, layer_names=[NAME], head_dim=64,
            model_fingerprint=value["model_fingerprint"], pr_fingerprint=OFFICIAL_COMMIT,
            device="cpu", testing=True, require_calibrated=True)


def test_small_teacher_energy_has_one_fixed_normalization():
    tiny = metric("validation", 1e-25)
    tiny["teacher_energy"] = tiny["requests"][0]["teacher_energy"] = 1e-27
    tiny["normalized_error"] = tiny["requests"][0]["normalized_error"] = 1e-5
    assert nonworse(tiny, copy.deepcopy(tiny))


def test_joint_layer_selection_rejects_local_only_gain_and_blocks_test_regression(monkeypatch):
    from tools import prepare_optr_rotations as prepare
    baseline = {"Rk": torch.eye(2).unsqueeze(0), "Rv": torch.eye(2).unsqueeze(0),
                "k_mean": torch.zeros(1, 2), "layer_id": 3}
    samples = {split: [[{"request_id": split}]] for split in ("train", "validation", "test")}
    def fitted(rk, rv, mean, train, validation, **kw):
        return -rk, rv, {"accepted": True}
    def outputs(constants, values, policy):
        split = values[0][0]["request_id"]
        return metric(split, 2 if constants["Rk"][0, 0, 0] < 0 else 1)
    monkeypatch.setattr(prepare, "_metrics", outputs)
    selected, record = prepare.refine_layer(NAME, baseline, samples, {}, fitter=fitted)
    assert not record["accepted"] and torch.equal(selected["Rk"], baseline["Rk"])
    def test_regression(constants, values, policy):
        split = values[0][0]["request_id"]
        changed = constants["Rk"][0, 0, 0] < 0
        return metric(split, (.5 if split == "validation" else 2) if changed else 1)
    monkeypatch.setattr(prepare, "_metrics", test_regression)
    with pytest.raises(ValueError, match="independent test regressed"):
        prepare.refine_layer(NAME, baseline, samples, {}, fitter=fitted)
