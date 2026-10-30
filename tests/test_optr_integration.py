"""Archive #86/#97/#148/#170/#173: artifacts and candidate quality boundaries."""
import copy
import hashlib
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
    row["frames"] = [{"frame_id": 0, "context": 0, "current_end": 1, "query_positions": [0],
                      "squared_error": error, "teacher_energy": 4., "normalized_error": error / 4.,
                      "terminal_query": {"position": 0, "squared_error": error,
                                         "teacher_energy": 4., "normalized_error": error / 4.}}]
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
    assert config["devices"] == selected["devices"] == [0, 1, 2, 3]
    assert config["port"] == selected["port"] == 7878
    assert runtime_feature_summary(selected, "candidate")["calibration_recipe_version"] == 5
    assert runtime_feature_summary(selected, "candidate")["optr_fitted_grouping"] == "layer"
    assert runtime_feature_summary(selected, "candidate")["optr_fitting_policy"] == "optr_oscar_shared_train_seed_ascend_v7"
    assert runtime_feature_summary(selected, "candidate")["optr_seed_partition"] == "train"
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


def test_refinement_cache_binds_online_math_and_separates_shared_zero_mean_seed(tmp_path, monkeypatch):
    from oscar_ascend.optr_oscar_seed import legacy_base_profile, seed_profile, source_provenance
    monkeypatch.delenv("OSCAR_ROTATION_ARTIFACT", raising=False)
    config = configuration(tmp_path)
    base = {k: v for k, v in config.items() if k != "rotation_refinement"}
    profile = calibration_cache_identity(config, "model")
    old = calibration_cache_identity(base, "model")
    assert profile["recipe_version"] == 5 and profile["accumulator_dtype"] == "float32"
    seed = profile["base_calibration_profile"]
    assert old["recipe_version"] == 3 and seed["recipe_version"] == 6
    assert seed == seed_profile(old) and seed != old
    assert seed["key_centering"] == "none" and seed["fitted_grouping"] == "layer"
    assert seed["source_provenance"] == source_provenance()
    assert legacy_base_profile(seed) == old
    assert rotation_artifact_path(config, "model", root=tmp_path) != rotation_artifact_path(base, "model", root=tmp_path)
    for key, value in (("sink_tokens", 32), ("recent_tokens", 512), ("k_clip_ratio", .97),
                       ("v_clip_ratio", .93), ("attention_splits", 2), ("cube_cores", 16),
                       ("max_num_batched_tokens", 8192), ("tensor_parallel_size", 4)):
        assert calibration_cache_identity({**config, key: value}, "model") != profile
    for override in ({"selection": "allow_regression"}, {"fitted_grouping": "layer_kv_head"}):
        with pytest.raises(ValueError, match="frozen"):
            optr_policy({**config, "optr_calibration": override})


def test_frame_aware_default_trains_all_deployment_lengths_without_changing_schedule():
    from oscar_ascend.optr_policy import legacy_capture_policy, compatible_capture_profiles
    current = optr_policy({})
    assert current["capture_lengths"] == {
        "train": [4096, 32768, 65536], "validation": [4096, 32768, 65536],
        "test": [4096, 32768, 65536]}
    assert (current["steps"], current["lr"], current["eval_every"]) == (80, .02, 20)
    assert current["selection"] == "shared_layer_each_request_frame_and_terminal_nonworse"
    assert current["policy_version"] == "optr_oscar_shared_train_seed_ascend_v7"
    assert current["fitted_grouping"] == "layer"
    assert current["fitting_objective"] == "upstream_fp32_independent_gqa_kl_output_k_then_conditional_v"
    for version in ("v1", "v2"):
        old = legacy_capture_policy({}, version)
        assert old["capture_lengths"]["train"] == [4096]
        assert sum(map(len, old["capture_lengths"].values())) == 7
        assert old["selection"] == "joint_all_heads_each_request_nonworse"
    # Never relabel old seven-request data as the new nine-request plan.
    assert [p["refinement_policy"]["policy_version"] for p in compatible_capture_profiles({}, {})] == [
        "optr_oscar_shared_upstream_ascend_v6", "optr_oscar_base_upstream_ascend_v5",
        "optr_joint_gqa_bf16_ascend_v4", "optr_frame_bf16_ascend_v3"]


@pytest.mark.parametrize("version,expected", [
    ("v1", "aa87ff99f250a374a8db465bf4956907df3378515c34b904c08b9b56936cdc09"),
    ("v2", "a20383a9b66cbf7a85bb73e4cacdcf64d10c62941c5ad41b6b897095856e3886"),
    ("v3", "68761932277b037026662af26b6485c95937d61702a03653f1bb45a3caab8756"),
    ("v4", "9b04defcb625bfadcb00b3fdd62da94fc053be61abaa233ebd95355071d034bd"),
    ("v5", "a2a22bd2e117ebb65a3b4dd799f6337ee5dc552a8e730fe060ff61e35dea37ca"),
    ("v6", "9601f02f7b913e9195154fbbd3bf18d694d9d3276134494e534ac4b3e2edc50a"),
])
def test_historical_capture_policy_exactly_preserves_prior_identity(version, expected):
    from oscar_ascend.optr_policy import legacy_capture_policy
    # Frozen before each version change, independent of reconstruction.
    policy = legacy_capture_policy({}, version)
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert (policy.get("fitted_grouping") == "layer") == (version == "v6")


@pytest.mark.parametrize("version,expected", [
    ("v1", "35168bd8260c391294f82bc28ddd657d77a608f70fdd7082257d4f3af5b193ec"),
    ("v2", "60257d325117e1d4bce2514ac5327ba00ddc09564d108def4d58eea656bbb66f"),
    ("v3", "87dbf77b471b1b42255484df26912ee8ad8e8be40fc86c80827e5f765b29aed4"),
    ("v4", "28f3a72d8b9eac5810a50ee04114f5e8fb54437f44f23edd462f4f137f7797da"),
    ("v5", "5ab04164eeb1d2871d4f33d97bdec18d737de15796735838d75ef34bc8086efc"),
    ("v6", "105ae433310b7b64a43d4aa23e87e1e8af8ea7c52d628d1d7bbcfc528073cd26"),
])
def test_historical_capture_profile_hashes_match_pre_partition_snapshots(version, expected):
    from oscar_ascend.optr_policy import legacy_capture_profile
    config = {"rotation_refinement": "optr", "rotation_variant": "perhead_center",
        "max_num_batched_tokens": 16384, "experimental_official_math": True,
        "oscar_dequant_policy": "fp32_metadata", "mamba_cache_dtype": "bfloat16",
        "mamba_ssm_cache_dtype": "bfloat16"}
    current = calibration_cache_identity(config, "model")
    historical = legacy_capture_profile(current, config, version)
    encoded = json.dumps(historical, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert historical != current


def test_historical_capture_profiles_restore_original_seed_and_exact_sampling_plan(tmp_path):
    from oscar_ascend.optr_policy import (compatible_capture_profiles,
        legacy_capture_policy, legacy_capture_profile)
    from oscar_ascend.optr_oscar_seed import legacy_seed_profile
    config = {**configuration(tmp_path), "optr_calibration": {"capture_lengths": {
        "train": [4, 32, 64], "validation": [4, 32, 64], "test": [4, 32, 64]},
        "decode_tokens": 3, "query_samples_per_frame": 2, "steps": 5}}
    profile = calibration_cache_identity(config, "preserved")
    legacy_config = {k: v for k, v in config.items() if k not in ("rotation_refinement", "optr_calibration")}
    original_seed = calibration_cache_identity(legacy_config, "preserved")
    compatible = compatible_capture_profiles(profile, config)
    assert compatible == [legacy_capture_profile(profile, config, version) for version in ("v6", "v5", "v4", "v3", "v2", "v1")]
    for candidate in compatible:
        policy = candidate["refinement_policy"]
        assert policy["capture_lengths"] == config["optr_calibration"]["capture_lengths"]
        assert policy["decode_tokens"] == 3 and policy["query_samples_per_frame"] == 2
        if policy["policy_version"] in ("optr_oscar_base_upstream_ascend_v5", "optr_oscar_shared_upstream_ascend_v6"):
            assert candidate["base_calibration_profile"] == legacy_seed_profile(profile["base_calibration_profile"])
            assert candidate["base_calibration_profile"] != profile["base_calibration_profile"]
            assert candidate["base_calibration_profile"]["recipe_version"] == 6
        else:
            assert candidate["base_calibration_profile"] == original_seed
            assert candidate["base_calibration_profile"] != profile["base_calibration_profile"]
            assert candidate["base_calibration_profile"]["recipe_version"] == 3
    assert legacy_capture_policy(config, "v1")["steps"] == 5
    assert legacy_capture_policy(config, "v2")["steps"] == 5
    assert legacy_capture_policy(config, "v1")["lr"] == .002
    assert legacy_capture_policy(config, "v2")["lr"] == .02


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5", "v6"])
def test_completed_older_fit_cannot_bypass_new_objective_via_artifact_cache(tmp_path, version):
    from oscar_ascend.optr_policy import legacy_capture_profile
    config = configuration(tmp_path)
    value = artifact(config)
    value["calibration_profile"] = legacy_capture_profile(value["calibration_profile"], config, version)
    value["refinement_certificate"]["policy"] = value["calibration_profile"]["refinement_policy"]
    if version == "v5":
        # v5 legitimately fitted distinct heads. Keep it readable as history,
        # while refusing to load even an internally valid PT as current v7.
        value["layers"][NAME]["Rk"][1].neg_()
        digest = matrix_digest(value["layers"])
        value["refinement_certificate"]["matrices_sha256"] = digest
        value["refinement_certificate"]["layers"][NAME].update(
            base_constants_sha256=digest, selected_constants_sha256=digest)
    path = tmp_path / "old-complete" / "calibrated.pt"
    # This is an internally valid completed historical artifact. Selecting
    # the current candidate must reject it even through an explicit path.
    publish_artifact(value, path)
    with pytest.raises(ValueError, match="profile"):
        load_active_artifact(config, path=path, device="cpu", testing=True)


def test_current_certificate_cannot_relabel_unpartitioned_seed(tmp_path):
    from oscar_ascend.optr_oscar_seed import legacy_seed_profile
    value = artifact(configuration(tmp_path))
    profile = value["calibration_profile"]
    profile["base_calibration_profile"] = legacy_seed_profile(profile["base_calibration_profile"])
    with pytest.raises(ValueError, match="profile/source"):
        validate_refinement_metadata(value)


def test_old_shared_fit_checkpoints_and_default_pt_path_are_isolated(tmp_path, monkeypatch):
    from oscar_ascend.optr_checkpoint import FitCheckpoints
    from oscar_ascend.optr_policy import legacy_capture_profile
    monkeypatch.delenv("OSCAR_ROTATION_ARTIFACT", raising=False)
    config = configuration(tmp_path)
    profile = calibration_cache_identity(config, "model")
    old = legacy_capture_profile(profile, config, "v6")
    old_digest = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    assert rotation_artifact_path(config, "model", root=tmp_path).parent.name != "optr_" + old_digest
    previous = FitCheckpoints(tmp_path / "fits", {"profile": old})
    current = FitCheckpoints(tmp_path / "fits", {"profile": profile})
    previous.save("layer-3", {"completed": True})
    previous.save("layer-3-shared", {"fitted": True})
    assert previous.load("layer-3") == {"completed": True}
    assert current.load("layer-3") is None and current.load("layer-3-shared") is None


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
    value["layers"][NAME]["Rk"][:, 0, 0] += 1
    digest = matrix_digest(value["layers"])
    cert = value["refinement_certificate"]
    cert["matrices_sha256"] = digest
    cert["layers"][NAME].update(base_constants_sha256=digest, selected_constants_sha256=digest)
    with pytest.raises(ValueError, match="non-orthogonal"):
        validate_artifact(value, layer_names=[NAME], head_dim=64,
            model_fingerprint=value["model_fingerprint"], pr_fingerprint=OFFICIAL_COMMIT,
            device="cpu", testing=True, require_calibrated=True)


@pytest.mark.parametrize("fault", ["Rk", "Rv", "mean", "mean_shape", "storage", "subset", "grouping", "selection", "seed"])
def test_shared_artifact_rejects_independent_heads_or_false_source_contract(tmp_path, fault):
    value = artifact(configuration(tmp_path))
    certificate = value["refinement_certificate"]
    record = certificate["layers"][NAME]
    if fault in ("Rk", "Rv"):
        value["layers"][NAME][fault][1].neg_()
    elif fault == "mean":
        value["layers"][NAME]["k_mean"][0, 0] = 1.
    elif fault == "mean_shape":
        value["layers"][NAME]["k_mean"] = torch.zeros(1, 64)
    elif fault == "storage":
        value["source_grouping"] = "layer"
    elif fault == "subset":
        record["head_subset_selection"] = {"selected_heads": [0]}
    elif fault == "grouping":
        certificate["policy"]["fitted_grouping"] = "layer_kv_head"
    elif fault == "selection":
        certificate["policy"]["selection"] = "joint_all_heads_each_request_frame_and_terminal_nonworse"
    elif fault == "seed":
        value["calibration_profile"]["base_calibration_profile"]["source_provenance"]["commit"] = "0" * 40
    # Rebinding hashes cannot authorize a different mathematical contract.
    digest = matrix_digest(value["layers"])
    certificate["matrices_sha256"] = digest
    record.update(base_constants_sha256=digest, selected_constants_sha256=digest)
    with pytest.raises(ValueError, match="shared|zero key mean|seed profile/source"):
        validate_refinement_metadata(value)


def test_small_teacher_energy_has_one_fixed_normalization():
    tiny = metric("validation", 1e-25)
    tiny["teacher_energy"] = tiny["requests"][0]["teacher_energy"] = 1e-27
    tiny["normalized_error"] = tiny["requests"][0]["normalized_error"] = 1e-5
    tiny["requests"][0]["frames"][0].update(teacher_energy=1e-27, normalized_error=1e-5)
    tiny["requests"][0]["frames"][0]["terminal_query"].update(teacher_energy=1e-27, normalized_error=1e-5)
    assert nonworse(tiny, copy.deepcopy(tiny))


def test_joint_layer_selection_rejects_local_only_gain_and_records_test_regression(monkeypatch):
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
    selected, record = prepare.refine_layer(NAME, baseline, samples, {}, fitter=fitted)
    assert record["accepted"] and torch.equal(selected["Rk"], -baseline["Rk"])
    assert not nonworse(record["test"]["baseline"], record["test"]["selected"])
