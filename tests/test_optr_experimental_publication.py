"""User-authorized experimental publication, never a model-quality claim."""
import json

import pytest
import torch

from oscar_ascend.optr_artifacts import (experimental_certificate, matrix_digest,
    validate_refinement_metadata)
from oscar_ascend.optr_checkpoint import FitCheckpoints, fitting_implementation_digest
from test_optr_integration import NAME, artifact, configuration, metric


def failed_experiment(tmp_path):
    config = configuration(tmp_path)
    value = artifact(config)
    value["layers"][NAME]["Rk"] = value["layers"][NAME]["Rk"].roll(1, dims=-1)
    cert = value["refinement_certificate"]
    record = cert["layers"][NAME]
    record["accepted"] = True
    record["selected_constants_sha256"] = cert["matrices_sha256"] = matrix_digest(value["layers"])
    record["validation"]["selected"] = metric("validation", .5)
    record["test"]["selected"] = metric("test", 2.)
    cert["accepted_layers"] = [NAME]
    value["refinement_certificate"] = experimental_certificate(cert)
    return config, value


@pytest.mark.parametrize("fault", ["none", "false_pass", "missing_status", "missing_layers",
    "not_experimental", "wrong_policy", "validation", "nonfinite", "different_teacher", "legacy"])
def test_experimental_certificate_allows_only_reported_test_regression(tmp_path, fault):
    _, value = failed_experiment(tmp_path)
    cert = value["refinement_certificate"]
    record = cert["layers"][NAME]
    if fault == "false_pass": cert["independent_test"]["status"] = "passed"
    if fault == "missing_status": cert.pop("independent_test")
    if fault == "missing_layers": cert["independent_test"]["regressed_layers"] = []
    if fault == "not_experimental": cert["experimental"] = False
    if fault == "wrong_policy": cert["publication_policy"] = "skip_validation"
    if fault == "validation": record["validation"]["selected"] = metric("validation", 2.)
    if fault == "nonfinite": record["test"]["selected"]["squared_error"] = float("nan")
    if fault == "different_teacher":
        changed = metric("test", 2.)
        changed["teacher_energy"] = changed["requests"][0]["teacher_energy"] = 8.
        changed["normalized_error"] = changed["requests"][0]["normalized_error"] = .25
        record["test"]["selected"] = changed
    if fault == "legacy":
        cert["schema_version"] = 1
        for key in ("publication_policy", "experimental", "independent_test"):
            cert.pop(key)
    if fault == "none":
        assert validate_refinement_metadata(value)["independent_test"] == {
            "status": "failed", "regressed_layers": [NAME]}
    else:
        with pytest.raises(ValueError):
            validate_refinement_metadata(value)


def test_experimental_publication_keeps_orthogonality_gate(tmp_path):
    from oscar_ascend.rotations import validate_artifact
    _, value = failed_experiment(tmp_path)
    # Preserve shared-head equality so this still isolates orthogonality.
    value["layers"][NAME]["Rk"][:, 0, 0] += 1
    digest = matrix_digest(value["layers"])
    value["refinement_certificate"]["matrices_sha256"] = digest
    value["refinement_certificate"]["layers"][NAME]["selected_constants_sha256"] = digest
    with pytest.raises(ValueError, match="non-orthogonal"):
        validate_artifact(value, layer_names=[NAME], head_dim=64,
            model_fingerprint=value["model_fingerprint"], pr_fingerprint=value["pr_fingerprint"],
            device="cpu", testing=True, require_calibrated=True)


@pytest.mark.parametrize("change", [None, "profile", "base_artifact_sha256", "trace_manifest_sha256", "test_only", "failed_report", "unknown_implementation", "corrupt"])
def test_old_joint_fitter_cannot_resume_as_official_sequential_fit(tmp_path, change):
    identity = {"profile": {"recipe_version": 5}, "base_artifact_sha256": "a" * 64,
        "trace_manifest_sha256": "b" * 64, "test_only": True,
        "fitting_implementation_sha256": "5ae615236f8751b9cddf5da9e649febcce72b04bdee7d7020a53ddf0c9ca095b"}
    legacy = FitCheckpoints(tmp_path, identity)
    payload = {"Rk": -torch.eye(4), "Rv": -torch.eye(4), "decision": {"accepted": False}}
    legacy.save("layer-3-head-0", payload)
    current_identity = {**identity, "fitting_implementation_sha256": fitting_implementation_digest()}
    if change in ("profile", "base_artifact_sha256", "trace_manifest_sha256", "test_only"):
        current_identity[change] = False if change == "test_only" else "different"
    if change == "unknown_implementation":
        current_identity["fitting_implementation_sha256"] = "0" * 64
    if change == "corrupt":
        pointer = json.loads((legacy.root / "layer-3-head-0.json").read_text())
        (legacy.root / pointer["file"]).write_bytes(b"bad")
    if change == "failed_report":
        (legacy.root / "layer-3-independent-test-failed.json").write_text('{"status":"failed"}')
    store = FitCheckpoints(tmp_path, current_identity)
    assert store.root != legacy.root
    assert store.load("layer-3-head-0") is None
    assert store.load("layer-3") is None


def test_regressed_first_layer_does_not_stop_later_layers(tmp_path, monkeypatch):
    from tools import prepare_optr_rotations as prepare
    from oscar_ascend import optr_traces
    from oscar_ascend.rotations import load_active_artifact
    from test_optr_quality_failure import samples, metrics, fitter
    config = configuration(tmp_path)
    model_path = tmp_path / "model" / "config.json"
    model = json.loads(model_path.read_text())
    model["num_hidden_layers"] = 8
    model_path.write_text(json.dumps(model))
    seed = artifact(config)["layers"][NAME]
    second = "model.layers.7.self_attn.attn"
    baseline = {"global_kv_heads": 2, "layers": {NAME: seed, second: {**seed, "layer_id": 7}}}
    seed_path = tmp_path / "seed.pt"
    torch.save(baseline, seed_path)
    monkeypatch.setattr(prepare, "_base_config", lambda *_: {"rotation_artifact": str(seed_path)})
    monkeypatch.setattr(prepare, "_prepare_base", lambda *_a, **_kw: baseline)
    monkeypatch.setattr(optr_traces, "load_trace_manifest", lambda *_a, **_kw: {})
    visited = []
    def load(_manifest, layer):
        visited.append(layer)
        return samples()
    monkeypatch.setattr(optr_traces, "load_layer_samples", load)
    def measure(constants, data, _policy, **_kw):
        split = data[0][0]["request_id"]
        changed = not torch.equal(constants["Rk"], seed["Rk"])
        errors = [4.] * 4 if not changed else [2.] * 4
        if changed and split == "test" and constants["layer_id"] == 3:
            errors = [1., 1., 4.25, 1.]
        return metrics(split, errors)
    monkeypatch.setattr(prepare, "_metrics", measure)
    output = tmp_path / "refined" / "refined.pt"
    result = prepare.prepare(config, output=output, testing=True, fitter=fitter,
                             capture_runner=lambda *_: {"fixture": "capture"})
    assert visited == [3, 7] and result["accepted_layers"] == [NAME, second]
    assert result["independent_test_regressed_layers"] == [NAME]
    loaded = load_active_artifact(config, path=output, device="cpu", testing=True)
    assert set(loaded["layers"]) == {NAME, second}
    assert loaded["refinement_certificate"]["independent_test"]["status"] == "failed"


def test_worker_experimental_status_survives_compact_terminal_filter():
    from tools.phase import _compact_should_mirror
    line = '(Worker_TP0 pid=123) [oscar] OPTR_ARTIFACT {"experimental":true,"independent_test":"failed"}'
    assert _compact_should_mirror(line, {})
