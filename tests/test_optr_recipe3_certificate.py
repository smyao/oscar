"""Recipe3 publication and reuse must reject independently failed fits."""
import copy

import pytest
import torch

from oscar_ascend import optr_recipe3_certificate as strict
from oscar_ascend.optr_artifacts import (experimental_certificate, matrix_digest,
    publish_artifact, validate_refinement_metadata)
from oscar_ascend.optr_checkpoint import fitting_implementation_digest
from oscar_ascend.rotations import load_active_artifact
from test_optr_integration import configuration, artifact, metric, NAME


def example(tmp_path):
    config = {**configuration(tmp_path), "optr_baseline": "recipe3"}
    result = artifact(config)
    constants = result["layers"][NAME]
    constants["Rk"][1] = constants["Rk"][1].T.clone()
    constants["k_mean"][:] = .125
    digest = matrix_digest(result["layers"])
    record = result["refinement_certificate"]["layers"][NAME]
    record.update(accepted=True, base_constants_sha256="e" * 64, selected_constants_sha256=digest)
    record["validation"]["selected"] = metric("validation", .5)
    record["test"]["selected"] = metric("test", .75)
    certificate = result["refinement_certificate"]
    certificate.update(accepted_layers=[NAME], matrices_sha256=digest,
        full_layer_names=[NAME], fitting_implementation_sha256=fitting_implementation_digest())
    result["refinement_certificate"] = strict.strict_certificate(certificate)
    return config, result


def test_distinct_head_rotations_and_original_nonzero_center_are_allowed(tmp_path):
    _, result = example(tmp_path)
    validate_refinement_metadata(result)
    strict.require_publishable(result)
    assert result["calibration_profile"]["base_calibration_profile"]["recipe_version"] == 3


@pytest.mark.parametrize("schema", [1, 2])
def test_old_report_only_or_unversioned_publication_cannot_enable_new_policy(tmp_path, schema):
    _, result = example(tmp_path)
    certificate = result["refinement_certificate"]
    if schema == 2:
        result["refinement_certificate"] = experimental_certificate(certificate)
    else:
        certificate["schema_version"] = schema
    with pytest.raises(ValueError, match="strict independent-test"):
        validate_refinement_metadata(result)


@pytest.mark.parametrize("view", [None, "bf16_output", "bf16_rank_output"])
def test_test_regression_cannot_publish_even_when_validation_improves(tmp_path, view):
    _, result = example(tmp_path)
    certificate = result["refinement_certificate"]
    selected = certificate["layers"][NAME]["test"]
    if view is None:
        selected["selected"] = metric("test", 1.5)
    else:
        selected["selected"]["projection_checks"][view] = metric("test", 1.5)
        selected["selected"]["projection_checks"][view].pop("projection_checks")
    result["refinement_certificate"] = strict.strict_certificate(certificate)
    destination = tmp_path / "failed" / "refined.pt"
    with pytest.raises(ValueError, match="worsens test"):
        publish_artifact(result, destination)
    assert not destination.exists()
    assert not (destination.parent / ".optr.complete.json").exists()


def test_unchanged_layer_checkpoint_valid_but_unchanged_whole_fit_not_published(tmp_path):
    _, result = example(tmp_path)
    certificate = result["refinement_certificate"]
    record = certificate["layers"][NAME]
    record.update(accepted=False, base_constants_sha256=record["selected_constants_sha256"])
    for split in ("validation", "test"):
        record[split]["selected"] = copy.deepcopy(record[split]["baseline"])
    certificate["accepted_layers"] = []
    result["refinement_certificate"] = strict.strict_certificate(certificate)
    validate_refinement_metadata(result)
    with pytest.raises(ValueError, match="no strict validation improvement"):
        publish_artifact(result, tmp_path / "refined.pt")


@pytest.mark.parametrize("fault", ["coverage", "digest", "status"])
def test_complete_current_implementation_is_required(tmp_path, fault):
    _, result = example(tmp_path)
    certificate = result["refinement_certificate"]
    if fault == "coverage": certificate["full_layer_names"].append("missing.layer")
    elif fault == "digest": certificate["fitting_implementation_sha256"] = "f" * 64
    else: certificate["independent_test"]["status"] = "failed"
    with pytest.raises(ValueError):
        strict.require_publishable(result)


def test_valid_cache_checks_original_recipe3_before_device_validation(tmp_path, monkeypatch):
    config, result = example(tmp_path)
    destination = tmp_path / "refined.pt"
    publish_artifact(result, destination)
    calls = []
    def reject_baseline(artifact, selected_config, *, testing):
        calls.append((selected_config, testing))
        raise ValueError("original recipe3 changed")
    monkeypatch.setattr(strict, "validate_baseline_binding", reject_baseline)
    with pytest.raises(ValueError, match="original recipe3 changed"):
        load_active_artifact(config, path=destination, device="cpu", testing=True)
    assert calls == [(config, True)]


def test_preparation_cpu_cache_check_does_not_grant_test_baseline_production_use(tmp_path, monkeypatch):
    from tools.prepare_optr_rotations import prepare
    config, result = example(tmp_path)
    result["test_only"] = False
    destination = tmp_path / "refined.pt"
    publish_artifact(result, destination)
    calls = []
    def bind(artifact, selected_config, *, testing):
        calls.append(testing)
        if not testing:
            raise ValueError("test-only recipe3 baseline cannot enter production")
    monkeypatch.setattr(strict, "validate_baseline_binding", bind)
    with pytest.raises(ValueError, match="test-only recipe3 baseline"):
        prepare(config, output=destination, testing=False)
    assert calls == [True, False]


@pytest.mark.parametrize("fault", ["file", "constants", "mean", "layers"])
def test_actual_baseline_identity_is_bound(tmp_path, monkeypatch, fault):
    from tools import prepare_optr_recipe3_base as base
    from oscar_ascend import optr_artifacts
    config, result = example(tmp_path)
    original = copy.deepcopy(result)
    record = result["refinement_certificate"]["layers"][NAME]
    record["base_constants_sha256"] = matrix_digest(original["layers"])
    monkeypatch.setattr(base, "base_config", lambda *_: {"rotation_artifact": "original.pt"})
    monkeypatch.setattr(base, "load_base", lambda *_, **__: original)
    monkeypatch.setattr(optr_artifacts, "file_sha256", lambda _: "c" * 64)
    strict.validate_baseline_binding(result, config, testing=True)
    if fault == "file": monkeypatch.setattr(optr_artifacts, "file_sha256", lambda _: "f" * 64)
    elif fault == "constants": original["layers"][NAME]["Rk"][0] *= -1
    elif fault == "mean": result["layers"][NAME]["k_mean"] += 1
    else: original["layers"].clear()
    with pytest.raises(ValueError, match="baseline"):
        strict.validate_baseline_binding(result, config, testing=True)
