"""Archive #86/#97/#166/#170/#172/#174: retain authenticated recipe3 constants.

Real CPU recipe3 fitting/pair publication and isolated-child contracts only;
synthetic tensors never become production calibration or quality evidence.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from oscar_ascend import rotations
from oscar_ascend.rotation_calibration import (
    fit_checkpoint_pair, checkpoint_pair_artifact, publish_checkpoint_pair,
)
from tools import prepare_optr_recipe3_base as baseline
from test_official_rotation_calibration import model, rank_states, NAME


@pytest.fixture
def case(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    original = {**model(model_dir), "rotation_variant": "perhead_center"}
    _, _, fingerprint = rotations.checkpoint_identity(model_dir)
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    real_path = rotations.rotation_artifact_path
    monkeypatch.setattr(rotations, "rotation_artifact_path",
        lambda config, fp, **kwargs: real_path(config, fp, root=tmp_path))
    config = baseline.base_config({**original, "rotation_refinement": "optr",
        "optr_baseline": "recipe3", "optr_seed_capture": {"previous": "capture"},
        "optr_calibration": {"steps": 80}, "rotation_seed": "old-shared",
        "rotation_artifact": str(tmp_path / "optr-final" / "refined.pt")}, fingerprint)
    profile = rotations.calibration_cache_identity(original, fingerprint)
    ranks = rank_states(fingerprint, prompt="e" * 64, n=13)
    k, v = fit_checkpoint_pair(ranks, layer_names=[NAME], head_dim=8,
        model_path=model_dir, model_fingerprint=fingerprint, global_q_heads=4,
        global_kv_heads=2, tp_size=2, prompt_sha256="e" * 64,
        max_tokens=profile["max_tokens"], calibration_profile=profile, testing=True)
    artifact = checkpoint_pair_artifact(k, v, layer_names=[NAME], head_dim=8,
        model_fingerprint=fingerprint, calibration_profile=profile, testing=True)
    artifact["capture_manifest"] = {"status": "native_post_rope_capture_complete",
        "model_fingerprint": fingerprint, "prompt_sha256": "e" * 64,
        "token_count": 13, "prompt_lengths": [5, 8], "calibration_profile": profile,
        "prompt_source_sha256": profile["prompt_source"]["sha256"]}
    return config, original, fingerprint, k, v, artifact


def publish(case):
    config, _, _, k, v, artifact = case
    destination = Path(config["rotation_artifact"])
    publish_checkpoint_pair(k, v, destination.parent, artifact=artifact,
                            artifact_name=destination.name)
    return destination


def test_base_path_is_original_recipe3_and_ignores_final_override(case, monkeypatch, tmp_path):
    config, original, fingerprint, *_ = case
    monkeypatch.setenv("OSCAR_ROTATION_ARTIFACT", str(tmp_path / "external-optr.pt"))
    inputs = {**original, "rotation_refinement": "optr", "optr_baseline": "recipe3",
        "optr_seed_capture": {"old": True}, "rotation_artifact": str(tmp_path / "final.pt")}
    preserved = copy.deepcopy(inputs)
    selected = baseline.base_config(inputs, fingerprint)
    assert selected == config
    assert inputs == preserved
    assert os.environ["OSCAR_ROTATION_ARTIFACT"] == str(tmp_path / "external-optr.pt")
    assert "qqt_sst_" in selected["rotation_artifact"]
    assert rotations.calibration_cache_identity(selected, fingerprint) == rotations.calibration_cache_identity(original, fingerprint)
    assert not set(baseline._REFINEMENT_FIELDS[:-1]) & selected.keys()


def test_original_pair_is_reused_with_nonzero_distinct_head_constants(case, tmp_path):
    destination = publish(case)
    config, _, _, _, _, expected = case
    before = {p.name: p.read_bytes() for p in destination.parent.iterdir() if p.suffix in (".pt", ".json")}
    def forbidden(*args, **kwargs):
        pytest.fail("matching recipe3 pair must not be refitted")
    actual = baseline.prepare_base(config, tmp_path / "work", testing=True, base_runner=forbidden)
    for key in ("Rk", "Rv", "k_mean"):
        assert torch.equal(actual["layers"][NAME][key], expected["layers"][NAME][key])
        assert actual["layers"][NAME][key].device.type == "cpu"
    assert torch.count_nonzero(actual["layers"][NAME]["k_mean"])
    assert not torch.equal(actual["layers"][NAME]["Rk"][0], actual["layers"][NAME]["Rk"][1])
    assert before == {p.name: p.read_bytes() for p in destination.parent.iterdir() if p.suffix in (".pt", ".json")}


def test_cpu_baseline_cannot_enter_production_even_with_testing_loader(case):
    publish(case)
    with pytest.raises(ValueError, match="test-only"):
        baseline.load_base(case[0])


def test_missing_base_uses_unchanged_pair_import_and_publishes_original_recipe(case, tmp_path, monkeypatch):
    config, _, _, key, value, expected = case
    key_path, value_path = tmp_path / "input-k.pt", tmp_path / "input-v.pt"
    torch.save(key, key_path)
    torch.save(value, value_path)
    monkeypatch.setenv("OSCAR_ROTATION_ARTIFACT", str(tmp_path / "wrong-final.pt"))
    before = (key_path.read_bytes(), value_path.read_bytes())
    result = baseline.prepare_base(config, tmp_path / "work", testing=True,
                                   import_pair=(key_path, value_path))
    assert result["calibration_profile"]["recipe_version"] == 3
    assert result["objective"] == "qqt_sst_r_h_pbr"
    for name in ("Rk", "Rv", "k_mean"):
        assert torch.equal(result["layers"][NAME][name], expected["layers"][NAME][name])
    assert (key_path.read_bytes(), value_path.read_bytes()) == before
    assert not (tmp_path / "wrong-final.pt").exists()


@pytest.mark.parametrize("fault", ["changed_bytes", "wrong_pair_constants", "pending", "missing_manifest"])
def test_corrupt_or_incomplete_pair_cannot_be_replaced_by_a_seed(case, fault, tmp_path):
    config, _, _, k, v, artifact = case
    destination = publish(case)
    if fault == "changed_bytes":
        destination.write_bytes(destination.read_bytes() + b"changed")
    elif fault == "wrong_pair_constants":
        artifact["layers"][NAME]["k_mean"] = artifact["layers"][NAME]["k_mean"] + 1
        publish_checkpoint_pair(k, v, destination.parent, artifact=artifact)
    elif fault == "pending":
        (destination.parent / ".oscar.pending.json").write_text("{}")
    else:
        (destination.parent / ".oscar.complete.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        baseline.load_base(config, testing=True)


@pytest.mark.parametrize("fault", ["optr", "seed", "path", "recipe4", "hadamard"])
def test_loader_rejects_wrong_artifact_domain_before_reading(case, tmp_path, fault):
    config = copy.deepcopy(case[0])
    sentinel = tmp_path / "wrong-domain.pt"
    sentinel.write_bytes(b"must not be read or overwritten")
    if fault == "optr": config["rotation_refinement"] = "optr"
    elif fault == "seed": config["rotation_seed"] = "oscar_base_shared_train_v2"
    elif fault == "path": config["rotation_artifact"] = str(sentinel)
    elif fault == "recipe4": config["rotation_k_clip"] = "per_head_calibrated"
    else: config["rotation_method"] = "hadamard"
    with pytest.raises(ValueError):
        baseline.load_base(config, testing=True)
    assert sentinel.read_bytes() == b"must not be read or overwritten"


def test_base_child_preserves_original_preparer_and_clears_only_artifact_override(case, monkeypatch, tmp_path):
    config = case[0]
    monkeypatch.setenv("OSCAR_ROTATION_ARTIFACT", str(tmp_path / "stale-final.pt"))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    environment = dict(os.environ)
    calls = []
    sentinel = object()
    monkeypatch.setattr(baseline, "load_base", lambda cfg, **kwargs: sentinel)
    def child(command, **options):
        calls.append(command)
        assert command[:3] == [sys.executable, "-m", "tools.prepare_rotations"]
        selected = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert selected == {k: v for k, v in config.items() if k != "rotation_artifact"}
        assert "OSCAR_ROTATION_ARTIFACT" not in options["env"]
        assert options["env"] == {k: v for k, v in environment.items() if k != "OSCAR_ROTATION_ARTIFACT"}
        assert options["check"] is True and options["timeout"] > 0
        assert not {"stdout", "stderr", "capture_output"} & options.keys()
    monkeypatch.setattr(baseline.subprocess, "run", child)
    assert baseline.prepare_base(config, tmp_path / "work") is sentinel
    assert len(calls) == 1 and os.environ == environment


def test_child_failure_stops_before_loading_and_injection_is_test_only(case, monkeypatch, tmp_path):
    def failure(command, **options):
        raise subprocess.CalledProcessError(7, command)
    monkeypatch.setattr(baseline.subprocess, "run", failure)
    monkeypatch.setattr(baseline, "load_base", lambda *_a, **_kw: pytest.fail("failed child must not load"))
    with pytest.raises(subprocess.CalledProcessError) as caught:
        baseline.prepare_base(case[0], tmp_path / "work")
    assert caught.value.returncode == 7
    with pytest.raises(ValueError, match="test-only"):
        baseline.prepare_base(case[0], tmp_path / "work", base_runner=lambda *_: None)


def test_capture_metadata_retains_verified_source_and_reports_missing_raw_manifest(case):
    artifact = case[-1]
    metadata = baseline.capture_source_metadata(artifact)
    assert metadata["schema_version"] == 1 and metadata["capture_manifest_available"]
    assert metadata["prompt_lengths"] == [5, 8] and metadata["token_count"] == 13
    assert metadata["prompt_sha256"] == "e" * 64
    assert metadata["calibration_profile"]["recipe_version"] == 3
    metadata["capture_manifest"]["prompt_lengths"][0] = 999
    assert artifact["capture_manifest"]["prompt_lengths"] == [5, 8]
    no_capture = {k: v for k, v in artifact.items() if k != "capture_manifest"}
    missing = baseline.capture_source_metadata(no_capture)
    assert missing["capture_manifest"] is None and not missing["capture_manifest_available"]
    assert missing["prompt_sha256"] == "e" * 64 and missing["token_count"] == 13
    assert "prompt_lengths" not in missing


@pytest.mark.parametrize("fault", ["tokens", "hash", "lengths", "profile"])
def test_capture_provenance_cannot_claim_different_recipe3_training_rows(case, fault):
    artifact = copy.deepcopy(case[-1])
    capture = artifact["capture_manifest"]
    if fault == "tokens": capture["token_count"] = 14
    elif fault == "hash": capture["prompt_sha256"] = "f" * 64
    elif fault == "lengths": capture["prompt_lengths"] = [3, 4]
    else: capture["calibration_profile"] = {"recipe_version": 99}
    with pytest.raises(ValueError, match="capture|prompt|provenance"):
        baseline.capture_source_metadata(artifact)
