"""Archive #86/#97/#165/#170/#174/#176: historical fit reuse is not publication.

Real CPU fixture checkpoints exercise authentication without NPU or replay in
the reader. Deliberately corrupt inputs must never become a new fitting seed.
"""
import builtins
import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from oscar_ascend.optr_artifacts import file_sha256
from oscar_ascend.optr_checkpoint import FitCheckpoints
from tools import optr_recipe3_source as source
from tools import prepare_optr_rotations as prepare
from test_optr_recipe3_pipeline import case, capture, proposal, forbidden
from test_official_rotation_calibration import NAME


@pytest.fixture
def completed(case, monkeypatch):
    from oscar_ascend import optr_checkpoint
    from tools import prepare_optr_recipe3
    case["config"].pop("optr_recipe3_repair", None)
    with monkeypatch.context() as historical:
        for module in (optr_checkpoint, prepare_optr_recipe3):
            historical.setattr(module, "fitting_implementation_digest", lambda: source.LEGACY_FITTING_SHA256)
        with pytest.raises(ValueError, match="no strict validation improvement"):
            prepare.prepare(case["config"], output=case["output"], testing=True,
                            capture_runner=capture(case), fitter=proposal(case, unchanged=True), base_runner=forbidden)
    case["generation"] = next((case["output"].parent / "fit-checkpoints").iterdir())
    case["trace"] = case["output"].parent / "native-traces" / "capture-manifest.json"
    case["config"]["optr_recipe3_repair"] = "single_head_trust_region_v1"
    return case


def discover(case, *, testing=True):
    return source.discover_source(case["config"], case["trace"],
        {"base_artifact_sha256": file_sha256(case["base_config"]["rotation_artifact"])}, [NAME], 2, 8, testing)


def load(spec, case):
    return source.load_source_layer(spec, NAME, case["baseline"]["layers"][NAME], 2, spec["policy"])


def edit_checkpoint(path, mutate):
    manifest = json.loads(path.read_text())
    payload = FitCheckpoints._load_from(path, manifest["key"], manifest["identity"])
    mutate(payload)
    FitCheckpoints(path.parent.parent, manifest["identity"]).save(manifest["key"], payload)


def test_source_reuses_exact_rejected_work_without_fit_replay_npu_or_trace_tensor_io(completed, monkeypatch):
    from oscar_ascend import optr_calibration, optr_recipe3_fit, optr_sequential, optr_traces
    for module, key in ((prepare, "_metrics"), (prepare, "_capture"), (prepare, "refine_layer"),
                        (optr_calibration, "replay_sample"), (optr_recipe3_fit, "fit_perhead"),
                        (optr_sequential, "fitting_device"), (optr_traces, "load_layer_samples"),
                        (optr_traces, "load_trace_manifest")):
        monkeypatch.setattr(module, key, forbidden)
    original_import, original_load = builtins.__import__, torch.load
    def guarded_import(name, *args, **kwargs):
        assert name != "torch_npu" and not name.startswith("torch_npu.")
        return original_import(name, *args, **kwargs)
    def guarded_load(path, *args, **kwargs):
        assert "native-traces" not in str(path)
        assert kwargs.get("map_location") == "cpu" and kwargs.get("weights_only") is True
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(torch, "load", guarded_load)
    before = {p: p.read_bytes() for p in completed["output"].parent.rglob("*") if p.is_file()}
    spec = json.loads(json.dumps(discover(completed)))
    result, record, provenance = load(spec, completed)
    assert spec["policy"]["policy_version"] == source.LEGACY_POLICY
    assert "recipe3_repair" not in spec["policy"]
    assert record["accepted"] is False and len(result["candidates"]) == 1
    assert provenance["fitting_implementation_sha256"] == source.LEGACY_FITTING_SHA256
    assert provenance["independent_test_evidence"] == "unchanged_recipe3_seed_only_not_rejected_proposals"
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not completed["output"].exists()


def test_source_absence_is_optional_but_partial_current_generation_is_not(completed):
    generation = completed["generation"]
    saved = generation.with_name("unrelated")
    generation.rename(saved)
    # A matching off-path generation is ambiguous, not optional.
    with pytest.raises(ValueError, match="ambiguous|noncanonical"):
        discover(completed)
    shutil.rmtree(saved)
    assert discover(completed) is None
    generation.mkdir()
    with pytest.raises(ValueError, match="coverage"):
        discover(completed)


@pytest.mark.parametrize("fault", ["bytes", "missing_layer", "missing_proposal", "extra_layer", "duplicate", "linked"])
def test_source_refuses_corrupt_incomplete_or_ambiguous_generations(completed, fault):
    generation = completed["generation"]
    path = generation / "layer-3.json"
    if fault == "bytes":
        manifest = json.loads(path.read_text())
        with (generation / manifest["file"]).open("ab") as stream:
            stream.write(b"tampered")
    elif fault == "missing_layer":
        path.unlink()
    elif fault == "missing_proposal":
        (generation / "layer-3-shared.json").unlink()
    elif fault == "extra_layer":
        shutil.copyfile(path, generation / "layer-7.json")
    elif fault == "duplicate":
        shutil.copytree(generation, generation.with_name("copied"))
    else:
        original = path.with_suffix(".original")
        path.rename(original)
        path.symlink_to(original)
    with pytest.raises(ValueError):
        discover(completed)


@pytest.mark.parametrize("field", ["Rk", "Rv", "k_mean", "layer_id"])
def test_reader_binds_real_recipe3_seed_not_just_cached_self_assertion(completed, field):
    spec = discover(completed)
    seed = copy.deepcopy(completed["baseline"]["layers"][NAME])
    if field == "layer_id":
        seed[field] = 7
    else:
        seed[field].reshape(-1)[0] += .01
    with pytest.raises(ValueError, match="actual recipe3 seed"):
        source.load_source_layer(spec, NAME, seed, 2, spec["policy"])


@pytest.mark.parametrize("fault", ["policy", "base", "implementation", "candidate_hash", "candidate_accept",
                                  "test_overlap", "test_changed", "selected_changed", "proposal_metrics"])
def test_authentic_hash_alone_cannot_bless_invalid_saved_semantics(completed, fault):
    path = completed["generation"] / "layer-3.json"
    def corrupt(payload):
        artifact = payload["artifact"]
        cert = artifact["refinement_certificate"]
        record = cert["layers"][NAME]
        selection = record["shared_layer_selection"]
        if fault == "policy":
            cert["policy"]["lr"] = .03
        elif fault == "base":
            cert["base_artifact_sha256"] = "f" * 64
        elif fault == "implementation":
            cert["fitting_implementation_sha256"] = "f" * 64
        elif fault == "candidate_hash":
            selection["evaluated_candidates"][0]["constants_sha256"] = "f" * 64
        elif fault == "candidate_accept":
            selection["evaluated_candidates"][0]["accepted"] = True
        elif fault == "test_overlap":
            record["test"] = copy.deepcopy(record["validation"])
        elif fault == "test_changed":
            record["test"]["selected"] = copy.deepcopy(record["validation"]["baseline"])
        elif fault == "selected_changed":
            record["accepted"] = True
        else:
            selection["proposal"] = None
    edit_checkpoint(path, corrupt)
    with pytest.raises(ValueError):
        discover(completed)


@pytest.mark.parametrize("key", ["layer-3", "layer-3-shared"])
def test_manifest_snapshot_prevents_replacement_after_discovery(completed, key):
    spec = discover(completed)
    edit_checkpoint(completed["generation"] / (key + ".json"), lambda payload: None)
    with pytest.raises(ValueError, match="manifest changed"):
        load(spec, completed)


def test_legitimate_source_pointer_replacement_cannot_reuse_old_v11_generation(completed, tmp_path):
    first = discover(completed)
    first_hash = source.source_snapshot_digest(first)
    # Reordering JSON keys is not a change to the authenticated snapshot.
    assert source.source_snapshot_digest(dict(reversed(list(first.items())))) == first_hash
    identity = {"profile": "new-v11-profile", "source_fit_snapshot_sha256": first_hash}
    previous = FitCheckpoints(tmp_path / "new-fit", identity)
    previous.save("layer-3", {"marker": "old completed v11 work"})
    edit_checkpoint(completed["generation"] / "layer-3-shared.json", lambda payload: None)
    second = discover(completed)
    second_hash = source.source_snapshot_digest(second)
    assert second_hash != first_hash
    current = FitCheckpoints(tmp_path / "new-fit", {**identity, "source_fit_snapshot_sha256": second_hash})
    assert current.root != previous.root and current.load("layer-3") is None
    assert previous.load("layer-3") == {"marker": "old completed v11 work"}


def test_trace_changes_and_foreign_policy_cannot_reuse_discovered_work(completed):
    spec = discover(completed)
    with pytest.raises(ValueError, match="specification"):
        source.load_source_layer(spec, NAME, completed["baseline"]["layers"][NAME], 2,
                                 {**spec["policy"], "lr": .03})
    completed["trace"].write_text(completed["trace"].read_text() + "\n")
    with pytest.raises(ValueError, match="trace manifest hash"):
        load(spec, completed)


def test_test_only_capture_requires_explicit_reader_permission(completed):
    with pytest.raises(ValueError, match="test-only"):
        discover(completed, testing=False)


def test_wrong_model_coverage_is_rejected_before_using_a_cache(completed):
    with pytest.raises(ValueError, match="actual model FULL"):
        source.discover_source(completed["config"], completed["trace"],
            {"base_artifact_sha256": file_sha256(completed["base_config"]["rotation_artifact"])}, [NAME, "model.layers.7.self_attn.attn"], 2, 8, True)


def test_production_trace_schema_can_omit_test_only_field(completed):
    # Test-only CPU fixture permission is kept explicit in the checkpoint;
    # production top-level traces legitimately omit the optional flag.
    spec = discover(completed)
    trace = json.loads(completed["trace"].read_text())
    trace.pop("test_only")
    completed["trace"].write_text(json.dumps(trace))
    identity = {**spec["identity"], "trace_manifest_sha256": file_sha256(completed["trace"])}
    assert source._trace(completed["trace"], spec["source_profile"], identity) == trace


def test_saved_false_flag_cannot_hide_a_candidate_that_passes_every_strict_gate(completed):
    from oscar_ascend.optr_artifacts import matrix_digest
    seed = completed["baseline"]["layers"][NAME]
    candidate = {"id": "k_step_1", "phase": "k", "step": 1,
                 "Rk": completed["candidate"]["Rk"], "Rv": seed["Rv"], "proxy_losses": [1., 1.]}
    proposal_path = completed["generation"] / "layer-3-shared.json"
    def add_candidate(payload):
        payload["result"]["candidates"].append(candidate)
        payload["result"]["decision"]["candidate_count"] = 2
    edit_checkpoint(proposal_path, add_candidate)
    def lower_errors(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("squared_error", "normalized_error"):
                    value[key] = item * .5
                else:
                    lower_errors(item)
        elif isinstance(value, list):
            for item in value:
                lower_errors(item)
    def add_false_rejection(payload):
        record = payload["artifact"]["refinement_certificate"]["layers"][NAME]
        record["shared_decision"]["candidate_count"] = 2
        measured = copy.deepcopy(record["validation"]["baseline"])
        lower_errors(measured)
        record["shared_layer_selection"]["evaluated_candidates"].append({
            "candidate": candidate["id"], "phase": candidate["phase"], "step": 1,
            "constants_sha256": matrix_digest({NAME: {**seed, "Rk": candidate["Rk"]}}),
            "accepted": False, "validation": measured})
    edit_checkpoint(completed["generation"] / "layer-3.json", add_false_rejection)
    with pytest.raises(ValueError, match="contradict.*strict gate"):
        discover(completed)


@pytest.mark.parametrize("fault", ["queries", "projection", "frames", "identity", "nan"])
def test_malformed_saved_measurement_cannot_count_as_a_rejected_candidate(completed, fault):
    def corrupt(payload):
        record = payload["artifact"]["refinement_certificate"]["layers"][NAME]
        # Break only the candidate evidence. Baseline/selected records remain
        # valid, so this specifically exercises the reusable candidate gate.
        metric = copy.deepcopy(record["shared_layer_selection"]["evaluated_candidates"][0]["validation"])
        record["shared_layer_selection"]["evaluated_candidates"][0]["validation"] = metric
        row = metric["requests"][0]
        if fault == "queries":
            row["frames"][0].pop("queries")
        elif fault == "projection":
            metric.pop("projection_checks")
        elif fault == "frames":
            row.pop("frames")
        elif fault == "identity":
            row["sample_sha256"] = "f" * 64
            for view in metric["projection_checks"].values():
                view["requests"][0]["sample_sha256"] = "f" * 64
        else:
            metric["normalized_error"] = float("nan")
    edit_checkpoint(completed["generation"] / "layer-3.json", corrupt)
    with pytest.raises(ValueError):
        discover(completed)
