"""Archive #86/#97/#170/#172/#174/#175/#176: explain authenticated rejected work.

CPU fixtures reproduce the saved-checkpoint shape, never claim NPU evidence.
The reporting pass must not recapture, fit, replay, or alter frozen gates.
"""
import builtins
import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from oscar_ascend.optr_artifacts import file_sha256, matrix_digest
from oscar_ascend.optr_checkpoint import FitCheckpoints, fitting_implementation_digest
from tools import optr_rejection_report as report
from tools import prepare_optr_rotations as prepare
from test_optr_recipe3_pipeline import case, capture, proposal, forbidden
from test_official_rotation_calibration import NAME


def metric(errors=(1., 1.)):
    frames = []
    for index, error in enumerate(errors):
        query = {"position": index, "squared_error": error,
                 "teacher_energy": 4., "normalized_error": error / 4.}
        frames.append({"frame_id": index, "context": index, "current_end": index + 1,
                       "query_positions": [index], "queries": [copy.deepcopy(query)],
                       "terminal_query": copy.deepcopy(query),
                       **{key: query[key] for key in ("squared_error", "teacher_energy", "normalized_error")}})
    row = {"request_id": "validation_4096_0:single_split", "sample_sha256": "a" * 64,
           "partition": "single_split", "frames": frames, "squared_error": sum(errors),
           "teacher_energy": 8., "normalized_error": sum(errors) / 8.}
    result = {**{key: row[key] for key in ("squared_error", "teacher_energy", "normalized_error")},
              "requests": [row]}
    result["projection_checks"] = {view: copy.deepcopy(result) for view in ("bf16_output", "bf16_rank_output")}
    return result


@pytest.mark.parametrize("kind,expected", [
    ("regression", "aggregate_error_regression"),
    ("equal", "no_strict_total_improvement"),
    ("fine", "fine_grained_regression"),
    ("pass", "passed"),
])
def test_exact_saved_gate_is_explained_without_relaxing_it(kind, expected):
    before = metric()
    after = metric({"regression": (2., 2.), "equal": (1., 1.),
                    "fine": (1.1, .5), "pass": (.5, .5)}[kind])
    value = report.metric_rejections(before, after)
    assert value["reason"] == expected
    assert value["accepted"] is (kind == "pass")
    if kind == "fine":
        assert value["strict_total_improvement"] and not value["nonworse"]
        assert value["regressed_checks"] == dict(total=0, request=0, frame=3, query=3, terminal=3)
        assert value["regressed_frame_stages"]["first_prefill"]["query"] == 3
        assert value["regressed_frame_stages"]["cached_context"]["query"] == 0
        assert value["top_failures"][0]["normalized_error"]["delta"] > 0


@pytest.mark.parametrize("view", ["bf16_output", "bf16_rank_output"])
@pytest.mark.parametrize("errors,reason", [((.5, 1.1), "fine_grained_regression"),
                                         ((2., 2.), "projection_total_regression")])
def test_bf16_only_failure_does_not_contaminate_fp32_diagnostic(view, errors, reason):
    before, after = metric(), metric((.5, .5))
    changed = metric(errors)
    changed.pop("projection_checks")
    after["projection_checks"][view] = changed
    value = report.metric_rejections(before, after)
    assert not value["accepted"] and value["bf16_only_regression"]
    assert value["views"]["fp32"]["nonworse"] is True
    assert value["views"][view]["nonworse"] is False
    assert value["reason"] == reason
    assert value["regressed_frame_stages"]["cached_context"]["query"] > 0


def test_zero_baseline_reports_absolute_increase_without_dividing_by_zero():
    value = report.metric_rejections(metric((0., 1.)), metric((.1, .5)))
    assert value["reason"] == "fine_grained_regression"
    gate = next(row for row in value["top_failures"] if row["scope"] == "query")
    assert gate["squared_error"]["positive_from_zero"] is True
    assert gate["squared_error"]["relative_delta"] is None


def test_changed_geometry_or_inconsistent_metrics_cannot_be_described_as_rejection():
    before, after = metric(), metric((.5, .5))
    after["requests"][0]["sample_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identity|coverage"):
        report.metric_rejections(before, after)
    after = metric((.5, .5))
    after["squared_error"] += 1
    with pytest.raises(ValueError, match="totals"):
        report.metric_rejections(before, after)


@pytest.fixture
def completed(case):
    with pytest.raises(ValueError, match="no strict validation improvement"):
        prepare.prepare(case["config"], output=case["output"], testing=True,
                        capture_runner=capture(case), fitter=proposal(case, unchanged=True), base_runner=forbidden)
    assert not case["output"].exists()
    case["generation"] = next((case["output"].parent / "fit-checkpoints").iterdir())
    return case


def load_checkpoint(path):
    manifest = json.loads(path.read_text())
    payload = FitCheckpoints._load_from(path, manifest["key"], manifest["identity"])
    return manifest, payload


def save_checkpoint(path, payload):
    manifest = json.loads(path.read_text())
    store = FitCheckpoints(path.parent.parent, manifest["identity"])
    assert store.root == path.parent
    store.save(manifest["key"], payload)


def test_completed_work_yields_report_without_any_capture_fit_replay_or_trace_tensor_read(completed, monkeypatch):
    from oscar_ascend import optr_calibration, optr_sequential, optr_traces, optr_recipe3_fit
    for module, name in ((prepare, "_metrics"), (prepare, "_capture"), (prepare, "refine_layer"),
                         (optr_calibration, "replay_sample"), (optr_sequential, "fitting_device"),
                         (optr_traces, "load_layer_samples"), (optr_traces, "load_trace_manifest"),
                         (optr_recipe3_fit, "fit_perhead")):
        monkeypatch.setattr(module, name, forbidden)
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
    before = {path: path.read_bytes() for path in completed["output"].parent.rglob("*") if path.is_file()}
    code_digest = fitting_implementation_digest()
    summary = report.report_no_improvement(completed["config"], output=completed["output"], testing=True)
    assert summary["layer_count"] == 1 and summary["candidate_count"] == 1
    assert not summary["recomputed_fit_or_replay"] and not summary["artifact_published"]
    assert summary["layers"][0]["reason"] == "no_strict_total_improvement"
    assert summary["layers"][0]["ranking_scope"] == "unchanged_only"
    saved = json.loads(Path(summary["report"]).read_text())
    assert saved["layers"][0]["independent_test"] == "baseline_compared_with_itself_no_proposal_tested"
    assert saved["model_quality"] == "not_measured"
    assert all(path.read_bytes() == data for path, data in before.items())
    assert fitting_implementation_digest() == code_digest
    assert not completed["output"].exists()


def test_diagnostic_refuses_test_artifacts_without_explicit_permission(completed):
    with pytest.raises(ValueError, match="test-only"):
        report.report_no_improvement(completed["config"], output=completed["output"])


@pytest.mark.parametrize("fault", ["bytes", "profile", "base", "trace", "implementation",
                                  "missing", "proposal_missing", "extra", "ambiguous"])
def test_report_rejects_damaged_incomplete_foreign_or_ambiguous_work(completed, monkeypatch, fault):
    from oscar_ascend import optr_checkpoint
    generation = completed["generation"]
    path = generation / "layer-3.json"
    if fault == "bytes":
        manifest = json.loads(path.read_text())
        with (generation / manifest["file"]).open("ab") as stream:
            stream.write(b"corruption")
    elif fault == "profile":
        completed["config"]["optr_calibration"]["steps"] = 2
    elif fault == "base":
        with Path(completed["base_config"]["rotation_artifact"]).open("ab") as stream:
            stream.write(b"changed baseline")
    elif fault == "trace":
        trace = completed["output"].parent / "native-traces" / "capture-manifest.json"
        trace.write_text(trace.read_text() + "\n")
    elif fault == "implementation":
        monkeypatch.setattr(optr_checkpoint, "fitting_implementation_digest", lambda: "a" * 64)
    elif fault == "missing":
        path.unlink()
    elif fault == "proposal_missing":
        (generation / "layer-3-shared.json").unlink()
    elif fault == "extra":
        (generation / "layer-7.json").write_bytes(path.read_bytes())
    elif fault == "ambiguous":
        shutil.copytree(generation, generation.parent / ("f" * 64))
    with pytest.raises((ValueError, FileNotFoundError)):
        report.report_no_improvement(completed["config"], output=completed["output"], testing=True)
    assert not (completed["output"].parent / "no-improvement-report.json").exists()


def test_diagnostic_rejects_unbound_candidate_metrics_even_with_valid_checkpoint_hash(completed):
    path = completed["generation"] / "layer-3.json"
    _, payload = load_checkpoint(path)
    record = payload["artifact"]["refinement_certificate"]["layers"][NAME]
    record["shared_layer_selection"]["evaluated_candidates"][0]["constants_sha256"] = "f" * 64
    save_checkpoint(path, payload)
    with pytest.raises(ValueError, match="saved candidate matrices"):
        report.report_no_improvement(completed["config"], output=completed["output"], testing=True)


def test_production_capture_schema_omits_test_only_while_identity_follows_prepare_testing(case):
    def production_schema(config_path, directory):
        result = capture(case)(config_path, directory)
        result.pop("test_only")
        return result
    with pytest.raises(ValueError, match="no strict validation improvement"):
        prepare.prepare(case["config"], output=case["output"], testing=True,
                        capture_runner=production_schema, fitter=proposal(case, unchanged=True), base_runner=forbidden)
    value = report.report_no_improvement(case["config"], output=case["output"], testing=True)
    saved = json.loads(Path(value["report"]).read_text())
    assert saved["identity"]["test_only"] is True
    assert value["layer_count"] == 1


def test_default_report_accepts_production_capture_without_test_only_key(completed, monkeypatch):
    """Exercise production metadata shape, without claiming CPU fixtures are NPU data."""
    from tools import prepare_optr_recipe3_base as base
    # Baseline production permission has an independent real-pair rejection
    # test above. Only this schema-shape test injects the already-read fixture.
    calls = []
    def permitted_baseline(_config, *, testing):
        calls.append(testing)
        return completed["baseline"]
    monkeypatch.setattr(base, "load_base", permitted_baseline)
    trace_path = completed["output"].parent / "native-traces" / "capture-manifest.json"
    trace = json.loads(trace_path.read_text())
    trace.pop("test_only")
    trace_path.write_text(json.dumps(trace))
    for key in ("layer-3", "layer-3-shared"):
        path = completed["generation"] / (key + ".json")
        manifest, payload = load_checkpoint(path)
        identity = {**manifest["identity"], "test_only": False,
                    "trace_manifest_sha256": file_sha256(trace_path)}
        if key == "layer-3":
            payload["artifact"]["test_only"] = False
            payload["artifact"]["refinement_certificate"]["trace_manifest_sha256"] = identity["trace_manifest_sha256"]
        FitCheckpoints(path.parent.parent, identity).save(key, payload)
    value = report.report_no_improvement(completed["config"], output=completed["output"])
    assert value["layer_count"] == 1 and calls == [False]
    assert json.loads(Path(value["report"]).read_text())["identity"]["test_only"] is False


def test_saved_candidate_accepted_flag_is_recomputed_from_frozen_gate(completed):
    path = completed["generation"] / "layer-3.json"
    _, payload = load_checkpoint(path)
    record = payload["artifact"]["refinement_certificate"]["layers"][NAME]
    record["shared_layer_selection"]["evaluated_candidates"][0]["accepted"] = True
    save_checkpoint(path, payload)
    with pytest.raises(ValueError, match="unchanged gate"):
        report.report_no_improvement(completed["config"], output=completed["output"], testing=True)


def test_best_changed_candidate_is_descriptive_and_never_overrides_rejection(completed):
    seed = completed["baseline"]["layers"][NAME]
    generation = completed["generation"]
    proposal_path = generation / "layer-3-shared.json"
    _, proposed = load_checkpoint(proposal_path)
    result = proposed["result"]
    changed = {"id": "k_step_1", "phase": "k", "step": 1,
               "Rk": -seed["Rk"], "Rv": seed["Rv"], "proxy_losses": [1., 1.]}
    result["candidates"].append(changed)
    result["decision"]["candidate_count"] = 2
    save_checkpoint(proposal_path, proposed)
    layer_path = generation / "layer-3.json"
    _, payload = load_checkpoint(layer_path)
    record = payload["artifact"]["refinement_certificate"]["layers"][NAME]
    record["shared_decision"] = copy.deepcopy(result["decision"])
    before = metric()
    # Replace both incumbent split sides consistently: these synthetic scalar
    # metrics exercise reporting only, not fitting or publication evidence.
    record["validation"] = {"baseline": before, "selected": copy.deepcopy(before)}
    selection = record["shared_layer_selection"]
    selection["proposal"] = copy.deepcopy(before)
    selection["evaluated_candidates"][0]["validation"] = copy.deepcopy(before)
    selection["evaluated_candidates"].append({"candidate": "k_step_1", "phase": "k", "step": 1,
        "constants_sha256": matrix_digest({NAME: {**seed, "Rk": changed["Rk"]}}),
        "accepted": False, "validation": metric((1.1, .5))})
    save_checkpoint(layer_path, payload)
    value = report.report_no_improvement(completed["config"], output=completed["output"], testing=True)
    row = value["layers"][0]
    assert row["best_total_candidate"] == "k_step_1" and row["total_relative_delta"] < 0
    assert row["reason"] == "fine_grained_regression" and row["regressed_queries"] == 3
    assert not value["artifact_published"] and not completed["output"].exists()
