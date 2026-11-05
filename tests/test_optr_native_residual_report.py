"""Archive #176/#177: v12 rejection reporting authenticates saved evidence.

CPU synthetic traces only. Reports may reconstruct small saved rotations for
authentication, but must not recapture, retrain or replay attention. Training
passes are reported separately and never become validation/model acceptance.
"""
from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from oscar_ascend import optr_calibration, optr_native_residual as native, optr_traces
from tools import optr_parallel, optr_rejection_report as report, prepare_rotations
from tools import prepare_optr_rotations as prepare
from tools.optr_rejection_summary import summarize_saved_report
import test_optr_recipe3_pipeline as old_pipeline
from test_optr_recipe3_pipeline import NAME, case, capture, forbidden
from test_optr_native_residual_pipeline import (
    _config, _prepare, _prescribe_teacher_and_direction, _read_checkpoint,
)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=["no_train_finalist", "train_passes_validation_rejects"])
def completed_native(case, monkeypatch, request):
    if request.param == "no_train_finalist":
        case["candidate"] = copy.deepcopy(case["baseline"]["layers"][NAME])
        collect = capture(case)
    else:
        _prescribe_teacher_and_direction(case, monkeypatch)
        original_replay = old_pipeline.replay_sample

        def validation_teacher(rk, rv, mean, sample, policy):
            # Deliberately prescribe a conflicting validation teacher while
            # keeping a real replay. No metric or strict acceptance is mocked.
            if sample["request_id"].startswith("validation_"):
                head = next(head for head in range(2) if torch.equal(rk, case["candidate"]["Rk"][head]))
                seed = case["baseline"]["layers"][NAME]
                rk, rv = seed["Rk"][head], seed["Rv"][head]
            return original_replay(rk, rv, mean, sample, policy)

        def collect(config_path, directory):
            with monkeypatch.context() as context:
                context.setattr(old_pipeline, "replay_sample", validation_teacher)
                return capture(case)(config_path, directory)

    with pytest.raises(ValueError, match="no strict validation improvement"):
        _prepare(case, capture_runner=collect)
    return {**case, "config": _config(case), "kind": request.param}


def _forbid_recomputation(monkeypatch, *, allow_trace_manifest=False):
    for module, name in ((native, "select_native_residual"), (native, "_gradient_direction"),
                         (optr_calibration, "replay_sample"), (optr_calibration, "_replay"),
                         (optr_traces, "load_layer_samples"),
                         (prepare, "_metrics"), (prepare, "_capture")):
        monkeypatch.setattr(module, name, forbidden)
    if not allow_trace_manifest:
        monkeypatch.setattr(optr_traces, "load_trace_manifest", forbidden)
    original_import, original_load = builtins.__import__, torch.load

    def guarded_import(name, *args, **kwargs):
        assert name != "torch_npu" and not name.startswith("torch_npu.")
        return original_import(name, *args, **kwargs)

    def guarded_load(path, *args, **kwargs):
        assert "native-traces" not in str(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(torch, "load", guarded_load)


def test_report_separates_train_acceptance_from_validation_and_handles_zero_finalists(completed_native, monkeypatch):
    case = completed_native
    _forbid_recomputation(monkeypatch)
    before = {path: path.read_bytes() for path in case["output"].parent.rglob("*") if path.is_file()}
    summary = report.report_no_improvement(case["config"], output=case["output"], testing=True)
    layer = summary["layers"][0]
    assert summary["recomputed_fit_or_replay"] is False and summary["artifact_published"] is False
    assert layer["train_candidates"] == summary["candidate_count"] == 8
    assert layer["strategy"] == "recipe3_native_residual_train_then_validation_v1"
    if case["kind"] == "no_train_finalist":
        assert layer["train_search"]["accepted_count"] == 0
        assert layer["validation_candidates"] == summary["validation_candidate_count"] == 0
        assert layer["validation_search"]["best_total"] is None
        assert layer["reason"] == "no_train_candidate_passed"
    else:
        assert layer["train_search"]["accepted_count"] >= 1
        assert layer["validation_candidates"] == summary["validation_candidate_count"] == 1
        assert layer["validation_search"]["accepted_count"] == 0
        assert layer["train_search"]["best_total"]["reason"] == "passed"
        assert layer["validation_search"]["best_total"]["reason"] == "aggregate_error_regression"
        assert layer["reason"] == "all_validation_finalists_rejected"
    saved = json.loads(Path(summary["report"]).read_text())
    detail = saved["layers"][0]
    assert all(row["phase"] == "native_train" for row in detail["candidates"])
    assert all(row["phase"] == "native_validation" for row in detail["validation_candidates"])
    assert not any(row["accepted"] for row in detail["validation_candidates"])
    assert detail["independent_test"] == "baseline_compared_with_itself_no_proposal_tested"
    assert summarize_saved_report(summary["report"])["layers"] == summary["layers"]
    assert all(path.read_bytes() == value for path, value in before.items())
    assert not case["output"].exists()


@pytest.mark.parametrize("corrupt", ["lock_file_hash", "binding", "tangent", "completed_decision"])
def test_report_rejects_tampered_locks_and_completed_records(completed_native, monkeypatch, corrupt):
    case = completed_native
    pointer, manifest, cache, saved = _read_checkpoint(case["output"], "layer-3-shared")
    if corrupt == "lock_file_hash":
        payload = pointer.parent / manifest["file"]
        payload.write_bytes(payload.read_bytes() + b"explicit corruption")
    elif corrupt == "completed_decision":
        _, _, completed_cache, completed = _read_checkpoint(case["output"], "layer-3")
        decision = completed["artifact"]["refinement_certificate"]["layers"][NAME]["native_residual_selection"]
        decision["evaluated_train_candidates"][0]["accepted"] = not decision["evaluated_train_candidates"][0]["accepted"]
        completed_cache.save("layer-3", completed)
    else:
        if corrupt == "binding":
            saved["native_residual_binding"]["base_constants_sha256"] = "f" * 64
        else:
            saved["proposal"]["directions"][0]["unit_tangent"][0, 1] += 1.
        cache.save("layer-3-shared", saved)
    _forbid_recomputation(monkeypatch)
    with pytest.raises(ValueError):
        report.report_no_improvement(case["config"], output=case["output"], testing=True)
    assert not (case["output"].parent / "no-improvement-report.json").exists()


def test_fixed_prepare_reports_completed_v12_rejections_without_workers_or_reselection(completed_native, monkeypatch, capsys):
    case = completed_native
    real_prepare = prepare.prepare

    def resume(config, **kwargs):
        return real_prepare(config, **kwargs, parallel_testing=True)

    # Preparation authenticates the existing JSON manifests before discovering
    # completed layers; that is neither capture nor reading trace tensors.
    _forbid_recomputation(monkeypatch, allow_trace_manifest=True)
    monkeypatch.setattr(prepare, "prepare", resume)
    monkeypatch.setattr(prepare, "refine_layer", forbidden)
    monkeypatch.setattr(optr_parallel.mp, "get_context", forbidden)
    capsys.readouterr()
    with pytest.raises(ValueError, match="no strict validation improvement"):
        prepare_rotations.prepare(case["config"], output=case["output"], testing=True, capture_runner=forbidden)
    events = [json.loads(line.split("OPTR_PROGRESS ", 1)[1])
              for line in capsys.readouterr().out.splitlines() if line.startswith("[oscar] OPTR_PROGRESS ")]
    assert next(row for row in events if row.get("phase") == "parallel_fit_plan")["workers"] == 0
    layers = [row for row in events if row.get("phase") == "no_improvement_layer"]
    assert len(layers) == 1 and layers[0]["train_candidates"] == 8
    assert any(row.get("phase") == "no_improvement_report" for row in events)
    assert not any(row.get("phase") == "no_improvement_report_failed" for row in events)


def test_saved_native_report_cli_remains_stdlib_only_compact_and_read_only(completed_native):
    case = completed_native
    result = report.report_no_improvement(case["config"], output=case["output"], testing=True)
    path = Path(result["report"])
    original = path.read_bytes()
    value = json.loads(original)
    layer = value["layers"][0]
    value["layers"] = [{**copy.deepcopy(layer), "layer": f"model.layers.{index * 4 + 3}.self_attn.attn"}
                       for index in range(16)]
    value["identity"]["full_layers"] = [row["layer"] for row in value["layers"]]
    standalone = path.parent / "standalone-native-summary.json"
    standalone.write_text(json.dumps(value))
    before = {file: file.read_bytes() for file in path.parent.rglob("*") if file.is_file()}
    script = '''
import runpy, sys
class ForbidModelImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'oscar_ascend'):
            raise AssertionError('saved-report must use only JSON and stdlib: ' + fullname)
sys.meta_path.insert(0, ForbidModelImports())
sys.argv = ['optr_rejection_report', '--saved-report', sys.argv[1]]
runpy.run_module('tools.optr_rejection_report', run_name='__main__')
'''
    completed = subprocess.run([sys.executable, "-S", "-c", script, str(standalone)],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line.split("OPTR_PROGRESS ", 1)[1]) for line in completed.stdout.splitlines()]
    assert len(rows) == 17 and len(completed.stdout.encode()) < 50 * 1024
    assert all(row["train_search"]["candidate_count"] == 8 for row in rows[:-1])
    assert rows[-1]["evidence_scope"] == "saved_json_only_checkpoints_not_reauthenticated"
    assert path.read_bytes() == original
    assert {file: file.read_bytes() for file in path.parent.rglob("*") if file.is_file()} == before


@pytest.mark.parametrize("fault", ["validation_phase", "missing_finalist", "accepted_validation", "train_acceptance"])
def test_saved_native_summary_does_not_merge_or_relabel_training_and_validation(completed_native, fault):
    case = completed_native
    result = report.report_no_improvement(case["config"], output=case["output"], testing=True)
    path = Path(result["report"])
    value = json.loads(path.read_text())
    layer = value["layers"][0]
    if fault == "train_acceptance":
        layer["candidates"][0]["accepted"] = not layer["candidates"][0]["accepted"]
    elif layer["validation_candidates"]:
        if fault == "validation_phase":
            layer["validation_candidates"][0]["phase"] = "native_train"
        elif fault == "accepted_validation":
            layer["validation_candidates"][0]["accepted"] = True
        else:
            layer["validation_candidates"] = []
            layer["validation_candidate_count"] = 0
    else:
        # Empty validation cannot be relabelled as one completed check.
        layer["validation_candidate_count"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        summarize_saved_report(path)
