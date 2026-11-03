"""Archive #176: explain a completed rejected fit without repeating its work.

The pair, capture provenance, checkpoint hashes and report authentication are
real CPU code. Synthetic traces/proposals make no model-quality/NPU claim.
"""
import json

import pytest

from oscar_ascend import optr_sequential, optr_traces
from tools import optr_parallel, prepare_optr_recipe3, prepare_rotations
from tools import prepare_optr_rotations
from tools.phase import _compact_should_mirror
from test_optr_recipe3_pipeline import case, capture, forbidden, proposal


MESSAGE = "recipe3 OptR found no strict validation improvement; no refined PT is published"


def test_fixed_preparer_reports_authenticated_completed_work_without_any_recomputation(
        case, monkeypatch, capsys):
    with pytest.raises(ValueError, match="no strict validation improvement"):
        prepare_optr_rotations.prepare(case["config"], output=case["output"], testing=True,
            capture_runner=capture(case), fitter=proposal(case, unchanged=True), base_runner=forbidden)
    assert case["calls"] == ["capture", "fit"]
    assert not case["output"].exists()
    capsys.readouterr()

    # The fixed entry does not expose this test-only switch. Exercise its real
    # dispatch and exception wrapper while selecting the production parallel
    # branch against the completed, authenticated CPU fixture.
    real_prepare = prepare_optr_rotations.prepare
    original_errors = []

    def resume(config, **kwargs):
        assert kwargs["testing"] is True
        try:
            return real_prepare(config, **kwargs, parallel_testing=True)
        except ValueError as error:
            original_errors.append(error)
            raise

    monkeypatch.setattr(prepare_optr_rotations, "prepare", resume)
    monkeypatch.setattr(prepare_optr_rotations, "refine_layer", forbidden)
    monkeypatch.setattr(prepare_optr_recipe3, "refine_layer", forbidden)
    monkeypatch.setattr(prepare_optr_rotations, "_metrics", forbidden)
    monkeypatch.setattr(optr_traces, "load_layer_samples", forbidden)
    monkeypatch.setattr(optr_traces, "aggregate_layer_error", forbidden)
    monkeypatch.setattr(optr_sequential, "fitting_device", forbidden)
    monkeypatch.setattr(optr_parallel.mp, "get_context", forbidden)
    # Covers both baseline calibration and raw trace child processes. Loading
    # the existing source pair and manifests remains the real implementation.
    monkeypatch.setattr(prepare_optr_rotations.subprocess, "run", forbidden)

    root = case["output"].parent.parent
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError) as failure:
        prepare_rotations.prepare(case["config"], output=case["output"], testing=True,
                                  capture_runner=forbidden)
    assert len(original_errors) == 1 and failure.value is original_errors[0]
    assert str(failure.value) == MESSAGE
    assert case["calls"] == ["capture", "fit"]
    assert not case["output"].exists()
    assert not (case["output"].parent / ".optr.complete.json").exists()
    assert all(path.read_bytes() == content for path, content in before.items())
    report_path = case["output"].parent / "no-improvement-report.json"
    after = {path for path in root.rglob("*") if path.is_file()}
    assert after - before.keys() == {report_path}

    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line.split("OPTR_PROGRESS ", 1)[1]) for line in lines
              if line.startswith("[oscar] OPTR_PROGRESS ")]
    assert any(event.get("phase") == "layer_checked_resumed" for event in events)
    plan = next(event for event in events if event.get("phase") == "parallel_fit_plan")
    assert plan["workers"] == 0
    summaries = [event for event in events if event.get("phase") == "no_improvement_report"]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["status"] == "no_strict_validation_improvement"
    assert summary["recomputed_fit_or_replay"] is False
    assert summary["artifact_published"] is False
    assert summary["layer_count"] == 1 and summary["candidate_count"] == 1
    layer_summaries = [event for event in events if event.get("phase") == "no_improvement_layer"]
    assert len(layer_summaries) == 1
    assert layer_summaries[0]["reason"] == "no_strict_total_improvement"
    assert all(_compact_should_mirror(line, {}) for line in lines
               if any(f'"phase": "{phase}"' in line for phase in (
                   "no_improvement_report", "no_improvement_layer")))

    report = json.loads(report_path.read_text())
    assert report["identity"]["test_only"] is True
    assert len(report["layers"]) == 1
    layer = report["layers"][0]
    assert layer["independent_test"] == "baseline_compared_with_itself_no_proposal_tested"
    assert layer["candidates"][0]["unchanged_baseline"] is True
    assert layer["candidates"][0]["accepted"] is False
