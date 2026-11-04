"""Archive #176/#177: expose hidden repair failures without repeating work."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.optr_rejection_report import metric_rejections
from tools.optr_rejection_summary import summarize_saved_report
from test_optr_rejection_report import metric


@pytest.fixture
def saved_report(tmp_path):
    before = metric()
    original = {"candidate": "primary", "phase": "primary", "unchanged_baseline": False,
                **metric_rejections(before, metric((1.1, .1)))}
    after = metric((.9, 1.))
    after["projection_checks"]["bf16_output"] = metric((1.01, .8))
    after["projection_checks"]["bf16_output"].pop("projection_checks")
    repair = {"candidate": "repair_h0_Rk_r0", "phase": "repair", "head": 0, "side": "Rk",
              "radius": .02, "unchanged_baseline": False, "same_bf16_operand_as_seed": False,
              **metric_rejections(before, after)}
    noop = {**repair, "candidate": "repair_h0_Rk_r1", "radius": .01,
            "same_bf16_operand_as_seed": True, **metric_rejections(before, before)}
    worse = {**repair, "candidate": "repair_h0_Rk_r2", "radius": .005,
             **metric_rejections(before, metric((1.01, 1.01)))}
    name = "model.layers.3.self_attn.attn"
    value = {"schema_version": 1, "status": "no_strict_validation_improvement",
             "identity": {"full_layers": [name], "test_only": True},
             "layers": [{"layer": name, "candidate_count": 4, "original_candidate_count": 1,
                         "repair_candidate_count": 3, "candidates": [original, repair, noop, worse]}]}
    path = tmp_path / "no-improvement-report.json"
    path.write_text(json.dumps(value, allow_nan=False))
    return path, value


def test_global_winner_cannot_hide_different_repair_failure_or_bf16_noop(saved_report):
    path, _ = saved_report
    summary = summarize_saved_report(path)
    layer = summary["layers"][0]
    assert layer["best_total_candidate"] == "primary"
    assert layer["regressed_queries"] == 3
    repair = layer["repair_search"]
    assert repair["candidate_count"] == 3
    assert repair["strict_total_improvement_count"] == 1
    assert repair["same_bf16_operand_count"] == 1
    assert repair["unchanged_constants_count"] == 0
    assert repair["bf16_only_regression_count"] == 1
    assert repair["reason_counts"] == {"fine_grained_regression": 1,
        "no_strict_total_improvement": 1, "aggregate_error_regression": 1}
    assert repair["fewest_regressed_query_checks_among_improving"] == 1
    best = repair["best_total"]
    assert best["candidate"] == "repair_h0_Rk_r0"
    assert best["total_relative_delta"] == pytest.approx(-.05)
    assert best["regressed_checks"]["query"] == 1
    assert best["worst_gate"][0]["projection_view"] == "bf16_output"
    assert best["worst_gate"][0]["context"] == 0
    assert summary["evidence_scope"] == "saved_json_only_checkpoints_not_reauthenticated"
    assert summary["report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert summary["test_only"] is True
    assert summary["artifact_published"] is False


def test_legacy_v10_and_empty_repair_are_not_reported_as_search_success(saved_report):
    path, value = saved_report
    layer = value["layers"][0]
    layer["candidates"] = layer["candidates"][:1]
    layer["candidate_count"] = 1
    del layer["repair_candidate_count"], layer["original_candidate_count"]
    path.write_text(json.dumps(value))
    repair = summarize_saved_report(path)["layers"][0]["repair_search"]
    assert repair["candidate_count"] == 0
    assert repair["best_total"] is None
    assert repair["fewest_regressed_query_checks_among_improving"] is None
    assert repair["reason_counts"] == {}


def test_zero_baseline_stays_undefined_relative_error_not_fake_zero(saved_report):
    path, value = saved_report
    for row in value["layers"][0]["candidates"]:
        row.update(metric_rejections(metric((0., 0.)), metric((.1, .1))))
        if row["phase"] == "repair":
            row["same_bf16_operand_as_seed"] = False
    path.write_text(json.dumps(value))
    best = summarize_saved_report(path)["layers"][0]["repair_search"]["best_total"]
    assert best["total_relative_delta"] is None
    assert best["worst_gate"][0]["positive_from_zero"] is True
    assert best["worst_gate"][0]["normalized_error_relative_delta"] is None


@pytest.mark.parametrize("fault", ["terminal_fragment", "coverage", "candidate_count", "duplicate",
    "projection", "nonfinite", "accepted", "reason", "strict", "count", "baseline", "relative",
    "repair_count", "radius", "scope"])
def test_incomplete_or_inconsistent_report_is_rejected(saved_report, fault):
    path, value = saved_report
    layer = value["layers"][0]
    row = layer["candidates"][1]
    if fault == "terminal_fragment":
        value = {"phase": "no_improvement_report", "candidate_count": 1216}
    elif fault == "coverage":
        value["identity"]["full_layers"].append("model.layers.7.self_attn.attn")
    elif fault == "candidate_count":
        layer["candidates"].pop()
    elif fault == "duplicate":
        row["candidate"] = "primary"
    elif fault == "projection":
        del row["views"]["bf16_output"]
    elif fault == "nonfinite":
        row["views"]["fp32"]["squared_error"]["candidate"] = float("nan")
    elif fault == "accepted":
        row["accepted"] = True
    elif fault == "reason":
        row["reason"] = "passed"
    elif fault == "strict":
        row["strict_total_improvement"] = False
    elif fault == "count":
        row["regressed_checks"]["query"] = -1
    elif fault == "baseline":
        row["views"]["fp32"]["squared_error"]["baseline"] = 3.
    elif fault == "relative":
        row["views"]["fp32"]["squared_error"]["relative_delta"] = -.9
    elif fault == "repair_count":
        layer["repair_candidate_count"] = 64
    elif fault == "radius":
        row["radius"] = 0
    else:
        del value["identity"]["test_only"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        summarize_saved_report(path)


def test_saved_cli_needs_only_json_and_stdlib_does_not_write_and_stays_compact(saved_report):
    path, value = saved_report
    template = value["layers"][0]
    value["layers"] = []
    for index in range(16):
        layer = copy.deepcopy(template)
        layer["layer"] = f"model.layers.{4 * index + 3}.self_attn.attn"
        value["layers"].append(layer)
    value["identity"]["full_layers"] = [layer["layer"] for layer in value["layers"]]
    path.write_text(json.dumps(value))
    before = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    script = '''
import runpy, sys
class ForbidModelImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'oscar_ascend'):
            raise AssertionError('saved-report must not import model code: ' + fullname)
sys.meta_path.insert(0, ForbidModelImports())
sys.argv = ['optr_rejection_report', '--saved-report', sys.argv[1]]
runpy.run_module('tools.optr_rejection_report', run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-S", "-c", script, str(path)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 17
    assert len(result.stdout.encode()) < 50 * 1024
    rows = [json.loads(line.split("OPTR_PROGRESS ", 1)[1]) for line in lines]
    assert all(row["repair_search"]["best_total"]["candidate"] == "repair_h0_Rk_r0" for row in rows[:-1])
    assert rows[-1]["evidence_scope"] == "saved_json_only_checkpoints_not_reauthenticated"
    assert rows[-1]["recomputed_fit_or_replay"] is False
    assert {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()} == before


def test_saved_cli_rejects_output_redirect_before_any_file_write(saved_report):
    path, _ = saved_report
    before = path.read_bytes()
    result = subprocess.run([sys.executable, "-S", "-m", "tools.optr_rejection_report", "--saved-report",
                             str(path), "--output", str(path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "never writes files" in result.stderr
    assert path.read_bytes() == before
