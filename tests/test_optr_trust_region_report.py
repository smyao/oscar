"""Archive #176: diagnose rejected v11 repairs from saved work, without replay."""
import builtins
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from oscar_ascend import optr_checkpoint, optr_trust_region
from oscar_ascend.rotations import calibration_cache_identity
from tools import optr_rejection_report as report
from tools import prepare_optr_recipe3, prepare_optr_rotations as prepare
from tools.optr_recipe3_source import LEGACY_FITTING_SHA256
from test_optr_recipe3_pipeline import case, capture, proposal, forbidden
from test_official_rotation_calibration import NAME
from test_optr_recipe3_source import edit_checkpoint


@pytest.fixture
def completed_repair(case, monkeypatch):
    source_case = {**case}
    case["candidate"] = case["baseline"]["layers"][NAME]
    profile = calibration_cache_identity(case["config"], case["fingerprint"])
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    pool = case["output"].parent / "generations"
    old_output = pool / ("optr_" + digest) / "refined.pt"
    with monkeypatch.context() as historical:
        for module in (optr_checkpoint, prepare_optr_recipe3):
            historical.setattr(module, "fitting_implementation_digest", lambda: LEGACY_FITTING_SHA256)
        with pytest.raises(ValueError, match="no strict validation improvement"):
            prepare.prepare(case["config"], output=old_output, testing=True,
                capture_runner=capture(case), fitter=proposal(source_case), base_runner=forbidden)
    config = {**case["config"], "optr_recipe3_repair": "single_head_trust_region_v1"}
    output = pool / "repair" / "refined.pt"
    with pytest.raises(ValueError, match="no strict validation improvement"):
        prepare.prepare(config, output=output, testing=True, capture_runner=forbidden,
                        fitter=forbidden, base_runner=forbidden)
    return {**case, "config": config, "output": output, "old_output": old_output,
            "generation": next((output.parent / "fit-checkpoints").iterdir())}


def test_v11_report_authenticates_source_and_all_saved_repair_rejections_without_replay(completed_repair, monkeypatch):
    from oscar_ascend import optr_calibration, optr_traces
    for module, name in ((prepare, "_metrics"), (prepare, "_capture"),
                         (prepare_optr_recipe3, "fit_perhead"), (optr_trust_region, "select_repair"),
                         (optr_calibration, "replay_sample"), (optr_traces, "load_layer_samples")):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(torch, "matrix_exp", forbidden)
    old_import, old_load = builtins.__import__, torch.load
    def safe_import(name, *args, **kwargs):
        assert name != "torch_npu" and not name.startswith("torch_npu.")
        return old_import(name, *args, **kwargs)
    def safe_load(path, *args, **kwargs):
        assert "native-traces" not in str(path)
        return old_load(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", safe_import)
    monkeypatch.setattr(torch, "load", safe_load)
    before = {p: p.read_bytes() for p in completed_repair["output"].parent.parent.rglob("*") if p.is_file()}
    summary = report.report_no_improvement(completed_repair["config"], output=completed_repair["output"], testing=True)
    assert summary["status"] == "no_strict_validation_improvement"
    assert summary["recomputed_fit_or_replay"] is False
    assert summary["layers"][0]["original_candidates"] == 1
    assert summary["layers"][0]["repair_candidates"] == 32
    repair_summary = summary["layers"][0]["repair_search"]
    assert repair_summary["candidate_count"] == 32
    assert sum(repair_summary["reason_counts"].values()) == 32
    saved = json.loads(Path(summary["report"]).read_text())
    layer = saved["layers"][0]
    from tools.optr_rejection_summary import summarize_saved_report
    assert summarize_saved_report(summary["report"])["layers"] == summary["layers"]
    assert len(layer["candidates"]) == 33
    assert all(not row["accepted"] for row in layer["candidates"])
    assert layer["source_fit_provenance"]["fitting_implementation_sha256"] == LEGACY_FITTING_SHA256
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not completed_repair["output"].exists()


@pytest.mark.parametrize("fault", ["source_provenance", "repair_accept", "repair_policy", "source_candidate_metrics",
                                  "repair_source_hash", "repair_direction_coverage"])
def test_v11_report_does_not_explain_forged_evidence_as_a_real_rejection(completed_repair, fault):
    path = completed_repair["generation"] / "layer-3.json"
    def corrupt(payload):
        record = payload["artifact"]["refinement_certificate"]["layers"][NAME]
        if fault == "source_provenance":
            record["source_fit_provenance"]["proposal_manifest_sha256"] = "f" * 64
        elif fault == "repair_accept":
            record["recipe3_repair_selection"]["evaluated_candidates"][0]["accepted"] = True
        elif fault == "repair_policy":
            record["recipe3_repair_selection"]["policy"]["radii"] = [.01]
        elif fault == "source_candidate_metrics":
            record["shared_layer_selection"]["evaluated_candidates"][0]["validation"] = copy.deepcopy(record["validation"]["baseline"])
        elif fault == "repair_source_hash":
            record["recipe3_repair_selection"]["source_candidates_sha256"] = "f" * 64
        else:
            repair = record["recipe3_repair_selection"]
            repair["evaluated_candidates"].pop()
            repair["candidate_count"] -= 1
    edit_checkpoint(path, corrupt)
    with pytest.raises(ValueError):
        report.report_no_improvement(completed_repair["config"], output=completed_repair["output"], testing=True)


def test_v11_fresh_capture_with_zero_directions_still_reports_all_rejected_originals(case):
    case["config"]["optr_recipe3_repair"] = "single_head_trust_region_v1"
    with pytest.raises(ValueError, match="no strict validation improvement"):
        prepare.prepare(case["config"], output=case["output"], testing=True,
            capture_runner=capture(case), fitter=proposal(case, unchanged=True), base_runner=forbidden)
    result = report.report_no_improvement(case["config"], output=case["output"], testing=True)
    assert result["layers"][0]["repair_candidates"] == 0
    assert result["layers"][0]["original_candidates"] == 1
