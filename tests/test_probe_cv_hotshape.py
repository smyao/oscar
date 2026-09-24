"""Archive #70-73/#85/#125/#126/#143-#146: paired CV host contracts.

They do not stand in for the target NPU numerical or Event measurements.
"""

import json
from pathlib import Path
import subprocess

import pytest

from tools import probe_cv_hotshape as hot
from tools.environment import file_fingerprint


def test_pinned_git_tree_fingerprint_ignores_dirty_checkout(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.email", "probe@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Probe"], cwd=tmp_path, check=True)
    source = tmp_path / "csrc/kernels/example.cpp"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"first build\n")
    subprocess.run(["git", "add", "csrc"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                              capture_output=True, text=True, check=True).stdout.strip()
    original = hot.git_csrc_fingerprint(tmp_path, revision)
    assert original == file_fingerprint(tmp_path / "csrc")
    source.write_bytes(b"candidate changed\n")
    assert hot.git_csrc_fingerprint(tmp_path, revision) == original
    assert file_fingerprint(tmp_path / "csrc") != original


def test_source_check_fails_closed_for_wrong_baseline(monkeypatch, tmp_path):
    (tmp_path / "csrc").mkdir()
    monkeypatch.setattr(hot, "git_csrc_fingerprint", lambda root: {"kernel.cpp": "old"})
    with pytest.raises(hot.BaselineUnavailable, match="does not match"):
        hot._source_check({"source": {"kernel.cpp": "new"}}, "baseline", root=tmp_path)
    assert hot._source_check({"source": {"kernel.cpp": "old"}}, "baseline", root=tmp_path)


def test_baseline_availability_reports_missing_evidence_without_torch(monkeypatch, tmp_path):
    target = tmp_path / "target.json"
    target.write_text('{"devices":[4,5,6,7],"soc_version":"ascend910b4"}')
    monkeypatch.setattr(hot, "_verify_artifact",
                        lambda *args: (_ for _ in ()).throw(hot.BaselineUnavailable("missing old SO")))
    assert hot.baseline_availability(tmp_path / "missing.json", target) == {
        "status": "not_available", "reason": "missing old SO",
        "source_revision": hot.BASELINE_REVISION}


def test_cli_baseline_unavailable_is_explicit_zero_rc_skip(monkeypatch, tmp_path):
    monkeypatch.setattr(hot, "run_probe",
                        lambda *args: (_ for _ in ()).throw(hot.BaselineUnavailable("old artifact deleted")))
    output = tmp_path / "result.json"
    rc = hot.main(["--variant", "baseline", "--output", str(output)])
    assert rc == 0
    assert json.loads(output.read_text())["status"] == "not_available"


def test_pinned_m128_geometry_matches_current_source_and_workspace():
    assert hot.BASELINE_REVISION == "fe0e925e7ef78bfb64217a300031502fc4a7b7bc"
    assert (hot.OLD_QUERY_ROWS, hot.OLD_KV_ROWS) == (128, 256)
    assert hot.splits_for_shape(11504, 32, query_rows=hot.OLD_QUERY_ROWS, capacity=16384) == 1
    assert hot.splits_for_shape(11504, 32, query_rows=128, capacity=16384) == 1
    assert hot.splits_for_shape(128, 32, query_rows=hot.OLD_QUERY_ROWS, capacity=16384) == 5
    assert hot.splits_for_shape(128, 32, query_rows=128, capacity=16384) == 5
    assert hot.workspace_per_core_bytes(256, 128, 256) == 917504


def test_small_synthetic_fixture_is_bitwise_replayable_and_page_disjoint(monkeypatch):
    torch = pytest.importorskip("torch")
    # Exercise old/current ranges across physical and virtual page boundaries.
    monkeypatch.setattr(hot, "QLENS", (4, 4, 4, 600))
    monkeypatch.setattr(hot, "CONTEXTS", (511, 513, 1025, 2049))
    monkeypatch.setattr(hot, "DIM", 64)
    first = hot.build_fixture(torch, scale=64 ** -0.5)
    second = hot.build_fixture(torch, scale=64 ** -0.5)
    assert first["hash"] == second["hash"]
    assert first["expected"].keys() == second["expected"].keys()
    assert len(first["expected"]) >= 10
    pages = [page for request in first["pages"] for page in request]
    assert sorted(pages) == list(range(first["blocks"]))
    for index, (context, length) in enumerate(zip(hot.CONTEXTS, hot.QLENS)):
        begin = sum(hot.QLENS[:index])
        assert first["cpu"]["slots"][begin] >= 0
        assert int(first["cpu"]["lens"][index]) == context + length
    assert all(bool(torch.isfinite(output).all()) and bool(torch.isfinite(lse).all())
               for output, lse in first["expected"].values())


def test_decode_fixture_keeps_disjoint_histories_and_full_causal_oracle(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(hot, "DIM", 64)
    monkeypatch.setattr(hot, "DECODE_QLENS", (4,) * 8)
    monkeypatch.setattr(hot, "DECODE_CONTEXTS", (65, 129, 257, 511) * 2)
    first = hot.build_decode_fixture(torch, scale=64 ** -0.5)
    second = hot.build_decode_fixture(torch, scale=64 ** -0.5)
    assert first["hash"] == second["hash"]
    assert first["source2_suppressed"] is False
    assert first["page_policy"] == "32_independent_disjoint_physical_histories"
    assert len(first["expected"]) == 32
    assert first["cpu"]["slots"].unique().numel() == 32
    pages = [page for request in first["pages"] for page in request]
    assert sorted(pages) == list(range(first["blocks"]))
    assert all(bool(torch.isfinite(output).all()) and bool(torch.isfinite(lse).all())
               for output, lse in first["expected"].values())


def test_profile_summary_keeps_source_ticks_and_overlapping_engines_separate():
    torch = pytest.importorskip("torch")
    raw = torch.zeros((2, 3, 4, 20), dtype=torch.int64)
    for engine in range(3):
        raw[0, engine, 3, 17] = 100 + engine
        raw[1, engine, 3, 17] = 200 + engine
        raw[0, engine, 0, 0] = 2
        raw[1, engine, 0, 0] = 3
        raw[0, engine, 3, 0] = 2
        raw[1, engine, 3, 0] = 3
        raw[0, engine, 0, 1] = raw[0, engine, 3, 1] = 1
        raw[1, engine, 0, 1] = raw[1, engine, 3, 1] = 1
    raw[0, 0, 0, 12] = raw[0, 0, 3, 12] = 50
    raw[1, 0, 0, 12] = raw[1, 0, 3, 12] = 100
    raw[0, 1, 0, 5] = raw[0, 1, 3, 5] = 30
    raw[1, 1, 0, 5] = raw[1, 1, 3, 5] = 40
    summary = hot.summarize_raw_profile(raw, cores=2)
    history_aic = summary["sources"]["history"]["aic"]
    assert history_aic["tasks"]["sum_across_cores"] == 5
    assert history_aic["aic_qk"]["p50"] == 50
    assert history_aic["aic_qk"]["p95"] == 100
    assert summary["sources"]["history"]["aic"]["process_span"]["max"] == 0
    assert summary["sources"]["total"]["aic"]["process_span"]["max"] == 200
    from tools.paired_concurrency_probe import cv_profile_terminal_rows
    case = {"profile": {"raw_counters": summary, "outer_event_ms": 5.0,
                        "outer_event_over_normal_median": 1.25}}
    lines = cv_profile_terminal_rows("main", case)
    assert len(lines) == 3
    assert all("pv_max_raw_ticks=" in line and
               "aiv_waitqk_max_raw_ticks=" in line and
               "aiv_waitpv_max_raw_ticks=" in line and
               "scope=instrumented_kernel_diagnostic" in line
               for line in lines)
    assert "source=history" in lines[0]
    assert "source=window" in lines[1]
    assert "source=current" in lines[2]
    raw[0, 0, 3, 12] = 51  # Source3 is not a second independent contribution.
    with pytest.raises(hot.HotShapeError, match="total counters differ"):
        hot.summarize_raw_profile(raw, cores=2)
    raw[0, 0, 3, 12] = 50
    raw[0, 1, 0, 1] = 2
    raw[0, 1, 3, 1] = 2
    with pytest.raises(hot.HotShapeError, match="Cube/AIV"):
        hot.summarize_raw_profile(raw, cores=2)


def test_profile_buffer_meets_binding_alignment_and_is_zeroed():
    torch = pytest.importorskip("torch")
    profile = hot._aligned_profile_tensor(torch, 3, "cpu")
    assert profile.shape == (3, 3, 4, 20)
    assert profile.dtype == torch.int64 and profile.is_contiguous()
    assert profile.data_ptr() % 64 == 0
    assert not bool(profile.any())
