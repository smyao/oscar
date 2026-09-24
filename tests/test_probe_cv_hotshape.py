"""Archive #70-73/#85/#125/#126/#143: paired CV diagnostic host contracts.

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
    target.write_text('{"devices":[0,1,2,3],"soc_version":"ascend910b4"}')
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


def test_shape_splits_capture_old_and_candidate_query_tile_difference():
    assert hot.splits_for_shape(11504, 32, query_rows=64, capacity=16384) == 1
    assert hot.splits_for_shape(11504, 32, query_rows=128, capacity=16384) == 1
    assert hot.splits_for_shape(128, 32, query_rows=64, capacity=16384) == 3
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
