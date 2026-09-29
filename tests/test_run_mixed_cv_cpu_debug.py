"""The official CPU-debug controller must preserve complete paired evidence.

Archive #126/#129/#151 and startup D.4: host tests here check orchestration,
not AscendC arithmetic or target NPU acceptance.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from tools import run_mixed_cv_cpu_debug as runner


def _case(tmp_path: Path):
    fixture = tmp_path / "c16_d256_q336_c65"
    fixture.mkdir()
    (fixture / "shape.txt").write_text("336 6 1 256 65 64 1 3 512 1 64 69632 2 1 1\n")
    (fixture / "raw.bin").write_bytes(b"unchanged golden")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"op": "mixed_cluster16", "path": str(fixture)}]))
    executable = tmp_path / "oscar_cv_cpu"
    executable.write_bytes(b"placeholder")
    return fixture, cases, executable


def test_complete_d256_case_uses_fresh_fe0_then_candidate_with_exact_artifacts(tmp_path, monkeypatch):
    fixture, cases, executable = _case(tmp_path)
    calls = []

    def phase(name, command, **kwargs):
        calls.append((name, command, kwargs))
        mode = command[1]
        reference = Path(command[3])
        if mode == "mixed_C16_reference":
            for item, size in runner._reference_sizes(fixture).items():
                (reference / item).write_bytes(bytes(size))
        else:
            assert mode == "mixed_C16_candidate"
            assert all((reference / item).is_file()
                       for item in ("partial.bin", "lse.bin", "status.bin"))
        log = tmp_path / f"{mode}.log"
        log.write_text(f"CPU_STAGE mode={mode} stage=ORACLE_DONE elapsed_s=1\n"
                       + json.dumps({"backend": "ascendc_cpu_debug", "op": mode,
                                     "status": "passed"}, separators=(",", ":")) + "\n")
        return SimpleNamespace(returncode=0, timed_out=False, cleanup_complete=True,
                               elapsed_seconds=1.0, log=str(log))

    monkeypatch.setattr(runner, "run_phase", phase)
    report_path = tmp_path / "report.json"
    assert runner.run_cases(executable=executable, cases_path=cases,
                            output=report_path, log_dir=tmp_path / "logs",
                            artifact_root=tmp_path / "artifacts") == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed"
    assert report["cases"][0]["execution"] == "separate_complete_kernels"
    assert [call[1][1] for call in calls] == ["mixed_C16_reference", "mixed_C16_candidate"]
    assert all(call[2]["timeout"] == 120 for call in calls)
    assert report["cases"][0]["golden_sha256"] == runner._fixture_sha256(fixture)


def test_reference_timeout_keeps_real_rc_and_blocks_candidate(tmp_path, monkeypatch):
    _, cases, executable = _case(tmp_path)
    calls = []

    def phase(name, command, **kwargs):
        calls.append(command[1])
        log = tmp_path / "timeout.log"
        log.write_text("CPU_STAGE mode=mixed_C16_reference stage=FE0_START elapsed_s=1\n")
        return SimpleNamespace(returncode=124, timed_out=True, cleanup_complete=True,
                               elapsed_seconds=120.0, log=str(log))

    monkeypatch.setattr(runner, "run_phase", phase)
    report_path = tmp_path / "report.json"
    assert runner.run_cases(executable=executable, cases_path=cases,
                            output=report_path, log_dir=tmp_path / "logs",
                            artifact_root=tmp_path / "artifacts") == 1
    report = json.loads(report_path.read_text())
    assert calls == ["mixed_C16_reference"]
    assert report["status"] == "failed"
    assert report["cases"][0]["phases"][0]["returncode"] == 124
    assert report["cases"][0]["phases"][0]["timed_out"] is True


def test_select_rejects_even_one_unknown_case(tmp_path, monkeypatch):
    _, cases, executable = _case(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("no phase may run for a misspelled selection")
    monkeypatch.setattr(runner, "run_phase", forbidden)
    try:
        runner.run_cases(executable=executable, cases_path=cases,
                         output=tmp_path / "report.json", log_dir=tmp_path / "logs",
                         artifact_root=tmp_path / "artifacts",
                         select={"c16_d256_q336_c65", "typo"})
    except ValueError as error:
        assert "typo" in str(error)
    else:
        raise AssertionError("unknown selected case was silently ignored")
