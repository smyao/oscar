"""Host checks for the isolated official CPU-debug controller.

Archive #126/#129/#145/#148-153 and startup D.4: these check full-case
pairing, bounded phases, and fail-closed evidence, not kernel arithmetic.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from tools import run_batched4_cpu_debug as runner


def _fixture(tmp_path: Path):
    case = tmp_path / "synthetic_active"
    case.mkdir()
    (case / "shape.txt").write_text("4 6 1 64 65 64 1 3 512 1 64 20480 2 1 1\n")
    (case / "raw.bin").write_bytes(b"frozen input bytes")
    executable = tmp_path / "oscar_batched4_cpu"
    executable.write_bytes(b"placeholder")
    return case, executable


def test_split_mode_binds_checked_reference_and_actual_candidate(tmp_path, monkeypatch):
    fixture, executable = _fixture(tmp_path)
    calls = []

    def run_phase(name, command, **kwargs):
        calls.append((command, kwargs))
        phase = command[1]
        artifact = Path(command[3])
        if phase == "old_only_reference":
            for file, size in runner._artifact_sizes(fixture).items():
                (artifact / file).write_bytes(bytes(size))
        else:
            assert phase == "old_only_candidate"
            assert all((artifact / file).is_file()
                       for file in runner._artifact_sizes(fixture))
        log = tmp_path / f"{phase}.log"
        log.write_text(f"BATCHED4_STAGE mode={phase} stage=ORACLE_DONE elapsed_s=1\n"
                       '{"backend":"ascendc_cpu_debug","op":"batched4","status":"passed"}\n')
        return SimpleNamespace(returncode=0, timed_out=False,
                               cleanup_complete=True, elapsed_seconds=1.0,
                               log=str(log))

    monkeypatch.setattr(runner, "run_phase", run_phase)
    output = tmp_path / "result.json"
    assert runner.run_case(executable=executable, fixture=fixture, mode="old_only",
                           output=output, log_dir=tmp_path / "logs",
                           artifact_root=tmp_path / "artifacts") == 0
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["reference_sha256"]
    assert [call[0][1] for call in calls] == ["old_only_reference", "old_only_candidate"]
    assert all(call[1]["timeout"] == 120 for call in calls)
    assert report["fixture_sha256"] == runner._fixture_sha(fixture)


def test_reference_timeout_preserves_rc_and_never_runs_candidate(tmp_path, monkeypatch):
    fixture, executable = _fixture(tmp_path)
    calls = []

    def run_phase(name, command, **kwargs):
        calls.append(command[1])
        log = tmp_path / "timeout.log"
        log.write_text("BATCHED4_STAGE mode=old_only_reference stage=REFERENCE_START elapsed_s=1\n")
        return SimpleNamespace(returncode=124, timed_out=True,
                               cleanup_complete=True, elapsed_seconds=120.0,
                               log=str(log))

    monkeypatch.setattr(runner, "run_phase", run_phase)
    output = tmp_path / "result.json"
    assert runner.run_case(executable=executable, fixture=fixture, mode="old_only",
                           output=output, log_dir=tmp_path / "logs",
                           artifact_root=tmp_path / "artifacts") == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["phases"][0]["returncode"] == 124
    assert calls == ["old_only_reference"]
