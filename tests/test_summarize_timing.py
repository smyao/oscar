# Archive #70-73/#133: debug-sync phase walls carry device-time evidence.
"""Unit tests for aggregating oscar-debug checkpoints into per-phase evidence."""
import json

from tools.summarize_timing import main, summarize_timing


def write_records(directory, pid, records):
    (directory / f"timing-{pid}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))


def test_debug_checkpoints_pair_into_per_phase_device_seconds(tmp_path):
    write_records(tmp_path, 11, [
        {"t": "oscar-debug", "pid": 11, "phase": "fia", "layer": "l3", "tokens": 16384,
         "state": "waiting_for_device", "wall_time": 100.0},
        {"t": "oscar-debug", "pid": 11, "phase": "fia", "layer": "l3", "tokens": 16384,
         "state": "device_completed", "wall_time": 107.4},
        {"t": "oscar-debug", "pid": 11, "phase": "fia", "layer": "l7", "tokens": 16384,
         "state": "waiting_for_device", "wall_time": 108.0},
        {"t": "oscar-debug", "pid": 11, "phase": "fia", "layer": "l7", "tokens": 16384,
         "state": "device_completed", "wall_time": 115.4},
        {"t": "oscar-debug", "pid": 11, "phase": "merge", "layer": "l3", "tokens": 16384,
         "state": "waiting_for_device", "wall_time": 116.0},
        {"t": "oscar-timing", "pid": 11, "phase_end": "fia", "host_s": 0.0001, "ts_unix": 116.1},
    ])
    report = summarize_timing(tmp_path)
    assert report["status"] == "observed" and report["unpaired_waiting"] == 1
    fia = report["phases"]["fia"]
    assert fia["device_count"] == 2 and abs(fia["device_s_sum"] - 14.8) < 1e-6
    assert abs(fia["device_s_p95"] - 7.4) < 1e-6 and fia["host_s_sum"] == 0.0001
    assert report["phases"]["merge"]["device_count"] == 0
    assert report["phases"]["merge"]["device_s_sum"] is None


def test_empty_dir_is_not_run_and_exit_code_two(tmp_path, capsys):
    assert main([str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "not_run"
    assert main([str(tmp_path), "--output", str(tmp_path / "out.json")]) == 2
    assert json.loads((tmp_path / "out.json").read_text())["status"] == "not_run"


def test_malformed_record_fails_loudly(tmp_path):
    (tmp_path / "timing-1.jsonl").write_text('{"t": "oscar-debug"\n')
    try:
        summarize_timing(tmp_path)
    except ValueError as error:
        assert "malformed timing record" in str(error)
    else:
        raise AssertionError("malformed record must raise")
