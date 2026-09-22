# Archive #70-73/#129: active progress must stay fresh during long requests.
"""Unit tests for process-local telemetry emission and throttled heartbeats."""
import json
import time

import pytest

from oscar_ascend import telemetry


def read_records(directory, pid):
    path = directory / f"worker-{pid}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_emit_once_writes_single_record_per_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("OSCAR_TRACE_DIR", str(tmp_path))
    telemetry.emit_once("unit_once_a", key="k1", layer="layer.0", tokens=128)
    telemetry.emit_once("unit_once_a", key="k1", layer="layer.0", tokens=128)
    telemetry.emit_once("unit_once_a", key="k2", layer="layer.1", tokens=128)
    records = read_records(tmp_path, telemetry.os.getpid())
    assert [r["layer"] for r in records] == ["layer.0", "layer.1"]
    assert all(r["event"] == "unit_once_a" and r["wall_time"] > 0 for r in records)


def test_emit_throttled_suppresses_within_interval_and_refreshes_after(tmp_path, monkeypatch):
    monkeypatch.setenv("OSCAR_TRACE_DIR", str(tmp_path))
    key = ("unit_throttle_a", "layer.0")
    telemetry._last.pop(key, None)
    telemetry.emit_throttled("unit_throttle_a", key="layer.0", min_interval=0.05,
                             layer="layer.0", tokens=16384, max_seq_len=16384)
    telemetry.emit_throttled("unit_throttle_a", key="layer.0", min_interval=0.05,
                             layer="layer.0", tokens=16384, max_seq_len=16384)
    time.sleep(0.06)
    telemetry.emit_throttled("unit_throttle_a", key="layer.0", min_interval=0.05,
                             layer="layer.0", tokens=16384, max_seq_len=32768)
    records = [r for r in read_records(tmp_path, telemetry.os.getpid()) if r["event"] == "unit_throttle_a"]
    assert len(records) == 2
    assert records[0]["max_seq_len"] == 16384 and records[1]["max_seq_len"] == 32768
    assert records[1]["wall_time"] > records[0]["wall_time"]


def test_emit_throttled_uses_env_interval_and_validates_it(tmp_path, monkeypatch):
    monkeypatch.setenv("OSCAR_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("OSCAR_PROGRESS_INTERVAL_SECONDS", "0.01")
    key = ("unit_throttle_b", "layer.2")
    telemetry._last.pop(key, None)
    telemetry.emit_throttled("unit_throttle_b", key="layer.2", layer="layer.2")
    time.sleep(0.02)
    telemetry.emit_throttled("unit_throttle_b", key="layer.2", layer="layer.2")
    records = [r for r in read_records(tmp_path, telemetry.os.getpid()) if r["event"] == "unit_throttle_b"]
    assert len(records) == 2
    for bad in ("0", "-1", "inf", "nan", "abc"):
        monkeypatch.setenv("OSCAR_PROGRESS_INTERVAL_SECONDS", bad)
        with pytest.raises(ValueError):
            telemetry.emit_throttled("unit_throttle_b", key="layer.2", layer="layer.2")


def test_emit_without_trace_dir_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("OSCAR_TRACE_DIR", raising=False)
    telemetry.emit_once("unit_noop", key="x")
    telemetry.emit_throttled("unit_noop", key="x", min_interval=0.01)
    assert not list(tmp_path.glob("worker-*.jsonl"))
