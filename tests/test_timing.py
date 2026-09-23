# Archive #70-73/D.4: host phase records must stay out of the terminal flood.
"""Host phase timing records route to the trace dir; stderr stays opt-in."""
import json

import pytest

from oscar_ascend import timing


@pytest.mark.parametrize("phase_name", ["fia", "current_source_suppress", "current_slot_guard", "current_native_fia"])
def test_timing_record_goes_to_trace_file_not_stderr(tmp_path, monkeypatch, capsys, phase_name):
    monkeypatch.setenv("OSCAR_TIMING", "1")
    monkeypatch.setenv("OSCAR_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("OSCAR_TIMING_STDERR", raising=False)
    with timing.phase(phase_name, layer="model.layers.3.self_attn.attn", tokens=4):
        pass
    assert capsys.readouterr().err == ""
    records = list(tmp_path.glob("timing-*.jsonl"))
    assert len(records) == 1
    lines = records[0].read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["t"] == "oscar-timing" and record["phase_end"] == phase_name
    assert record["device_ms"] is None and record["host_s"] >= 0


def test_timing_record_stderr_opt_in_and_invalid_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OSCAR_TIMING", "1")
    monkeypatch.setenv("OSCAR_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("OSCAR_TIMING_STDERR", "1")
    with timing.phase("rotate", layer="model.layers.3.self_attn.attn", tokens=4):
        pass
    assert "oscar-timing" in capsys.readouterr().err
    assert list(tmp_path.glob("timing-*.jsonl")) == []
    monkeypatch.setenv("OSCAR_TIMING_STDERR", "loud")
    with pytest.raises(ValueError):
        with timing.phase("rotate", layer="model.layers.3.self_attn.attn", tokens=4):
            pass


def test_timing_disabled_writes_nothing(tmp_path, monkeypatch, capsys):
    for name in ("OSCAR_TIMING", "OSCAR_PROFILER", "OSCAR_DEBUG_SYNC", "OSCAR_TIMING_STDERR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OSCAR_TRACE_DIR", str(tmp_path))
    with timing.phase("fia", layer="model.layers.3.self_attn.attn", tokens=4):
        pass
    assert list(tmp_path.glob("timing-*.jsonl")) == []
    assert capsys.readouterr().err == ""
