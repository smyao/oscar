"""Archive #94/#95/#125/#130/#131: incomplete streams and logs cannot pass.

Native completion protocol supplies token usage/finish and optional cached
tokens. These tests exercise reporting, not OSCAR numerical equivalence.
"""
import io
import json
import subprocess

import pytest

from tools.prefix_cache_probe import read_sse
from tools import prefix_cache_probe
from tools.extract_prefix_evidence import extract, json_objects


def stream(*, done=True, finish="length", usage=None):
    events = [dict(choices=[dict(text="ok", token_ids=[1, 2], finish_reason=finish)])]
    if usage is not None:
        events.append(dict(choices=[], usage=usage))
    text = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return io.BytesIO((text + ("data: [DONE]\n\n" if done else "")).encode())


def test_completed_stream_does_not_invent_zero_hit():
    result = read_sse(stream(usage={"prompt_tokens": 4609, "completion_tokens": 2}), 4609, 2)
    assert result["cached_tokens"] is None
    assert result["sse_done"] and result["token_ids"] == [1, 2]


@pytest.mark.parametrize("kwargs", [
    {"done": False, "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    {"finish": None, "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    {}, {"usage": {"prompt_tokens": 4, "completion_tokens": 1}},
])
def test_missing_completion_evidence_fails(kwargs):
    with pytest.raises(ValueError):
        read_sse(stream(**kwargs), 4, 2)


def test_explicit_zero_and_nonzero_hits_are_preserved():
    for hit in (0, 2304):
        result = read_sse(stream(usage={"prompt_tokens": 4609, "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": hit}}), 4609, 2)
        assert result["cached_tokens"] == hit


def test_adjacent_json_and_separate_rank_units(tmp_path):
    line = '(Worker) {"t":"oscar-timing","pid":1,"host_s":0.02,"phase_end":"fia"}'
    line += '{"t":"oscar-device-event","pid":2,"rank":1,"device_ms":3,"phase":"fia","device_evidence":"npu_event"}'
    assert len(list(json_objects(line))) == 2
    source = tmp_path / "test.log"
    source.write_text(line + '\n{"t":"oscar-device-event",\n')
    result = extract([source], tmp_path / "out")
    assert result["records"] == 2
    assert result["unrecognized_or_incomplete_json_lines"] == 1
    assert {row["measurement"] for row in result["phase_groups"]} == {"host_launch_seconds", "npu_event_ms_unqualified"}
    assert result["performance_acceptance"] == "not_established"


def test_partial_outer_json_does_not_promote_nested_event():
    assert list(json_objects('{"unfinished":{"t":"oscar-timing","host_s":2}')) == []


def test_incomplete_stream_preserves_raw_evidence():
    source = stream(done=False, usage={"prompt_tokens": 4, "completion_tokens": 2})
    raw = io.BytesIO()
    with pytest.raises(ValueError):
        read_sse(source, 4, 2, raw)
    assert raw.getvalue() == source.getvalue()


def test_absolute_child_deadline_records_failure(monkeypatch, tmp_path):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(prefix_cache_probe.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="absolute deadline"):
        prefix_cache_probe.isolated_http(tmp_path, "slow", {"timeout": 0.1})
    result = json.loads((tmp_path / "slow.response.json").read_text())
    assert result["status"] == "failed" and result["deadline_s"] == 0.1


def test_table_rows_and_missing_run_identity_remain_separate(tmp_path):
    paths = [tmp_path / "one.log", tmp_path / "two.log"]
    for path in paths:
        path.write_text('│ Output Token Throughput │ 18.4419 │\n'
                        '{"t":"oscar-timing","pid":1,"host_s":0.02}\n')
    result = extract(paths, tmp_path / "out")
    assert len(result["phase_groups"]) == 2
    assert len(result["benchmark_table_rows"]) == 2
    assert all(row["count"] == 1 for row in result["phase_groups"])
