# Archive #138: ladder arm windows split production heartbeat deltas.
"""Unit tests for reconstructing per-state residence walls from heartbeats."""
import json

import pytest

from tools.summarize_progress import main, summarize_progress


def write_records(directory, pid, records):
    (directory / f"worker-{pid}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records))


def heart(pid, wall, *, requests, kv, layer="l3", tokens=16384):
    return {"event": "attention_progress", "pid": pid, "rank": 0, "layer": layer,
            "tokens": tokens, "requests": requests, "max_seq_len": kv, "wall_time": wall}


def test_deltas_bucket_by_arm_requests_and_kv_with_gap_separation(tmp_path):
    write_records(tmp_path, 1, [
        heart(1, 100.0, requests=1, kv=16384),
        heart(1, 102.0, requests=1, kv=16384, layer="l7"),
        heart(1, 200.0, requests=4, kv=32768),
        heart(1, 208.0, requests=4, kv=32768, layer="l7"),
        {"event": "attention_dispatched", "pid": 1, "wall_time": 209.0},
    ])
    windows = [[95.0, 150.0, "K=1"], [195.0, 300.0, "K=4"]]
    report = summarize_progress(tmp_path, windows)
    assert report["status"] == "observed"
    buckets = {(row["arm"], row["requests"], row["kv_bucket_k"]): row for row in report["buckets"]}
    assert abs(buckets[("K=1", 1, 16)]["wall_s_sum"] - 2.0) < 1e-9
    assert abs(buckets[("K=4", 4, 32)]["wall_s_sum"] - 8.0) < 1e-9
    # The 102s->200s cross-window delta is neither arm's residence time.
    assert abs(buckets[("between_arms", 1, 16)]["wall_s_sum"] - 98.0) < 1e-9


def test_empty_or_missing_windows_behave(tmp_path):
    assert summarize_progress(tmp_path)["status"] == "not_run"
    write_records(tmp_path, 1, [heart(1, 1.0, requests=1, kv=16384),
                                heart(1, 2.5, requests=1, kv=16384, layer="l7")])
    report = summarize_progress(tmp_path)
    assert report["status"] == "observed"
    assert report["buckets"][0]["arm"] == "outside_arms"
    assert report["buckets"][0]["wall_s_sum"] == 1.5
    assert main([str(tmp_path), "--output", str(tmp_path / "out.json")]) == 0


def test_malformed_progress_record_fails_loudly(tmp_path):
    (tmp_path / "worker-1.jsonl").write_text('{"event": "attention_progress"\n')
    with pytest.raises(ValueError, match="malformed progress record"):
        summarize_progress(tmp_path)
