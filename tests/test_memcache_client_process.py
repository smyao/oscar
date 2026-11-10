"""Archive #94/#125/#177: real child isolation, visible SDK errors and bounded cleanup.

The SDK module below is a temporary fake; no real service, SDK or NPU is used.
"""
import json
import os
from pathlib import Path
import subprocess

import pytest

from tools import memcache_service

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sdk_child(tmp_path, monkeypatch):
    trace = tmp_path / "client.jsonl"
    local = tmp_path / "mmc-local.conf"
    local.touch()
    (tmp_path / "memcache_hybrid.py").write_text('''import json, os, time
def record(event):
    with open(os.environ["SDK_TEST_TRACE"], "a") as out:
        out.write(json.dumps({"pid":os.getpid(), "event":event}) + "\\n")
class DistributedObjectStore:
    def init(self, rank, *, init_bm):
        assert rank == 0 and init_bm is False
        record("init")
        mode = os.environ["SDK_TEST_MODE"]
        print("SDK_TEST_INIT_" + mode, flush=True)
        if mode == "hang": time.sleep(60)
        return -19 if mode == "failure" else 0
    def batch_is_exist(self, keys, flags):
        assert len(keys) == 1 and flags == 0
        record("query")
        return [0]
    def close(self):
        record("close")
''')
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path), str(ROOT))))
    monkeypatch.setenv("SDK_TEST_TRACE", str(trace))
    monkeypatch.setenv("SDK_TEST_MODE", "success")
    return local, trace


def test_real_child_query_does_not_load_sdk_into_launcher(sdk_child, capfd):
    local, trace = sdk_child
    memcache_service.verify_metadata(local)
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [item["event"] for item in events] == ["init", "query", "close"]
    assert len({item["pid"] for item in events}) == 1 and events[0]["pid"] != os.getpid()
    assert "SDK_TEST_INIT_success" in capfd.readouterr().out


def test_real_sdk_child_failure_is_visible_and_not_reuse(sdk_child, monkeypatch, capfd):
    local, trace = sdk_child
    monkeypatch.setenv("SDK_TEST_MODE", "failure")
    with pytest.raises(RuntimeError) as caught:
        memcache_service.verify_metadata(local)
    assert isinstance(caught.value.__cause__, subprocess.CalledProcessError)
    assert caught.value.__cause__.returncode != 0
    assert [json.loads(line)["event"] for line in trace.read_text().splitlines()] == ["init", "close"]
    output = capfd.readouterr()
    assert "SDK_TEST_INIT_failure" in output.out
    assert "metadata client initialization failed: -19" in output.err


def test_hung_sdk_child_is_killed_and_reaped(sdk_child, monkeypatch):
    local, trace = sdk_child
    monkeypatch.setenv("SDK_TEST_MODE", "hang")
    monkeypatch.setattr(memcache_service, "METADATA_TIMEOUT_SECONDS", 1)
    with pytest.raises(RuntimeError) as caught:
        memcache_service.verify_metadata(local)
    assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [item["event"] for item in events] == ["init"]
    with pytest.raises(ProcessLookupError):
        os.kill(events[0]["pid"], 0)
