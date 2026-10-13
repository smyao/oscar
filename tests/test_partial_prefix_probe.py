"""Archive #94/#95/#125/#127/#128: probe evidence must distinguish absent states.

These are client orchestration tests, not a model or NPU execution substitute.
"""
import json
from types import SimpleNamespace

import pytest

from tools import partial_prefix_probe as probe


def result(observed=None, controlled=11520):
    return {"plan": probe.plan(), "requests": {
        "long_seed": {"cached_tokens": None}, "partial_after_long_seed": {"cached_tokens": observed},
        "checkpoint_anchor": {"cached_tokens": 0}, "partial_after_anchor": {"cached_tokens": controlled}}}


@pytest.mark.parametrize("observed", [None, 0, 2304, 6912, 11520])
def test_sparse_zero_is_not_a_bug_claim_but_controlled_hit_is_required(observed):
    report = probe.validate(result(observed))
    assert report["controlled_hit"] == "passed"
    assert report["observed_partial_hit"] == observed
    assert report["coordinator_bug_established"] is False


@pytest.mark.parametrize("controlled", [None, 0, True, 6912, 128, 13824])
def test_missing_or_inconsistent_controlled_witness_fails(controlled):
    with pytest.raises(ValueError):
        probe.validate(result(controlled=controlled))


@pytest.mark.parametrize("arm,hit", [("long_seed",2304),("checkpoint_anchor",2304),
    ("partial_after_long_seed",13824),("partial_after_long_seed",-1),("partial_after_long_seed",True)])
def test_wrong_salt_or_unsafe_partial_boundary_fails(arm, hit):
    report = result(); report["requests"][arm]["cached_tokens"] = hit
    with pytest.raises(ValueError):
        probe.validate(report)


def test_four_bounded_requests_have_two_isolated_salts_and_exact_fork(tmp_path, monkeypatch):
    calls = []
    def http(directory, name, task):
        if name == "tokenize":
            return {"body": {"tokens": [3, 5, 7]}}
        calls.append((name, task))
        return {"status": "completed", "cached_tokens": 11520 if name == "partial_after_anchor" else None}
    monkeypatch.setattr(probe, "isolated_http", http)
    args = SimpleNamespace(seed_tokens=32768, shared_tokens=13824, output=tmp_path/"evidence",
        base_url="http://127.0.0.1:7878", model="qwen3.5", output_tokens=8, timeout=120, total_timeout=600)
    assert probe.run(args) == 0
    names, tasks = zip(*calls)
    payloads = [t["payload"] for t in tasks]
    assert list(names) == probe.plan()["sequence"]
    assert [len(p["prompt"]) for p in payloads] == [32768,32768,12288,32768]
    assert all(0 < t["timeout"] <= 120 for t in tasks)
    a, b, c, d = payloads
    assert a["cache_salt"] == b["cache_salt"] != c["cache_salt"] == d["cache_salt"]
    assert a["prompt"][:13824] == b["prompt"][:13824]
    assert a["prompt"][13824] != b["prompt"][13824]
    assert c["prompt"] == a["prompt"][:12288] == d["prompt"][:12288]
    report = json.loads((args.output/"summary.json").read_text())
    assert report["status"] == "completed" and report["model_quality"] == "not_established"


def test_plan_does_not_create_output_or_send_http(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(probe, "isolated_http", lambda *a: pytest.fail("plan sent HTTP"))
    assert probe.main(["--output", str(tmp_path/"unused")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["requests"] == 4 and report["expected_controlled_hit"] == 11520
    assert not (tmp_path/"unused").exists()


def test_failed_http_is_recorded_and_stops_later_requests(tmp_path, monkeypatch):
    calls = []
    def http(directory, name, task):
        calls.append(name)
        if name == "tokenize": return {"body": {"tokens": [3, 5]}}
        raise RuntimeError("HTTP child rc=1")
    monkeypatch.setattr(probe, "isolated_http", http)
    args = SimpleNamespace(seed_tokens=32768, shared_tokens=13824, output=tmp_path/"evidence",
        base_url="http://127.0.0.1:7878", model="qwen3.5", output_tokens=8, timeout=120, total_timeout=600)
    assert probe.run(args) == 1 and calls == ["tokenize", "long_seed"]
    report = json.loads((args.output/"summary.json").read_text())
    assert report["status"] == "failed" and "HTTP child rc=1" in report["error"]
