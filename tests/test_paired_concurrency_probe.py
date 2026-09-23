"""Archive #70-#73/#94/#95/#125/#129-#139: one-call pairing and fail-closed cleanup."""

import json

from benchmarks.compare import canonical_sha256
from tools import paired_concurrency_probe as paired


def _report(variant, *, latency=1.0, throughput=1.0, tpot=10.0):
    requests = []
    for index, length in enumerate((20000, 23000, 27000, 30000)):
        requests.append({"request_id": f"synthetic-{index}-{length}",
                         "status": "completed", "prompt_sha256": f"prompt-{index}",
                         "prompt_tokens": length, "ttft_ms": 100.0 * latency,
                         "tpot_ms": tpot * latency, "e2e_ms": 200.0 * latency,
                         "cache_salt_sha256": canonical_sha256(f"salt-{index}")})
    sample = {"repeat": 0, "status": "completed", "completed_requests": 4,
              "failed_requests": 0, "timeouts": 0, "requests": requests,
              "throughput_tps": {"prompt": 5000.0 * throughput,
                                 "generation": 50.0 * throughput},
              "latency_ms": {name: {"p50": value * latency, "p95": value * latency}
                             for name, value in (("ttft_ms", 100), ("tpot_ms", tpot),
                                                 ("e2e_ms", 200))}}
    manifest = [{"id": row["request_id"], "prompt_tokens": row["prompt_tokens"],
                 "prompt_sha256": row["prompt_sha256"]} for row in requests]
    identity = {"target_config_sha256": "same-config", "model_fingerprint": "same-model",
                "manifest": manifest, "output_tokens": 64,
                "arrival": "simultaneous_barrier", "cache_salt": "distinct deterministic per request"}
    return {"mode": "synthetic_mixed", "variant": variant, "status": "measured",
            "synthetic_mixed": {"status": "measured", "pair_sha256": canonical_sha256(identity),
                                "pair_identity": identity,
                                "prompt_lengths": [20000, 23000, 27000, 30000],
                                "prompt_manifest": manifest, "sample": sample}}


def _acceptance():
    return {"frozen_before_measurement": True,
            "performance": {"max_latency_ratio": 1.0,
                            "min_throughput_ratio": 1.0}}


def test_paired_comparison_requires_every_ratio_and_exact_workload():
    native = _report("native")
    oscar = _report("oscar", latency=.8, throughput=1.2)
    result = paired.compare_synthetic_reports(native, oscar, _acceptance())
    assert result["status"] == "passed"
    assert result["performance_acceptance"] == "not_run"
    assert len(result["batches"][0]["requests"]) == 4
    oscar["synthetic_mixed"]["sample"]["requests"][2]["ttft_ms"] = 101.0
    result = paired.compare_synthetic_reports(native, oscar, _acceptance())
    assert result["status"] == "failed"
    assert any("ttft_ms" in issue for issue in result["issues"])
    oscar["synthetic_mixed"]["prompt_manifest"][0]["prompt_sha256"] = "different"
    assert paired.compare_synthetic_reports(native, oscar, _acceptance())["status"] == "failed"


def test_zero_tpot_fails_closed_and_full_service_warmup_is_disclosed():
    native = _report("native", tpot=0.0)
    oscar = _report("oscar", tpot=0.0)
    full_service = {"status": "passed", "server": {"variant": "oscar"},
                    "performance": oscar["synthetic_mixed"]}
    result = paired.compare_synthetic_reports(native, full_service, _acceptance())
    assert result["warmup_pairing"] == "unpaired"
    assert result["performance_acceptance"] == "not_run"
    assert result["status"] == "needs_evidence"
    assert any("tpot_unresolved_SSE_burst" in issue for issue in result["issues"])


def test_native_failure_stops_before_oscar_and_preserves_child_exit(tmp_path, monkeypatch):
    config_path = tmp_path / "target.json"
    acceptance_path = tmp_path / "acceptance.json"
    config_path.write_text(json.dumps({"devices": [0, 1, 2, 3]}))
    acceptance_path.write_text(json.dumps(_acceptance()))
    variants = []

    def variant(name, *_args):
        variants.append(name)
        return {"status": "failed", "returncode": 7, "probe_error": "device failure",
                "log": str(tmp_path / "native.log"), "report": str(tmp_path / "native.json"),
                "owned_server_cleanup_complete": True, "runner_cleanup_complete": True}

    monkeypatch.setattr(paired, "_run_variant", variant)
    monkeypatch.setattr(paired, "_observe_resources",
                        lambda *_args, before=None: ({"status": "passed"} if before else
                                                     {"devices": [0, 1, 2, 3], "memory": []}))
    report = paired.run_paired(config_path, output=tmp_path / "paired.json",
                               log_dir=tmp_path / "logs", acceptance_path=acceptance_path)
    assert variants == ["native"]
    assert report["status"] == "failed" and report["exit_code"] == 7
    assert report["oscar"] == "not_run"
    assert json.loads((tmp_path / "paired.json").read_text())["exit_code"] == 7


def test_native_only_requires_release_before_reporting_success(tmp_path, monkeypatch):
    config_path = tmp_path / "target.json"
    acceptance_path = tmp_path / "acceptance.json"
    config_path.write_text(json.dumps({"devices": [0, 1, 2, 3]}))
    acceptance_path.write_text(json.dumps(_acceptance()))
    monkeypatch.setattr(paired, "_run_variant", lambda name, *_args: {
        "status": "passed", "returncode": 0, "probe_error": None,
        "log": str(tmp_path / "native.log"), "report": str(tmp_path / "native.json"),
        "owned_server_cleanup_complete": True, "runner_cleanup_complete": True})
    monkeypatch.setattr(paired, "_observe_resources",
                        lambda *_args, before=None: ({"status": "failed", "reason": "memory held"}
                                                     if before else {"devices": [0, 1, 2, 3], "memory": []}))
    report = paired.run_paired(config_path, output=tmp_path / "paired.json",
                               log_dir=tmp_path / "logs", acceptance_path=acceptance_path,
                               native_only=True)
    assert report["status"] == "failed"
    assert report["native"]["resource_release"] == "failed"
    assert report["oscar"] == "not_run"


def test_one_call_runs_native_then_oscar_and_writes_paired_ratios(tmp_path, monkeypatch):
    config_path = tmp_path / "target.json"
    acceptance_path = tmp_path / "acceptance.json"
    config_path.write_text(json.dumps({"devices": [0, 1, 2, 3]}))
    acceptance_path.write_text(json.dumps(_acceptance()))
    order = []

    def variant(name, _config_path, _config, directory):
        order.append(name)
        directory.mkdir(parents=True)
        path = directory / "report.json"
        path.write_text(json.dumps(_report(name, latency=.8 if name == "oscar" else 1,
                                           throughput=1.2 if name == "oscar" else 1)))
        return {"status": "passed", "returncode": 0, "probe_error": None,
                "log": str(directory / "console.log"), "report": str(path),
                "owned_server_cleanup_complete": True, "runner_cleanup_complete": True}

    monkeypatch.setattr(paired, "_run_variant", variant)
    monkeypatch.setattr(paired, "_observe_resources",
                        lambda *_args, before=None: ({"status": "passed"} if before else
                                                     {"devices": [0, 1, 2, 3], "memory": []}))
    report = paired.run_paired(config_path, output=tmp_path / "paired.json",
                               log_dir=tmp_path / "logs", acceptance_path=acceptance_path)
    assert order == ["native", "oscar"]
    assert report["status"] == "passed" and report["exit_code"] == 0
    assert report["comparison"]["status"] == "passed"
    assert report["comparison"]["warmup_pairing"] == "fresh_service_before_batch"
    assert report["comparison"]["performance_acceptance"] == "not_run"
    assert (tmp_path / "logs/comparison.json").exists()
