"""Archive #27/#37-49/#127/#128/#157/#163/#171: do not fake service evidence.

Host-only evidence validators and capacity planning. No server/NPU is started.
"""
import copy
import json
from pathlib import Path

import pytest

from tools.probe_kv_offload_service import (
    read_geometry, read_pool_states, pressure_workload, pressure_plan, require_swap_evidence, require_prefix_evidence,
    _mode_config, _probe_environment, _request, PREEMPTIONS,
)
from tools.service_probe import ServiceProbeError


def geometry(mode="align"):
    return {"rank": 0, "num_blocks": 600, "mamba_cache_mode": mode, "groups": [
        {"group_id": 0, "family": "oscar_full", "block_size": 2304, "num_speculative_blocks": 0, "layer_count": 16},
        *[{"group_id": i, "family": "gdn", "block_size": 768 if mode == "align" else 16384,
           "num_speculative_blocks": 3, "layer_count": 16} for i in range(1, 4)]]}


def test_geometry_requires_complete_consistent_tp_evidence():
    rows = ["(TP) [oscar] KV_OFFLOAD_GEOMETRY " + json.dumps(dict(geometry(), rank=rank)) for rank in range(4)]
    assert read_geometry("\n".join(rows))["num_blocks"] == 600
    with pytest.raises(ServiceProbeError, match="missing TP"):
        read_geometry("\n".join(rows[:3]))
    rows[3] = rows[3].replace('"num_blocks": 600', '"num_blocks": 500')
    with pytest.raises(ServiceProbeError, match="inconsistent"):
        read_geometry("\n".join(rows))


def pool_records(*, pre_gdn=(4, 4, 4), post_gdn=None, null_id=0):
    post_gdn = pre_gdn if post_gdn is None else post_gdn
    rows = []
    for request in ("request-one", "request-two"):
        for base, scheduled, full_pages, gdn in ((0, 1024, 1, pre_gdn),
                (1024, 1024, 1, pre_gdn), (2048, 192, 1, pre_gdn),
                # Lookahead already owns FULL page2, before end reaches2304.
                (2298, 4, 2, post_gdn), (2362, 4, 2, post_gdn)):
            page = 1
            groups = []
            for count in (full_pages, *gdn):
                groups.append(list(range(page, page + count)))
                page += count
            groups[-1] += [null_id, null_id]  # Native logical padding is not ownership.
            rows.append({"request": request, "base": base, "scheduled": scheduled,
                "end": base + scheduled, "groups": groups,
                "live_pages": page - 1, "null_id": null_id})
    return rows


def pool_log(rows):
    return "\n".join("[scheduler] KV_OFFLOAD_POOL_STATE " + json.dumps(row) for row in rows)


@pytest.mark.parametrize("mode,gdn,capacity", [("none", (4, 4, 4), 27),
    ("align", (5, 5, 5), 33), ("align", (2, 5, 1), 19)])
def test_pressure_uses_measured_native_owned_pages_not_state_or_layer_estimate(mode, gdn, capacity):
    observed = geometry(mode)
    rows = read_pool_states(pool_log(pool_records(pre_gdn=gdn)), observed)
    plan = pressure_plan(observed, mode, rows)
    assert plan["num_gpu_blocks"] == capacity
    assert plan["pre_cross_peak"] == 1 + sum(gdn)
    assert plan["post_cross_peak"] == 2 + sum(gdn)
    assert plan["prompt_tokens"] < 2304 < plan["prompt_tokens"] + plan["output_tokens"]
    assert plan["both_crossed_pages"] > capacity
    assert plan["measured_requests"]["request-one"]["post_steps"] == 2
    assert plan["baseline_pool_records"] == rows
    assert plan["scope"].endswith("observed preemption mandatory")


def test_pool_records_require_real_ownership_consistent_end_and_shared_null():
    observed = geometry()
    rows = pool_records(null_id=599)
    assert read_pool_states(pool_log(rows), observed) == rows
    for field, bad in (("end", 1025), ("live_pages", 14), ("null_id", 600),
                       ("scheduled", 0), ("groups", [[1]]), ("request", "")):
        changed = copy.deepcopy(rows)
        changed[0][field] = bad
        with pytest.raises(ServiceProbeError, match="pool-state"):
            read_pool_states(pool_log(changed), observed)
    with pytest.raises(ServiceProbeError, match="missing native"):
        read_pool_states("service ready", observed)
    changed = copy.deepcopy(rows)
    changed[0]["groups"][-1][-2:] = [0, 0]
    changed[0]["null_id"] = 0
    with pytest.raises(ServiceProbeError, match="null page changed"):
        read_pool_states(pool_log(changed), observed)


def test_pressure_requires_two_complete_observed_transitions_and_increasing_demand():
    observed = geometry()
    rows = pool_records()
    with pytest.raises(ServiceProbeError, match="both serial"):
        pressure_plan(observed, "align", rows[:5])
    with pytest.raises(ServiceProbeError, match="both sides"):
        pressure_plan(observed, "align", [row for row in rows if len(row["groups"][0]) == 1])
    no_increase = read_pool_states(pool_log(pool_records(post_gdn=(3, 4, 4))), observed)
    with pytest.raises(ServiceProbeError, match="did not increase"):
        pressure_plan(observed, "align", no_increase)
    with pytest.raises(ServiceProbeError, match="baseline pool"):
        pressure_plan(dict(observed, num_blocks=27), "align", rows)


@pytest.mark.parametrize("actual,requested", [("none", "align"), ("align", "none"), (None, "none")])
def test_probe_cannot_claim_an_unobserved_mamba_mode(actual, requested):
    observed = dict(geometry(), mamba_cache_mode=actual)
    with pytest.raises(ServiceProbeError, match="mamba_cache_mode"):
        pressure_workload(observed, requested)


def test_geometry_requires_same_mamba_mode_on_every_worker():
    values = [dict(geometry(), rank=rank) for rank in range(4)]
    values[3]["mamba_cache_mode"] = "none"
    text = "\n".join("KV_OFFLOAD_GEOMETRY " + json.dumps(value) for value in values)
    with pytest.raises(ServiceProbeError, match="inconsistent"):
        read_geometry(text)
    for value in values:
        value.pop("mamba_cache_mode")
    with pytest.raises(ServiceProbeError, match="incomplete"):
        read_geometry("\n".join("KV_OFFLOAD_GEOMETRY " + json.dumps(value) for value in values))


def swap_logs(version="1", missing_rank=None):
    return "\n".join(
        f"KV_OFFLOAD_{event} request=req-a tokens=2307 rank={rank} generation=3 version={ver}{kind}"
        for rank in range(4) if rank != missing_rank
        for event, ver, kind in (("SWAP_SAVE", "1", ""), ("RESTORE", version, " kind=swap")))


def test_preemption_requires_native_counter_and_same_generation_joint_restore():
    before, after = f"{PREEMPTIONS} 0\n", f"{PREEMPTIONS} 1\n"
    assert require_swap_evidence(swap_logs(), before, after)["native_preemptions_delta"] == 1
    for events in (swap_logs(missing_rank=2), swap_logs(version="2"), "KV_OFFLOAD_READY rank=0"):
        with pytest.raises(ServiceProbeError, match="same-generation"):
            require_swap_evidence(events, before, after)
    with pytest.raises(ServiceProbeError, match="counter"):
        require_swap_evidence(swap_logs(), before, before)


def test_http200_and_local_cache_hit_are_not_external_restore_evidence():
    with pytest.raises(ServiceProbeError, match="external joint-prefix"):
        require_prefix_evidence("HTTP200 reset_prefix_cache success")
    logs = "KV_OFFLOAD_PREFIX_HIT request=req tokens=2304\n" + "\n".join(
        f"KV_OFFLOAD_RESTORE request=req tokens=2304 kind=prefix rank={rank}" for rank in range(4))
    assert require_prefix_evidence(logs)["joint_restores"] == [["req", "2304"]]


def test_probe_mode_does_not_change_production_config_or_native_mtp_graph(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "configs/target.json").read_text())
    prior = json.dumps(original, sort_keys=True)
    for mode in ("none", "align"):
        config = _mode_config(original, mode)
        assert config["speculative_config"] == original["speculative_config"]
        assert config["compilation_config"] == original["compilation_config"]
        assert config["devices"] == original["devices"]
        assert config["enable_prefix_caching"] == (mode == "align")
        assert config["max_num_batched_tokens"] == 1024
    assert json.dumps(original, sort_keys=True) == prior
    monkeypatch.delenv("OSCAR_OFFLOAD_SERVICE_PROBE", raising=False)
    import os
    with _probe_environment():
        assert os.environ["OSCAR_OFFLOAD_SERVICE_PROBE"] == "1"
    assert "OSCAR_OFFLOAD_SERVICE_PROBE" not in os.environ


def test_service_probe_requires_generated_token_ids_not_only_text(monkeypatch):
    monkeypatch.setattr("tools.probe_kv_offload_service.request_with_deadline", lambda *args, **kwargs:
        {"choices": [{"text": "looks correct"}], "usage": {"prompt_tokens": 2, "completion_tokens": 16}})
    with pytest.raises(ServiceProbeError, match="generated-token"):
        _request(None, {"served_model_name": "model"}, [1, 2], output_tokens=16, label="proof", timeout=5)


@pytest.mark.parametrize("disabled_stats", [False, True])
def test_service_workflow_sizes_pressure_after_observing_baseline_and_preserves_geometry(tmp_path, monkeypatch, disabled_stats):
    """Fake only external services; exercise actual evidence readers and orchestration."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    import tools.probe_kv_offload_service as probe

    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "input.json"
    config_path.write_text(json.dumps(dict(json.loads((root / "configs/target.json").read_text()),
                                          disable_log_stats=disabled_stats)))
    launches = []
    completed_serial = []
    monkeypatch.setattr(probe, "target_env", lambda config: {})
    monkeypatch.setattr(probe, "load_tokenizer", lambda model: SimpleNamespace(encode=lambda *a, **k: [4]))
    monkeypatch.setattr(probe, "exact_prompt", lambda tokenizer, length: [1] * length)
    monkeypatch.setattr(probe, "_http", lambda *a, **k: "")
    def metrics(server, directory, name, timeout):
        assert not disabled_stats, "disabled stats must use host events"
        return f"{PREEMPTIONS} {int(name == 'after')}\n"
    monkeypatch.setattr(probe, "_metrics", metrics)
    monkeypatch.setattr(probe, "evaluate_mtp_metrics", lambda before, after: {"status": "passed"})

    @contextmanager
    def owned(config, directory, report):
        mode = "align" if config["enable_prefix_caching"] else "none"
        pressure = "offload_probe_num_gpu_blocks" in config
        if pressure:
            assert len([label for label in completed_serial if label.startswith(mode)]) == 2
        launches.append((mode, pressure, copy.deepcopy(config)))
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / "server.log"
        observed = dict(geometry(mode), num_blocks=config.get("offload_probe_num_gpu_blocks", 600))
        log.write_text("\n".join("KV_OFFLOAD_GEOMETRY " + json.dumps(dict(observed, rank=rank))
                                  for rank in range(4)) + "\n")
        lifecycle = {}
        if disabled_stats:
            from tools.probe_evidence import configure
            configure({}, lifecycle, directory)
        yield SimpleNamespace(log=log, base_url="http://unit-test.invalid", lifecycle=lifecycle)

    def request(server, config, tokens, *, output_tokens, label, timeout):
        events = []
        if "serial" in label:
            records = pool_records(pre_gdn=(2, 3, 4))[:5]
            events.append(pool_log([dict(row, request=label) for row in records]))
            completed_serial.append(label)
        elif "pressure" in label:
            events.append(swap_logs().replace("req-a", label))
            if disabled_stats:
                from oscar_ascend.probe_evidence import _emit
                info = server.lifecycle["probe_evidence"]
                scope = json.loads((Path(info["directory"]) / "active.json").read_text())
                window = (Path(info["directory"]), info["run_id"], scope["window_id"])
                _emit(window, "preempted", label, preemptions_before=0, preemptions_after=1)
                _emit(window, "mtp_verified", label, scheduled_draft_slots=3,
                      num_accepted_tokens=0, invalid_draft_slots=5)
        elif label == "prefix-first":
            events.append("KV_OFFLOAD_PREFIX_PUBLISHED request=prefix tokens=2304")
        elif label == "prefix-external":
            events.append("KV_OFFLOAD_PREFIX_HIT request=prefix tokens=2304")
            events.extend(f"KV_OFFLOAD_RESTORE request=prefix tokens=2304 kind=prefix rank={rank}"
                          for rank in range(4))
        with server.log.open("a") as stream:
            stream.write("\n".join(events) + "\n")
        return {"generated_token_ids": [tokens[0]] * output_tokens}

    monkeypatch.setattr(probe, "_owned", owned)
    monkeypatch.setattr(probe, "_request", request)
    report = probe.run_probe(config_path, tmp_path / "probe")
    assert report["status"] == "passed"
    for mode in ("none", "align"):
        baseline = next(config for name, pressure, config in launches if name == mode and not pressure)
        pressure = next(config for name, pressure, config in launches if name == mode and pressure)
        assert pressure.pop("offload_probe_num_gpu_blocks") == 21  # 1 + 2 * observed10.
        assert pressure == baseline
        assert pressure["max_model_len"] == 16384
        assert report["arms"][mode]["pressure_plan"]["pre_cross_peak"] == 10
        if disabled_stats:
            assert report["arms"][mode]["pressure"]["mtp"]["source"] == "native_scheduler_probe_events"
            assert report["arms"][mode]["pressure"]["swap_evidence"]["completed_preemptions"] == 2
