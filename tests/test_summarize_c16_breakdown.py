"""Archive #70-73/#145: actor accounting is not a cross-core critical path.

Synthetic files exercise summary/validation only; no NPU or model is loaded.
"""
import copy
import json
from pathlib import Path
import sys
import types

import pytest

from tools.summarize_c16_breakdown import summarize, summary_lines, main, _bounded


@pytest.fixture
def schema(monkeypatch):
    module = types.ModuleType("oscar_ascend.c16_timer_schema")
    module.SCHEMA_ID = "test-exclusive-cycle-schema"
    module.PHASE_NAMES = ("qk", "pv", "unpack_k", "cross_core_wait", "task_other")
    module.PHASE_CATEGORIES = {"qk": "compute", "pv": "compute", "unpack_k": "mixed",
                               "cross_core_wait": "wait", "task_other": "mixed"}
    module.CLOCK_HZ = 50_000_000
    module.CLOCK_SOURCE = "fixed native clock test fixture"
    monkeypatch.setitem(sys.modules, "oscar_ascend.c16_timer_schema", module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def actor(schema, engine, phases, *, total=1000, core=0):
    entries = {name: {"cycles": phases.get(name, 0), "count": 1 if phases.get(name, 0) else 0}
               for name in schema.PHASE_NAMES}
    return {"core": core, "engine": engine, "total_cycles": total,
            "other_cycles": total - sum(phases.values()), "phases": entries, "ready": 1, "flags": 0}


def call_record(root, schema, rank=0, index=1, layer=3, draft=False):
    raw = root / "device" / f"call-rank{rank}-{index:04d}-ticks.json"
    write_json(raw, [[[123, 1]]])
    return {"schema_version": 1, "phase_schema_id": schema.SCHEMA_ID,
            "rank": rank, "call_index": index, "layer": f"{'mtp' if draft else 'model.language_model'}.layers.{layer}.self_attn.attn",
            "is_draft": draft, "draft_index": 0, "status": "complete", "counters_valid": True, "counter_issues": [],
            "instrumentation": "local_completion_fences", "inference_uses": "original_production_outputs",
            "baseline_ms": 2.0, "diagnostic_ms": 5.0, "counter_path": str(raw),
            "clock_hz": schema.CLOCK_HZ, "clock_source": schema.CLOCK_SOURCE,
            "timebase_check": {"status": "passed", "max_actor_ms": 1200 * 1000 / schema.CLOCK_HZ, "event_ms": 5.0},
            "build": {"status": "built", "library_sha256": "build-sha"},
            "geometry": {"tokens": 16384, "hq": 6, "hk": 1, "dim": 256, "splits": 1, "requests": 1,
                         "cores": 1, "max_seq_len": 32768, "max_query_len": 16384, "dequant_mode": 0},
            "validation": {"status": "passed", "bitwise": {name: True for name in
                ("partial", "lse", "status", "cluster_stats", "baseline_status_zero")}},
            "actors": [actor(schema, "AIC", {"qk": 400, "pv": 300, "cross_core_wait": 100}),
                       actor(schema, "AIV0", {"unpack_k": 600, "cross_core_wait": 100}),
                       actor(schema, "AIV1", {"unpack_k": 200, "cross_core_wait": 600}, total=1200)]}


def manifest(root, schema, *, ranks=2, layers=(3, 7)):
    result = {"status": "completed", "run_id": "c16-test", "expected_ranks": ranks,
              "device_dir": str(root / "device"), "effective_config": {"tensor_parallel_size": ranks},
              "model_identity": {"full_layers": [f"model.layers.{layer}.self_attn.attn" for layer in layers]},
              "timer_build": {"status": "built", "library_sha256": "build-sha", "phase_schema_id": schema.SCHEMA_ID},
              "workload": {"prompt_tokens": 32768, "output_tokens": 16},
              "service": {"request": {"status": "completed", "completion_check": {"status": "passed"}},
                          "lifecycle": {"cleanup_complete": True}, "resource_release": {"status": "passed"}}}
    write_json(root / "run.json", result)
    return result


def save_call(root, record):
    path = root / "device" / f"call-rank{record['rank']}-{record['call_index']:04d}.json"
    write_json(path, record)
    return path


def complete_fixture(root, schema):
    manifest(root, schema)
    for rank in range(2):
        for index, layer in enumerate((3, 7), 1):
            save_call(root, call_record(root, schema, rank, index, layer))


def test_complete_coverage_and_actor_accounts_not_phase_max_frankenstein(tmp_path, schema):
    complete_fixture(tmp_path, schema)
    report = summarize(tmp_path)
    assert report["status"] == "complete", report["issues"]
    assert len(report["calls"]) == 4  # -ticks.json files were not parsed as calls.
    assert [r["baseline_event_sum_ms"] for r in report["ranks"]] == [4, 4]
    assert [r["diagnostic_event_sum_ms"] for r in report["ranks"]] == [10, 10]
    assert report["ranks"][0]["diagnostic_over_baseline"] == 2.5
    aiv = report["ranks"][0]["primary_actors"]["AIV"]
    assert aiv["engine"] == "AIV1" and aiv["total_cycles"] == 1200
    phase = {p["name"]: p for p in aiv["phases"]}
    assert phase["unpack_k"]["cycles"] == 200  # Not AIV0's larger 600.
    assert phase["cross_core_wait"]["cycles"] == 600
    assert aiv["phase_sum_cycles"] + aiv["other_cycles"] == aiv["total_cycles"]
    assert aiv["categories"]["wait"]["local_share"] == .5
    assert aiv["categories"]["mixed"]["local_share"] == pytest.approx(1 / 6)
    assert report["ranks"][0]["target_aiv_nonwait_phase_winners"] == {"unpack_k": 2}
    assert phase["cross_core_wait"]["diagnostic_ms"] == pytest.approx(.012)
    lines = summary_lines(report)
    assert any("first_inspect=cross_core_wait" in line for line in lines)
    assert any("C16_CONSISTENCY rank=0" in line and '"unpack_k":2' in line for line in lines)
    assert any("Do not multiply shares by old 1199 ms" in line for line in lines)
    assert not any("critical_path_ms" in line for line in lines)


def test_draft_is_observed_not_required_seventeen_or_target_layer_zero(tmp_path, schema):
    complete_fixture(tmp_path, schema)
    assert summarize(tmp_path)["status"] == "complete"
    save_call(tmp_path, call_record(tmp_path, schema, rank=0, index=3, layer=0, draft=True))
    report = summarize(tmp_path)
    assert report["status"] == "complete"
    assert report["ranks"][0]["draft_calls"] == 1
    assert report["ranks"][0]["target_calls"] == 2
    assert report["ranks"][0]["observed_target_layers"] == [3, 7]
    assert report["ranks"][1]["draft_calls"] == 0


def test_missing_rank_and_target_layer_never_complete(tmp_path, schema):
    manifest(tmp_path, schema)
    save_call(tmp_path, call_record(tmp_path, schema))
    report = summarize(tmp_path)
    assert report["status"] == "partial"
    assert "rank_missing:1" in report["issues"]
    assert report["ranks"][0]["missing_target_layers"] == [7]
    assert report["ranks"][1]["missing_target_layers"] == [3, 7]


@pytest.mark.parametrize("mutation", [
    lambda r: r["actors"][0].update(other_cycles=201),
    lambda r: r["actors"][0]["phases"]["qk"].update(count=0),
    lambda r: r["actors"][0]["phases"]["qk"].update(cycles=-1),
    lambda r: r["actors"][0]["phases"].pop("pv"),
    lambda r: r["actors"][0]["phases"].update(fake_phase={"cycles": 1, "count": 1}),
    lambda r: r["actors"].pop(),
    lambda r: r["actors"].append(copy.deepcopy(r["actors"][0])),
    lambda r: r["actors"][0].update(flags=1),
    lambda r: r["actors"][0].update(ready=0),
    lambda r: r["actors"][0].update(engine=[]),
    lambda r: r.update(counters_valid=False),
    lambda r: r.update(phase_schema_id="old"),
    lambda r: r["validation"]["bitwise"].update(partial=False),
    lambda r: r["validation"]["bitwise"].pop("lse"),
    lambda r: r.update(baseline_ms=0),
    lambda r: r.update(diagnostic_ms=float("nan")),
    lambda r: r["build"].update(library_sha256="wrong"),
])
def test_invalid_counter_or_outputs_are_errors(tmp_path, schema, mutation):
    manifest(tmp_path, schema, ranks=1, layers=(3,))
    row = call_record(tmp_path, schema)
    mutation(row)
    save_call(tmp_path, row)
    report = summarize(tmp_path)
    assert report["status"] == "error"
    assert report["calls"][0]["issues"]
    # Invalid calls cannot be chosen as clean representative actors.
    assert not report["ranks"][0]["primary_actors"]
    assert "C16_ERROR" in "\n".join(summary_lines(report))


def test_actual_bytes_preserved_no_invented_traffic(tmp_path, schema):
    manifest(tmp_path, schema, ranks=1, layers=(3,))
    row = call_record(tmp_path, schema)
    row["actors"][2]["phases"]["unpack_k"].update(logical_bytes=8192, byte_scope="recorded logical payload")
    save_call(tmp_path, row)
    report = summarize(tmp_path)
    phases = report["ranks"][0]["primary_actors"]["AIV"]["phases"]
    assert next(p for p in phases if p["name"] == "unpack_k")["logical_bytes"] == 8192
    assert all("bytes" not in p and "logical_bytes" not in p for p in phases if p["name"] != "unpack_k")
    assert any("recorded_bytes=8192" in line for line in summary_lines(report))


def test_failed_request_and_cleanup_do_not_hide_good_counters(tmp_path, schema):
    complete_fixture(tmp_path, schema)
    document = json.loads((tmp_path / "run.json").read_text())
    document["service"]["request"]["completion_check"]["status"] = "failed"
    document["service"]["lifecycle"]["cleanup_complete"] = False
    write_json(tmp_path / "run.json", document)
    report = summarize(tmp_path)
    assert report["status"] == "partial"
    assert "real_32k_request_completion_not_passed" in report["issues"]
    assert "owned_service_or_resource_cleanup_not_passed" in report["issues"]


def test_empty_and_failed_build_directories_still_print_summary(tmp_path, schema, capsys):
    write_json(tmp_path / "run.json", {"status": "failed", "error": "compiler failed"})
    assert main([str(tmp_path)]) == 2
    text = capsys.readouterr().out
    assert "compiler failed" in text
    assert "OSCAR_C16_BREAKDOWN_END" in text
    assert json.loads((tmp_path / "c16-breakdown-summary.json").read_text())["status"] == "error"


def test_counter_file_and_source_schema_are_required(tmp_path, schema):
    manifest(tmp_path, schema, ranks=1, layers=(3,))
    row = call_record(tmp_path, schema)
    Path(row["counter_path"]).unlink()
    save_call(tmp_path, row)
    assert summarize(tmp_path)["status"] == "error"


def test_terminal_line_and_byte_budget_preserves_errors_coverage_paths():
    lines = ["=== OSCAR_C16_BREAKDOWN_BEGIN ===", "C16_RUN x status=error"]
    lines += [f"C16_CALL {i} " + "数据🔥" * 500 for i in range(200)]
    lines += ["C16_COVERAGE rank=3 missing=[63]", "C16_ERROR failed validation", "C16_FILES /tmp/raw", "=== OSCAR_C16_BREAKDOWN_END ==="]
    result = _bounded(lines)
    assert len(result) <= 150
    assert len(("\n".join(result) + "\n").encode()) <= 40 * 1024
    assert all(line in result for line in lines[-4:])
    assert result[-1] == lines[-1]


def test_main_writes_json_without_modifying_evidence(tmp_path, schema, capsys):
    complete_fixture(tmp_path, schema)
    before = (tmp_path / "run.json").read_bytes()
    assert main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "C16_PHASES" in output and "C16_CATEGORIES" in output
    assert (tmp_path / "run.json").read_bytes() == before
    report = json.loads((tmp_path / "c16-breakdown-summary.json").read_text())
    assert report["status"] == "complete"


def test_real_phase_schema_sixty_four_target_calls_fit_bounded_summary(tmp_path):
    import oscar_ascend.c16_timer_schema as live_schema

    assert live_schema.PHASE_NAMES[19] == "bf16_cast"
    assert live_schema.PHASE_CATEGORIES["bf16_cast"] == "compute"
    layers = tuple(range(3, 64, 4))
    manifest(tmp_path, live_schema, ranks=4, layers=layers)
    for rank in range(4):
        for index, layer in enumerate(layers, 1):
            row = call_record(tmp_path, live_schema, rank=rank, index=index, layer=layer)
            row["actors"][2]["phases"]["bf16_cast"] = {"cycles": 40, "count": 2}
            row["actors"][2]["other_cycles"] -= 40
            save_call(tmp_path, row)
    report = summarize(tmp_path)
    assert report["status"] == "complete", report["issues"]
    assert len(report["phase_categories"]) == 24
    assert len(report["calls"]) == 64
    lines = summary_lines(report)
    assert any("bf16_cast=" in line and "compute" in line for line in lines if line.startswith("C16_PHASES"))
    assert sum(line.startswith("C16_CALL ") for line in lines) == 64
    assert len(lines) <= 150
    assert len(("\n".join(lines) + "\n").encode()) <= 40 * 1024


@pytest.mark.parametrize("key,value", [("clock_hz", 25_000_000), ("clock_source", "unverified"),
    ("timebase_check", {"status": "failed", "max_actor_ms": .024, "event_ms": 5.0})])
def test_unverified_clock_never_creates_phase_milliseconds(tmp_path, schema, key, value):
    manifest(tmp_path, schema, ranks=1, layers=(3,))
    row = call_record(tmp_path, schema)
    row[key] = value
    save_call(tmp_path, row)
    report = summarize(tmp_path)
    assert report["status"] == "error"
    assert not report["calls"][0]["timebase_valid"]
    assert all(phase["diagnostic_ms"] is None for a in report["calls"][0]["actors"] for phase in a["phases"])
    assert report["calls"][0]["actors"][0]["total_cycles"] == 1000
