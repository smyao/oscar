"""Archive #26/#37-49/#127/#128/#157: APC probe host contracts, not NPU evidence."""
import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools import probe_prefix_cache as probe
from tools.serving_variants import variant_config


def candidate():
    return variant_config(json.loads((probe.ROOT / "configs/target.json").read_text()), "candidate")


def synthetic_report():
    return {"status": "passed", "scope": probe.SCOPE, "prefix_caching": True,
        "device_completion": "passed", "precision": "passed", "graph_capture": "passed",
        "graph_replay": "passed", "graph_same_addresses": True, "graph_eager_bitwise": "passed",
        "writer": "rotate_clip_store_striped_out", "rotation_scope": "identity_storage_contract",
        "cases": [dict(case=name, contexts=contexts, status="passed", oracle_queries=8,
            shared_pages_unchanged=True, suffix_isolation=True, source_leaders={"0": 2, "1": 2, "2": 2},
            input_sha256="input" + str(i), tasks_sha256="metadata" + str(i))
            for i, (name, contexts) in enumerate(zip(probe.STATES, ([4608, 4608], [4608, 2304], [512, 2304])))]}


def test_fixture_uses_true_snapshot_alias_and_two_private_tails():
    layout = probe.fixture_layout()
    assert layout.page.block_size == 2304 and layout.page.virtual_blocks_per_page == 18
    raw = torch.zeros(probe.BLOCKS * layout.page.page_size_bytes, dtype=torch.uint8)
    key, value, tags = layout.views(raw, probe.BLOCKS)
    assert key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() == raw.data_ptr()
    logical = tuple((torch.zeros((4608, 1, 256)), torch.zeros((4608, 1, 256))) for _ in range(2))
    inputs = probe.make_inputs(torch, logical, ((3, 1, 5), (3, 1, 2)), 1203)
    assert torch.equal(inputs["table"][0, :36], inputs["table"][1, :36])
    assert inputs["slots"].tolist() == list(range(5 * 2304, 5 * 2304 + 4)) + list(range(2 * 2304, 2 * 2304 + 4))
    key[5, 64:68].fill_(9); value[2, 64:68].fill_(8); tags[5, 64:68] = torch.arange(4)
    assert torch.count_nonzero(probe.page_bytes(torch, raw, layout, 3)) == 0
    assert torch.count_nonzero(probe.page_bytes(torch, raw, layout, 1)) == 0


def test_changed_contexts_preserve_buffer_shapes_and_change_virtual_tables():
    generator = torch.Generator().manual_seed(711)
    shared = tuple(torch.randn((4608, 1, 256), generator=generator).to(torch.bfloat16) for _ in range(2))
    full = probe.make_inputs(torch, (shared, shared), ((3, 1, 5), (3, 1, 2)), 33)
    partial = probe.make_inputs(torch, (shared, tuple(t[:2304] for t in shared)), ((3, 1, 5), (3, 4)), 34)
    reused = probe.make_inputs(torch, (tuple(t[:512] for t in shared), tuple(t[:2304] for t in shared)), ((5,), (3, 4)), 35)
    assert {k: (v.shape, v.dtype) for k, v in full.items()} == {k: (v.shape, v.dtype) for k, v in partial.items()}
    assert {k: (v.shape, v.dtype) for k, v in full.items()} == {k: (v.shape, v.dtype) for k, v in reused.items()}
    assert partial["lens"].tolist() == [4612, 2308]
    assert reused["lens"].tolist() == [516, 2308]
    assert not torch.equal(full["table"], partial["table"])
    assert not torch.equal(partial["table"], reused["table"])


def test_oracle_is_finite_and_changes_for_new_page_owner():
    generator = torch.Generator().manual_seed(712)
    logical = tuple(tuple(torch.randn((512, 1, 256), generator=generator).to(torch.bfloat16) for _ in range(2)) for _ in range(2))
    inputs = probe.make_inputs(torch, logical, ((3,), (2,)), 45)
    before, lse = probe.oracle(torch, logical, inputs)
    changed = ((logical[0][0], -logical[0][1]), logical[1])
    after, _ = probe.oracle(torch, changed, inputs)
    assert before.shape == (8, 6, 256) and lse.shape == (8, 6)
    assert torch.isfinite(before).all() and torch.isfinite(lse).all()
    assert not torch.equal(before[:4], after[:4]) and torch.equal(before[4:], after[4:])


def test_prefix_gate_uses_candidate_q4_and_rejects_research_layout():
    config = candidate()
    assert probe.selected_reader(config) == "attention_cv_mdl_decode_out"
    config["experimental_mdl_decode"] = False
    assert probe.selected_reader(config) == "attention_cv_bundle_decode_out"
    for key, value in (("enable_prefix_caching", False), ("experimental_request_windows", True)):
        invalid = candidate(); invalid[key] = value
        with pytest.raises(ValueError): probe.selected_reader(invalid)


def test_complete_device_graph_report_cannot_be_replaced_by_host_summary():
    report = synthetic_report()
    assert probe.validate_evidence(report) == []
    for key in ("device_completion", "graph_replay", "precision", "graph_same_addresses", "rotation_scope"):
        broken = copy.deepcopy(report); broken[key] = "host_passed"
        assert probe.validate_evidence(broken)
    for key in ("shared_pages_unchanged", "suffix_isolation", "oracle_queries"):
        broken = copy.deepcopy(report); broken["cases"][0][key] = False
        assert probe.validate_evidence(broken)
    broken = copy.deepcopy(report); broken["cases"][2]["input_sha256"] = broken["cases"][0]["input_sha256"]
    assert "changed_inputs" in probe.validate_evidence(broken)
    broken = copy.deepcopy(report); broken["cases"][1]["source_leaders"]["1"] = 0
    assert any("sources" in name for name in probe.validate_evidence(broken))


def test_source_binding_exists_and_follows_current_source():
    assert len(probe.source_binding()) == len(probe.SOURCE_PATHS)


def test_no_real_npu_is_failure(monkeypatch, tmp_path):
    path = tmp_path / "config.json"; path.write_text(json.dumps(candidate()))
    monkeypatch.setattr(probe.reuse, "_active_device", lambda _: None)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False)
    with pytest.raises(RuntimeError, match="real NPU"):
        probe.probe(path, probe.ROOT / "configs/acceptance.json")


def test_cli_prints_actual_error_and_preserves_failed_report(monkeypatch, tmp_path, capsys):
    def fail(*args): raise RuntimeError("intentional prefix writer failure")
    monkeypatch.setattr(probe, "probe", fail)
    output = tmp_path / "report.json"
    assert probe.main(["--config", str(tmp_path / "config.json"), "--output", str(output)]) == 2
    assert json.loads(output.read_text())["first_error"] == "intentional prefix writer failure"
    captured = capsys.readouterr()
    assert "intentional prefix writer failure" in captured.err and "PERF_PREFIX_CACHE" in captured.out


def test_observe_requires_current_reader_source_and_artifact(monkeypatch, tmp_path):
    from tools import observe_serve as observe
    from oscar_ascend.ops import loader
    config = candidate()
    manifest = {"signature": "current", "sha256": {"extension": "current-extension"}}
    monkeypatch.setattr(loader, "validate_build_artifacts", lambda *args: manifest)
    monkeypatch.setattr(observe, "read_npu_resources", lambda *args, **kwargs: {})
    monkeypatch.setattr(observe, "wait_for_release", lambda *args, **kwargs: {"status": "passed"})
    report = synthetic_report()
    report.update(artifact_signature=manifest["signature"], artifact_sha256=manifest["sha256"],
                  source_binding=probe.source_binding(), reader=probe.selected_reader(config))
    def phase(name, argv, **kwargs):
        assert name == "prefix-cache-npu" and "tools.probe_prefix_cache" in argv
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report))
    monkeypatch.setattr(observe, "_phase", phase)
    status = {}
    observe._prefix_cache_gate(tmp_path / "config.json", config, {}, tmp_path, status)
    assert status["prefix_cache_gate"]["status"] == "passed"
    for key in ("reader", "source_binding", "artifact_signature", "graph_replay"):
        saved = report[key]; report[key] = "stale"
        with pytest.raises(RuntimeError, match=key):
            observe._prefix_cache_gate(tmp_path / "config.json", config, {}, tmp_path, {})
        report[key] = saved


def test_observe_prefix_failure_releases_resources_and_direct_keeps_same_route(monkeypatch, tmp_path, capsys):
    from tools import observe_serve as observe, install_serve
    calls = []
    monkeypatch.setattr(observe, "read_npu_resources", lambda *args, **kwargs: {})
    def release(*args, **kwargs):
        calls.append("release"); return {"status": "passed"}
    def fail(*args, **kwargs):
        raise RuntimeError("actual prefix writer failure")
    monkeypatch.setattr(observe, "wait_for_release", release)
    monkeypatch.setattr(observe, "_phase", fail)
    with pytest.raises(RuntimeError, match="actual prefix writer failure"):
        observe._prefix_cache_gate(tmp_path / "config.json", candidate(), {}, tmp_path, {})
    assert calls == ["release"]
    assert observe.main(["--variant", "candidate", "--plan", "--probe-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert "prefix_cache_gate" in plan["phases"]
    assert install_serve.main(["--variant", "candidate", "--plan"]) == 0
    direct = json.loads(capsys.readouterr().out)
    assert direct["runtime_selection"] == plan["runtime_selection"]
    assert direct["runtime_selection"]["prefix_caching"] is True
