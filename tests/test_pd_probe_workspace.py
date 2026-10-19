"""Archive #34/#36/#148/#153/#155/#164-165: production probe allocation seams.

Execute the shared runtime constructor and actual GraphWorkspace allocations
on CPU using a scoped allocation adapter. No forward, NPU op, CANN resource,
numerical acceptance or speed is simulated or claimed by this host test.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from oscar_ascend import runtime
from oscar_ascend.integration.runtime_api import OscarReadinessError
from tools import probe_current_only as shared
from tools import probe_first_mtp_current as first
from tools import probe_mixed_decode_split as mixed
from tools.serving_variants import variant_config


def target():
    return variant_config(json.loads((Path(__file__).resolve().parents[1] /
        "configs/target.json").read_text()), "candidate")


@pytest.fixture
def cpu_allocations(monkeypatch):
    """Exercise every real workspace shape without pretending CPU is NPU."""
    actual_workspace = runtime.GraphWorkspace
    actual_empty = torch.empty
    actual_device = torch.device
    allocations = []

    def allocate(geometry, device):
        assert actual_device(device).type == "cpu"
        def host_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return actual_empty(*args, **kwargs)
        # This temporary adapter changes only the guard/device argument while
        # invoking the real allocator; tensor shapes, dtypes and bytes are real.
        with patch.object(torch, "device", lambda _: SimpleNamespace(type="npu")), \
                patch.object(torch, "empty", host_empty):
            workspace = actual_workspace(geometry, device)
        allocations.append(workspace)
        return workspace

    monkeypatch.setattr(runtime, "GraphWorkspace", allocate)
    # _runtime only uses object.__new__; no model, native imports or forward
    # is needed to test its real provider, LayerState and workspace binding.
    monkeypatch.setitem(sys.modules, "oscar_ascend.integration.impl",
        SimpleNamespace(OscarAttentionImpl=type("UnexecutedAttentionImpl", (), {})))
    return allocations


@pytest.fixture
def fixture():
    return shared.reuse.make_fixture(torch,
        shared.reuse.Shape("pd_probe_host_allocation", (4, 17), (17, 0),
            256, 1, 1, False), rotation_mode="identity")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("explicit_precision", [False, True])
def test_shared_runtime_reserves_actual_bf16_c16_arena(
        fixture, cpu_allocations, enabled, explicit_precision):
    config = target()
    if not explicit_precision:
        config.pop("experimental_bf16_history")
        config.pop("experimental_c4_member_pipeline")
        config.pop("experimental_mdl_decode")
    before = deepcopy(config)
    impl, state = shared._runtime(torch, fixture, fixture["cpu"], "cpu", 2,
        config, enabled)
    assert config == before
    assert impl.provider.layers["probe.full"] is state
    assert cpu_allocations == [state.workspace]
    g = state.workspace.geometry
    assert g.tokens == fixture["tokens"]
    assert g.history_cluster_size == 16 and g.c4_member_pipeline and g.bf16_history
    assert impl.provider.config["experimental_bf16_history"] is True
    assert impl.provider.config["experimental_c4_member_pipeline"] is True
    # Independent typed C16 formula: Q16 in BF16; normalized input one tile.
    m, b, d = 128, 256, 256
    float_bytes = (2*m*d + m*b + 16*m*d + 16*2*m)*4
    operand_bytes = (16*m*d + 2*b*d + m*b)*2
    assert g.cv_bytes == 2*(float_bytes+operand_bytes) == 7_766_016
    assert replace(g, bf16_history=False).cv_bytes > g.cv_bytes
    actual = {name: getattr(state.workspace, name).numel() *
        getattr(state.workspace, name).element_size() for name in g.buffer_bytes}
    assert actual == g.buffer_bytes
    assert sum(actual.values()) == g.total_bytes
    assert shared._reader(impl.provider.config, tokens=4, max_query_len=4) == \
        "attention_bf16_history_mdl_out"
    assert shared._reader(impl.provider.config, tokens=2, max_query_len=1,
        q1_draft=True) == "attention_bf16_history_mdl_q1_out"


@pytest.mark.parametrize("flag", ["experimental_bf16_history",
    "experimental_c4_member_pipeline", "experimental_mdl_decode"])
def test_explicit_retired_or_incomplete_runtime_fails_before_allocation(
        fixture, cpu_allocations, flag):
    config = {**target(), flag: False}
    with pytest.raises(OscarReadinessError):
        shared._runtime(torch, fixture, fixture["cpu"], "cpu", 2, config, True)
    assert not cpu_allocations


def test_unadapted_production_workspace_still_rejects_cpu():
    with pytest.raises(OscarReadinessError, match="requires NPU"):
        runtime.GraphWorkspace(runtime.WorkspaceGeometry(1, 6, 1, 256), "cpu")


@pytest.mark.parametrize("enabled", [False, True])
def test_first_mtp_comparison_isolates_current_fia_and_keeps_bf16(enabled):
    config = target()
    # Negative control: the old baseline's dependent partition flag prevents
    # provider construction before any real numerical test can start.
    with pytest.raises(OscarReadinessError, match="qualified first-MTP current"):
        runtime.AscendRuntimeProvider({**config,
            "experimental_first_mtp_current_fia": False})
    before = deepcopy(config)
    selected = first._case_configuration(config, enabled)
    provider = runtime.AscendRuntimeProvider(selected)
    assert provider.config["experimental_first_mtp_current_fia"] is enabled
    assert provider.config["experimental_mixed_decode_split"] is False
    assert provider.config["experimental_current_only"] is False
    assert provider.config["experimental_bf16_history"] is True
    assert config == before


@pytest.mark.parametrize("new_reader,old_reader", [
    ("attention_bf16_history_mdl_out", "attention_cv_striped_decode_out"),
    ("attention_bf16_history_mdl_out", "attention_cv_bundle_decode_out"),
    ("attention_bf16_history_mdl_out", "attention_cv_mdl_decode_out"),
    ("attention_bf16_history_mdl_q1_out", "attention_cv_striped_q1_out"),
    ("attention_bf16_history_mdl_q1_out", "attention_cv_bundle_q1_out"),
    ("attention_bf16_history_mdl_q1_out", "attention_cv_mdl_decode_q1_out"),
])
def test_report_requires_production_bf16_reader_witness(new_reader, old_reader):
    from test_probe_current_only import complete_report
    report = complete_report()
    config = target()
    for row in report["cases"]:
        row["rotation"] = shared._rotation_label(shared._rotation_mode(config, row["first_draft"]))
    acceptance = {"performance": {"warmup": 2, "repeats": 5}}
    shared.validate_case_evidence(report, config, acceptance)
    calls = report["cases"][0]["real_operator_calls"]
    calls[old_reader] = calls.pop(new_reader)
    with pytest.raises(RuntimeError, match="reader evidence"):
        shared.validate_case_evidence(report, config, acceptance)


@pytest.mark.parametrize("probe", [shared, first, mixed])
def test_npu_probe_requests_actual_bf16_symbols_before_any_case(probe, monkeypatch, tmp_path):
    from oscar_ascend.ops import loader
    from oscar_ascend.ops.cv_dispatch import BF16_HISTORY_CV_OPS
    from tools import probe_native_current_fia as native
    config = tmp_path / "target.json"
    config.write_text(json.dumps(target()))
    acceptance = Path(__file__).resolve().parents[1] / "configs/acceptance.json"
    monkeypatch.setattr(native, "_select_target_npu", lambda _: None)
    monkeypatch.setattr(native, "_target_geometry", lambda _: (6, 1, 256))
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: True,
        set_device=lambda _: None, get_device_name=lambda _: "Ascend910B4"), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm_ascend", SimpleNamespace(ops=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", SimpleNamespace())
    monkeypatch.setattr(loader, "validate_build_artifacts", lambda _: {})
    seen = []
    class ReachedCapabilityCheck(Exception):
        pass
    def require(names, *args, **kwargs):
        seen.append(set(names))
        raise ReachedCapabilityCheck
    monkeypatch.setattr(loader, "require_capabilities", require)
    monkeypatch.setattr(probe, "run_case", lambda *a: pytest.fail("must stop before device cases"))
    with pytest.raises(ReachedCapabilityCheck):
        probe.probe(config, acceptance)
    assert len(seen) == 1 and BF16_HISTORY_CV_OPS <= seen[0]
    assert "rotate_bf16_query_out" in seen[0]
    assert not any(name.startswith("attention_cv_") for name in seen[0])
