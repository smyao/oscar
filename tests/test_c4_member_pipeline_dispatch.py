"""Archive #148/#150/#153/#154: candidate routing and capacity, host only.

Mocked calls establish the production Python ABI path, never NPU math/graph
acceptance. Real kernels and the frozen observe gate are validated separately.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oscar_ascend.integration.cache_storage_ops import (
    REQUEST_FORMAT, REQUEST_CLUSTER_READERS, REQUEST_MEMBER_PIPELINE_CV_OP,
    storage_reader,
)
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.ops.cv_dispatch import CLUSTER_CV_OPS, MEMBER_PIPELINE_CV_OP, select_cv_op
from oscar_ascend.runtime import AscendRuntimeProvider, WorkspaceGeometry
from tools.serving_variants import variant_config


FLAGS = dict(fast_unpack=True, mixed_cv=True, striped_cache=True, decode_bundle=True)


@pytest.mark.parametrize("tokens,max_query,draft,expected", [
    (128, 4, False, "attention_cv_bundle_decode_out"),
    (16384, 1, True, "attention_cv_bundle_q1_out"),
    (512, 8, False, "attention_cv_window_range_balanced_out"),
    (83, 83, False, "attention_cv_window_range_out"),
    (84, 84, False, MEMBER_PIPELINE_CV_OP),
    (512, 128, False, MEMBER_PIPELINE_CV_OP),
    (512, None, False, MEMBER_PIPELINE_CV_OP),
    (8192, 8192, False, "attention_cv_window_range_cluster16_out"),
])
def test_only_existing_c4_route_is_replaced(tokens, max_query, draft, expected):
    chosen = select_cv_op(16, 6, 1, tokens, max_query, q1_draft=draft,
                         c4_member_pipeline=True, **FLAGS)
    assert chosen == expected
    old = select_cv_op(16, 6, 1, tokens, max_query, q1_draft=draft, **FLAGS)
    assert (chosen != old) == (old == "attention_cv_window_range_cluster4_out")


def test_request_windows_cannot_silently_select_the_old_c4_reader():
    regular = select_cv_op(16, 6, 1, 512, 128, c4_member_pipeline=True, **FLAGS)
    assert storage_reader(SimpleNamespace(cache_format=REQUEST_FORMAT), regular) == REQUEST_MEMBER_PIPELINE_CV_OP
    assert storage_reader(SimpleNamespace(cache_format="striped_v1"), regular) == regular
    assert regular in CLUSTER_CV_OPS and REQUEST_MEMBER_PIPELINE_CV_OP in REQUEST_CLUSTER_READERS


def test_pipeline_workspace_contains_both_kv_pairs_and_score_slots():
    old = WorkspaceGeometry(2048, 6, 1, 256, cube_cores=20, history_cluster_size=4)
    pipeline = replace(old, c4_member_pipeline=True)
    assert pipeline.cv_bytes == 20 * 2494464
    assert pipeline.cv_bytes - old.cv_bytes == 20 * 655360
    assert replace(pipeline, tokens=128).cv_bytes == pipeline.cv_bytes
    assert replace(pipeline, history_cluster_size=16).cv_bytes == 20 * 4997120
    assert replace(old, history_cluster_size=16).cv_bytes == replace(pipeline, history_cluster_size=16).cv_bytes
    with pytest.raises(ValueError, match="cluster workspace"):
        replace(pipeline, history_cluster_size=1)


@pytest.mark.parametrize("bad", [None, 1, "true"])
def test_pipeline_flag_is_boolean_at_every_host_boundary(bad):
    with pytest.raises(ValueError, match="explicit boolean"):
        variant_config({"experimental_c4_member_pipeline": bad}, "candidate")
    with pytest.raises(OscarReadinessError, match="explicit boolean"):
        AscendRuntimeProvider({"experimental_c4_member_pipeline": bad})
    with pytest.raises(ValueError, match="explicit boolean"):
        select_cv_op(16, 6, 1, 512, 128, c4_member_pipeline=bad, **FLAGS)


def test_pipeline_requires_its_existing_striped_decode_base():
    with pytest.raises(ValueError, match="requires"):
        variant_config({"experimental_c4_member_pipeline": True, "experimental_decode_bundle": False}, None)
    with pytest.raises(OscarReadinessError, match="requires"):
        AscendRuntimeProvider({"experimental_c4_member_pipeline": True, "experimental_decode_bundle": False})
    with pytest.raises(ValueError, match="requires"):
        select_cv_op(4, 6, 1, 512, 128, c4_member_pipeline=True)


@pytest.mark.parametrize("request_windows", [False, True])
def test_missing_new_capability_fails_before_any_device_work(monkeypatch, request_windows):
    from oscar_ascend.ops import loader
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/target.json").read_text())
    config["experimental_c4_member_pipeline"] = True
    config = variant_config(config, "candidate")
    config["experimental_request_windows"] = request_windows
    if request_windows:
        with pytest.raises(OscarReadinessError, match="physical-page snapshots"):
            AscendRuntimeProvider(config)
        return
    provider = AscendRuntimeProvider(config)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", ",".join(map(str, config["devices"])))
    monkeypatch.setattr(loader, "require_production_ops", lambda: None)
    missing = "attention_bf16_history_member_pipeline_cluster4_out"
    seen = []

    def require(names):
        seen.append(set(names))
        if missing in names:
            raise RuntimeError("missing new member-pipeline capability")

    monkeypatch.setattr(loader, "require_capabilities", require)
    with pytest.raises(RuntimeError, match="missing new member-pipeline capability"):
        provider.assert_ready()
    assert any(missing in item for item in seen) and provider._ready is False


def test_runtime_workspace_records_the_effective_pipeline_flag(monkeypatch):
    from oscar_ascend import runtime
    config = variant_config({"devices": [0, 1, 2, 3], "port": 8989,
                             "experimental_c4_member_pipeline": True}, "candidate")
    # Runtime receives the native engine's resolved capacity after CLI omission.
    config["max_num_batched_tokens"] = 2048
    provider = AscendRuntimeProvider(config)
    monkeypatch.setattr(provider, "_device_cube_cores", lambda _: 20)
    monkeypatch.setattr(provider, "_prepare_rotations", lambda *_: None)
    monkeypatch.setattr(runtime, "GraphWorkspace", lambda geometry, device:
                        SimpleNamespace(geometry=geometry, device=device))
    workspace = provider.ensure_workspace(6, 1, 256, "npu:0")
    assert workspace.geometry.c4_member_pipeline is True
    assert workspace.geometry.cv_bytes >= 20 * 2494464


def test_production_forward_passes_cluster_stats_to_new_operator():
    from tests.test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        256, 20, 1, 256, 16, False, True, True, True, _member_pipeline=True)
