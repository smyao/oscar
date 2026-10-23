"""Archive #86/#97/#148/#164: TP-owned clip constants and common writer.

CPU integration doubles only; no NPU, graph, performance or quality claim.
"""
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from oscar_ascend import rotations, runtime
from oscar_ascend.integration import official_math
from oscar_ascend.integration.cache_storage_ops import store_current
from oscar_ascend.integration.runtime_api import OscarReadinessError
from test_center_runtime import setup_provider
from test_official_runtime import workspace, state


@pytest.mark.parametrize("rank", range(4))
def test_worker_loads_its_head_clip_once_and_draft_keeps_config_default(monkeypatch, rank):
    provider, name, _, _ = setup_provider(monkeypatch, rank=rank)
    provider.config["rotation_k_clip"] = "per_head_calibrated"
    artifact = rotations.load_active_artifact()
    artifact["calibration_profile"] = {"rotation_k_clip": "per_head_calibrated"}
    ratios = [.90, .94, .98, 1.0]
    artifact["layers"][name]["k_clip_ratio"] = torch.tensor(ratios, dtype=torch.float64)
    prepared = provider._prepare_rotations(256, "cpu")
    assert type(prepared[name].key_clip_ratio) is float
    assert prepared[name].key_clip_ratio == ratios[rank]
    assert prepared[None].key_clip_ratio is None
    monkeypatch.setattr(rotations, "load_active_artifact", lambda *a, **kw: pytest.fail("reloaded PT"))
    assert provider._prepare_rotations(256, "cpu") is prepared


@pytest.mark.parametrize("clip", [None, 0.0, -0.1, 1.01, float("nan"), float("inf"),
                                   torch.tensor(.94), True, ".94"])
def test_required_head_clip_is_checked_before_mtp_identity_or_runtime_cache(monkeypatch, clip):
    provider, name, _, _ = setup_provider(monkeypatch)
    provider.config["rotation_k_clip"] = "per_head_calibrated"
    prepare = rotations.prepare_device_rotations

    def wrong_clip(*args, **kwargs):
        selected = prepare(*args, **kwargs)
        values = dict(vars(selected[name]), key_clip_ratio=clip)
        selected[name] = NS(**values)
        return selected

    monkeypatch.setattr(rotations, "prepare_device_rotations", wrong_clip)
    with pytest.raises(OscarReadinessError, match="finite host K clip ratio"):
        provider._prepare_rotations(256, "cpu")
    assert provider.rotations == {}


def test_old_center_recipe_without_clip_remains_explicitly_usable(monkeypatch):
    provider, name, _, _ = setup_provider(monkeypatch)
    prepared = provider._prepare_rotations(256, "cpu")
    assert getattr(prepared[name], "key_clip_ratio", None) is None
    assert getattr(prepared[None], "key_clip_ratio", None) is None


def test_reshape_binds_target_scalar_but_not_draft_scalar(monkeypatch):
    # Exercise the real LayerState construction without importing vLLM/NPU.
    module = ModuleType("oscar_ascend.integration.specs")
    module.OscarFullAttentionSpec = type("Spec", (), {})
    module.packed_view = lambda raw, *args: raw
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = module.OscarFullAttentionSpec()
    spec.page_size_bytes, spec.head_size, spec.num_kv_heads = 128, 256, 1
    spec.layout, spec.block_size = object(), 128
    names = ("model.layers.3.self_attn.attn", "mtp.layers.0.self_attn.attn")
    runner = NS(device="cpu", runner_only_attn_layers=set(), parallel_config=None,
                _get_layer_kv_cache_specs=lambda _: dict.fromkeys(names, spec),
                model_config=NS(get_num_attention_heads=lambda _: 6),
                vllm_config=NS(scheduler_config=NS(max_num_batched_tokens=1),
                              compilation_config=NS(cudagraph_capture_sizes=[])))
    provider = object.__new__(runtime.AscendRuntimeProvider)
    provider.config, provider.layers = {}, {}
    provider._request_windows, provider._cache_format = False, "striped_rotated_fp32_v3"
    matrices = dict(key_transposed=torch.eye(256), value_transposed=torch.eye(256),
                    inverse_value_transposed=torch.eye(256), hadamard=False)
    provider._rotations = lambda *_: {names[0]: NS(**matrices, key_clip_ratio=.945),
                                     names[1]: NS(**matrices, key_clip_ratio=None)}
    provider._snapshot_layout = lambda _: NS(required_bytes=0,
        views=lambda *_: (torch.empty(1), torch.empty(1), torch.empty(1)))
    provider.ensure_workspace = lambda *_: NS(geometry=NS(tokens=1, total_bytes=0))
    raw = {name: torch.empty(128, dtype=torch.uint8) for name in names}
    provider.reshape_kv_cache_tensors(runner, NS(num_blocks=1), raw, lambda *_: {})
    assert provider.layers[names[0]].key_clip_ratio == .945
    assert provider.layers[names[1]].key_clip_ratio is None


@pytest.mark.parametrize("clip,expected", [(None, .96), (.945, .945)])
@pytest.mark.parametrize("reuse_rotated", [False, True])
def test_writer_uses_loaded_head_scalar_and_preserves_exact_window(clip, expected, reuse_rotated):
    w = workspace(n=1)
    s = state(w)
    s.key_clip_ratio = clip
    k, v = torch.ones_like(w.key_input), torch.full_like(w.value_input, 2)
    calls = []

    def store(*args):
        assert args[0].data_ptr() == args[2].data_ptr()
        assert args[1].data_ptr() == args[3].data_ptr()
        calls.append(args[-2:])

    ops = NS(store_aligned_int2_out=store)
    precomputed = (k, v) if reuse_rotated else None
    store_current(ops, s, k, v, torch.arange(1), torch.arange(1), None, None,
                  {"experimental_official_math": True, "k_clip_ratio": .96, "v_clip_ratio": .92},
                  rotated_kv=precomputed)
    assert calls == [(expected, .92)]


@pytest.mark.parametrize("n,later,reader", [
    (8, True, "attention_cv_official_decode_q1_out"),
    (8, False, "attention_cv_official_decode_out"),
    (256, False, "attention_cv_official_cluster4_out"),
    (8192, False, "attention_cv_official_cluster16_out"),
])
def test_actual_q1_q4_c4_c16_forward_all_pass_head_clip_to_same_writer(monkeypatch, n, later, reader):
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order as forward
    original = official_math.store_rotated
    seen = []

    def check_writer(ops, state, *args, **kwargs):
        state.key_clip_ratio = .945
        old_store = ops.store_aligned_int2_out

        def check_scalar(*values):
            assert values[-2:] == (.945, .92)
            seen.append(values[-2])
            return old_store(*values)

        monkeypatch.setattr(ops, "store_aligned_int2_out", check_scalar)
        return original(ops, state, *args, **kwargs)

    monkeypatch.setattr(official_math, "store_rotated", check_writer)
    forward(n, 1, 1, n, 16, later, True, True, True,
            _member_pipeline=True, _c16_l1=True, _bf16_expected=reader,
            _aligned=True, _official=True)
    assert seen == [.945]


def test_candidate_false_legacy_writer_keeps_explicit_config_scalar():
    w = workspace(n=1)
    s = state(w)
    s.cache_format = "canonical_v1"
    s.rotation_k_transpose = s.rotation_v_transpose = torch.eye(256)
    s.hadamard = False
    s.key_clip_ratio = .945
    calls = []
    ops = NS(rotate_clip_store_out=lambda *args: calls.append(args[-3:]))
    store_current(ops, s, w.key_input, w.value_input, torch.arange(1), torch.arange(1),
                  None, None, {"k_clip_ratio": .96, "v_clip_ratio": .92})
    assert calls == [(.96, .92, False)]


def test_summary_does_not_relabel_old_center_recipe_as_calibrated_clip():
    from tools.serving_variants import runtime_feature_summary
    old = runtime_feature_summary({"rotation_variant": "perhead_center"}, "candidate")
    assert old["calibration_recipe_version"] == 3 and old["clip_policy"] == "scalar"
