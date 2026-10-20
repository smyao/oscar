"""Archive #26-49/#69/#127-129/#142/#155: external lifecycle host contracts.

Pinned native NPU _update_states and GPU synchronize_input_prep bodies execute
with host dependencies. CPU storage is injected at allocation seams only;
these tests do not prove target NPU allocation, stream execution or graph math.
"""
import ast
from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
import torch

from oscar_ascend import plugin
from oscar_ascend.ops import loader
from oscar_ascend.runtime import (AscendRuntimeProvider, RequestWindowSidecar,
                                 REQUEST_WINDOW_CAPABILITIES)
from oscar_ascend.request_window_storage import FixedWindowMap
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.integration.metadata import (
    OscarMetadata, OscarMetadataError, GraphMetadataBindings, slice_window_map,
)
from oscar_ascend.integration.mixed_decode_split import _slice_metadata
from tests.test_runtime_native import native_specs, definitions

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    config = json.loads((ROOT / "configs/target.json").read_text())
    return config | dict(max_num_seqs=4, sink_tokens=2, recent_tokens=4,
        experimental_history_reuse=True, experimental_fast_unpack=True,
        experimental_mixed_cv=True, experimental_striped_cache=True,
        experimental_decode_bundle=True,
        experimental_request_windows=True)


def output(*, new=(), finished=(), preempted=(), resumed=(), cached=()):
    return NS(scheduled_new_reqs=[NS(req_id=key) for key in new],
        scheduled_cached_reqs=NS(resumed_req_ids=set(resumed), req_ids=list(cached),
                                num_computed_tokens=[1] * len(cached)),
        finished_req_ids=set(finished), preempted_req_ids=set(preempted))


def native_runner_type(base):
    nodes = []
    for filename, owner, method in (
        ("references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", "_update_states"),
        ("references/vllm/vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner", "synchronize_input_prep"),
    ):
        tree = ast.parse((ROOT / filename).read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
        nodes.append(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method))
    cls = ast.ClassDef(name="NPUModelRunner", bases=[ast.Name(id="FixtureBase", ctx=ast.Load())],
                       keywords=[], body=nodes, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"FixtureBase": base, "contextmanager": contextmanager}
    exec(compile(module, "pinned-native-request-window-seams", "exec"), scope)
    return scope["NPUModelRunner"]


@pytest.fixture
def integrated(monkeypatch, tmp_path):
    plugin.unregister()
    config = configuration()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"model_type": "qwen3_5_text", "mamba_ssm_dtype": "float32"}}))
    config["model"] = str(model)
    provider = AscendRuntimeProvider(config)
    stream = object()
    timeline, allocations, context = [], [], {}
    budget_open = [False]
    monkeypatch.setattr(torch, "npu", NS(current_stream=lambda: stream), raising=False)
    monkeypatch.setattr(plugin, "require_runtime", lambda: provider)
    monkeypatch.delenv("OSCAR_PASSIVE_TIMING_CONTROL", raising=False)
    monkeypatch.delenv("OSCAR_TRACE_DIR", raising=False)
    monkeypatch.setattr(provider, "assert_ready", lambda: None)
    monkeypatch.setattr(plugin, "_request_windows_configured", lambda: True)

    class Scheduler:
        def _preempt_request(self, request, timestamp): pass
        def schedule(self): return output()
    Scheduler.__module__ = "vllm.v1.core.sched.scheduler"
    Scheduler.__qualname__ = "Scheduler"
    plugin._patch_request_window_scheduler(NS(Scheduler=Scheduler))

    def sidecar(layout):
        assert budget_open[0], "sidecar escaped native model-load accounting"
        allocations.append(layout.required_bytes)
        raw = torch.empty(layout.required_bytes, dtype=torch.uint8)
        key, value, tags = layout.views(raw, testing=True)
        tags.fill_(-1)
        return RequestWindowSidecar(layout, raw, key, value, tags,
                                    torch.zeros(layout.max_num_seqs, dtype=torch.int32))

    def window_map(layout):
        assert budget_open[0]
        provider._request_window_stream = stream
        host = torch.empty(layout.map_shape, dtype=torch.int64)
        device = torch.empty_like(host)
        host[:, 0].fill_(-1); host[:, 1].zero_(); device.copy_(host)
        return FixedWindowMap(layout, host, device, testing=True)

    monkeypatch.setattr(provider, "_allocate_request_window_sidecar", sidecar)
    monkeypatch.setattr(provider, "_allocate_request_window_map", window_map)

    class Attention:
        def __init__(self, name):
            self.layer_name = name
            self.impl = NS(provider=provider, num_kv_heads=1, head_size=256)
            context[name] = self  # native registration happens before posthook
    plugin._patch_attention_registration(NS(Attention=Attention))
    names = [f"model.layers.{4 * index + 3}.self_attn.attn" for index in range(16)]
    names.append("mtp.layers.0.self_attn.attn")
    deferred = lambda: "native deferred correction"

    class NativeEvent:
        def synchronize(self): timeline.append("wait prior H2D")
        def record(self):
            timeline.append("record new H2D")

    class Base:
        def load_model(self):
            budget_open[0] = True
            try:
                for name in names:
                    Attention(name)
            finally:
                budget_open[0] = False
            timeline.append("load target and draft complete")
            return "native load result"
        def _update_states(self, scheduler_output):
            self.native_updates += 1
            timeline.extend(["native parent update", "condense", "reorder", "refresh"])
            self.input_batch.req_ids = list(self.next_req_ids)
            return deferred
        def get_kv_cache_spec(self): return {}
        def _allocate_kv_cache_tensors(self, config): return {}
        def _reshape_kv_cache_tensors(self, config, raw): return {}
        def _dummy_run(self): return None

    Runner = native_runner_type(Base)
    plugin._patch_runner(NS(NPUModelRunner=Runner))
    runner = Runner()
    runner.model_config = NS(max_model_len=config["max_model_len"], model=str(model),
        quantization=None, dtype=torch.bfloat16, revision=None, hf_overrides={})
    runner.compilation_config = NS(static_forward_context=context)
    runner.vllm_config = NS(
        model_config=runner.model_config,
        cache_config=NS(enable_prefix_caching=False, mamba_cache_mode="none",
                        mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="bfloat16"),
        scheduler_config=NS(scheduler_cls=None, get_scheduler_cls=lambda: Scheduler,
                            max_num_seqs=config["max_num_seqs"],
                            max_num_batched_tokens=16384),
        compilation_config=NS(cudagraph_capture_sizes=[512]),
        parallel_config=NS(), kv_transfer_config=None,
        speculative_config=NS(num_speculative_tokens=3))
    runner.use_async_scheduling, runner.prepare_inputs_event = True, NativeEvent()
    runner.native_updates = 0
    runner.requests = {"a": NS(num_computed_tokens=3, prev_num_draft_len=2)}
    runner.input_batch, runner.next_req_ids = NS(req_ids=[]), []
    try:
        yield NS(provider=provider, runner=runner, context=context, names=names,
                 timeline=timeline, allocations=allocations, stream=stream,
                 deferred=deferred, Attention=Attention)
    finally:
        plugin.unregister()


def test_all_registered_target_and_draft_sidecars_are_allocated_before_load_returns(integrated):
    x = integrated
    assert x.runner.load_model() == "native load result"
    assert len(x.allocations) == 17
    assert set(x.provider._request_window_layers) == set(x.names)
    assert x.provider._request_windows_sealed
    assert x.provider._request_window_binding.device_map.shape == (4, 2)
    assert all(side.status.shape == (4,) for side in x.provider._request_window_layers.values())
    with pytest.raises(OscarReadinessError, match="after.*sealed"):
        x.Attention("model.layers.100.self_attn.attn")


def test_one_native_update_preserves_rewind_deferred_return_and_publishes_after_reorder(integrated):
    x = integrated; runner = x.runner
    runner.load_model()
    runner.next_req_ids = ["b", "a"]
    pointer = x.provider._request_window_binding.device_map.data_ptr()
    with runner.synchronize_input_prep():
        result = runner._update_states(output(new=("a", "b"), cached=("a",)))
        assert result is x.deferred
        assert runner.requests["a"].prev_num_draft_len == 0  # actual native rewind body
        assert x.provider.request_window_map(2).tolist() == [[0, 1], [1, 2]]
        assert x.timeline[-4:] == ["native parent update", "condense", "reorder", "refresh"]
    assert runner.native_updates == 1 and x.timeline[-1] == "record new H2D"
    runner.next_req_ids = ["a", "b"]
    with runner.synchronize_input_prep(): runner._update_states(output())
    assert x.provider.request_window_map(4).tolist() == [[1, 2], [0, 1], [-1, 0], [-1, 0]]
    assert x.provider.request_window_map(4).data_ptr() == pointer
    runner.next_req_ids = []
    with runner.synchronize_input_prep(): runner._update_states(output(finished=("a", "b")))
    assert x.provider.request_window_map(4).tolist() == [[-1, 0]] * 4
    assert runner.native_updates == 3
    with pytest.raises(OscarReadinessError, match="input-prep scope"):
        runner._update_states(output())


def test_scope_cleans_up_and_other_stream_cannot_reuse_leases(integrated, monkeypatch):
    x = integrated; x.runner.load_model(); x.runner.next_req_ids = ["a"]
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    with pytest.raises(OscarReadinessError, match="changed compute stream"):
        with x.runner.synchronize_input_prep():
            x.runner._update_states(output(new=("a",)))
    assert x.provider._request_window_prep_runner is None
    assert x.provider._request_window_leases.active_count == 0


@pytest.mark.parametrize("field,value", [("enable_prefix_caching", True), ("mamba_cache_mode", "align")])
def test_unsupported_native_mode_fails_at_seal(integrated, field, value):
    setattr(integrated.runner.vllm_config.cache_config, field, value)
    with pytest.raises(OscarReadinessError, match="prefix caching disabled"):
        integrated.runner.load_model()


def test_explicit_flag_keeps_native_scheduler_and_requires_full_capability_bundle(monkeypatch):
    cfg = configuration()
    for bad in (1, "true", None):
        with pytest.raises(OscarReadinessError, match="explicit boolean"):
            AscendRuntimeProvider(cfg | {"experimental_request_windows": bad})
    with pytest.raises(OscarReadinessError, match="native scheduler"):
        AscendRuntimeProvider(cfg | {"scheduler_cls": "native.AsyncScheduler"})
    with pytest.raises(OscarReadinessError, match="history override"):
        AscendRuntimeProvider(cfg | {"history_token_capacity": 16896})
    with pytest.raises(OscarReadinessError, match="whole-prefill budget"):
        AscendRuntimeProvider(cfg | {"experimental_bounded_whole_prefill": True})
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(loader, "require_production_ops", lambda: None)
    seen = []
    def require(capabilities):
        seen.append(frozenset(capabilities))
        if capabilities == REQUEST_WINDOW_CAPABILITIES:
            raise OscarReadinessError("request-window bundle missing")
    monkeypatch.setattr(loader, "require_capabilities", require)
    provider = AscendRuntimeProvider(cfg)
    with pytest.raises(OscarReadinessError, match="bundle missing"):
        provider.assert_ready()
    assert REQUEST_WINDOW_CAPABILITIES in seen and not provider._ready
    old = AscendRuntimeProvider({})
    old.register_request_window_layer(None); old.seal_request_windows(None)
    old.publish_request_window_map(None, None)
    assert old.request_window_map(0) is None and not old._request_window_layers


def metadata():
    return OscarMetadata(
        query_start_loc=torch.tensor([0, 4, 12], dtype=torch.int32),
        seq_lens=torch.tensor([100, 200], dtype=torch.int32),
        block_tables=torch.zeros((2, 4), dtype=torch.int32),
        slot_mapping=torch.arange(12, dtype=torch.int64), num_reqs=2,
        num_actual_tokens=12, max_query_len=8, max_seq_len=200,
        num_input_tokens=12, query_start_loc_cpu=torch.tensor([0, 4, 12], dtype=torch.int32),
        window_map=torch.tensor([[3, 9], [1, 12]], dtype=torch.int64))


def test_request_slicing_and_graph_address_signature_do_not_use_token_indices():
    parent = metadata()
    child = _slice_metadata(parent, 1, 2, 4, 12, 12, "long_suffix")
    assert child.window_map.tolist() == [[1, 12]]
    assert child.window_map.data_ptr() == parent.window_map[1:].data_ptr()
    assert slice_window_map(parent, 0, 1).tolist() == [[3, 9]]
    capture = replace(parent, capture_origin=True)
    binding = GraphMetadataBindings(); binding.bind(capture)
    parent.window_map.copy_(torch.tensor([[1, 30], [0, 31]]))
    assert binding.bind(parent).window_map.tolist() == [[1, 30], [0, 31]]
    with pytest.raises(OscarMetadataError, match="buffers changed"):
        binding.bind(replace(parent, window_map=parent.window_map.clone()))
    with pytest.raises(OscarMetadataError, match="request rows"):
        slice_window_map(parent, 0, 3)


def test_dual_reshape_reuses_sidecars_and_shared_map_while_native_gdn_views_stay_original(
        integrated, native_specs, monkeypatch):
    x = integrated; runner = x.runner; runner.load_model()
    interface, specs_module, _ = native_specs
    native_gdn = interface.MambaSpec(block_size=262144, shapes=((6, 2560), (12, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16), page_size_padded=817152,
        mamba_cache_mode="none", num_speculative_blocks=3)
    full = interface.FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256,
                                      dtype=torch.bfloat16, page_size_padded=817152)
    original = {name: full for name in x.names} | {"gdn": native_gdn}
    specs = x.provider.transform_kv_cache_specs(runner, original)
    assert specs["gdn"] is native_gdn and all(specs[name].block_size == 5632 for name in x.names)
    groups = [interface.KVCacheGroupSpec(x.names, specs[x.names[0]]),
              interface.KVCacheGroupSpec(["gdn"], native_gdn)]
    config = interface.KVCacheConfig(2, [], groups)
    raw = {name: torch.full((2 * 817152,), 37, dtype=torch.int8) for name in x.names}
    raw["gdn"] = raw[x.names[0]]
    before = raw["gdn"][:2 * (30720 + 393216)].clone()
    native = definitions("references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
        "request_window_native_reshape", set(), {"torch": torch, "math": math, **interface.__dict__},
        monkeypatch, methods=[("NPUModelRunner", "_get_layer_kv_cache_specs"),
                             ("NPUModelRunner", "_reshape_kv_cache_tensors")])
    runner._get_layer_kv_cache_specs = MethodType(native._get_layer_kv_cache_specs, runner)
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        NS(backend=object(), layer_names=g.layer_names, kv_cache_spec=g.kv_cache_spec) for g in groups)
    runner.runner_only_attn_layers, runner.use_compress, runner.device = set(), False, "cpu"
    runner.parallel_config = NS(tensor_parallel_size=4)
    runner.model_config = NS(get_num_attention_heads=lambda parallel: 6)
    rotation = NS(key_transposed=torch.eye(256), value_transposed=torch.eye(256),
                  inverse_value_transposed=torch.eye(256), hadamard=False)
    monkeypatch.setattr(x.provider, "_rotations", lambda *_: {name: rotation for name in x.names})
    monkeypatch.setattr(x.provider, "ensure_workspace", lambda *_: NS(geometry=NS(tokens=16384, total_bytes=1)))
    views = x.provider.reshape_kv_cache_tensors(runner, config, raw,
        lambda cfg, buffers: native._reshape_kv_cache_tensors(runner, cfg, buffers))
    for name in x.names:
        state, sidecar = x.provider.layers[name], x.provider._request_window_layers[name]
        assert views[name].ndim == 1 and views[name].data_ptr() == raw[name].data_ptr()
        assert state.window_key is sidecar.key and state.window_status is sidecar.status
        assert state.window_map is x.provider._request_window_binding.device_map
        assert state.cache_format == "striped_request_window_v1"
    conv, ssm = views["gdn"]
    assert conv.shape == (2, 6, 2560) and ssm.shape == (2, 12, 128, 128)
    assert torch.equal(raw["gdn"][:before.numel()], before)
    assert runner.runner_only_attn_layers == set()
