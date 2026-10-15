"""Archive #157-163: keep real prefill state ownership across native graph dispatch.

Execute the pinned runner dispatcher and builder fold AST unchanged. Host
fixtures only replace device-independent surrounding dependencies; these are
not NPU graph-replay or service-completion claims. MTP stays at three drafts.
"""
from dataclasses import dataclass
from enum import Enum
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from oscar_ascend import plugin
from oscar_ascend.integration.dummy_context import native_dummy_run
from oscar_ascend.integration.runtime_api import clear_runtime, install_runtime
from test_plugin import Provider, runner_module
from test_runtime_native import definitions


class Mode(Enum):
    NONE = 0
    FULL = 1


@dataclass
class Descriptor:
    num_tokens: int
    num_reqs: int | None = None


@pytest.fixture(autouse=True)
def clean_plugin():
    plugin.unregister()
    clear_runtime()
    yield
    plugin.unregister()
    clear_runtime()


def _native_modules(monkeypatch):
    runner_native = definitions(
        "references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
        "native_prefill_dispatch_contract", set(),
        {"np": SimpleNamespace(all=lambda values: bool(values.all())),
         "CUDAGraphMode": Mode, "BatchDescriptor": Descriptor,
         "enable_sp": lambda config: False}, monkeypatch,
        methods=[("NPUModelRunner", "_determine_batch_execution_and_padding")])
    runner_mod = runner_module()
    runner_mod.NPUModelRunner._determine_batch_execution_and_padding = (
        runner_native._determine_batch_execution_and_padding)
    builder_native = definitions(
        "references/vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py",
        "native_prefill_fold_contract", set(), {"torch": torch}, monkeypatch,
        methods=[("AscendGDNAttentionMetadataBuilder",
                  "_fold_spec_sized_prefill_chunks_into_spec")])
    builder_mod = ModuleType("vllm_ascend.ops.gdn_attn_builder")
    builder_mod.AscendGDNAttentionMetadataBuilder = type(
        "AscendGDNAttentionMetadataBuilder", (), {
            "_fold_spec_sized_prefill_chunks_into_spec":
                builder_native._fold_spec_sized_prefill_chunks_into_spec})
    for module in (runner_mod, builder_mod):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return runner_mod, builder_mod


def _runner(cls, *, prefix=False, mode="none", computed=(32764, 32768)):
    runner = cls()
    cache = SimpleNamespace(enable_prefix_caching=prefix, mamba_cache_mode=mode)
    parallel = SimpleNamespace(data_parallel_size=1, tensor_parallel_size=4)
    spec = SimpleNamespace(method="mtp", num_speculative_tokens=3)
    runner.cache_config = cache
    runner.speculative_config = spec
    runner.vllm_config = SimpleNamespace(
        cache_config=cache, speculative_config=spec, parallel_config=parallel,
        observability_config=SimpleNamespace(cudagraph_metrics=False))
    runner.model_config = SimpleNamespace(is_encoder_decoder=False, is_hybrid=True)
    runner.uniform_decode_query_len = 4
    runner._pad_for_sequence_parallelism = lambda tokens: tokens
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=torch.tensor(computed),
        num_prompt_tokens=torch.tensor([32768, 32768]),
        num_prompt_tokens_cpu=torch.tensor([32768, 32768]),
        req_ids=["prefill", "decode"], lora_id_to_lora_request={})
    runner.requests = {name: SimpleNamespace(num_prompt_tokens=32768)
                       for name in runner.input_batch.req_ids}
    calls = []

    def dispatch(**kwargs):
        calls.append(kwargs)
        return (Mode.FULL if kwargs["uniform_decode"] else Mode.NONE,
                Descriptor(kwargs["num_tokens"]))

    runner.cudagraph_dispatcher = SimpleNamespace(dispatch=dispatch)
    return runner, calls


def _invoke(runner, *, positional=False, **extra):
    fn = runner._determine_batch_execution_and_padding
    if positional:
        return fn(8, 2, torch.tensor([4, 4]), 4, False, False, False, **extra)
    return fn(num_tokens=8, num_reqs=2, num_scheduled_tokens_np=torch.tensor([4, 4]),
              max_num_scheduled_tokens=4, use_cascade_attn=False, **extra)


def _register(monkeypatch):
    provider = Provider()
    provider.config = {"enable_prefix_caching": False}
    install_runtime(provider)
    monkeypatch.setenv("OSCAR_ENABLED", "1")
    plugin.register()


@pytest.mark.parametrize("positional", [False, True])
def test_real_native_warm_prefill_graph_collision_is_removed(monkeypatch, positional):
    runner_mod, _ = _native_modules(monkeypatch)
    runner, calls = _runner(runner_mod.NPUModelRunner)
    # Actual native dispatch mistakes four prompt tokens for a uniform decode.
    assert _invoke(runner, positional=positional)[0] is Mode.FULL
    assert calls[-1]["uniform_decode"] is True
    calls.clear()
    _register(monkeypatch)
    assert _invoke(runner, positional=positional)[0] is Mode.NONE
    assert calls == []  # Native force_eager branch, not a changed dispatcher.
    assert runner.speculative_config.num_speculative_tokens == 3
    # The same runner/address later decodes normally; no sticky eager state.
    runner.input_batch.num_computed_tokens_cpu[0] = 32768
    assert _invoke(runner, positional=positional)[0] is Mode.FULL
    assert len(calls) == 1


@pytest.mark.parametrize("prefix,cache_mode", [(True, "align"), (True, "none")])
def test_prefix_enabled_dispatch_keeps_existing_selection(monkeypatch, prefix, cache_mode):
    runner_mod, _ = _native_modules(monkeypatch)
    runner, calls = _runner(runner_mod.NPUModelRunner, prefix=prefix, mode=cache_mode)
    _register(monkeypatch)
    assert _invoke(runner)[0] is Mode.FULL
    assert calls[-1]["uniform_decode"] is True


def test_native_dummy_and_forced_capture_selection_remain_unchanged(monkeypatch):
    runner_mod, _ = _native_modules(monkeypatch)
    runner, calls = _runner(runner_mod.NPUModelRunner)
    _register(monkeypatch)
    with native_dummy_run():
        assert _invoke(runner)[0] is Mode.FULL
    assert _invoke(runner, force_uniform_decode=True)[0] is Mode.FULL
    assert len(calls) == 2


@pytest.mark.parametrize("prefix,mode,expected", [
    (False, "none", 1), (True, "align", 4), (True, "none", 4)])
def test_native_fold_hook_preserves_prefill_and_other_spec_counts(monkeypatch, prefix, mode, expected):
    runner_mod, builder_mod = _native_modules(monkeypatch)
    runner, _ = _runner(runner_mod.NPUModelRunner, prefix=prefix, mode=mode)
    builder = builder_mod.AscendGDNAttentionMetadataBuilder()
    builder.vllm_config, builder.num_spec = runner.vllm_config, 3
    common = SimpleNamespace(is_prefilling=torch.tensor([True, False]),
        seq_lens_cpu_upper_bound=torch.tensor([32768, 32772]),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32))
    masks, accepted = torch.tensor([False, True]), torch.tensor([1, 2], dtype=torch.int32)
    _register(monkeypatch)
    output_masks, output_accepted = builder._fold_spec_sized_prefill_chunks_into_spec(
        common, masks, accepted)
    assert output_accepted.tolist() == [expected, 2]
    assert output_masks.tolist() == [expected == 4, True]
    assert masks.tolist() == [False, True] and accepted.tolist() == [1, 2]
    if expected == 1:
        assert output_masks is masks and output_accepted is accepted


def test_registration_restores_both_native_descriptors_exactly(monkeypatch):
    runner_mod, builder_mod = _native_modules(monkeypatch)
    runner_cls = runner_mod.NPUModelRunner
    builder_cls = builder_mod.AscendGDNAttentionMetadataBuilder
    rn = "_determine_batch_execution_and_padding"
    bn = "_fold_spec_sized_prefill_chunks_into_spec"
    original_runner, original_builder = getattr(runner_cls, rn), getattr(builder_cls, bn)
    _register(monkeypatch)
    assert getattr(runner_cls, rn) is not original_runner
    assert getattr(builder_cls, bn) is not original_builder
    patched_runner, patched_builder = getattr(runner_cls, rn), getattr(builder_cls, bn)
    plugin.register()
    assert getattr(runner_cls, rn) is patched_runner
    assert getattr(builder_cls, bn) is patched_builder
    plugin.unregister()
    assert getattr(runner_cls, rn) is original_runner
    assert getattr(builder_cls, bn) is original_builder


def test_new_deferred_hooks_keep_registration_lightweight():
    script = """
import os, sys, json
os.environ['OSCAR_ENABLED'] = '1'
before = set(sys.modules)
from oscar_ascend.plugin import register, unregister
register()
print(json.dumps(sorted(x for x in set(sys.modules) - before
    if x.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'vllm_ascend'})))
unregister()
"""
    assert json.loads(subprocess.check_output([sys.executable, "-c", script], text=True)) == []
