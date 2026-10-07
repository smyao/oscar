"""Archive #140/#142/#148/#155: native first-MTP host contracts only.

Execute pinned native methods unchanged. CPU results do not validate native
FIA numerical differences, device completion, graph replay or model quality.
"""
from abc import ABC, abstractmethod
from enum import Enum
from types import MethodType, SimpleNamespace
from typing import ClassVar, Generic, TypeVar

import pytest
import torch

from oscar_ascend.integration.current_attention import use_native_current
from oscar_ascend.integration.dummy_context import native_dummy_run
from oscar_ascend.integration.first_draft_current_only import supports_first_draft_current_only
from oscar_ascend.integration.metadata import from_common
from test_runtime_native import definitions, load_our_module


class AscendAttentionState(Enum):
    ChunkedPrefill = 1
    SpecDecoding = 2


def config():
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp", enforce_eager=True, parallel_drafting=False,
            disable_padded_drafter_batch=False,
            draft_model_config=SimpleNamespace(
                uses_mrope=True,
                hf_config=SimpleNamespace(model_type="qwen3_5_mtp", architectures=["Qwen3_5MTP"]))),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1,
                                        decode_context_parallel_size=1))


def common():
    return SimpleNamespace(
        causal=True, query_start_loc=torch.tensor([0, 4, 11], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 11], dtype=torch.int32),
        seq_lens=torch.tensor([504, 7], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([504, 7], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 8, dtype=torch.int32),
        slot_mapping=torch.tensor([500, 501, 502, 503, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int32),
        num_reqs=2, num_actual_tokens=11, num_input_tokens=11,
        max_query_len=7, max_seq_len=504, attn_state=AscendAttentionState.ChunkedPrefill)


@pytest.mark.parametrize("rejected", range(4))
@pytest.mark.parametrize("mrope", [False, True])
def test_native_first_pass_keeps_rejected_prefix_and_fresh_seven(monkeypatch, rejected, mrope):
    native = definitions(
        "references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "native_first_mtp_inputs", set(), {"torch": torch}, monkeypatch,
        methods=[("AscendSpecDecodeBaseProposer", "set_inputs_first_pass")])
    positions_native = definitions(
        "references/vllm/vllm/v1/spec_decode/llm_base_proposer.py",
        "native_first_mtp_positions", set(), {"torch": torch}, monkeypatch,
        methods=[("SpecDecodeBaseProposer", "_set_positions")])
    cad = common()
    before = {name: getattr(cad, name).clone() for name in (
        "query_start_loc", "query_start_loc_cpu", "seq_lens", "seq_lens_cpu_upper_bound", "slot_mapping")}
    inputs = torch.arange(101, 112)
    positions = torch.arange(11) if not mrope else torch.arange(33).view(3, 11)
    hidden = torch.arange(44).view(11, 4)
    sample = torch.tensor([3 - rejected, 10])
    next_ids = torch.tensor([900, 901])
    proposer = SimpleNamespace(
        needs_extra_input_slots=False, runner=SimpleNamespace(pcp_manager=None),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(uses_mrope=mrope)),
        input_ids=torch.full((11,), -99), hidden_states=torch.empty_like(hidden),
        positions=torch.empty(11, dtype=positions.dtype),
        mrope_positions=torch.empty(3, 11, dtype=positions.dtype),
        uses_mrope=mrope, uses_xdrope_dim=0, draft_uses_xdrope_dim=0)
    proposer._set_positions = MethodType(positions_native._set_positions, proposer)
    count, actual_sample, returned, long_seq_args = native.set_inputs_first_pass(
        proposer, inputs, next_ids, positions, hidden, sample, cad,
        torch.tensor([rejected, 0]))
    assert count == 11 and returned is cad and actual_sample is sample and long_seq_args is None
    expected_ids = torch.cat((inputs[1:], inputs[-1:]))
    expected_ids[sample] = next_ids
    assert torch.equal(proposer.input_ids, expected_ids)
    assert torch.equal(proposer.hidden_states, hidden)
    assert torch.equal(proposer.mrope_positions if mrope else proposer.positions, positions)
    for name, expected in before.items():
        assert torch.equal(getattr(cad, name), expected)
    metadata = from_common(cad, is_draft=True, current_only=True, first_draft_current_only=True)
    plan = metadata.current_only_plan
    assert (plan.prefix_requests, plan.prefix_tokens, plan.fresh_requests, plan.fresh_tokens) == (1, 4, 1, 7)
    assert plan.cumulative == (7,)
    assert metadata.is_draft and metadata.draft_index == 0 and metadata.first_draft_current_only
    assert not use_native_current(metadata)  # The original MTP prefix remains full CV.


@pytest.mark.parametrize("section,name,value", [
    ("spec", "method", "eagle"), ("spec", "enforce_eager", False),
    ("spec", "enforce_eager", None), ("spec", "parallel_drafting", True),
    ("spec", "disable_padded_drafter_batch", True),
    ("hf", "model_type", "deepseek_mtp"), ("hf", "architectures", ["Qwen3_5MoeMTP"]),
    ("parallel", "prefill_context_parallel_size", 2),
    ("parallel", "decode_context_parallel_size", 2),
    ("parallel", "decode_context_parallel_size", True),
])
def test_only_audited_native_config_can_enable_first_mtp(section, name, value):
    cfg = config()
    assert supports_first_draft_current_only(cfg)
    target = {"spec": cfg.speculative_config, "parallel": cfg.parallel_config,
              "hf": cfg.speculative_config.draft_model_config.hf_config}[section]
    setattr(target, name, value)
    assert not supports_first_draft_current_only(cfg)
    assert not supports_first_draft_current_only(object())


class NoCPUReads(SimpleNamespace):
    def __getattribute__(self, name):
        if name in {"query_start_loc_cpu", "seq_lens_cpu_upper_bound"}:
            raise AssertionError(f"unexpected CPU metadata read: {name}")
        return super().__getattribute__(name)


def builder(monkeypatch, *, enabled=True, cfg=None):
    definitions(
        "references/vllm/vllm/v1/attention/backend.py", "vllm.v1.attention.backend",
        {"AttentionType", "MultipleOf", "AttentionBackend", "AttentionCGSupport", "AttentionMetadataBuilder"},
        {"ABC": ABC, "abstractmethod": abstractmethod, "Enum": Enum, "Generic": Generic,
         "ClassVar": ClassVar, "M": TypeVar("M"), "torch": torch}, monkeypatch)
    backend = load_our_module("oscar_ascend.integration.backend", monkeypatch)
    monkeypatch.setattr(backend, "require_runtime", lambda: SimpleNamespace(config={"experimental_current_only": enabled}))
    return backend.OscarMetadataBuilder(object(), ["mtp.layers.0.self_attn.attn"], cfg or config(), "cpu")


def test_builder_only_explicit_first_pass_reads_cpu_and_plans(monkeypatch):
    build = builder(monkeypatch)
    first = build.build(0, common(), object())  # Pinned native first call supplies model.
    assert first.first_draft_current_only and first.current_only_plan.fresh_tokens == 7
    main = build.build(0, common())
    assert not main.is_draft and not main.first_draft_current_only and use_native_current(main)
    assert main.current_only_plan.fresh_tokens == 7
    for index in (0, 1, 2):
        later = build.build_for_drafting(NoCPUReads(**vars(common())), index)
        assert later.is_draft and later.draft_index == index
        assert later.current_only_plan is None and later.current_cumulative is None
        assert not later.first_draft_current_only
    with native_dummy_run():
        dummy = build.build(0, NoCPUReads(**vars(common())), object())
    assert dummy.dummy_origin and dummy.current_only_plan is None
    capture = build.build_for_cudagraph_capture(NoCPUReads(**vars(common())))
    assert capture.capture_origin and capture.current_only_plan is None
    # Restore slots changed by capture; a fresh builder does not conflate addresses.
    real = builder(monkeypatch).build(0, common(), object())
    assert real.current_only_plan.fresh_tokens == 7


def test_disabled_or_non_eager_first_draft_never_reads_cpu(monkeypatch):
    disabled = builder(monkeypatch, enabled=False)
    assert disabled.build(0, NoCPUReads(**vars(common())), object()).current_only_plan is None
    cfg = config()
    cfg.speculative_config.enforce_eager = False
    unavailable = builder(monkeypatch, cfg=cfg)
    assert unavailable.build(0, NoCPUReads(**vars(common())), object()).current_only_plan is None
    # Even an explicit stale qualification cannot reach a later draft read.
    later = from_common(NoCPUReads(**vars(common())), is_draft=True, draft_index=1,
                        current_only=True, first_draft_current_only=True)
    assert later.current_only_plan is None and not later.first_draft_current_only


def test_forward_dispatches_qualified_first_draft_without_widening_native_current(monkeypatch):
    from oscar_ascend.integration import current_only_dispatch
    import sys

    builder(monkeypatch)
    monkeypatch.setattr(sys.modules["vllm.v1.attention.backend"], "AttentionImpl", object, raising=False)
    module = load_our_module("oscar_ascend.integration.impl", monkeypatch)
    impl = object.__new__(module.OscarAttentionImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 6, 1, 64
    cache = torch.empty(1, dtype=torch.uint8)
    state = SimpleNamespace(packed=cache, cache_format="canonical_v1")
    impl.provider = SimpleNamespace(config={"experimental_current_only": True}, layer_state=lambda _: state)
    metadata = from_common(common(), is_draft=True, current_only=True, first_draft_current_only=True)
    calls = []
    def suffix(*args):
        calls.append(args[6])
        return args[7]
    monkeypatch.setattr(current_only_dispatch, "dispatch_current_suffix", suffix)
    q = torch.zeros(11, 6 * 64, dtype=torch.bfloat16)
    kv = torch.zeros(11, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q)
    assert not use_native_current(metadata)
    assert impl.forward(SimpleNamespace(layer_name="mtp"), q, kv, kv, cache, metadata, output) is output
    assert calls == [metadata]


@pytest.mark.parametrize("changes", [
    {"first_draft_current_only": False}, {"draft_index": 1},
    {"dummy_origin": True}, {"capture_origin": True},
])
def test_dispatch_rejects_unqualified_later_dummy_and_capture_before_ops(changes):
    from dataclasses import replace
    from oscar_ascend.integration.current_only_dispatch import dispatch_current_suffix
    from oscar_ascend.integration.runtime_api import OscarReadinessError
    metadata = from_common(common(), is_draft=True, current_only=True, first_draft_current_only=True)
    metadata = replace(metadata, **changes)
    with pytest.raises(OscarReadinessError, match="qualified first MTP"):
        dispatch_current_suffix(None, None, None, None, None, None, metadata, None, metadata.current_only_plan)
