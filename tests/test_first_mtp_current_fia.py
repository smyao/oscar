"""Archive #126/#140/#142/#148/#155: first-MTP source2 domain contracts.

Pinned native input shift executes on CPU. FP32 mathematical decomposition is
independently checked here; native FIA precision/graph/model evidence is absent.
"""
from dataclasses import replace
from types import MethodType, SimpleNamespace
import sys

import pytest
import torch

from oscar_ascend.integration.current_attention import use_native_current, _populate_slot_error_status
from oscar_ascend.integration.dummy_context import native_dummy_run
from oscar_ascend.integration.metadata import from_common
from oscar_ascend.ops.reference import attention, encode_kv, decode_kv
from test_current_attention import AscendAttentionState
from test_first_draft_current_only import builder as base_builder, config, common, NoCPUReads
from test_runtime_native import definitions


def _builder(monkeypatch, enabled=True):
    base_builder(monkeypatch,enabled=False)
    backend=sys.modules["oscar_ascend.integration.backend"]
    monkeypatch.setattr(backend,"require_runtime",lambda:SimpleNamespace(config={
        "experimental_first_mtp_current_fia":enabled,"experimental_current_only":False}))
    return backend.OscarMetadataBuilder(object(),["mtp"],config(),"cpu")


@pytest.mark.parametrize("stage",[AscendAttentionState.ChunkedPrefill,AscendAttentionState.PrefillCacheHit,
                                  AscendAttentionState.SpecDecoding])
def test_only_explicit_first_build_qualifies_positive_history_current_partial(monkeypatch,stage):
    build=_builder(monkeypatch)
    cm=common();cm.attn_state=stage;cm.seq_lens=torch.tensor([515,776])
    del cm.seq_lens_cpu_upper_bound  # No zero-history proof is used or needed.
    first=build.build(0,cm,object())
    assert first.is_draft and first.draft_index==0 and first.first_draft_current_fia
    assert use_native_current(first) and first.current_cumulative==(4,11)
    assert first.current_only_plan is None and not first.first_draft_current_only
    for index in (0,1,2):
        later=build.build_for_drafting(NoCPUReads(**vars(cm)),index)
        assert not later.first_draft_current_fia and not use_native_current(later)
    with native_dummy_run():
        dummy=build.build(0,NoCPUReads(**vars(cm)),object())
    assert not use_native_current(dummy) and dummy.current_cumulative is None
    capture=build.build_for_graph_capture(NoCPUReads(**vars(cm)))
    assert not use_native_current(capture)


def test_disabled_flag_and_index_zero_alone_do_not_read_cpu(monkeypatch):
    first=_builder(monkeypatch,False).build(0,NoCPUReads(**vars(common())),object())
    assert first.draft_index==0 and not first.first_draft_current_fia and not use_native_current(first)


@pytest.mark.parametrize("rejected",range(4))
@pytest.mark.parametrize("mrope",[False,True])
def test_native_shift_preserves_source2_set_and_three_source_attention(monkeypatch,rejected,mrope):
    native=definitions("references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "native_first_current_partial",set(),{"torch":torch},monkeypatch,
        methods=[("AscendSpecDecodeBaseProposer","set_inputs_first_pass")])
    native_pos=definitions("references/vllm/vllm/v1/spec_decode/llm_base_proposer.py",
        "native_first_current_partial_positions",set(),{"torch":torch},monkeypatch,
        methods=[("SpecDecodeBaseProposer","_set_positions")])
    starts=(0,4,11);contexts=(511,769);n=11;dim=64;heads=6
    cm=common();cm.seq_lens=torch.tensor([515,776]);cm.attn_state=AscendAttentionState.ChunkedPrefill
    cm.slot_mapping=torch.tensor(list(range(511,515))+list(range(769,776)))
    cm.positions=torch.tensor(list(range(511,515))+list(range(769,776)))
    positions=torch.stack((cm.positions,cm.positions+1000,cm.positions+2000)) if mrope else cm.positions
    hidden=torch.randn(n,heads*dim,generator=torch.Generator().manual_seed(42))
    proposer=SimpleNamespace(needs_extra_input_slots=False,runner=SimpleNamespace(pcp_manager=None),
        input_ids=torch.empty(n,dtype=torch.int64),hidden_states=torch.empty_like(hidden),
        uses_mrope=mrope,uses_xdrope_dim=0,draft_uses_xdrope_dim=0,
        positions=torch.empty(n,dtype=torch.int64),mrope_positions=torch.empty(3,n,dtype=torch.int64),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(uses_mrope=mrope)))
    proposer._set_positions=MethodType(native_pos._set_positions,proposer)
    before_slots=cm.slot_mapping.clone();sample=torch.tensor([3-rejected,10])
    _,actual_sample,returned,_=native.set_inputs_first_pass(proposer,torch.arange(n),torch.tensor([50,51]),
        positions,hidden,sample,cm,torch.tensor([rejected,0]))
    assert returned is cm and actual_sample is sample and torch.equal(cm.slot_mapping,before_slots)
    assert torch.equal(proposer.hidden_states,hidden)
    assert torch.equal(proposer.mrope_positions if mrope else proposer.positions,positions)
    meta=from_common(cm,is_draft=True,first_draft_current_fia=True)
    assert use_native_current(meta) and meta.current_only_plan is None
    # Deterministic per-row Q/K/V from the ACTUAL shifted native inputs.
    ids=proposer.input_ids.float()[:,None,None]
    q=torch.sin(hidden.view(n,heads,dim)+ids*.01).to(torch.bfloat16)
    k=torch.cos(hidden[:,:dim].view(n,1,dim)+ids*.02).to(torch.bfloat16)
    v=torch.sin(hidden[:,-dim:].view(n,1,dim)-ids*.03).to(torch.bfloat16)
    generator=torch.Generator().manual_seed(47022)
    for request,context in enumerate(contexts):
        begin,end=starts[request:request+2]
        current=attention(q[begin:end],k[begin:end],v[begin:end],causal=True)
        old_k=torch.randn(context,1,dim,generator=generator).to(torch.bfloat16)
        old_v=torch.randn(context,1,dim,generator=generator).to(torch.bfloat16)
        decoded_k,decoded_v=decode_kv(encode_kv(old_k.float(),old_v.float()),dim)
        for local,index in enumerate(range(begin,end)):
            # C++ source2 global positions [context,context+local+1) map
            # exactly to this native request's current tensor prefix.
            assert [begin+p-context for p in range(context,context+local+1)]==list(range(begin,index+1))
            cut=min(context,max(64,context+local+1-256))
            h=attention(q[index:index+1],decoded_k[64:cut],decoded_v[64:cut],causal=False)
            precise_k=torch.cat((old_k[:64],old_k[cut:]));precise_v=torch.cat((old_v[:64],old_v[cut:]))
            w=attention(q[index:index+1],precise_k,precise_v,causal=False)
            lses=torch.stack((h.lse[0],w.lse[0],current.lse[local]),-1)
            # Production FIA returns BF16 current output before the FP32
            # three-source merge; include that rounding in this CPU check.
            outputs=torch.stack((h.output[0],w.output[0],current.output[local].to(torch.bfloat16).float()),-2)
            merged=(outputs*torch.softmax(lses,-1)[...,None]).sum(-2)
            selected_k=old_k.float();selected_v=old_v.float()
            selected_k[64:cut]=decoded_k[64:cut];selected_v[64:cut]=decoded_v[64:cut]
            full=attention(q[index:index+1],torch.cat((selected_k,k[begin:index+1].float())),
                torch.cat((selected_v,v[begin:index+1].float())),causal=False)
            torch.testing.assert_close(merged,full.output[0],atol=.005,rtol=.005)
            torch.testing.assert_close(torch.logsumexp(lses,-1),full.lse[0],atol=.005,rtol=.005)


def test_unexpected_internal_slot_hole_is_still_an_error():
    status=torch.empty(11,6,dtype=torch.int32)
    slots=torch.arange(16);slots[2]=-1;slots[11:]=-1
    checked=_populate_slot_error_status(status,slots,11)
    assert bool(checked[2].eq(1).all()) and int(checked.sum())==6


def test_native_unpadded_first_view_excludes_graph_padding_request(monkeypatch):
    native=definitions("references/vllm-ascend/vllm_ascend/attention/utils.py",
        "native_unpadded_first_partial",set(),{"AscendCommonAttentionMetadata":SimpleNamespace},monkeypatch,
        methods=[("AscendCommonAttentionMetadata","unpadded")])
    cm=common();cm.query_start_loc=torch.tensor([0,4,11,16],dtype=torch.int32)
    cm.query_start_loc_cpu=cm.query_start_loc.clone();cm.seq_lens=torch.tensor([515,776,0])
    cm.slot_mapping=torch.tensor(list(range(11))+[-1]*5)
    cm.num_input_tokens=16;cm.num_reqs=3;cm.max_seq_len=776
    cm.seq_lens_cpu_upper_bound=cm.seq_lens.clone();cm.block_table_tensor=torch.zeros(3,8,dtype=torch.int32)
    cm.actual_seq_lengths_q=[4,11,16];cm.decode_token_per_req=4
    for name in ("seq_lens_cpu","num_computed_tokens_cpu","positions","positions_cpu",
                 "prefill_context_parallel_metadata","_seq_lens_cpu","_num_computed_tokens_cpu",
                 "dcp_local_seq_lens","dcp_local_seq_lens_cpu","is_prefilling","encoder_seq_lens",
                 "encoder_seq_lens_cpu","logits_indices_padded","num_logits_indices"):
        setattr(cm,name,None)
    live=native.unpadded(cm,11,2)
    assert live.query_start_loc_cpu.tolist()==[0,4,11] and live.num_actual_tokens==11
    assert live.slot_mapping is cm.slot_mapping and live.num_input_tokens==16
    meta=from_common(live,is_draft=True,first_draft_current_fia=True)
    assert meta.current_cumulative==(4,11) and use_native_current(meta)
    assert meta.current_only_plan is None
