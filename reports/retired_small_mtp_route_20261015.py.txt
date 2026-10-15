"""Archive #160-162: native three-token MTP with all later padding removed.

Execute the pinned native merged loop, MTP forward and public compile bypass.
The existing row-local CPU layer fixture establishes input/output equivalence;
this is not real W8A8 quantization, NPU graph or whole-model precision evidence.
"""
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from oscar_ascend.integration import compact_later_mtp as compact
from test_compact_later_mtp import FLAGS, _case
from test_runtime_native import definitions

UNCACHED = SimpleNamespace(config={"enable_prefix_caching": False})


def _small_case(monkeypatch, batch, n, mrope=True):
    model, context, values, public_call, seen = _case(monkeypatch, mrope=mrope)
    table=torch.arange(batch*8,dtype=torch.int32).reshape(batch,8)+10
    positions=torch.arange(batch,dtype=torch.int64)+504
    slots=table.gather(1,(positions//128)[:,None]).flatten().to(torch.int64)*128+positions%128
    slots=torch.cat((slots,torch.full((n-batch,),-1,dtype=torch.int64)))
    metadata=replace(context.attn_metadata["mtp"],
        query_start_loc=torch.arange(batch+1,dtype=torch.int32),
        seq_lens=(positions+1).to(torch.int32),block_tables=table,slot_mapping=slots,
        num_reqs=batch,num_actual_tokens=batch,max_seq_len=int(positions[-1])+1,
        num_input_tokens=n)
    context.attn_metadata={"mtp":metadata}
    context.num_tokens=n;context.num_accept_tokens=batch
    logical=torch.cat([torch.arange(500+row,504+row) for row in range(batch)])
    logical=torch.cat((logical,torch.zeros(n-logical.numel(),dtype=torch.int64)))
    values=dict(input_ids=torch.arange(n),hidden_states=torch.arange(n*4).view(n,4).float(),
        positions=torch.stack((logical,logical+1000,logical+2000)) if mrope else logical,
        inputs_embeds=None)
    return model,context,values,public_call,seen


@pytest.mark.parametrize("batch,n", [(32,128),(31,128),(24,96),(12,48)])
@pytest.mark.parametrize("mrope", [False,True])
def test_native_three_drafts_small_padding_equivalence(monkeypatch,batch,n,mrope):
    model,ctx,values,public_call,seen=_small_case(monkeypatch,batch,n,mrope)
    native=definitions("references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "compact_small_native_loop",set(),{"torch":torch,"get_forward_context":lambda:ctx,
        "_EXTRA_CTX":ctx,"lmhead_tp_enable":lambda:False,
        "get_ascend_config":lambda:SimpleNamespace(enable_reduce_sample=False)},monkeypatch,
        methods=[("AscendSpecDecodeBaseProposer",name) for name in (
            "_run_merged_draft","maybe_pad_and_reduce","maybe_all_gather_and_unpad","model_returns_tuple")])
    position_methods=definitions("references/vllm/vllm/v1/spec_decode/llm_base_proposer.py",
        "compact_small_native_positions",set(),{"torch":torch},monkeypatch,
        methods=[("SpecDecodeBaseProposer","_get_positions"),("SpecDecodeBaseProposer","_set_positions")])
    later=ctx.attn_metadata["mtp"]
    first=replace(later,query_start_loc=torch.arange(batch+1,dtype=torch.int32)*4,
        seq_lens=later.seq_lens-1,num_actual_tokens=batch*4,max_query_len=4,draft_index=0)
    snapshots=[]

    def execute(enabled):
        ctx.attn_metadata={"mtp":first};ctx.num_tokens=n;ctx.num_accept_tokens=batch;ctx.skip_compiled=False
        active_outputs=[]

        class Caller:
            def __call__(self,**kwargs):
                is_later=ctx.attn_metadata["mtp"].draft_index>0
                if is_later:
                    # Native updates only its leading B hidden rows on each
                    # later step. Poison the never-consumed tail explicitly.
                    kwargs["hidden_states"][batch:].fill_(torch.nan)
                saved=(ctx.attn_metadata,ctx.num_tokens,ctx.skip_compiled)
                if enabled and is_later:
                    def checked_original(instance,**live):
                        assert live["hidden_states"].shape[0]==batch
                        assert torch.isfinite(live["hidden_states"]).all()
                        assert live["input_ids"].numel()==batch
                        assert live["positions"].shape[-1]==batch
                        assert ctx.skip_compiled is True
                        assert ctx.attn_metadata["mtp"].cv_shape_tokens==n
                        return public_call(instance,**live)
                    output=compact._compact_call(checked_original,model,(),kwargs,UNCACHED,ctx,FLAGS,(1,1,4))
                else:
                    # The CPU baseline executes the same public/native forward,
                    # without claiming to run an AOT/NPU graph on this host.
                    ctx.skip_compiled=True
                    try:output=public_call(model,**kwargs)
                    finally:ctx.skip_compiled=saved[2]
                assert ctx.attn_metadata is saved[0]
                assert (ctx.num_tokens,ctx.skip_compiled)==saved[1:]
                if is_later:active_outputs.append(output[:batch].clone())
                return output

        Caller.model=model.model
        Caller.compute_logits=staticmethod(model.compute_logits)
        proposer=SimpleNamespace(model=Caller(),method="mtp",_share_mtp_indices=False,
            pass_hidden_states_to_model=True,parallel_drafting=False,num_speculative_tokens=3,
            device="cpu",pcp_size=1,dcp_size=1,use_cuda_graph=False,uses_mrope=mrope,
            uses_xdrope_dim=0,draft_uses_xdrope_dim=0,is_multimodal_model=False,
            enable_shared_expert_dp=False,supports_mm_inputs=False,runner=SimpleNamespace(pcp_manager=None),
            draft_model_config=model.vllm_config.speculative_config.draft_model_config,
            vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=50000,uses_mrope=mrope)),
            input_ids=values["input_ids"].clone(),hidden_states=values["hidden_states"].clone(),
            positions=values["positions"].clone() if not mrope else torch.empty(n,dtype=torch.int64),
            mrope_positions=values["positions"].clone() if mrope else torch.empty(3,n,dtype=torch.int64),
            arange=torch.arange(n))
        for name in ("maybe_pad_and_reduce","maybe_all_gather_and_unpad","model_returns_tuple"):
            setattr(proposer,name,MethodType(getattr(native,name),proposer))
        proposer._get_positions=MethodType(position_methods._get_positions,proposer)
        proposer._set_positions=MethodType(position_methods._set_positions,proposer)
        result=native._run_merged_draft(proposer,n,batch,torch.arange(batch)*4+3,values["positions"],None,
            [{"mtp":first},{"mtp":later},{"mtp":replace(later,draft_index=2,seq_lens=later.seq_lens+1)}],n,True)
        snapshots.append(active_outputs)
        return result

    expected=execute(False);baseline=seen[:];seen.clear()
    actual=execute(True)
    assert actual.shape==(batch,3) and torch.equal(actual,expected)
    assert len(snapshots[0])==len(snapshots[1])==2
    assert all(torch.equal(a,b) for a,b in zip(*snapshots))
    assert [row[0] for row in baseline]==[n,n,n]
    assert [row[0] for row in seen]==[n,batch,batch]
    assert all(row[3]==n and row[4] is True for row in seen[1:])
    assert ctx.num_tokens==n and ctx.skip_compiled is False


def test_small_compaction_keeps_tail_failures_and_exception_restoration(monkeypatch):
    model,ctx,values,original,seen=_small_case(monkeypatch,31,128)
    saved=dict(vars(ctx))
    ctx.attn_metadata["mtp"].slot_mapping[-1]=0
    with pytest.raises(RuntimeError,match="tail slot trap"):
        compact._compact_call(original,model,(),values,UNCACHED,ctx,FLAGS,(1,1,4))
    assert not seen and vars(ctx)==saved
    ctx.attn_metadata["mtp"].slot_mapping[-1]=-1
    def fail(instance,**live):
        assert live["hidden_states"].shape[0]==31 and ctx.skip_compiled
        raise RuntimeError("real model failure")
    with pytest.raises(RuntimeError,match="real model failure"):
        compact._compact_call(fail,model,(),values,UNCACHED,ctx,FLAGS,(1,1,4))
    assert vars(ctx)==saved


def test_apc_on_preserves_the_previously_delivered_small_batch_path(monkeypatch):
    model,ctx,values,original,seen=_small_case(monkeypatch,32,128)
    before=dict(vars(ctx)); marker=object()
    provider=SimpleNamespace(config={"enable_prefix_caching": True})
    monkeypatch.setattr(compact,"_validate_model",lambda *a:pytest.fail("APC-on dispatch must stay unchanged"))
    monkeypatch.setattr(compact,"_tail_slot_guard",lambda *a:pytest.fail("APC-on short path must stay unchanged"))
    result=compact._compact_call(lambda *a,**k:marker,model,(),values,provider,ctx,FLAGS,(1,1,4))
    assert result is marker and vars(ctx)==before and not seen


@pytest.mark.parametrize("n,b", [(128,32),(128,31),(96,24),(48,12),(1024,128),(16384,32)])
def test_apc_policy_matches_pre_fix_predicate(n,b):
    assert compact.should_compact_rows(n,b,prefix_caching=True) == (n>=1024 and n>=8*b)
