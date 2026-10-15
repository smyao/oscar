"""Archive #34/#36/#140/#142/#148/#155: compact caller/restore contracts.

Pinned native later-loop, MTP forward and compile bypass AST execute on CPU.
Matrix/attention stand-ins establish row/metadata lifecycle, not NPU precision.
"""
import ast
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from oscar_ascend.integration import compact_later_mtp as compact
from oscar_ascend.integration.metadata import OscarMetadata
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.runtime import WorkspaceGeometry
from test_runtime_native import definitions

ROOT=Path(__file__).resolve().parents[1]
FLAGS={name:False for name in ("enable_sp","enable_sp_by_pass","flashcomm2_enable","shared_expert_dp_enabled",
    "lmhead_tp_enable","embedding_tp_enable","mlp_tp_enable","oproj_tp_enable","olora_tp_enable","matmul_allreduce_enable","o_shard_enable")}


def _function(path, name, namespace):
    tree=ast.parse((ROOT/path).read_text())
    function=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==name)
    function.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),function],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(ROOT/path),"exec"),namespace)
    return namespace[name]


def _case(monkeypatch, *, mrope=True, negative_live=False):
    # Native q4 + fresh1020 first pass: two later requests in N1024 buffers.
    n,b,h=1024,2,4
    cfg=SimpleNamespace(speculative_config=SimpleNamespace(method="mtp",enforce_eager=True,
        parallel_drafting=False,disable_padded_drafter_batch=False,draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_mtp",architectures=["Qwen3_5MTP"]))),
        parallel_config=SimpleNamespace(data_parallel_size=1,pipeline_parallel_size=1,tensor_parallel_size=4,
            prefill_context_parallel_size=1,decode_context_parallel_size=1),lora_config=None)
    later=OscarMetadata(torch.arange(b+1,dtype=torch.int32),torch.tensor([505,1021],dtype=torch.int32),
        torch.zeros(b,8,dtype=torch.int32),torch.tensor([504,-1 if negative_live else 1020]+[-1]*(n-b)),
        b,b,1,1021,num_input_tokens=n,is_draft=True,draft_index=1)
    context=SimpleNamespace(attn_metadata={"mtp":later},num_tokens=n,num_accept_tokens=b,skip_compiled=False,
        is_draft_model=True,cudagraph_runtime_mode=SimpleNamespace(name="NONE"),in_profile_run=False,
        capturing=False,flash_comm_v1_enabled=False,flashcomm_v2_enabled=False,pad_size=0)
    namespace={"torch":torch,"get_pp_group":lambda:SimpleNamespace(is_last_rank=True),
        "is_forward_context_available":lambda:True,"get_forward_context":lambda:context}
    public_call=_function("references/vllm/vllm/compilation/decorators.py","__call__",namespace.copy())
    native_forward=_function("references/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py",
        "qwen3_5_mtp_forward",namespace.copy())
    seen=[]
    class Norm:
        def __call__(self,x,residual=None):
            return x if residual is None else (x+residual,None)
    scheme=type("AscendW8A8LinearMethod",(),{})()
    linear=SimpleNamespace(quant_method=SimpleNamespace(quant_method=scheme))
    attn=SimpleNamespace(impl=type("OscarAttentionImpl",(),{})(),calculate_kv_scales=False,query_quant=None)
    class DenseLayer:
        layer_type="full_attention"
        self_attn=SimpleNamespace(qkv_proj=linear,o_proj=linear,attn=attn)
        mlp=SimpleNamespace(gate_up_proj=linear,down_proj=linear)
        def __call__(self,positions,hidden_states,residual):
            meta=context.attn_metadata["mtp"]
            seen.append((hidden_states.shape[0],positions.clone(),meta.num_input_tokens,meta.cv_shape_tokens,
                         context.skip_compiled,context.num_tokens))
            assert meta.num_input_tokens==hidden_states.shape[0]
            return hidden_states*0.5,hidden_states
    class FC:
        quant_method=linear.quant_method
        def __call__(self,x):return x[:,:h]+x[:,h:]
    predictor=type("Qwen3_5MultiTokenPredictor",(),{"__call__":public_call,"forward":native_forward})()
    predictor.config=SimpleNamespace(model_type="qwen3_5_text")
    predictor.num_mtp_layers=1;predictor.layers=[DenseLayer()];predictor.fc=FC()
    predictor.pre_fc_norm_hidden=predictor.pre_fc_norm_embedding=predictor.norm=Norm()
    predictor.embed_input_ids=lambda ids:ids.float()[:,None].expand(-1,h)
    predictor.do_not_compile=False
    def outer_forward(self,**kwargs):return self.model(**kwargs)
    model=type("Qwen3_5MTP",(),{"__call__":public_call,"forward":outer_forward})()
    model.model=predictor;model.vllm_config=cfg;model.do_not_compile=False
    model.compute_logits=lambda hidden:torch.stack((torch.sin(hidden[:,0]),torch.cos(hidden[:,0])),dim=-1)
    logical_positions=torch.cat((torch.arange(500,504),torch.arange(1020)))
    positions=torch.stack((logical_positions,logical_positions+1000,logical_positions+2000)) if mrope else logical_positions
    values=dict(input_ids=torch.arange(n),hidden_states=torch.arange(n*h).view(n,h).float(),
        positions=positions,inputs_embeds=None)
    def cpu_tail_guard(provider,slots,begin,end):
        if bool(compact._tail_slot_errors(slots,begin,end).any()):raise RuntimeError("tail slot trap")
    monkeypatch.setattr(compact,"_tail_slot_guard",cpu_tail_guard)
    return model,context,values,public_call,seen


@pytest.mark.parametrize("mrope",[False,True])
@pytest.mark.parametrize("embeds",[False,True])
def test_public_compile_bypass_compacts_metadata_and_restores_context(monkeypatch,mrope,embeds):
    model,ctx,values,original,seen=_case(monkeypatch,mrope=mrope,negative_live=True)
    if embeds:values["inputs_embeds"]=torch.ones_like(values["hidden_states"])
    before=dict(vars(ctx));old=ctx.attn_metadata
    result=compact._compact_call(original,model,(),values,None,ctx,FLAGS,(1,1,4))
    assert result.shape==(2,4)
    assert seen[0][0]==seen[0][2]==seen[0][5]==2 and seen[0][3]==1024 and seen[0][4] is True
    assert torch.equal(seen[0][1],values["positions"][...,:2])
    assert vars(ctx)==before and ctx.attn_metadata is old
    assert old["mtp"].cv_shape_tokens is None and old["mtp"].num_input_tokens==1024


def test_tail_guard_and_exception_restore_preserve_failures(monkeypatch):
    model,ctx,values,original,seen=_case(monkeypatch)
    ctx.attn_metadata["mtp"].slot_mapping[-1]=0
    with pytest.raises(RuntimeError,match="tail slot trap"):
        compact._compact_call(original,model,(),values,None,ctx,FLAGS,(1,1,4))
    assert not seen and ctx.num_tokens==1024 and ctx.skip_compiled is False
    ctx.attn_metadata["mtp"].slot_mapping[-1]=-1
    original_context=dict(vars(ctx))
    def fail(model,**kwargs):
        assert ctx.skip_compiled and ctx.num_tokens==2
        raise RuntimeError("original model failure")
    with pytest.raises(RuntimeError,match="original model failure"):
        compact._compact_call(fail,model,(),values,None,ctx,FLAGS,(1,1,4))
    assert vars(ctx)==original_context


@pytest.mark.parametrize("failure",["sp","dp","lora","moe","quant","kvscale","compiled_contract"])
def test_unsupported_actual_model_or_native_runtime_fails_closed(monkeypatch,failure):
    model,ctx,values,original,_=_case(monkeypatch)
    flags=dict(FLAGS)
    if failure=="sp":flags["enable_sp"]=True
    elif failure=="dp":model.vllm_config.parallel_config.data_parallel_size=2
    elif failure=="lora":model.vllm_config.lora_config=object()
    elif failure=="moe":model.model.config.model_type="qwen3_5_moe_text"
    elif failure=="quant":model.model.fc.quant_method=object()
    elif failure=="kvscale":model.model.layers[0].self_attn.attn.calculate_kv_scales=True
    elif failure=="compiled_contract":del model.do_not_compile
    with pytest.raises(OscarReadinessError,match="compact later MTP"):
        compact._compact_call(original,model,(),values,None,ctx,flags,(1,1,4))
    assert ctx.num_tokens==1024 and ctx.skip_compiled is False


@pytest.mark.parametrize("mrope",[False,True])
def test_native_merged_loop_consumes_only_b_rows_after_compact_return(monkeypatch,mrope):
    model,ctx,values,public_call,seen=_case(monkeypatch,mrope=mrope)
    native=definitions("references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "compact_native_later_loop",set(),{"torch":torch,"get_forward_context":lambda:ctx,
        "_EXTRA_CTX":ctx,"lmhead_tp_enable":lambda:False,
        "get_ascend_config":lambda:SimpleNamespace(enable_reduce_sample=False)},monkeypatch,
        methods=[("AscendSpecDecodeBaseProposer",name) for name in (
            "_run_merged_draft","maybe_pad_and_reduce","maybe_all_gather_and_unpad","model_returns_tuple")])
    positions_native=definitions("references/vllm/vllm/v1/spec_decode/llm_base_proposer.py",
        "compact_native_positions",set(),{"torch":torch},monkeypatch,
        methods=[("SpecDecodeBaseProposer","_get_positions"),("SpecDecodeBaseProposer","_set_positions")])
    first=replace(ctx.attn_metadata["mtp"],query_start_loc=torch.tensor([0,4,1024]),
        seq_lens=torch.tensor([504,1020]),num_actual_tokens=1024,max_query_len=1020,max_seq_len=1020,draft_index=0)
    later=ctx.attn_metadata["mtp"]
    ctx.attn_metadata={"mtp":first}
    def execute(enabled):
        ctx.attn_metadata={"mtp":first};ctx.num_tokens=1024;ctx.skip_compiled=False
        class Caller:
            def __call__(self,**kwargs):
                if enabled and ctx.attn_metadata["mtp"].draft_index>0:
                    return compact._compact_call(public_call,model,(),kwargs,None,ctx,FLAGS,(1,1,4))
                # Execute the same native MTP forward for the baseline/first;
                # this fixture does not pretend to execute a compiled graph.
                saved=ctx.skip_compiled;ctx.skip_compiled=True
                try:return public_call(model,**kwargs)
                finally:ctx.skip_compiled=saved
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
            positions=values["positions"].clone() if not mrope else torch.empty(1024,dtype=torch.int64),
            mrope_positions=values["positions"].clone() if mrope else torch.empty(3,1024,dtype=torch.int64),
            arange=torch.arange(1024))
        for name in ("maybe_pad_and_reduce","maybe_all_gather_and_unpad","model_returns_tuple"):
            setattr(proposer,name,MethodType(getattr(native,name),proposer))
        proposer._get_positions=MethodType(positions_native._get_positions,proposer)
        proposer._set_positions=MethodType(positions_native._set_positions,proposer)
        return native._run_merged_draft(proposer,1024,2,torch.tensor([3,1023]),values["positions"],None,
            [{"mtp":first},{"mtp":later},{"mtp":replace(later,draft_index=2,seq_lens=later.seq_lens+1)}],1024,True)
    expected=execute(False);baseline=seen[:];seen.clear()
    actual=execute(True)
    assert torch.equal(actual,expected) and actual.shape==(2,3)
    assert [v[0] for v in baseline]==[1024,1024,1024]
    assert [v[0] for v in seen]==[1024,2,2]
    assert torch.equal(seen[1][1],baseline[1][1][...,:2])
    assert torch.equal(seen[2][1],baseline[2][1][...,:2])
    assert ctx.num_tokens==1024 and ctx.skip_compiled is False


def test_original_padded_shape_keeps_cv_split_choice():
    geometry=WorkspaceGeometry(16384,6,1,256,cube_cores=20)
    assert geometry.splits_for_tokens(16384)==1 and geometry.splits_for_tokens(1)==20
    row=OscarMetadata(None,None,None,None,1,1,1,1,num_input_tokens=1,is_draft=True,
        draft_index=1,cv_shape_tokens=16384)
    assert geometry.splits_for_tokens(row.cv_shape_tokens)==1


@pytest.mark.parametrize("compact_shape,expected_splits",[(None,20),(16384,1)])
def test_real_forward_keeps_actual_b_buffers_with_original_n_split(monkeypatch,compact_shape,expected_splits):
    from oscar_ascend.integration.current_attention import use_native_current
    native=definitions("oscar_ascend/integration/impl.py","compact_impl_shape_contract",set(),
        {"torch":torch,"use_native_current":use_native_current,"OscarReadinessError":OscarReadinessError},
        monkeypatch,methods=[("OscarAttentionImpl","forward")])
    geometry=WorkspaceGeometry(16384,6,1,256,cube_cores=20)
    received=[]
    def partial_views(tokens,splits):
        received.append((tokens,splits))
        raise RuntimeError("stop after buffer geometry")
    workspace=SimpleNamespace(validate=lambda *args:None,geometry=geometry,
        tasks=torch.empty(49152,16,dtype=torch.int64),attention_status=torch.empty(49152,2,dtype=torch.int32),
        query_rot=torch.empty(1,6,256),partial_views=partial_views)
    raw=torch.empty(1,dtype=torch.uint8)
    state=SimpleNamespace(packed=raw,cache_format="canonical_v1",workspace=workspace)
    impl=SimpleNamespace(num_heads=6,num_kv_heads=1,head_size=256,
        provider=SimpleNamespace(config={},layer_state=lambda _:state))
    metadata=OscarMetadata(torch.tensor([0,1]),torch.tensor([501]),torch.zeros(1,5,dtype=torch.int32),
        torch.tensor([500]),1,1,1,501,num_input_tokens=1,is_draft=True,draft_index=1,cv_shape_tokens=compact_shape)
    q=torch.empty(1,6,256,dtype=torch.bfloat16);kv=torch.empty(1,1,256,dtype=torch.bfloat16)
    with pytest.raises(RuntimeError,match="stop after buffer geometry"):
        native.forward(impl,SimpleNamespace(layer_name="mtp"),q,kv,kv,raw,metadata,output=torch.empty_like(q))
    assert received==[(1,expected_splits)]


def test_plugin_wraps_compile_call_before_forward_and_uninstalls(monkeypatch):
    from types import ModuleType
    from oscar_ascend import plugin
    model,ctx,values,original,seen=_case(monkeypatch)
    module=ModuleType("vllm.model_executor.models.qwen3_5_mtp")
    module.Qwen3_5MTP=type(model)
    config={"experimental_compact_later_mtp":False}
    monkeypatch.delenv("OSCAR_PASSIVE_TIMING_CONTROL",raising=False)
    monkeypatch.setattr(plugin,"require_runtime",lambda:SimpleNamespace(config=config))
    monkeypatch.setattr(compact,"compact_later_mtp_call",lambda orig,m,args,kwargs,provider:
        compact._compact_call(orig,m,args,kwargs,provider,ctx,FLAGS,(1,1,4)))
    plugin.unregister()
    try:
        plugin._patch_mtp_forward(module)
        # Default off delegates unchanged; explicit true reaches skip_compiled
        # before the native decorated call attempts to load compiled code.
        ctx.skip_compiled=True
        assert model(**values).shape==(1024,4)
        ctx.skip_compiled=False
        config["experimental_compact_later_mtp"]=True
        assert model(**values).shape==(2,4)
        assert [item[0] for item in seen]==[1024,2]
    finally:
        plugin.unregister()
    assert type(model).__dict__["__call__"] is original


@pytest.mark.parametrize("n,b,expected",[(128,32,False),(1023,1,False),(1024,129,False),(1024,128,True),(16384,32,True)])
def test_conservative_large_padding_policy(n,b,expected):
    assert compact.should_compact_rows(n,b) is expected


def test_small_decode_uses_original_call_without_guards_or_context_change(monkeypatch):
    model,ctx,values,original,seen=_case(monkeypatch)
    n,b=128,32
    ctx.num_tokens=n;ctx.num_accept_tokens=b
    ctx.attn_metadata={"mtp":replace(ctx.attn_metadata["mtp"],num_actual_tokens=b,num_reqs=b,num_input_tokens=n)}
    values={key:(value[..., :n] if key=="positions" else value[:n]) if value is not None else None
            for key,value in values.items()}
    marker=object();before=dict(vars(ctx))
    monkeypatch.setattr(compact,"_validate_model",lambda *a:pytest.fail("small decode must not audit compact model"))
    monkeypatch.setattr(compact,"_tail_slot_guard",lambda *a:pytest.fail("small decode must not run compact slot guard"))
    assert compact._compact_call(lambda *a,**k:marker,model,(),values,None,ctx,{},()) is marker
    assert vars(ctx)==before and not seen
