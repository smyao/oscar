"""Independent TEST-ONLY BF16 attention contract; never a serving fallback.

Archive #13-20/#126/#145/#148: keep original quantizer/metadata, all heads,
padding and nonfinite checks; numerical completion is not NPU/model proof.
D.4: fused FIA only; no online full-history restoration (old6499.8-6655.1ms).
This offline oracle may reconstruct a bounded test fixture. The kernel must
keep fixed KV256 workspace, FP32 softmax/merge and source ownership. The
2026-10-16 user authorization permits BF16 operands, not lower accumulators
or postmeasurement tolerance changes. Closed FIA rounding is not inferred.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import torch
from .reference import AttentionResult, decode_kv, merge_attention

CONTRACT = "oscar_bf16_attention_v2"
ACCEPTANCE = Path(__file__).resolve().parents[2] / "configs/acceptance_bf16.json"

def load_acceptance(path: Path = ACCEPTANCE) -> dict:
    value=json.loads(path.read_text())
    if value.get("contract")!=CONTRACT or value.get("frozen_before_measurement") is not True:
        raise ValueError("explicit premeasurement BF16 contract required")
    if value["fused_attention"]!={"atol":.005,"rtol":.005,"per_element":True,
                                 "reference":"independent declared BF16 online-tile oracle"}:
        raise ValueError("BF16 oracle tolerance changed")
    return value

def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name+" contains nonfinite values")

def round_bf16(value: torch.Tensor) -> torch.Tensor:
    """RNE BF16 values represented exactly in FP32 for an explicit FP32 dot."""
    if not value.is_floating_point():raise ValueError("floating input required")
    _finite("operand before BF16 cast",value)
    result=value.to(torch.bfloat16).float()
    _finite("operand after BF16 cast",result)
    return result

def attention_tiles(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *,
                    visible: torch.Tensor | None = None, scale: float | None = None,
                    tile_size: int = 256, value_rotation: torch.Tensor | None = None) -> AttentionResult:
    """Declared BF16 QK/PV with FP32 online state, for one actual task segment.

    Q[Q,Hq,D], K/V[K,Hkv,D], visible[Q,K]. Key row0 is the actual source/split
    start, so tile boundaries must not be shifted to global sequence multiples.
    Optional Rv maps normalized history output back using BF16 operands and
    FP32 accumulation. Both the normalized partial and Rv round by RNE. This is not
    dense-softmax-then-cast: unnormalized P is rounded at EACH online tile.
    """
    if query.ndim!=3 or key.ndim!=3 or key.shape!=value.shape:
        raise ValueError("Q[Q,Hq,D] and K/V[K,Hkv,D] required")
    nq,hq,dim=query.shape;nk,hk,kdim=key.shape
    if dim!=kdim or dim<=0 or hk<=0 or hq<=0 or hq%hk or tile_size!=256:
        raise ValueError("invalid GQA/head dimension or noncontract tile size")
    if query.device!=key.device or query.device!=value.device:
        raise ValueError("one oracle device required")
    if visible is None:visible=torch.ones((nq,nk),dtype=torch.bool,device=query.device)
    if visible.dtype!=torch.bool or visible.shape!=(nq,nk) or visible.device!=query.device:
        raise ValueError("explicit [Q,K] visibility required")
    factor=dim**-.5 if scale is None else scale
    if not math.isfinite(factor) or factor<=0:raise ValueError("invalid scale")
    q,k,v=round_bf16(query),round_bf16(key),round_bf16(value)
    heads=torch.arange(hq,device=query.device)//(hq//hk)
    k=k.index_select(1,heads);v=v.index_select(1,heads)
    maximum=torch.full((nq,hq),-math.inf,dtype=torch.float32,device=q.device)
    denominator=torch.zeros_like(maximum)
    accumulator=torch.zeros((nq,hq,dim),dtype=torch.float32,device=q.device)
    for begin in range(0,nk,tile_size):
        end=min(begin+tile_size,nk)
        scores=torch.einsum("qhd,khd->qhk",q,k[begin:end])*factor
        _finite("QK scores before mask",scores)
        mask=visible[:,None,begin:end]
        scores=scores.masked_fill(~mask,-math.inf)
        new_max=torch.maximum(maximum,scores.amax(-1))
        safe_max=torch.where(torch.isfinite(new_max),new_max,torch.zeros_like(new_max))
        alpha=torch.where(torch.isfinite(maximum),torch.exp(maximum-safe_max),torch.zeros_like(maximum))
        probability=torch.exp(scores-safe_max.unsqueeze(-1))
        # The denominator consumes FP32 P. Only the Cube PV operand rounds.
        denominator=denominator*alpha+probability.sum(-1)
        pv=torch.einsum("qhk,khd->qhd",round_bf16(probability),v[begin:end])
        accumulator=accumulator*alpha.unsqueeze(-1)+pv
        maximum=new_max
    live=denominator>0
    safe_sum=torch.where(live,denominator,torch.ones_like(denominator))
    output=accumulator/safe_sum.unsqueeze(-1)
    lse=torch.where(live,maximum+torch.log(safe_sum),torch.full_like(maximum,-math.inf))
    if value_rotation is not None:
        if value_rotation.dtype not in (torch.float32,torch.bfloat16) or value_rotation.shape!=(dim,dim):
            raise ValueError("Rv must be original FP32 or initialized BF16 [D,D]")
        output=round_bf16(output)@round_bf16(value_rotation).T
    _finite("normalized output",output)
    return AttentionResult(output,lse)

def merge_parts(parts: list[AttentionResult] | tuple[AttentionResult,...]) -> AttentionResult:
    """Original independent FP32 LSE merge; no BF16 partial rounding."""
    return merge_attention(parts)

def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape!=expected.shape:raise ValueError("comparison shape mismatch")
    if bool((torch.isnan(expected)|torch.isposinf(expected)).any()):
        raise ValueError("reference must be finite or negative infinity")
    finite=torch.isfinite(expected)
    same_special=torch.equal(torch.isneginf(actual),torch.isneginf(expected))
    valid=bool(torch.isfinite(actual[finite]).all()) and same_special and not bool(torch.isnan(actual).any())
    delta=(actual.float()[finite]-expected.float()[finite]).abs()
    tolerance=.005+.005*expected.float()[finite].abs()
    return {"passed":bool(valid and (delta<=tolerance).all()),
            "max_abs":float(delta.max()) if delta.numel() else 0.,
            "bad_elements":int((delta>tolerance).sum()) if valid else int(actual.numel()),
            "atol":.005,"rtol":.005}

def compare_results(actual: AttentionResult, expected: AttentionResult, *,
                    old_fp32: AttentionResult | None = None) -> dict:
    load_acceptance()
    report={"contract":CONTRACT,"output":error_metrics(actual.output,expected.output),
            "lse":error_metrics(actual.lse,expected.lse)}
    report["passed"]=report["output"]["passed"] and report["lse"]["passed"]
    if old_fp32 is not None:
        report["old_fp32_diagnostic_only"]={"output":error_metrics(actual.output,old_fp32.output),
                                             "lse":error_metrics(actual.lse,old_fp32.lse),
                                             "used_as_BF16_gate":False}
    return report

def fixture_tasks_reference(fixture: dict, tasks: torch.Tensor) -> dict:
    """Offline oracle for ALL real prepare-generated tasks, all three sources.

    Reads original test raw slots and native physical table/window tags. It
    never consumes candidate output or imports production routing. Invalid
    input/status tests use exact status gates separately; this numerical
    oracle rejects them instead of declaring a different error precedence.
    """
    data=fixture["cpu"];spec=fixture["spec"];q=data["q"];qr=data["qr"]
    n,hq,dim=q.shape;hk=spec.kv_heads;segments=3*spec.splits;group=hq//hk
    if tasks.dtype!=torch.int64 or tasks.shape!=(n*hk*segments,16) or tasks.device.type!="cpu":
        raise ValueError("complete actual CPU prepare table required")
    if q.device.type!="cpu":raise ValueError("fixture oracle is offline CPU only")
    slot_bytes=dim//2+8;block_tokens=fixture["stride"]//(hk*slot_bytes)
    prefix=data["raw"].numel()-fixture["blocks"]*fixture["stride"]
    raw=data["raw"][prefix:].view(fixture["blocks"],block_tokens,hk,slot_bytes)
    sink=64;recent=spec.recent_tokens
    partial=torch.zeros((n,hq,segments,dim),dtype=torch.float32)
    lse=torch.full((n,hq,segments),-math.inf,dtype=torch.float32)
    written=torch.zeros((n,hq,segments),dtype=torch.bool)
    def physical(request,positions):
        columns=positions//128
        if bool((columns<0).any()) or bool((columns>=data["table"].shape[1]).any()):raise ValueError("table column invalid")
        virtual=data["table"][request,columns].long();ratio=block_tokens//128
        block=virtual//ratio;within=(virtual%ratio)*128+positions%128
        if bool((virtual<0).any()) or bool((block>=fixture["blocks"]).any()):raise ValueError("physical page invalid")
        return block,within
    for row in tasks.tolist():
        qb,count,head,begin,end,request,kind,segment,context,request_begin,error=row[:11]
        if error:raise ValueError("error task belongs to exact-status qualification")
        if count<=0:continue
        if kind not in (0,1,2) or not (0<=qb<qb+count<=n) or not (0<=head<hk) or not (0<=segment<segments):
            raise ValueError("task owner invalid")
        candidate=torch.arange(begin,end,dtype=torch.int64)
        query_positions=context+torch.arange(qb,qb+count)-request_begin
        cut=torch.maximum(torch.tensor(sink),query_positions+1-recent)
        head_slice=slice(head*group,(head+1)*group)
        if kind==0:
            block,within=physical(request,candidate)
            if candidate.numel():key,value=decode_kv(raw[block,within,head:head+1],dim)
            else:key=torch.empty((0,1,dim));value=torch.empty_like(key)
            visible=(candidate[None,:]>=sink)&(candidate[None,:]<context)&(candidate[None,:]<cut[:,None])&(candidate[None,:]<=query_positions[:,None])
            query=qr[qb:qb+count,head_slice];rv=data["rv"]
        elif kind==1:
            sink_count=min(sink,context)
            tail_begin=min(context,max(sink,context+qb-request_begin+1-recent))
            positions=torch.where(candidate<sink_count,candidate,tail_begin+candidate-sink_count)
            block,within=physical(request,positions)
            ring=recent+3
            window_row=torch.where(positions<sink,positions,sink+within%ring)
            if not torch.equal(data["tags"][block,window_row],within):raise ValueError("window tag invalid")
            key=data["wk"][block,window_row,head:head+1];value=data["wv"][block,window_row,head:head+1]
            visible=((positions[None,:]<sink)|(positions[None,:]>=cut[:,None]))&(positions[None,:]<=query_positions[:,None])
            query=q[qb:qb+count,head_slice];rv=None
        else:
            positions=candidate;token=request_begin+positions-context
            key=data["ck"][token,head:head+1];value=data["cv"][token,head:head+1]
            visible=positions[None,:]<=query_positions[:,None]
            query=q[qb:qb+count,head_slice];rv=None
        if begin==end:
            key=torch.empty((0,1,dim));value=torch.empty_like(key)
        answer=attention_tiles(query,key,value,visible=visible,scale=fixture["scale"],value_rotation=rv)
        if bool(written[qb:qb+count,head_slice,segment].any()):raise ValueError("duplicate output owner")
        partial[qb:qb+count,head_slice,segment]=answer.output
        lse[qb:qb+count,head_slice,segment]=answer.lse
        written[qb:qb+count,head_slice,segment]=True
    live_slots=data["slots"]>=0
    if not bool(written[live_slots].all()):
        raise ValueError("full oracle requires all live source/split output owners; current-source suppression needs a separately supplied native result")
    merged=merge_parts([AttentionResult(partial[:,:,s],lse[:,:,s]) for s in range(segments)])
    return {"partial":partial,"lse":lse,"output":merged.output,"output_lse":merged.lse,
            "contract":CONTRACT,"all_task_slots":int(tasks.shape[0]),"live_output_rows":int(written.sum())}
