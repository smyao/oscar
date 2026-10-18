# Archive #111/#126/#145/#148/#164: independent precision and lifecycle boundaries.
"""Independent test-only official-math codec; never import in serving.

The official sglang-rebase writer uses FP32 normalized-zero parameters.
Its unified decode and mixed-prefix prefill deliberately reconstruct in
different dtypes. The byte permutation here is this project's versioned
striped format; it is not claimed to equal SGLang's physical byte order.
"""
from __future__ import annotations
import math
from pathlib import Path
import json
import torch
from tools.bf16_reference import attention_tiles, round_bf16, merge_parts
from oscar_ascend.ops.reference import AttentionResult

CONTRACT="oscar_bf16_attention_v3"
CACHE_FORMAT="striped_fp32_v2"
ACCEPTANCE=Path(__file__).resolve().parents[1]/"configs/acceptance_aligned_bf16.json"

def load_acceptance():
    value=json.loads(ACCEPTANCE.read_text())
    if value["contract"]!=CONTRACT or not value["frozen_before_measurement"]:
        raise ValueError("independent v3 premeasurement contract required")
    if value["fused_attention"]!={"atol":.005,"rtol":.005,"per_element":True}:
        raise ValueError("v3 tolerance changed")
    return value

def _finite(name,x):
    if not bool(torch.isfinite(x).all()):raise ValueError(name+" must be finite")

def rotate_bf16(value:torch.Tensor,rotation:torch.Tensor):
    """BF16 operands with FP32 dot and BF16 output; offline declaration."""
    if value.shape[-1]!=rotation.shape[-2] or rotation.ndim!=2:
        raise ValueError("shared [D,D] rotation required")
    return round_bf16(round_bf16(value)@round_bf16(rotation)).to(torch.bfloat16)

def clip_rows(value:torch.Tensor,ratio:float):
    if value.dtype!=torch.bfloat16 or value.shape[-1] not in (64,128,256):
        raise ValueError("pre-rotated BF16 rows with D64/128/256 required")
    if not math.isfinite(ratio) or not 0<=ratio<=1:raise ValueError("invalid clip ratio")
    x=value.float();_finite("rotated input",x)
    if ratio>0:
        index=min(int(ratio*x.shape[-1]),x.shape[-1]-1)
        threshold=x.abs().sort(-1).values[...,index,None]
        x=torch.minimum(torch.maximum(x,-threshold),threshold)
    return x

def quantize_rows(value:torch.Tensor,ratio:float=0.):
    x=clip_rows(value,ratio)
    minimum=x.amin(-1,keepdim=True);maximum=x.amax(-1,keepdim=True)
    scale=(maximum-minimum).clamp_min(1e-8)/3.
    zero=-minimum/scale
    _finite("FP32 scale",scale);_finite("FP32 normalized zero",zero)
    # No metadata downcast before these FP32 operations; uint8 conversion
    # truncates nonnegative values. Clamp protects endpoint roundoff only.
    codes=(x/scale+zero+.5).clamp(0,3).to(torch.uint8)
    return codes,scale.squeeze(-1),zero.squeeze(-1)

def dequantize_rows(codes:torch.Tensor,scale:torch.Tensor,zero:torch.Tensor,mode:int):
    if codes.dtype!=torch.uint8 or bool((codes>3).any()):raise ValueError("INT2 codes required")
    if scale.dtype!=torch.float32 or zero.dtype!=torch.float32 or scale.shape!=codes.shape[:-1] or zero.shape!=scale.shape:
        raise ValueError("one FP32 scale/normalized-zero per row required")
    _finite("scale",scale);_finite("zero",zero)
    if bool((scale<=0).any()):raise ValueError("scale must be positive")
    if mode==0:
        return round_bf16((codes.float()-zero[...,None])*scale[...,None])
    if mode==1:
        s,z=round_bf16(scale),round_bf16(zero)
        return round_bf16(round_bf16(codes.float()-z[...,None])*s[...,None])
    raise ValueError("dequant_mode must be0(prefill) or1(decode)")

def _pack_codes(codes):
    dim=codes.shape[-1];width=dim//8
    planes=codes.long().reshape(*codes.shape[:-1],8,width)
    shift=torch.arange(8,device=codes.device)*2
    words=(planes<<shift[:,None]).sum(-2)
    return torch.stack((words&255,words>>8),-1).to(torch.uint8).flatten(-2)

def pack_kv(key:torch.Tensor,value:torch.Tensor,k_clip:float=.96,v_clip:float=.92):
    if key.shape!=value.shape:raise ValueError("matching K/V rows required")
    k,ks,kz=quantize_rows(key,k_clip);v,vs,vz=quantize_rows(value,v_clip)
    metadata=torch.stack((ks,kz,vs,vz),-1).contiguous().view(torch.uint8)
    return torch.cat((_pack_codes(k),_pack_codes(v),metadata),-1)

def unpack_kv(packed:torch.Tensor,dim:int,mode:int):
    if packed.dtype!=torch.uint8 or dim not in (64,128,256) or packed.shape[-1]!=dim//2+16:
        raise ValueError("striped_fp32_v2 slot shape/dtype required")
    metadata=packed[...,dim//2:].contiguous().view(torch.float32)
    decoded=[]
    for side in (0,1):
        raw=packed[...,side*dim//4:(side+1)*dim//4].long()
        words=raw[...,0::2]|(raw[...,1::2]<<8)
        levels=torch.stack([(words>>(2*plane))&3 for plane in range(8)],-2).flatten(-2).to(torch.uint8)
        decoded.append(dequantize_rows(levels,metadata[...,side*2],metadata[...,side*2+1],mode))
    return tuple(decoded)


def fixture_tasks_reference(fixture: dict, tasks: torch.Tensor, dequant_mode: int) -> dict:
    """Offline oracle for ALL real prepare-generated tasks, all three sources.

    Reads striped_fp32_v2 test raw slots and native physical table/window tags. It
    never consumes candidate output or imports production routing. Invalid
    input/status tests use exact status gates separately; this numerical
    oracle rejects them instead of declaring a different error precedence.
    """
    data=fixture["cpu"];spec=fixture["spec"];q=data["q"];qr=data["qr"]
    if qr.dtype!=torch.bfloat16:raise ValueError("aligned QR storage must be BF16")
    n,hq,dim=q.shape;hk=spec.kv_heads;segments=3*spec.splits;group=hq//hk
    if tasks.dtype!=torch.int64 or tasks.shape!=(n*hk*segments,16) or tasks.device.type!="cpu":
        raise ValueError("complete actual CPU prepare table required")
    if q.device.type!="cpu":raise ValueError("fixture oracle is offline CPU only")
    slot_bytes=dim//2+16;block_tokens=fixture["stride"]//(hk*slot_bytes)
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
        # attention_tasks.cpp ABI: field6 is kind*S+split (output segment),
        # field7 is kind. They coincide only when S==1.
        qb,count,head,begin,end,request,segment,kind,context,request_begin,error=row[:11]
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
            if candidate.numel():key,value=unpack_kv(raw[block,within,head:head+1],dim,dequant_mode)
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
