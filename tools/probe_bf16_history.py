"""Unified BF16 attention NPU gate (archive #126/#145/#148/#154/#157).

This diagnostic calls all-source BF16 symbols directly. It never relaxes the old MDL/C4
bitwise gates or the frozen oracle, and never enables a serving configuration.
D.4 forbids serving-side full-history restoration: only this CPU oracle may
materialize history. Device workspace remains bounded per Cube core. Device,
graph, CV timing, full attention timing and model quality are separate facts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

from . import probe_history_reuse as reuse
from . import probe_mdl_decode as mdl
from . import probe_striped_cache as shared
from .phase import atomic_json
from .striped_fixture import raw_to_striped

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'oscar.bf16_history.npu.v2'
from .experiments.bf16_direct_decode import CONTRACT
MIN_SPEEDUP = 1.15
CPU_SCREEN_REPORT = ROOT / 'reports/bf16_all_sources_screen_20261016_v3.json'
ROUTES = {
    'mdl': ('attention_cv_mdl_decode_out', 'attention_bf16_history_mdl_out', 1, 1146880),
    'mdl_q1': ('attention_cv_mdl_decode_q1_out', 'attention_bf16_history_mdl_q1_out', 1, 1146880),
    'c4': ('attention_cv_member_pipeline_cluster4_out', 'attention_bf16_history_member_pipeline_cluster4_out', 4, 2494464),
    'c16': ('attention_cv_window_range_cluster16_out', 'attention_bf16_history_window_range_cluster16_out', 16, 4997120),
    'general': ('attention_cv_window_range_out', 'attention_bf16_history_window_range_out', 1, 917504),
    'balanced': ('attention_cv_window_range_balanced_out', 'attention_bf16_history_window_range_balanced_out', 1, 917504),
}
SCOPES = ('complete_cv_all_three_sources_device_events',
          'complete_cv_all_three_sources_merge_bf16_copy_device_events')
SOURCE_PATHS = (
    'tools/probe_bf16_history.py', 'tools/bf16_history_pipeline.py',
    'tools/probe_history_reuse.py', 'tools/probe_mdl_decode.py',
    'oscar_ascend/integration/impl.py', 'oscar_ascend/integration/current_attention.py',
    'oscar_ascend/integration/guard_diag.py', 'oscar_ascend/integration/cache_storage_ops.py',
    'oscar_ascend/runtime.py', 'oscar_ascend/ops/cv_dispatch.py',
    'oscar_ascend/integration/mixed_decode_split.py', 'tools/serving_variants.py',
    'tools/probe_native_current_fia.py', 'tools/striped_fixture.py',
    'oscar_ascend/bf16_history_policy.py',
    'csrc/kernels/rotate_vector_init.cpp', 'csrc/kernels/rotate_clip_store_striped.cpp',
    'csrc/rotation_bindings.cpp', 'csrc/include/oscar_rotation_vector_init_launch.h',
    'csrc/kernels/status_guard.cpp',
    'tools/probe_striped_cache.py', 'tools/experiments/bf16_history_screen.py',
    'tools/experiments/bf16_all_sources_screen.py', 'tools/experiments/bf16_direct_decode.py',
    'csrc/kernels/attention_striped_unpack_bf16.h', 'tools/experiments/generate_bf16_history_kernels.py',
    'oscar_ascend/ops/reference.py', 'configs/acceptance.json',
    'csrc/kernels/attention_bf16_history_mdl.cpp',
    'csrc/kernels/attention_bf16_history_member_pipeline_cluster4.cpp',
    'csrc/kernels/attention_bf16_history_window_range_cluster16.cpp',
    'csrc/kernels/attention_bf16_history_window_range.cpp',
    'csrc/kernels/attention_bf16_history_window_range_balanced.cpp',
    'csrc/include/oscar_bf16_history_launch.h', 'csrc/bf16_history_bindings.cpp',
    'csrc/kernels/attention_bf16_history_operands.h',
    'csrc/kernels/attention_bf16_precise_check.h',
    'csrc/kernels/attention_striped_unpack_padded.h', 'csrc/kernels/attention_range_mask.h',
    'csrc/include/oscar_attention_launch.h', 'csrc/include/oscar_mdl_decode_launch.h',
    'csrc/include/oscar_member_pipeline_launch.h',
    'csrc/kernels/attention_cv_mdl_decode.cpp',
    'csrc/kernels/attention_cv_member_pipeline_cluster4.cpp',
    'csrc/kernels/attention_cv_window_range_cluster16.cpp',
    'csrc/kernels/attention_cv_window_range.cpp',
    'csrc/kernels/attention_cv_window_range_balanced.cpp',
    'csrc/kernels/attention_tasks.cpp', 'csrc/kernels/merge_lse.cpp',
)


@dataclass(frozen=True)
class Case:
    shape: reuse.Shape
    route: str
    rotation: str = 'hadamard'
    amplitude: float = 1.0
    cancellation: bool = False
    graph: bool = False
    timed: bool = False


def case_plan():
    """Fixed before measurement; preserve cancellation and all six real routes."""
    short = reuse.Shape('mdl_q4', (4, 4), (1341, 1373), 256, 1, 3, False, padded_tokens=12)
    q1 = replace(short, name='mdl_q1', qlens=(1, 1), padded_tokens=8, slot_context=True)
    c4 = reuse.Shape('c4_mature', (336,), (1341,), 256, 1, 1, True, padded_tokens=352)
    c16 = reuse.Shape('c16_mature', (672,), (1341,), 256, 1, 1, True, padded_tokens=688)
    general = reuse.Shape('general_multiblock', (17,), (641,), 256, 2, 3, False, padded_tokens=24)
    balanced = reuse.Shape('balanced_mixed', (4, 17), (641, 769), 256, 1, 3, False, padded_tokens=64)
    rows = [Case(short, 'mdl', graph=True, timed=True),
            Case(q1, 'mdl_q1', graph=True, timed=True),
            Case(c4, 'c4', graph=True, timed=True),
            Case(c16, 'c16', graph=True, timed=True),
            Case(general, 'general', graph=True, timed=True),
            Case(balanced, 'balanced', graph=True, timed=True),
            Case(reuse.Shape('identity_exact_cancellation', (1,), (1023,), 256, 1, 1, False),
                 'mdl', rotation='identity', cancellation=True),
            Case(reuse.Shape('hadamard_large_values', (4,), (1025,), 256, 1, 3, False),
                 'mdl', amplitude=128.0),
            Case(replace(short, name='ordinary_qr'), 'mdl', rotation='qr'),
            Case(reuse.Shape('empty_history', (4,), (0,), 256, 1, 3, False, padded_tokens=8), 'mdl'),
            Case(reuse.Shape('frontier_q2', (2,), (320,), 256, 2, 3, False), 'mdl'),
            Case(reuse.Shape('kv_tail_q3', (3,), (575,), 256, 1, 3, False), 'mdl')]
    # Same old task/lane errors for each independently implemented family.
    for case in tuple(rows[:6]):
        rows += [replace(case, shape=replace(case.shape, name=case.shape.name+'_bad_metadata',
                         corrupt_metadata=True, corrupt_position=80), graph=False, timed=False),
                 replace(case, shape=replace(case.shape, name=case.shape.name+'_bad_qr',
                         corrupt_qr=True, corrupt_qr_token=0), graph=False, timed=False),
                 replace(case, shape=replace(case.shape, name=case.shape.name+'_dead_tail',
                         corrupt_dead_tail=True), graph=False, timed=False)]
    return tuple(rows)


def source_binding(root=ROOT):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def _invalid(case):
    return case.shape.corrupt_metadata or case.shape.corrupt_qr


def make_fixture(torch, case, *, dense_oracle=True):
    from oscar_ascend.ops.reference import attention, decode_kv, encode_kv
    logical = {}
    def retain(request, begin, q, k, v, ck, cv):
        if case.amplitude != 1.0:
            v.copy_((v.float()*case.amplitude).bfloat16())
            cv.copy_((cv.float()*case.amplitude).bfloat16())
        if case.cancellation:
            q.zero_();k[:] = torch.tensor([0, 1, 2, 3]).bfloat16().repeat(64)
            ck.copy_(k[:len(ck)])
            vector = torch.tensor([2048, 2112, 2176, 2304]).bfloat16().repeat(64)
            v[::2] = vector;v[1::2] = -vector
            cv[:] = -vector  # context 1023: current token has odd absolute position.
        logical[request] = (begin, q.clone(), k.clone(), v.clone(), ck.clone(), cv.clone())
    fixture = reuse.make_fixture(torch, case.shape, on_request=retain,
                                  rotation_mode='identity' if case.rotation == 'identity' else 'hadamard')
    rk = torch.eye(256) if case.rotation == 'identity' else reuse._hadamard(torch, 256)
    rv = rk.clone()
    if case.rotation == 'qr':
        rng = torch.Generator().manual_seed(1943)
        rk = torch.linalg.qr(torch.randn((256,256), generator=rng)).Q.contiguous()
        rv = torch.linalg.qr(torch.randn((256,256), generator=rng)).Q.contiguous()
        fixture['cpu']['rv'] = rv
    fixture['cpu']['qr'] = (fixture['cpu']['q'].float() @ rk).contiguous()
    fixture['cpu']['rk_transpose'] = rk.T.contiguous()
    fixture['query_hadamard'] = case.rotation == 'hadamard'
    if case.shape.corrupt_qr:
        fixture['cpu']['qr'][case.shape.corrupt_qr_token,0,0] = float('nan')
    canonical = fixture['cpu']['raw'][reuse.PREFIX:].view(fixture['blocks'], reuse.BLOCK_TOKENS,
                                                       case.shape.kv_heads, 136)
    original = {}
    for request, values in logical.items():
        begin,q,k,v,ck,cv = values
        if len(k):
            packed = encode_kv(k.float() @ rk, v.float() @ rv)
            dk,dv = decode_kv(packed, 256)
            dk,dv = dk @ rk.T, dv @ rv.T
            # QR requires repacking every actual physical page with the new rotation.
            if case.rotation == 'qr':
                for page,physical in enumerate(fixture['page_assignments'][request]):
                    first=page*reuse.BLOCK_TOKENS;count=min(reuse.BLOCK_TOKENS,len(k)-first)
                    if count>0:canonical[physical,:count] = packed[first:first+count]
        else:dk,dv=k.float(),v.float()
        original[request] = (begin,q,k.float(),v.float(),ck.float(),cv.float(),dk,dv)
    # Valid precision cases compare every query, including large C4/C16 groups.
    fixture['expected'] = {}
    if not _invalid(case) and dense_oracle:
        for request,(begin,q,k,v,ck,cv,dk,dv) in original.items():
            context=len(k)
            for local in range(len(q)):
                cut=min(context,max(reuse.SINK,context+local+1-case.shape.recent_tokens))
                kk,vv=k.clone(),v.clone();kk[reuse.SINK:cut]=dk[reuse.SINK:cut];vv[reuse.SINK:cut]=dv[reuse.SINK:cut]
                value=attention(q[local:local+1],torch.cat((kk,ck[:local+1])),
                                torch.cat((vv,cv[:local+1])),scale=fixture['scale'],causal=False)
                fixture['expected'][begin+local]=(value.output[0],value.lse[0])
    fixture['logical'] = original
    fixture['cpu']['raw'] = raw_to_striped(fixture)
    fixture['input_sha256'] = reuse._fingerprint(fixture['cpu'])
    return fixture


def partial_oracle(torch, fixture, tasks):
    """Independent raw-space attention per exact descriptor plus dense full oracle.

    Tasks select ranges; the separately computed full oracle does not trust their
    partition. Every output slot must have exactly one task owner, including
    padding and empty splits. No candidate value enters either oracle.
    """
    from oscar_ascend.ops.reference import attention
    spec=fixture['spec'];n=fixture['tokens'];h=spec.heads;s=3*spec.splits
    output=torch.zeros((n,h,s,256));lse=torch.full((n,h,s),-torch.inf)
    covered=torch.zeros((n,h,s),dtype=torch.int32)
    for row in tasks.tolist():
        token,count,hk,first,last,request,part,kind,context,begin,error=row[:11]
        if error:raise ValueError('invalid task entered numerical oracle')
        if count==0:continue
        heads=slice(hk*6,(hk+1)*6)
        if count<0:
            covered[token,heads,part]+=1
            continue
        _,q,k,v,ck,cv,dk,dv=fixture['logical'][request]
        for index in range(token,token+count):
            local=index-begin;position=context+local
            cut=max(reuse.SINK,position+1-spec.recent_tokens)
            if kind==0:
                ids=torch.arange(first,last);ids=ids[(ids>=reuse.SINK)&(ids<cut)&(ids<=position)]
                kk,vv=dk[ids,hk:hk+1],dv[ids,hk:hk+1]
            elif kind==1:
                tail=min(context,max(reuse.SINK,context+token-begin+1-spec.recent_tokens))
                mapping=torch.cat((torch.arange(min(reuse.SINK,context)),torch.arange(tail,context)))
                ids=mapping[first:last];ids=ids[((ids<reuse.SINK)|(ids>=cut))&(ids<=position)]
                kk,vv=k[ids,hk:hk+1],v[ids,hk:hk+1]
            else:
                ids=torch.arange(first,last);ids=ids[ids<=position]-context
                kk,vv=ck[ids,hk:hk+1],cv[ids,hk:hk+1]
            covered[index,heads,part]+=1
            if len(kk):
                result=attention(q[local:local+1,heads],kk,vv,scale=fixture['scale'],causal=False)
                output[index,heads,part]=result.output[0];lse[index,heads,part]=result.lse[0]
    if not bool((covered==1).all()):raise RuntimeError('task oracle has missing or duplicate partial owners')
    return {'partial':output,'lse':lse}


def candidate_workspace_per_core(route):
    """Typed arena matching Bf16AttentionWorkspace; all audited routes are D256."""
    if route not in ROUTES:raise ValueError('unknown BF16 attention route')
    m=32 if route in ('mdl','mdl_q1') else 128
    c=ROUTES[route][2];buffers=2 if route in ('mdl','mdl_q1','c4') else 1
    slots=2 if route=='c4' else 1;n=d=256
    floats=2*m*d+slots*m*n+(c*m*d+2*c*m if c>1 else 0)
    operands=c*m*d+buffers*2*n*d+slots*m*n
    return floats*4+operands*2


def allocate(torch,fixture,device,cores,route,*,candidate):
    _,_,cluster,old_bytes=ROUTES[route]
    buffers=reuse._allocate(torch,fixture,device,cores,candidate=cluster>1)
    buffers['workspace']=torch.empty(cores*(candidate_workspace_per_core(route) if candidate else old_bytes),dtype=torch.uint8,device=device)
    n,h,d=fixture['tokens'],fixture['spec'].heads,256
    buffers.update(output=torch.empty((n*h,d),dtype=torch.float32,device=device),
                   output_lse=torch.empty(n*h,dtype=torch.float32,device=device),
                   merge_status=torch.empty(n*h,dtype=torch.int32,device=device),
                   bf16=torch.empty((n*h,d),dtype=torch.bfloat16,device=device))
    if candidate:
        buffers['qr_bf16']=torch.empty((n,h,d+16),dtype=torch.bfloat16,device=device)
        buffers['rotate_status']=torch.empty((n,h),dtype=torch.int32,device=device)
    return buffers


def produce_bf16_query(ops,tensors,fixture,buffers):
    """Real producer ABI, outside CV-only timing but inside changed-input graphs.

    The independent FP32 CPU qr remains the old-reader input. The candidate
    consumes BF16[D+16], including the producer's original FP32 sum trailer.
    """
    ops.rotate_bf16_query_out(tensors['q'],tensors['rk_transpose'],buffers['qr_bf16'],
        buffers['rotate_status'],fixture['query_hadamard'],tensors['slots'])
    if fixture['spec'].corrupt_qr:
        token=fixture['spec'].corrupt_qr_token
        # Deliberate post-producer corruption, equivalent to the old negative
        # qr fixture: poison both operand and its original FP32 check value.
        buffers['qr_bf16'][token,0,0]=float('nan')
        buffers['qr_bf16'][token,0,256:258].view(dtype=buffers['output'].dtype).fill_(float('nan'))


def launch(ops,name,tensors,fixture,buffers,cores,cluster,*,merge=True):
    inputs=({**tensors,'qr':buffers['qr_bf16']}
            if name.startswith('attention_bf16_history_') else tensors)
    shared._launch(ops,name,inputs,fixture,buffers,cores,cluster)
    if merge:
        n,h,s=fixture['tokens'],fixture['spec'].heads,3*fixture['spec'].splits
        ops.merge_lse_out(buffers['partial'].view(n*h,s,256),buffers['lse'].view(n*h,s),
                          buffers['output'],buffers['output_lse'],buffers['merge_status'])
        buffers['bf16'].copy_(buffers['output'])


def snapshot(buffers,*,merge=True):
    result=mdl.snapshot(buffers,merge=merge)
    if buffers['cluster_stats'] is not None:result['cluster_stats']=buffers['cluster_stats'].clone()
    return result


def same_bits(torch,expected,buffers,label):
    for name,value in expected.items():
        if not shared._store_bits_equal(torch,value,buffers[name]):
            raise RuntimeError(label+' changed '+name+' bits')


def check_cluster_stats(stats, cluster, *, expected_leaders, require_cluster):
    if tuple(stats.shape[1:]) != (8,) or stats.ndim != 2:
        raise RuntimeError('BF16 clusterStats geometry differs')
    for row in stats.tolist():
        if any(value < 0 for value in row) or row[1] != cluster*row[0] or row[4] != (cluster-1)*row[3] or row[7] != row[0]:
            raise RuntimeError('BF16 clusterStats per-core ownership differs')
    totals=stats.sum(0).tolist()
    if totals[6] != totals[1] or totals[1]+totals[2] != expected_leaders:
        raise RuntimeError('BF16 lost or duplicated source 0 leader')
    if require_cluster and (totals[0] <= 0 or totals[3] <= 0):
        raise RuntimeError('BF16 mature case did not execute shared history')
    if expected_leaders == 0 and any(totals):
        raise RuntimeError('BF16 padding executed cluster work')
    return dict(zip(reuse.STATS_FIELDS,totals))


def prepare(torch,ops,tensors,fixture,buffers):
    if fixture['actual_tokens']:
        return reuse._prepare(torch,ops,tensors,fixture,buffers)
    spec=fixture['spec']
    args=(tensors['starts'],tensors['lens'],tensors['slots'],buffers['tasks'],buffers['positions'],
          spec.heads,spec.kv_heads,reuse.SINK,spec.recent_tokens,spec.splits)
    if spec.slot_context:ops.prepare_attention_tasks_out(*args,tensors['table'],True)
    else:ops.prepare_attention_tasks_out(*args)
    torch.npu.synchronize()
    tasks=buffers['tasks'].cpu()
    if not bool((buffers['positions']==-1).all()) or not bool((tasks[:,1]==-1).all()) or bool(tasks[:,10].any()):
        raise RuntimeError('all-padding graph preparation is not harmless')
    return {'task_sha256':reuse._fingerprint({'tasks':tasks}),'source0_leaders':0}


def mdl_history_stage_diagnostics(torch,case,fixture,buffers,rotation):
    """#171: inspect retained Norm, without launching/changing any device kernel.

    MDL Norm is a separate FP32 plane written only by nonempty history tasks.
    Later precise/padding tasks leave it intact. Match each core's LAST writer,
    not every task ever assigned to that core. Mirrors CvTaskSchedule's
    segment-major ordering; only active rows are compared with final partial.
    """
    if case.route not in ('mdl','mdl_q1') or 'workspace' not in buffers:
        return {'status':'not_available'}
    stride=candidate_workspace_per_core(case.route)
    workspace=buffers['workspace'].detach().cpu()
    if workspace.dtype!=torch.uint8 or workspace.numel()%stride:
        raise ValueError('unexpected MDL workspace geometry')
    cores=workspace.numel()//stride
    if cores<=0:raise ValueError('empty MDL workspace')
    spec=fixture['spec']
    n=fixture['tokens'];group=spec.heads//spec.kv_heads
    owner=1 if case.route=='mdl_q1' else 4
    tiles=(n+owner-1)//owner;per_token=spec.kv_heads*3*spec.splits
    tasks=buffers['tasks'].detach().cpu().reshape(n*per_token,16).tolist()
    last={};last_operand_writer={}
    for work in range(tiles*per_token):
        begin=(work%tiles)*owner;segment=work//tiles
        for token in range(begin,min(begin+owner,n)):
            task_id=token*per_token+segment
            row=tasks[task_id]
            qbegin,count,hk,first,end,request,split,kind,context,request_begin,error=row[:11]
            # Empty/padding tasks publish output/status but never touch operands.
            # A later precise task, even with invalid data, makes these old
            # history operand planes unsuitable as evidence.
            if count>0 and first<end:
                last_operand_writer[work%cores]=task_id
            if count<=0 or kind!=0 or first>=end or error:continue
            if not (0<=qbegin<n and count<=owner and qbegin+count<=n
                    and 0<=hk<spec.kv_heads and 0<=split<3*spec.splits
                    and 0<=first<end<=context and 0<=request<len(spec.qlens)
                    and 0<=request_begin<=qbegin and count*group<=32):
                raise ValueError('invalid history descriptor in stage diagnostic')
            last[work%cores]=(task_id,row)
    arena=workspace.reshape(cores,stride)
    partial=buffers['partial'].detach().cpu()
    rv=None if rotation is None else rotation.detach().cpu().float()
    if rv is not None and tuple(rv.shape)!=(256,256):
        raise ValueError('unexpected shared rotation geometry')
    result={'scope':'last_nonempty_history_per_core_only','cores':cores,
            'rotation_finite':None if rv is None else bool(torch.isfinite(rv).all()),
            'tasks':[]}
    for core,(task_id,row) in sorted(last.items()):
        qbegin,count,hk,_,_,_,split=row[:7]
        norm=arena[core,:32*256*4].contiguous().view(torch.float32).reshape(32,256)[:count*group]
        output=partial[qbegin:qbegin+count,hk*group:(hk+1)*group,split].reshape(count*group,256)
        finite=torch.isfinite(norm)
        detail={'core':core,'task_id':task_id,'qbegin':qbegin,'qcount':count,'split':split,
                'norm_nonfinite_elements':int((~finite).sum()),
                'norm_bad_columns':(~finite).any(dim=0).nonzero().flatten().tolist(),
                'norm_max_abs':float(norm.abs().max()) if bool(finite.all()) else None,
                'partial_nonfinite_elements':int((~torch.isfinite(output)).sum())}
        if last_operand_writer.get(core)==task_id:
            # Bf16AttentionWorkspace{256,32,1,2,1}: three FP32 MxD planes,
            # then BF16 Q, K0/V0, K1/V1 and P. Only inspect published tiles;
            # the unused double-buffer slot may intentionally remain poisoned.
            operand=arena[core,3*32*256*4:].contiguous().view(torch.bfloat16)
            tile_count=(row[4]-row[3]+255)//256
            values=[]
            for slot in range(min(tile_count,2)):
                last_tile=tile_count-1-((tile_count-1-slot)%2)
                offset=32*256+256*256+slot*2*256*256
                v=operand[offset:offset+256*256].reshape(256,256).float()
                bad=~torch.isfinite(v)
                values.append({'slot':slot,'tile_index':last_tile,
                               'nonfinite_elements':int(bad.sum()),
                               'bad_columns':bad.any(dim=0).nonzero().flatten().tolist(),
                               'first_bad_coordinates':bad.nonzero()[:8].tolist()})
            p=operand[32*256+4*256*256:][:count*group*256].float()
            detail['retained_operands']={'status':'available',
                                         'value_tiles':values,
                                         'probability_nonfinite_elements':int((~torch.isfinite(p)).sum())}
        else:
            detail['retained_operands']={'status':'overwritten_by_later_task'}
        if rv is not None:
            # CPU FP32 diagnostic reference, not device arithmetic equivalence
            # or a replacement for any frozen numerical gate.
            rotated=norm@rv.T
            detail['cpu_rotation_nonfinite_elements']=int((~torch.isfinite(rotated)).sum())
        result['tasks'].append(detail)
    return result


def failure_diagnostics(torch,case,fixture,old,new,device_rotation=None):
    """#170: bounded host evidence from this offline probe only, never serving.

    Inspect both arms even when the first one fails. -Inf LSE is legal for
    empty sources; nonfinite partials remain failures under the existing gate.
    Coordinates identify token/head/source-split, not a claimed device cause.
    """
    result={'case':case.shape.name,'route':case.route,'arms':{}}
    sources=('history','sink_recent','current')
    splits=fixture['spec'].splits
    for arm,buffers in (('baseline',old),('candidate',new)):
        detail={'operator':ROUTES[case.route][0 if arm=='baseline' else 1]}
        result['arms'][arm]=detail
        try:
            partial=buffers['partial'].detach().cpu()
            lse=buffers['lse'].detach().cpu()
            bad_partial=~torch.isfinite(partial)
            bad_lse=~torch.isfinite(lse)&~torch.isneginf(lse)
            detail['partial_shape']=list(partial.shape)
            detail['lse_shape']=list(lse.shape)
            detail['sources']={}
            for index,name in enumerate(sources):
                sl=slice(index*splits,(index+1)*splits)
                detail['sources'][name]={
                    'nonfinite_partial_elements':int(bad_partial[:,:,sl].sum()),
                    'invalid_lse_elements':int(bad_lse[:,:,sl].sum()),
                    'empty_lse_elements':int(torch.isneginf(lse[:,:,sl]).sum())}
            for name,mask in (('partial',bad_partial),('lse',bad_lse)):
                detail[name+'_first_bad_coordinates']=mask.nonzero()[:16].tolist()
            for name in ('status','rotate_status','merge_status'):
                if name not in buffers:continue
                status=buffers[name].detach().cpu()
                bad=status!=0
                detail[name]={'nonzero_elements':int(bad.sum()),
                              'first_bad_coordinates':bad.nonzero()[:16].tolist(),
                              'first_bad_values':status[bad][:16].tolist()}
        except Exception as error:
            # Preserve the original failure when an unhealthy device cannot
            # supply further diagnostics. Never replace it with a read error.
            detail['diagnostic_error']=f'{type(error).__name__}: {error}'
    try:
        result['history_stage']=mdl_history_stage_diagnostics(
            torch,case,fixture,new,device_rotation)
    except Exception as error:
        result['history_stage']={'diagnostic_error':f'{type(error).__name__}: {error}'}
    return result


def compare_pair(torch,case,fixture,old,new,task_hash,acceptance,oracle=None,*,device_rotation=None):
    try:
        return _compare_pair(torch,case,fixture,old,new,task_hash,acceptance,oracle)
    except Exception as error:
        detail=failure_diagnostics(torch,case,fixture,old,new,device_rotation)
        error.bf16_failure_diagnostics=detail
        print('[oscar] BF16_HISTORY_FAILURE '+json.dumps(detail,allow_nan=False),flush=True)
        raise


def _compare_pair(torch,case,fixture,old,new,task_hash,acceptance,oracle=None):
    from .experiments.bf16_history_screen import compare, relative_l2
    invalid=_invalid(case)
    if 'rotate_status' in new and not bool((new['rotate_status']==0).all()):
        raise RuntimeError('BF16 query producer reported a nonzero status')
    for buffers in (old,new):
        reuse._check_status(torch,buffers,invalid=invalid)
        if reuse._fingerprint({'tasks':buffers['tasks'].cpu()})!=task_hash:
            raise RuntimeError('BF16 attention modified prepared tasks')
    if not torch.equal(old['status'],new['status']):raise RuntimeError('BF16 changed exact error code/lane ownership')
    if old['cluster_stats'] is not None:
        if not torch.equal(old['cluster_stats'],new['cluster_stats']):raise RuntimeError('BF16 changed clusterStats')
        check_cluster_stats(new['cluster_stats'].cpu(), ROUTES[case.route][2],
                            expected_leaders=int((old['tasks'][:,7].eq(0)&old['tasks'][:,1].gt(0)).sum()),
                            require_cluster=case.shape.expect_clusters and fixture['actual_tokens']>0)
    # Every source now has BF16 Cube operands. Exact input BF16 storage is
    # unchanged, but rounded P changes source1/2 PV, so old partial bitwise
    # parity is no longer the numerical contract. Keep empty protocol exact.
    empty=torch.isneginf(old['lse'])
    if not torch.equal(empty,torch.isneginf(new['lse'])):
        raise RuntimeError('BF16 changed empty partial ownership')
    for buffers in (old,new):
        if bool((buffers['partial'][empty]!=0).any()):raise RuntimeError('BF16 changed empty partial zero')
    if not shared._store_bits_equal(torch,old['partial'][empty],new['partial'][empty]):
        raise RuntimeError('BF16 changed empty partial bits')
    if invalid:
        # Source 0 finite values may legitimately change; status and nonfinite
        # classification must match, never require new arithmetic to equal old.
        for name in ('partial','lse'):
            a,b=old[name],new[name]
            for predicate in (torch.isnan,torch.isposinf,torch.isneginf):
                if not torch.equal(predicate(a),predicate(b)):raise RuntimeError('BF16 changed invalid '+name+' classification')
        return {'passed':True,'oracle':'expected_error_domain','queries':0,'status_lane_bits':True,
                'empty_partial_bits':True,'operand_contract':CONTRACT,'comparisons':{}}
    expected_ids=list(range(fixture['actual_tokens']))
    if sorted(fixture['expected'])!=expected_ids:raise RuntimeError('independent oracle omitted query rows')
    n,h=fixture['tokens'],fixture['spec'].heads
    gold=(torch.stack([fixture['expected'][i][0] for i in expected_ids]) if expected_ids
          else torch.empty((0,h,256),dtype=torch.float32))
    gold_lse=(torch.stack([fixture['expected'][i][1] for i in expected_ids]) if expected_ids
              else torch.empty((0,h),dtype=torch.float32))
    oracle=oracle or partial_oracle(torch,fixture,old['tasks'].cpu())
    comparisons={}
    for arm,buffers in (('baseline',old),('candidate',new)):
        if not bool((buffers['merge_status']==0).all()):raise RuntimeError('merge left invalid or unwritten status')
        actual={'partial':buffers['partial'].cpu(),'lse':buffers['lse'].cpu(),
                'output':buffers['output'].view(n,h,256)[expected_ids].cpu(),
                'output_lse':buffers['output_lse'].view(n,h)[expected_ids].cpu(),
                'bf16':buffers['bf16'].view(n,h,256)[expected_ids].float().cpu()}
        expected={**oracle,'output':gold,'output_lse':gold_lse,'bf16':gold}
        comparisons[arm]={name:compare(value,expected[name],acceptance['fused_attention'],
                                      allow_negative_infinity=name in ('lse','output_lse')) for name,value in actual.items()}
        comparisons[arm]['relative_l2']=relative_l2(actual['output'],gold,acceptance['prefill_window']['relative_l2_lt'])
    return {'passed':all(v['passed'] for arm in comparisons.values() for v in arm.values()),
            'oracle':'all_actual_queries_raw_space','queries':len(expected_ids),'status_lane_bits':True,
            'empty_partial_bits':True,'operand_contract':CONTRACT,'comparisons':comparisons}


def _timing(torch,execute,pair,names):
    samples={scope:{name:[] for name in names} for scope in SCOPES}
    references=[snapshot(b) for b in pair]
    for merge,scope in zip((False,True),SCOPES):
        for repeat in range(7):
            for i in ((0,1) if repeat<2 or repeat%2==0 else (1,0)):
                mdl.poison(torch,pair[i]);begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                begin.record();execute(i,merge=merge);end.record();end.synchronize()
                elapsed=float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid BF16 device event time')
                wanted=references[i] if merge else {k:v for k,v in references[i].items() if k in ('partial','lse','status','cluster_stats')}
                same_bits(torch,wanted,pair[i],'same-arm timed repetition')
                if repeat>=2:samples[scope][names[i]].append(elapsed)
    result={}
    for scope,values in samples.items():
        a,b=(statistics.median(values[name]) for name in names)
        result[scope]={'samples_ms':values,'baseline_ms':a,'candidate_ms':b,'speedup':a/b,
                      'minimum_speedup':MIN_SPEEDUP,'warmup':2,'repeats':5,
                      'gate':'passed' if a/b>=MIN_SPEEDUP else 'failed'}
    return result


def graph_gate(torch,ops,case,fixture,tensors,pair,cores,acceptance):
    names=ROUTES[case.route][:2];cluster=ROUTES[case.route][2]
    def addresses():
        return {**{k:v.data_ptr() for k,v in tensors.items()},
                **{str(i)+k:v.data_ptr() for i,b in enumerate(pair) for k,v in b.items() if v is not None}}
    before=addresses();graphs=[torch.npu.NPUGraph(),torch.npu.NPUGraph()]
    def execute(i):
        if i:produce_bf16_query(ops,tensors,fixture,pair[i])
        launch(ops,names[i],tensors,fixture,pair[i],cores,cluster)
    for i,graph in enumerate(graphs):
        with torch.npu.graph(graph,capture_error_mode='thread_local',auto_dispatch_capture=True):execute(i)
    states=[]
    for label,shape,negate,remap in (
        ('captured_inputs',case.shape,False,False),
        ('changed_q_qr',case.shape,True,False),
        ('changed_cache_table_tasks',replace(case.shape,contexts=tuple(c+16 for c in case.shape.contexts)),False,True),
        ('smaller_live_rows',replace(case.shape,qlens=tuple(max(1,q-1) for q in case.shape.qlens)),False,False),
        ('all_padding',case.shape,False,False),
        ('restored_live_rows',case.shape,False,False)):
        replacement=make_fixture(torch,replace(case,shape=shape))
        if negate:
            # Recompute oracle from the independently retained logical inputs.
            replacement['cpu']['q'].neg_();replacement['cpu']['qr'].neg_()
            for request,values in list(replacement['logical'].items()):
                begin,q,*other=values;replacement['logical'][request]=(begin,-q,*other)
            _refresh_dense_oracle(torch,replacement)
        if remap:mdl.remap_physical_pages(torch,replacement)
        if label=='all_padding':
            replacement['cpu']['slots'].fill_(-1)
            replacement['actual_tokens']=0;replacement['expected']={}
        for name,tensor in tensors.items():
            value=replacement['cpu'][name]
            if value.shape!=tensor.shape or value.dtype!=tensor.dtype:raise RuntimeError('graph replacement changes allocation: '+name)
            tensor.copy_(value)
        fixture=replacement
        receipt=prepare(torch,ops,tensors,fixture,pair[0])
        for buffers in pair:mdl.poison(torch,buffers)
        for graph in graphs:graph.replay()
        torch.npu.synchronize()
        detail=compare_pair(torch,replace(case,shape=shape),fixture,*pair,receipt['task_sha256'],acceptance)
        captured=[snapshot(b) for b in pair]
        for i,buffers in enumerate(pair):mdl.poison(torch,buffers);execute(i)
        torch.npu.synchronize()
        for i,buffers in enumerate(pair):same_bits(torch,captured[i],buffers,'graph/eager same-arm')
        if addresses()!=before:raise RuntimeError('graph tensor address changed')
        states.append({'state':label,'queries':fixture['actual_tokens'],'precision':detail,
                       'task_sha256':receipt['task_sha256'],
                       'raw_sha256':reuse._fingerprint({'raw':fixture['cpu']['raw']}),
                       'table_sha256':reuse._fingerprint({'table':fixture['cpu']['table']}),
                       'q_sha256':reuse._fingerprint({'q':fixture['cpu']['q']})})
    if states[0]['q_sha256']==states[1]['q_sha256']:raise RuntimeError('graph did not vary Q')
    if any(states[0][k]==states[2][k] for k in ('task_sha256','raw_sha256','table_sha256')):
        raise RuntimeError('graph did not vary all cache/table/task inputs')
    return {'capture':'passed','replay':'passed','same_addresses':True,'eager_bitwise':True,
            'precision':'passed' if all(s['precision']['passed'] for s in states) else 'failed','states':states}


def _refresh_dense_oracle(torch,fixture):
    from oscar_ascend.ops.reference import attention
    fixture['expected']={}
    for begin,q,k,v,ck,cv,dk,dv in fixture['logical'].values():
        for local in range(len(q)):
            cut=min(len(k),max(reuse.SINK,len(k)+local+1-fixture['spec'].recent_tokens))
            kk,vv=k.clone(),v.clone();kk[reuse.SINK:cut]=dk[reuse.SINK:cut];vv[reuse.SINK:cut]=dv[reuse.SINK:cut]
            result=attention(q[local:local+1],torch.cat((kk,ck[:local+1])),torch.cat((vv,cv[:local+1])),scale=fixture['scale'],causal=False)
            fixture['expected'][begin+local]=(result.output[0],result.lse[0])


def run_case(torch,ops,case,device,cores,acceptance):
    fixture=make_fixture(torch,case);names=ROUTES[case.route][:2];cluster=ROUTES[case.route][2]
    tensors={k:v.to(device) for k,v in fixture['cpu'].items()}
    pair=[allocate(torch,fixture,device,cores,case.route,candidate=bool(i)) for i in range(2)]
    receipt=reuse._prepare(torch,ops,tensors,fixture,pair[0])
    pair[1]['tasks'],pair[1]['positions']=pair[0]['tasks'],pair[0]['positions']
    original_inputs={k:v.clone() for k,v in tensors.items()}
    produce_bf16_query(ops,tensors,fixture,pair[1])
    def execute(i,merge=True):launch(ops,names[i],tensors,fixture,pair[i],cores,cluster,merge=merge)
    for i,buffers in enumerate(pair):mdl.poison(torch,buffers);execute(i,merge=not _invalid(case))
    torch.npu.synchronize()
    detail=compare_pair(torch,case,fixture,*pair,receipt['task_sha256'],acceptance,
                        device_rotation=tensors['rv'])
    for i,buffers in enumerate(pair):
        expected=snapshot(buffers,merge=not _invalid(case));mdl.poison(torch,buffers);execute(i,merge=not _invalid(case))
        torch.npu.synchronize();same_bits(torch,expected,buffers,'same-arm repeat')
    for name,value in original_inputs.items():
        if not shared._store_bits_equal(torch,value,tensors[name]):raise RuntimeError('BF16 CV mutated input '+name)
    # This first pass never times even a locally passing case. Performance
    # waits for the complete CPU/device/graph matrix, including cancellation.
    graph=graph_gate(torch,ops,case,fixture,tensors,pair,cores,acceptance) if case.graph and detail['passed'] else None
    passed=detail['passed'] and (not case.graph or graph['precision']=='passed')
    return {'case':case.shape.name,'route':case.route,'baseline_operator':names[0],'candidate_operator':names[1],
            'status':'passed' if passed else 'failed','device_completion':'passed',
            'input_sha256':fixture['input_sha256'],'task_sha256':receipt['task_sha256'],
            'queries':fixture['actual_tokens'],'padding':fixture['tokens']-fixture['actual_tokens'],
            'precision':detail,'old_repeat_bitwise':True,'candidate_repeat_bitwise':True,
            'input_bytes_unchanged':True,'timing':None,'graph':graph}


def numerical_matrix_errors(rows):
    """The whole fixed device matrix must finish before any event timing."""
    bad=[]
    plan=case_plan()
    if not isinstance(rows,list) or len(rows)!=len(plan) or any(not isinstance(r,dict) for r in rows):return ['cases']
    for case,row in zip(plan,rows):
        names=ROUTES[case.route][:2];detail=row.get('precision') or {};invalid=_invalid(case)
        expected={'case':case.shape.name,'route':case.route,'baseline_operator':names[0],
                  'candidate_operator':names[1],'status':'passed','device_completion':'passed',
                  'queries':sum(case.shape.qlens),'padding':(case.shape.padded_tokens or sum(case.shape.qlens))-sum(case.shape.qlens),
                  'old_repeat_bitwise':True,'candidate_repeat_bitwise':True,'input_bytes_unchanged':True}
        if any(row.get(k)!=v for k,v in expected.items()):bad.append(case.shape.name+'.contract')
        if any(detail.get(k)!=v for k,v in {'passed':True,'queries':0 if invalid else sum(case.shape.qlens),
                'oracle':'expected_error_domain' if invalid else 'all_actual_queries_raw_space',
                'status_lane_bits':True,'empty_partial_bits':True,'operand_contract':CONTRACT}.items()):bad.append(case.shape.name+'.precision')
        if not invalid:
            comparisons=detail.get('comparisons') or {}
            if set(comparisons)!={'baseline','candidate'}:bad.append(case.shape.name+'.oracle_arms')
            for arm in ('baseline','candidate'):
                checks=comparisons.get(arm) or {}
                if set(checks)!={'partial','lse','output','output_lse','bf16','relative_l2'} or any(v.get('passed') is not True for v in checks.values()):
                    bad.append(case.shape.name+'.'+arm+'.oracle')
        if case.graph:
            graph=row.get('graph') or {};states=graph.get('states') or []
            if any(graph.get(k)!=v for k,v in {'capture':'passed','replay':'passed','same_addresses':True,'eager_bitwise':True,'precision':'passed'}.items()):bad.append(case.shape.name+'.graph')
            labels=['captured_inputs','changed_q_qr','changed_cache_table_tasks','smaller_live_rows','all_padding','restored_live_rows']
            if len(states)!=6 or [s.get('state') for s in states]!=labels or any((s.get('precision') or {}).get('passed') is not True for s in states):bad.append(case.shape.name+'.graph_states')
            elif (states[0].get('q_sha256')==states[1].get('q_sha256') or any(states[0].get(k)==states[2].get(k) for k in ('task_sha256','raw_sha256','table_sha256'))):bad.append(case.shape.name+'.graph_changed_inputs')
    return bad


from .experiments.bf16_all_sources_screen import SOURCE_PATHS as CPU_SOURCE_PATHS


def direct_decode_errors(report):
    if not isinstance(report,dict):return ['missing']
    expected={'contract':CONTRACT,'status':'cpu_direct_decode_passed','returncode':0,'packed_words':65536,
        'packed_code_positions_checked':524288,'valid_scale_patterns':31743,'finite_zero_patterns':63488,
        'metadata_pairs':634872,'levels_checked':2539488,'metadata_cartesian_product_exhaustive':False,
        'negative_zero_code0_bits':0,'device_completion':'not_run','production_qualified':False}
    bad=[key for key,value in expected.items() if report.get(key)!=value]
    if type(report.get('exact_halfway_values_checked')) is not int or report['exact_halfway_values_checked']<=0:
        bad.append('RNE_halfway_coverage')
    return bad


def cpu_screen_identity_errors(report):
    """An existing failed screen is reusable only for exactly these sources."""
    from .experiments import bf16_all_sources_screen as screen
    if not isinstance(report,dict):return ['missing']
    expected_cases=[asdict(case) for case in screen.CASES]
    acceptance=json.loads((ROOT/'configs/acceptance.json').read_text())
    expected={'schema_version':2,'operand_contract':CONTRACT,'required_cases_complete':True,'required_cases':expected_cases,
              'frozen_attention_gate':acceptance['fused_attention'],
              'frozen_prefill_window_gate':acceptance['prefill_window'],
              'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in CPU_SOURCE_PATHS}}
    bad=[key for key,value in expected.items() if report.get(key)!=value]
    bad += ['direct_decode.'+key for key in direct_decode_errors(report.get('direct_decode'))]
    rows=report.get('cases')
    if not isinstance(rows,list) or len(rows)!=len(expected_cases):return bad+['cases']
    for expected_case,row in zip(expected_cases,rows):
        if not isinstance(row,dict) or row.get('case')!=expected_case or any(type(row.get(key)) is not bool for key in (
                'baseline_passed','candidate_passed','input_bytes_unchanged','codec_oracle_identity','all_queries_and_heads_checked')):
            bad.append(expected_case['name']+'.identity')
    if bad:return bad
    passed=all(row['baseline_passed'] and row['candidate_passed'] and row['input_bytes_unchanged'] for row in rows)
    for key,value in {'status':'cpu_screen_passed_not_qualified' if passed else 'cpu_screen_failed',
                      'returncode':0 if passed else 2,
                      'next_stage':'device_design_allowed' if passed else 'numerical_gate_failed_no_performance_qualification'}.items():
        if report.get(key)!=value:bad.append(key)
    return bad


def cpu_screen_errors(report):
    """Reject partial/stale/failed CPU screens, including failed controls."""
    bad=cpu_screen_identity_errors(report)
    if bad:return bad
    if report['returncode']!=0:bad.append('returncode')
    for row in report['cases']:
        if any(row[key] is not True for key in ('baseline_passed','candidate_passed','input_bytes_unchanged',
                                               'codec_oracle_identity','all_queries_and_heads_checked')):
            bad.append(row['case']['name']+'.numerics')
    return bad


def run_cpu_screen():
    from .experiments import bf16_all_sources_screen as screen
    # Reuse the already completed full falsification only when every fixture,
    # frozen threshold and input source hash still matches. Failed evidence is
    # reusable as a failure; it never becomes a performance authorization.
    if CPU_SCREEN_REPORT.is_file():
        saved=json.loads(CPU_SCREEN_REPORT.read_text())
        if not cpu_screen_identity_errors(saved):
            print('[oscar] BF16_HISTORY_CPU_SCREEN '+json.dumps({'status':saved['status'],
                  'source_bound_report':str(CPU_SCREEN_REPORT),'reused':True}),flush=True)
            return saved
    # Deliberately no caller-provided subset or skip flag. This separate CPU
    # precision experiment is never a device execution substitute.
    return screen.run(cases=screen.CASES,progress=lambda row:
        print('[oscar] BF16_HISTORY_CPU_CASE '+json.dumps(row),flush=True))


def time_case(torch,ops,case,device,cores,acceptance,first_receipt):
    """Rebuild only a timed fixture after global qualification, binding inputs."""
    fixture=make_fixture(torch,case)
    if fixture['input_sha256']!=first_receipt['input_sha256']:
        raise RuntimeError('BF16 timing fixture differs from the numerical pass')
    names=ROUTES[case.route][:2];cluster=ROUTES[case.route][2]
    tensors={key:value.to(device) for key,value in fixture['cpu'].items()}
    pair=[allocate(torch,fixture,device,cores,case.route,candidate=bool(i)) for i in range(2)]
    receipt=reuse._prepare(torch,ops,tensors,fixture,pair[0])
    if receipt['task_sha256']!=first_receipt['task_sha256']:
        raise RuntimeError('BF16 timing tasks differ from the numerical pass')
    pair[1]['tasks'],pair[1]['positions']=pair[0]['tasks'],pair[0]['positions']
    produce_bf16_query(ops,tensors,fixture,pair[1])
    def execute(i,merge=True):launch(ops,names[i],tensors,fixture,pair[i],cores,cluster,merge=merge)
    for i,buffers in enumerate(pair):mdl.poison(torch,buffers);execute(i)
    torch.npu.synchronize()
    precision=compare_pair(torch,case,fixture,*pair,receipt['task_sha256'],acceptance)
    if not precision['passed']:
        raise RuntimeError('BF16 numerical gate changed when rebuilding the timing fixture')
    return _timing(torch,execute,pair,names)


def performance_phase(torch,ops,rows,cpu_report,device,cores,acceptance,config):
    """One mandatory barrier; no locally passing case can bypass this gate."""
    errors=['cpu_screen.'+value for value in cpu_screen_errors(cpu_report)]
    errors += numerical_matrix_errors(rows)
    if errors:
        return {'status':'not_run_numerical_gate_failed','blocking_errors':errors}
    from . import bf16_history_pipeline as complete
    # The full step has additional math (native current FIA, rotate and store).
    # Complete all six full-path numerical receipts before timing any scope.
    for case,row in zip(case_plan(),rows):
        if not case.timed:continue
        row['full_attention']=complete.numerical_case(torch,ops,case,device,cores,config,acceptance,row)
    execution=complete.execution_contract(config,cores)
    full_errors=[error for case,row in zip(case_plan(),rows) if case.timed
                 for error in complete.validate_receipt(case,row,timed=False,execution=execution)]
    if full_errors:
        for case,row in zip(case_plan(),rows):
            if case.timed and complete.validate_receipt(case,row,timed=False,execution=execution):row['status']='failed'
        return {'status':'not_run_numerical_gate_failed','blocking_errors':full_errors}
    # Retain the first numerical/graph receipts and update only measured fields.
    for case,row in zip(case_plan(),rows):
        if not case.timed:continue
        row['timing']=time_case(torch,ops,case,device,cores,acceptance,row)
        row['full_attention']['timing']=complete.timing_case(torch,ops,case,device,cores,config,acceptance,row)
        if (not all(entry['gate']=='passed' for entry in row['timing'].values()) or
                row['full_attention']['timing']['gate']!='passed'):row['status']='failed'
        gc.collect();torch.npu.empty_cache()
    cv_passed=all(all(entry['gate']=='passed' for entry in row['timing'].values())
                  for case,row in zip(case_plan(),rows) if case.timed)
    full_passed=all(row['full_attention']['timing']['gate']=='passed' for case,row in zip(case_plan(),rows) if case.timed)
    return {'status':'passed' if cv_passed and full_passed else 'failed',
            'cv_performance':'passed' if cv_passed else 'failed',
            'full_attention_performance':'passed' if full_passed else 'failed','blocking_errors':[]}


def validate_evidence(report):
    """Validate operator evidence; numerical receipts precede performance."""
    if not isinstance(report,dict):return ['report']
    wanted={'schema':SCHEMA,'operand_contract':CONTRACT,'status':'passed','device_completion':'passed',
            'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
            'cv_performance':'passed','full_attention_performance':'passed',
            'model_quality':'not_established','production_qualified':False}
    bad=[key for key,value in wanted.items() if report.get(key)!=value]
    from .bf16_history_pipeline import validate_execution
    configuration=report.get('configuration') or {}
    execution=configuration.get('execution') or {}
    if not validate_execution(execution):bad.append('execution_contract')
    if (not isinstance(configuration.get('model_config_sha256'),str) or len(configuration['model_config_sha256'])!=64 or
        configuration.get('target_geometry')!=[6,1,256] or configuration.get('soc_version')!='ascend910b4' or
        not isinstance(configuration.get('devices'),list) or not configuration['devices'] or
        configuration.get('probe_physical_device')!=configuration['devices'][0]):bad.append('configuration')
    bad += ['cpu_screen.'+value for value in cpu_screen_errors(report.get('cpu_screen'))]
    rows=report.get('cases');bad += numerical_matrix_errors(rows)
    if not isinstance(rows,list) or len(rows)!=len(case_plan()) or any(not isinstance(r,dict) for r in rows):return bad
    for case,row in zip(case_plan(),rows):
        names=ROUTES[case.route][:2]
        if case.timed:
            from .bf16_history_pipeline import validate_receipt
            bad += validate_receipt(case,row,timed=True,execution=execution)
            timing=row.get('timing') or {}
            if set(timing)!=set(SCOPES):bad.append(case.shape.name+'.timing_scope')
            for scope in SCOPES:
                entry=timing.get(scope) or {};samples=entry.get('samples_ms') or {}
                valid=set(samples)==set(names) and all(isinstance(samples[name],list) and len(samples[name])==5 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in samples[name]) for name in names)
                if valid:
                    a,b=(statistics.median(samples[name]) for name in names)
                    valid=a/b>=MIN_SPEEDUP and all(entry.get(k)==v for k,v in {'baseline_ms':a,'candidate_ms':b,'speedup':a/b,'minimum_speedup':MIN_SPEEDUP,'warmup':2,'repeats':5,'gate':'passed'}.items())
                if not valid:bad.append(case.shape.name+'.device_latency.'+scope)
    return bad



def configuration_binding(config,cores):
    from .bf16_history_pipeline import execution_contract
    return {'execution':execution_contract(config,cores),
            'model_config_sha256':hashlib.sha256((Path(config['model'])/'config.json').read_bytes()).hexdigest(),
            'target_geometry':[6,1,256],'soc_version':config['soc_version'],
            'devices':list(config['devices']),'probe_physical_device':config['devices'][0]}


def probe(config_path,acceptance_path,*,output_path=None):
    from .serving_variants import variant_config
    config=variant_config(json.loads(config_path.read_text()), 'candidate')
    if config.get('experimental_bf16_history') is not True:raise RuntimeError('BF16 probe requires the BF16 attention configuration')
    acceptance=json.loads(acceptance_path.read_text())
    if (acceptance.get('frozen_before_measurement') is not True or acceptance['fused_attention']!={'atol':0.005,'rtol':0.005}
            or acceptance['prefill_window']['relative_l2_lt']!=0.02 or acceptance['performance']!={'max_latency_ratio':1.0,'min_throughput_ratio':1.0,'repeats':5,'warmup':2,'statistic':'median','per_case_required':True,'noise_allowance':0.0}):
        raise RuntimeError('frozen BF16 accuracy/performance contract changed')
    reuse._active_device(config)
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as error:raise RuntimeError('BF16 gate requires real NPU; no CPU substitution') from error
    if not torch.npu.is_available():raise RuntimeError('BF16 gate requires real NPU; no CPU substitution')
    torch.npu.set_device(0);torch.set_num_threads(min(4,torch.get_num_threads()))
    from .build_ops import normalize_soc
    from .probe_native_current_fia import _target_geometry
    cores=reuse._core_count(torch,config)
    configuration=configuration_binding(config,cores)
    if _target_geometry(config)!=(6,1,256):raise RuntimeError('BF16 target gate requires actual Hq6/Hkv1/D256')
    if configuration_binding(config,cores)!=configuration:raise RuntimeError('model configuration changed during probe binding')
    if normalize_soc(torch.npu.get_device_name(0))!=config['soc_version']:raise RuntimeError('BF16 target SoC differs')
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    require_capabilities({name for values in ROUTES.values() for name in values[:2]}|{'prepare_attention_tasks_out','merge_lse_out','rotate_out','rotate_vector_init_out','rotate_bf16_query_out',
                         'rotate_clip_store_striped_out','status_guard'})
    result={'schema':SCHEMA,'operand_contract':CONTRACT,'status':'running','device_completion':'not_run','cases':[],
            'source_binding':source_binding(),'acceptance_sha256':hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
            'artifact_signature':manifest['signature'],'artifact_sha256':manifest['sha256'], 'configuration':configuration,
            'full_attention_performance':'not_established','model_quality':'not_established','production_qualified':False,
            'scope_note':'CV scopes compute current inside CV; separate complete eager attention scope includes actual applicable native FIA, prepare, rotate, merge/copy, store and guards; excludes cache reset and whole-model/GDN work'}
    result['cpu_screen']=run_cpu_screen()
    result['cv_performance']=('not_run_numerical_gate_failed' if cpu_screen_errors(result['cpu_screen'])
                              else 'not_run_numerical_matrix_incomplete')
    if output_path:atomic_json(output_path,result)
    for case in case_plan():
        try:row=run_case(torch,torch.ops.oscar_ascend_ops,case,torch.device('npu:0'),cores,acceptance)
        except Exception as error:
            result.update(status='failed',device_completion='incomplete',first_error=f'{case.shape.name}: {error}')
            detail=getattr(error,'bf16_failure_diagnostics',None)
            if detail is not None:result['failure_diagnostics']=detail
            error.bf16_report=result
            if output_path:atomic_json(output_path,result)
            raise
        result['cases'].append(row)
        if output_path:atomic_json(output_path,result)
        print('[oscar] BF16_HISTORY_CASE '+json.dumps({'case':row['case'],'operator':row['candidate_operator'],
              'status':row['status'],'precision':row['precision']['passed'],
              'graph':row['graph']['replay'] if row['graph'] else 'not_run'}),flush=True)
        gc.collect();torch.npu.empty_cache()
    rows=result['cases']
    try:
        performance=performance_phase(torch,torch.ops.oscar_ascend_ops,rows,result['cpu_screen'],
                                      torch.device('npu:0'),cores,acceptance,config)
    except Exception as error:
        result.update(status='failed',cv_performance='incomplete',first_error=f'performance: {error}')
        error.bf16_report=result
        if output_path:atomic_json(output_path,result)
        raise
    result['performance_gate']=performance
    passed=all(r['status']=='passed' for r in rows) and performance['status']=='passed'
    result.update(status='passed' if passed else 'failed',device_completion='passed',
                  precision='passed' if all(r['precision']['passed'] for r in rows) else 'failed',
                  error_domain='passed' if all(r['precision']['passed'] for r,c in zip(rows,case_plan()) if _invalid(c)) else 'failed',
                  graph_capture='passed' if all((r['graph'] or {}).get('capture')=='passed' for r,c in zip(rows,case_plan()) if c.graph) else 'not_complete',
                  graph_replay='passed' if all((r['graph'] or {}).get('replay')=='passed' for r,c in zip(rows,case_plan()) if c.graph) else 'not_complete',
                  cv_performance=performance.get('cv_performance',performance['status']),
                  full_attention_performance=performance.get('full_attention_performance',performance['status']))
    mismatches=validate_evidence(result)
    if mismatches:result.update(status='failed',evidence_mismatches=mismatches)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,default=ROOT/'configs/acceptance.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    args.output.unlink(missing_ok=True)
    try:report=probe(args.config,args.acceptance,output_path=args.output)
    except Exception as error:
        traceback.print_exc()
        # Keep completed cases from this invocation; never revive a stale file.
        report=getattr(error,'bf16_report',None) or {'schema':SCHEMA,'operand_contract':CONTRACT,'status':'failed','first_error':str(error),
            'device_completion':'not_established','graph_capture':'not_run','graph_replay':'not_run',
            'precision':'not_run','cv_performance':'not_run','full_attention_performance':'not_established',
            'model_quality':'not_established','production_qualified':False}
        report.setdefault('source_binding',{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                           for name in SOURCE_PATHS if (ROOT/name).is_file()})
        report['missing_sources']=[name for name in SOURCE_PATHS if not (ROOT/name).is_file()]
        if args.acceptance.is_file():
            report['acceptance_sha256']=hashlib.sha256(args.acceptance.read_bytes()).hexdigest()
    atomic_json(args.output,report)
    print('[oscar] BF16_HISTORY_RESULT '+json.dumps({'status':report['status'],'report':str(args.output)}),flush=True)
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
