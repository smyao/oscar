"""Archive #34/#36/#129/#142/#157/#161: real mixed-prepare region gate.

The selected legacy/official prepare, boundary guards and identical device
consumer copies are executed. V4 retains source2 FP32 CV tasks; only the
legacy path applies its existing source2 suppression. No attention/model/AISBench
is substituted. One native ForwardContext owns each synthetic main-model step;
same-address32K->64K and invalid-length steps must refresh both regions.
The complete metadata-chain timing has a non-regression1.0 gate, never a CV
or end-to-end performance claim. Without a real NPU this command must fail.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback
from types import SimpleNamespace

from oscar_ascend.integration.metadata import OscarMetadata
from oscar_ascend.integration.mixed_decode_split import (
    _slice_metadata, guard_mixed_metadata, plan_mixed_decode,
)
from oscar_ascend.integration.prepare_reuse import (
    invalidate_prepare_reuse, mixed_prepare_eligible, prepare_views,
    remember_prepared, reused_slots,
)
from oscar_ascend.runtime import WorkspaceGeometry
from .phase import atomic_json
from .probe_cv_hotshape import _tensor_hash

ROOT=Path(__file__).resolve().parents[1]
TOKENS,HEADS,KV_HEADS,DIM,LAYERS=16384,6,1,256,16
SHORT_REQUESTS,SHORT_TOKENS=31,124
LONG_TOKENS=TOKENS-SHORT_TOKENS
SINK,RECENT=64,256
COLUMNS=math.ceil((65536+LONG_TOKENS)/128)
CASE_NAMES=('context32k','context64k_same_addresses','invalid_length_same_addresses')
MODES=('off','on')
MIN_SPEEDUP=1.0
TIMING_SCOPE='complete_mixed_metadata_prepare_suppress_boundary_guards_identical_device_consumers'
SOURCE_PATHS=(
    'csrc/kernels/attention_tasks.cpp','csrc/kernels/attention_official_tasks.cpp','csrc/kernels/status_guard.cpp',
    'csrc/kernels/oscar_common.h','csrc/torch_bindings.cpp',
    'oscar_ascend/integration/prepare_reuse.py','oscar_ascend/integration/metadata.py',
    'oscar_ascend/integration/mixed_decode_split.py','oscar_ascend/integration/impl.py',
    'oscar_ascend/integration/current_attention.py','oscar_ascend/integration/guard_diag.py',
    'oscar_ascend/runtime.py','oscar_ascend/ops/cv_dispatch.py','tools/probe_mixed_prepare_reuse.py',
    'tools/official_reference.py','configs/acceptance_official_bf16.json',
)


def source_binding(root=ROOT):
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def input_values(torch,context,*,invalid_length=False):
    """Private virtual128 slots, fixed capacity, no physical KV allocation."""
    starts=torch.tensor([*(4*i for i in range(SHORT_REQUESTS+1)),TOKENS],dtype=torch.int32)
    lengths=torch.tensor([context+4]*SHORT_REQUESTS+[context+LONG_TOKENS],dtype=torch.int32)
    table=torch.arange((SHORT_REQUESTS+1)*COLUMNS,dtype=torch.int32).view(SHORT_REQUESTS+1,COLUMNS)
    positions=torch.cat([torch.arange(context,context+4,dtype=torch.int64)
                         for _ in range(SHORT_REQUESTS)]+[
                         torch.arange(context,context+LONG_TOKENS,dtype=torch.int64)])
    requests=torch.bucketize(torch.arange(TOKENS),starts[1:].to(torch.int64),right=True)
    columns=positions//128
    slots=table[requests,columns].to(torch.int64)*128+positions%128
    expected_positions=positions.clone()
    if invalid_length:
        lengths[-1]=0;expected_positions[SHORT_TOKENS:]=-1
    return {'starts':starts,'lengths':lengths,'table':table,'slots':slots,
            'expected_positions':expected_positions}


def allocate(torch,device):
    geometry=WorkspaceGeometry(TOKENS,HEADS,KV_HEADS,DIM,cube_cores=20,history_cluster_size=16)
    workspace=SimpleNamespace(geometry=geometry,
        tasks=torch.empty((geometry.task_count,16),dtype=torch.int64,device=device),
        positions=torch.empty(TOKENS,dtype=torch.int64,device=device),
        slots=torch.empty(TOKENS,dtype=torch.int64,device=device),
        store_status=torch.empty((TOKENS,KV_HEADS),dtype=torch.int32,device=device))
    audit={'tasks':torch.empty((LAYERS,geometry.task_count,16),dtype=torch.int64,device=device),
           'positions':torch.empty((LAYERS,TOKENS),dtype=torch.int64,device=device)}
    return workspace,audit


def parent_metadata(torch,values,device):
    tensors={name:values[name].to(device).clone() for name in ('starts','lengths','table','slots')}
    metadata=OscarMetadata(tensors['starts'],tensors['lengths'],tensors['table'],tensors['slots'],
        SHORT_REQUESTS+1,TOKENS,LONG_TOKENS,65536+LONG_TOKENS,
        num_input_tokens=TOKENS,query_start_loc_cpu=values['starts'].clone(),
        current_cumulative=tuple(values['starts'][1:].tolist()),mixed_decode_split=True)
    return metadata,tensors


@contextmanager
def native_context(metadata,names):
    from vllm.forward_context import ForwardContext,override_forward_context
    context=ForwardContext(no_compile_layers={},attn_metadata={name:metadata for name in names},slot_mapping={})
    context.capturing=False;context.in_profile_run=False;context.num_tokens=TOKENS
    with override_forward_context(context):
        yield context


def execute_epoch(torch,ops,workspace,audit,metadata,*,enabled,epoch,
                  suppress=None,scope=native_context,official=False):
    """Actual prepare calls; injectable CPU seams exist only for host tests."""
    if suppress is None:
        from oscar_ascend.integration.current_attention import suppress_current_source_tasks
        suppress=suppress_current_source_tasks
    names=tuple(f'prepare_probe.{epoch}.full{layer}' for layer in range(LAYERS))
    counts={'prepare':0,'suppress':0,'boundary_guards':0,'consumer_copies':0,'task_bytes_written':0}
    splits=workspace.geometry.splits_for_tokens(TOKENS)
    factor=KV_HEADS*3*splits
    with scope(metadata,names) as context:
        context_class=f'{type(context).__module__}.{type(context).__name__}'
        for layer,name in enumerate(names):
            reuse=mixed_prepare_eligible(metadata,name,enabled=enabled,native_current=True)
            if reuse is not enabled:raise RuntimeError('requested prepare mode did not reach its native owner scope')
            plan=plan_mixed_decode(metadata,TOKENS,query_heads=HEADS,kv_heads=KV_HEADS,immutable_starts=reuse)
            if plan is None or plan.prefix_tokens!=SHORT_TOKENS:
                raise RuntimeError('probe did not enter the real mixed partition')
            guard_mixed_metadata(ops,workspace,metadata,plan,TOKENS);counts['boundary_guards']+=1
            for rb,re,begin,end,segment in (
                (0,SHORT_REQUESTS,0,SHORT_TOKENS,'short_prefix'),
                (SHORT_REQUESTS,SHORT_REQUESTS+1,SHORT_TOKENS,TOKENS,'long_suffix')):
                child=_slice_metadata(metadata,rb,re,begin,end,TOKENS,segment,prepare_reuse=reuse)
                n=end-begin;geometry=(n,HEADS,KV_HEADS,SINK,RECENT,splits)
                if official:geometry=(*geometry,True)
                if reuse:
                    tasks,positions,slot_out=prepare_views(workspace,child,n,KV_HEADS,splits,name,context)
                    cached=reused_slots(workspace,child,name,geometry,torch.npu.current_stream(),context)
                else:
                    invalidate_prepare_reuse(workspace);cached=None
                    tasks,positions,slot_out=(workspace.tasks[:n*factor],workspace.positions[:n],workspace.slots[:n])
                slots=child.slot_mapping if cached is None else cached
                if cached is None:
                    if slots.dtype==torch.int32:
                        slot_out.copy_(slots);slots=slot_out
                    prepare=(ops.prepare_official_attention_tasks_out if official else ops.prepare_attention_tasks_out)
                    prepare(child.query_start_loc,child.seq_lens,slots,tasks,
                        positions,HEADS,KV_HEADS,SINK,RECENT,splits,None,False)
                    if not official:
                        suppress(tasks,n,KV_HEADS,splits)
                        counts['suppress']+=1
                    counts['prepare']+=1
                    counts['task_bytes_written']+=tasks.numel()*tasks.element_size()
                    if reuse:remember_prepared(workspace,child,name,geometry,torch.npu.current_stream(),slots,context)
                # Identical consumers in both arms; these copies intentionally
                # expose a later segment overwriting an earlier cached region.
                audit['tasks'][layer,begin*factor:end*factor].copy_(tasks)
                audit['positions'][layer,begin:end].copy_(positions)
                counts['consumer_copies']+=2
    return {**counts,**({'prepare_operator':'prepare_official_attention_tasks_out','current_source':'CV_FP32'} if official else {}),
            'native_context_class':context_class,'native_context_id':id(context),
            'region_order':['short_prefix','long_suffix'],'extra_production_hbm_bytes':0}


def _snapshot(audit):
    return {name:value.cpu().clone() for name,value in audit.items()}


def _check(torch,snapshot,values,*,invalid_length=False):
    expected=values['expected_positions'][None,:].expand(LAYERS,-1)
    if not torch.equal(snapshot['positions'],expected):raise RuntimeError('prepared logical positions changed')
    errors=snapshot['tasks'][...,10]
    wanted=torch.zeros_like(errors)
    if invalid_length:wanted[:,SHORT_TOKENS*3:]=1
    if not torch.equal(errors,wanted):raise RuntimeError('prepare metadata error domain changed or was cleared')
    return {'task_error':'invalid_long_length_code1_preserved' if invalid_length else 'all_zero',
            'complete_task_fields':True,'complete_positions':True}


def _counts_valid(receipt,mode,official=False):
    launches=2 if mode=='on' else 2*LAYERS
    return (receipt.get('prepare')==launches and receipt.get('suppress')==(0 if official else launches)
            and (not official or (receipt.get('prepare_operator')=='prepare_official_attention_tasks_out' and receipt.get('current_source')=='CV_FP32'))
            and receipt.get('boundary_guards')==LAYERS and receipt.get('consumer_copies')==LAYERS*4
            and receipt.get('task_bytes_written')==TOKENS*3*16*8*(1 if mode=='on' else LAYERS)
            and receipt.get('native_context_class')=='vllm.forward_context.ForwardContext'
            and receipt.get('extra_production_hbm_bytes')==0)


def validate_evidence(report):
    required={'status':'passed','device_completion':'passed','precision':'all_task_fields_and_positions_bitwise',
              'scope':'mixed_prepare_only_no_CV_no_model','same_addresses':True,'source_inputs_unchanged':True,
              'tokens':TOKENS,'layers':LAYERS,'split_tokens':[SHORT_TOKENS,LONG_TOKENS]}
    bad=[key for key,value in required.items() if not isinstance(report,dict) or report.get(key)!=value]
    cases=report.get('cases',[]) if isinstance(report,dict) else []
    if not isinstance(cases,list) or [c.get('case') for c in cases]!=list(CASE_NAMES):return bad+['cases']
    previous=None
    for index,case in enumerate(cases):
        if (case.get('status')!='passed' or case.get('bitwise') is not True
                or case.get('task_error')!=('invalid_long_length_code1_preserved' if index==2 else 'all_zero')
                or not case.get('complete_task_fields') or not case.get('complete_positions')):
            bad.append(case['case']+'.fields')
        for mode in MODES:
            if not _counts_valid(case.get(mode,{}) or {},mode,report.get('official_math',False)):bad.append(case['case']+'.'+mode)
        current=case.get('on',{}).get('native_context_id')
        if current is None or current==previous:bad.append(case['case']+'.new_context')
        previous=current
    if isinstance(report,dict) and report.get('validation_mode')=='correctness_only_user_requested':
        if report.get('timing') is not None or report.get('performance')!='not_run_user_requested':
            bad.append('unexpected_performance_evidence')
        return bad
    timing=report.get('timing',{}) if isinstance(report,dict) else {}
    samples=timing.get('samples_ms',{})
    valid=(set(samples)==set(MODES) and all(isinstance(samples[m],list) and len(samples[m])==5
           and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in samples[m]) for m in MODES))
    if valid:
        old,new=(statistics.median(samples[m]) for m in MODES)
        valid=(new<=old and timing.get('off_ms')==old and timing.get('on_ms')==new
               and timing.get('ratio')==new/old and timing.get('speedup')==old/new
               and timing.get('minimum_speedup')==MIN_SPEEDUP and timing.get('scope')==TIMING_SCOPE
               and timing.get('gate')=='passed')
    if not valid:bad.append('complete_metadata_latency')
    return bad


def probe(config_path,acceptance_path):
    target=json.loads(config_path.read_text());acceptance=json.loads(acceptance_path.read_text())
    correctness_only=target.get('experimental_bf16_attention',False) is True
    official=target.get('experimental_official_math',False) is True
    from .probe_history_reuse import _active_device
    _active_device(target)
    if target.get('experimental_prepare_reuse') is not True:
        raise RuntimeError('explicit prepare reuse candidate config required')
    if acceptance['performance']!={'max_latency_ratio':1.0,'min_throughput_ratio':1.0,'repeats':5,
            'warmup':2,'statistic':'median','per_case_required':True,'noise_allowance':0.0}:
        raise RuntimeError('frozen performance policy changed')
    import torch
    import torch_npu  # noqa: F401
    if not torch.npu.is_available():raise RuntimeError('target NPU required; no CPU substitution')
    torch.npu.set_device(0)
    import vllm_ascend.ops  # noqa: F401; archive #28/#78 native import order
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    require_capabilities({'prepare_official_attention_tasks_out' if official else 'prepare_attention_tasks_out','status_guard'})
    ops=torch.ops.oscar_ascend_ops;device=torch.device('npu:0')
    workspace,audit=allocate(torch,device)
    metadata,tensors=parent_metadata(torch,input_values(torch,32768),device)
    addresses={'input_'+name:t.data_ptr() for name,t in tensors.items()}
    addresses.update({'arena_'+name:getattr(workspace,name).data_ptr() for name in ('tasks','positions','slots')})
    goldens={};rows={name:{'case':name} for name in CASE_NAMES};epoch=0
    for mode in MODES:
        # No invalidation between candidate steps: changed inputs and fresh
        # native contexts, not a manual cache clear or repeated layer IDs,
        # must refresh the regions. Names are disjoint across these epochs.
        for index,name in enumerate(CASE_NAMES):
            values=input_values(torch,32768 if index==0 else 65536,invalid_length=index==2)
            for key,tensor in tensors.items():tensor.copy_(values[key])
            original={key:value.clone() for key,value in tensors.items()}
            receipt=execute_epoch(torch,ops,workspace,audit,metadata,enabled=mode=='on',epoch=epoch,official=official);epoch+=1
            torch.npu.synchronize();snapshot=_snapshot(audit)
            detail=_check(torch,snapshot,values,invalid_length=index==2)
            if any(not torch.equal(value,original[key]) for key,value in tensors.items()):
                raise RuntimeError('prepare probe mutated native source inputs')
            rows[name][mode]=receipt;rows[name].update(detail)
            if mode=='off':goldens[name]=snapshot
            else:
                if any(not torch.equal(snapshot[key],goldens[name][key]) for key in snapshot):
                    raise RuntimeError('prepare reuse changed a descriptor or position')
                rows[name].update(status='passed',bitwise=True,task_sha256=_tensor_hash(snapshot))
    values=input_values(torch,32768)
    for key,tensor in tensors.items():tensor.copy_(values[key])
    samples={mode:[] for mode in MODES}
    for repeat in range(0 if correctness_only else 7):
        order=MODES if repeat<2 or (repeat-2)%2==0 else tuple(reversed(MODES))
        for mode in order:
            begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            begin.record()
            receipt=execute_epoch(torch,ops,workspace,audit,metadata,enabled=mode=='on',epoch=epoch,official=official);epoch+=1
            end.record();end.synchronize();elapsed=float(begin.elapsed_time(end))
            snapshot=_snapshot(audit)
            if (not _counts_valid(receipt,mode,official) or not math.isfinite(elapsed) or elapsed<=0
                    or any(not torch.equal(snapshot[key],goldens[CASE_NAMES[0]][key]) for key in snapshot)):
                raise RuntimeError('timed full metadata chain changed fields/counts')
            if repeat>=2:samples[mode].append(elapsed)
    old,new=(None,None) if correctness_only else (statistics.median(samples[mode]) for mode in MODES)
    now={'input_'+name:t.data_ptr() for name,t in tensors.items()}
    now.update({'arena_'+name:getattr(workspace,name).data_ptr() for name in ('tasks','positions','slots')})
    if now!=addresses:raise RuntimeError('metadata input/output addresses moved')
    timing=None if correctness_only else {'off_ms':old,'on_ms':new,'ratio':new/old,'speedup':old/new,'samples_ms':samples,
            'minimum_speedup':MIN_SPEEDUP,'scope':TIMING_SCOPE,'gate':'passed' if new<=old else 'failed'}
    result={'status':'passed' if correctness_only or new<=old else 'failed','device_completion':'passed',
        'precision':'all_task_fields_and_positions_bitwise','scope':'mixed_prepare_only_no_CV_no_model',
        'same_addresses':True,'source_inputs_unchanged':True,'tokens':TOKENS,'layers':LAYERS,
        'split_tokens':[SHORT_TOKENS,LONG_TOKENS],'cases':list(rows.values()),'timing':timing,
        'artifact_signature':manifest['signature'],'artifact_sha256':manifest['sha256'],
        'source_binding':source_binding(),'whole_service_performance':'not_established',
        'validation_mode':'correctness_only_user_requested' if correctness_only else 'metadata_latency',
        'performance':'not_run_user_requested' if correctness_only else ('passed' if new<=old else 'failed')}
    if official:result.update(official_math=True,math_contract='oscar_bf16_attention_v4')
    bad=validate_evidence(result)
    if bad:result.update(status='failed',evidence_mismatches=bad)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,default=ROOT/'configs/acceptance.json')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:result=probe(args.config,args.acceptance)
    except Exception as error:
        traceback.print_exc();result={'status':'failed','first_error':str(error)}
    atomic_json(args.output,result)
    print('[oscar] PREPARE_REUSE_RESULT '+json.dumps({'status':result['status'],'report':str(args.output)}),flush=True)
    raise SystemExit(0 if result['status']=='passed' else 2)
