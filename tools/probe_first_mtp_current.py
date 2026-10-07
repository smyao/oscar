"""Archive #126/#140/#142/#148/#155: first-MTP current partial, real NPU only.

History/window CV and the INT2 writer remain production operators. Both arms
start from identical cache snapshots for every timed iteration; a repeated
prefill must not read a ring overwritten by the previous iteration. No graph
or whole-model performance is inferred from this eager primitive gate.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import statistics
import traceback

from . import probe_current_only as shared
from . import probe_history_reuse as reuse
from .phase import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def case_plan():
    return tuple((qlens, slot) for qlens in ((4,385),(1,17)) for slot in ('int32','int64'))


def case_name(case):
    qlens,slot=case
    return f'first_mtp_history511_769_q{qlens[0]}_{qlens[1]}_{slot}'


@contextmanager
def counted_native_current():
    from oscar_ascend.integration import impl as module
    original=module.native_current_partial
    calls=[]
    def call(query,key,value,cumulative,**kwargs):
        result=original(query,key,value,cumulative,**kwargs)
        calls.append({'tokens':query.shape[0],'cumulative':tuple(cumulative)})
        return result
    module.native_current_partial=call
    try:yield calls
    finally:module.native_current_partial=original


def validate_case_evidence(report, acceptance):
    rows=report.get('cases')
    expected={case_name(case):case for case in case_plan()}
    if not isinstance(rows,list) or len(rows)!=len(expected) or any(not isinstance(r,dict) for r in rows):
        raise RuntimeError('first-MTP current evidence requires all four cases')
    names=[r.get('case') for r in rows]
    if any(not isinstance(n,str) for n in names) or set(names)!=set(expected):
        raise RuntimeError('first-MTP current cases missing, duplicated, or unexpected')
    policy=acceptance['performance']
    forwards=1+policy['warmup']+policy['repeats']
    for row in rows:
        qlens,slot=expected[row['case']]
        required={'status':'passed','ab_output':'frozen_tolerance_passed','ab_lse':'frozen_tolerance_passed',
                  'independent_oracle':'passed','store_bitwise':'passed','rotation':'draft_identity',
                  'slot_dtype':slot,'cache_reset_before_each_forward':True}
        if any(row.get(k)!=v for k,v in required.items()) or row.get('qlens')!=list(qlens):
            raise RuntimeError('first-MTP current lacks exact shape/rotation/numerical evidence')
        if row.get('contexts')!=[511,769] or row.get('current_only_calls')!=0:
            raise RuntimeError('first-MTP current incorrectly tested a fresh-suffix shortcut')
        if row.get('native_current_calls')!={'baseline':0,'candidate':forwards}:
            raise RuntimeError('first-MTP current lacks actual native FIA witnesses')
        for side in ('baseline','candidate'):
            values=row.get('device_event_ms',{}).get(side)
            median=row.get('device_event_median_ms',{}).get(side)
            if (not isinstance(values,list) or len(values)!=policy['repeats'] or
                any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in values) or
                type(median) not in (float,int) or not math.isclose(median,statistics.median(values),rel_tol=1e-12)):
                raise RuntimeError('first-MTP current timing evidence incomplete or invalid')
        follow=row.get('decode_after_store',{})
        if any(follow.get(k)!=v for k,v in {'status':'passed','reader':'production_subsequent_q1',
            'output_bitwise':'passed','independent_oracle':'passed','subsequent_store_bitwise':'passed',
            'sampled_queries':2}.items()):
            raise RuntimeError('first-MTP current lacks subsequent q1 cache-reader evidence')


def run_case(torch,target,acceptance,device,cores,case):
    qlens,slot=case;name=case_name(case)
    shape=reuse.Shape(name,qlens,(511,769),256,1,1,False)
    fixture=reuse.make_fixture(torch,shape,rotation_mode='identity')
    tensors={key:value.to(device) for key,value in fixture['cpu'].items()}
    tensors['slots']=tensors['slots'].to(getattr(torch,slot))
    sides={};snapshots={};outputs={};lses={};events={k:[] for k in ('baseline','candidate')}
    witnesses={k:0 for k in events};policy=acceptance['performance'];tol=acceptance['fused_attention']
    cache_names=('raw','window_key','window_value','window_tags')
    for side,enabled in (('baseline',False),('candidate',True)):
        cfg={**target,'experimental_current_only':False,'experimental_first_mtp_current_fia':enabled}
        impl,state=shared._runtime(torch,fixture,tensors,device,cores,cfg,False,hadamard=False)
        metadata=shared._metadata(torch,fixture,tensors,draft=True,enabled=False)
        if enabled:
            # A separate native-builder contract qualifies the actual first
            # call. This primitive test supplies the same explicit marker.
            metadata=replace(metadata,first_draft_current_fia=True,
                current_cumulative=tuple(fixture['cpu']['starts'][1:].tolist()))
        assert metadata.current_only_plan is None
        sides[side]=(impl,state,metadata)
        snapshots[side]={key:getattr(state,key).clone() for key in cache_names}
    for iteration in range(1+policy['warmup']+policy['repeats']):
        for side in ('baseline','candidate'):
            impl,state,metadata=sides[side]
            for key,value in snapshots[side].items():getattr(state,key).copy_(value)
            output=torch.empty_like(tensors['q'])
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            with counted_native_current() as calls:
                start.record()
                shared._forward(torch,impl,state,tensors,metadata,output)
                end.record();torch.npu.synchronize()
            expected_calls=0 if side=='baseline' else 1
            if len(calls)!=expected_calls or (calls and calls[0]!={'tokens':sum(qlens),'cumulative':(qlens[0],sum(qlens))}):
                raise RuntimeError('first-MTP native-current call geometry mismatch')
            witnesses[side]+=len(calls)
            ms=float(start.elapsed_time(end))
            if not math.isfinite(ms) or ms<=0:raise RuntimeError('invalid first-MTP device duration')
            if iteration>policy['warmup']:events[side].append(ms)
            outputs[side]=output;lses[side]=state.workspace.lse[:fixture['tokens']].clone()
        if iteration==0:
            for output in outputs.values():shared._assert_output(torch,output,fixture['expected'],tol)
            torch.testing.assert_close(outputs['candidate'].float(),outputs['baseline'].float(),**tol)
            torch.testing.assert_close(lses['candidate'],lses['baseline'],**tol)
            indexes=sorted(fixture['expected'])
            oracle_lse=torch.stack([fixture['expected'][i][1] for i in indexes])
            torch.testing.assert_close(lses['candidate'][indexes].cpu(),oracle_lse,**tol)
        shared._assert_stores(torch,sides['baseline'][1],sides['candidate'][1])
    old_impl,old,_=sides['baseline'];new_impl,new,_=sides['candidate']
    with counted_native_current() as later_calls:
        follow=shared._decode_after_store(torch,fixture,tensors,old_impl,old,new_impl,new,device,tol)
    if later_calls:raise RuntimeError('later q1 incorrectly selected first-MTP current FIA')
    copy_calls=sum(impl.provider.ops.calls['copy_validate_current_out'] for impl,_,_ in sides.values())
    if copy_calls:raise RuntimeError('first-MTP partial gate selected whole current-only shortcut')
    medians={k:statistics.median(v) for k,v in events.items()}
    row={'case':name,'status':'passed','qlens':list(qlens),'contexts':[511,769],
         'slot_dtype':slot,'rotation':'draft_identity','ab_output':'frozen_tolerance_passed',
         'ab_lse':'frozen_tolerance_passed','independent_oracle':'passed','store_bitwise':'passed',
         'cache_reset_before_each_forward':True,'current_only_calls':copy_calls,
         'native_current_calls':witnesses,'decode_after_store':follow,'device_event_ms':events,
         'device_event_median_ms':medians,'candidate_over_baseline':medians['candidate']/medians['baseline'],
         'timing_scope':'same_input_eager_first_MTP_full_attention_forward',
         'full_model_performance':'not_established'}
    print('[oscar] PERF_FIRST_MTP_CURRENT '+json.dumps({
        'case':name,'precision':'passed','cache_and_q1':'bitwise_passed',
        'baseline_ms':medians['baseline'],'candidate_ms':medians['candidate'],
        'ratio':row['candidate_over_baseline'],'scope':'eager_full_attention_forward'},sort_keys=True),flush=True)
    return row


def probe(config_path,acceptance_path):
    target=json.loads(config_path.read_text());acceptance=json.loads(acceptance_path.read_text())
    if target.get('experimental_first_mtp_current_fia') is not True:
        raise RuntimeError('requires explicit experimental_first_mtp_current_fia=true')
    shared._configuration(target,False)
    if acceptance.get('frozen_before_measurement') is not True:raise RuntimeError('frozen acceptance required')
    from .probe_native_current_fia import _select_target_npu,_target_geometry
    _select_target_npu(target)
    import torch
    import torch_npu  # noqa: F401 -- never substitute CPU
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if not torch.npu.is_available():raise RuntimeError('requires a real NPU')
    torch.npu.set_device(0)
    from .build_ops import normalize_soc
    if normalize_soc(torch.npu.get_device_name(0))!=target['soc_version']:
        raise RuntimeError('explicit target SoC mismatch')
    if _target_geometry(target)!=(6,1,256):raise RuntimeError('requires target Hq6/Hkv1/D256')
    import vllm_ascend.ops  # noqa: F401
    from oscar_ascend.ops.loader import require_capabilities,validate_build_artifacts
    from oscar_ascend.ops.cv_dispatch import STRIPED_CV_OPS
    path=ROOT/'build/ascendc/build_manifest.json';manifest=validate_build_artifacts(path)
    require_capabilities(STRIPED_CV_OPS|{'prepare_attention_tasks_out','rotate_out','merge_lse_out',
        'status_guard','rotate_clip_store_striped_out'},path)
    rows=[];device=torch.device('npu:0');cores=reuse._core_count(torch,target)
    for case in case_plan():
        rows.append(run_case(torch,target,acceptance,device,cores,case))
        gc.collect();torch.npu.empty_cache()
    report={'status':'passed','device_completion':'passed','precision':'passed','store_bitwise':'passed',
            'decode_after_store':'passed','cases':rows,'artifact_signature':manifest['signature'],
            'artifact_sha256':manifest['sha256'],'graph_capture':'not_run_eager_scope',
            'graph_replay':'not_run_eager_scope','model_quality':'not_established',
            'full_model_performance':'not_established'}
    validate_case_evidence(report,acceptance)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/target.json')
    parser.add_argument('--acceptance',type=Path,default=ROOT/'configs/acceptance.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:report=probe(args.config,args.acceptance)
    except Exception as exc:
        traceback.print_exc();report={'status':'failed','first_error':str(exc)}
    atomic_json(args.output,report)
    print('[oscar] PERF_FIRST_MTP_CURRENT_RESULT '+json.dumps({'status':report['status'],'report':str(args.output)}),flush=True)
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
