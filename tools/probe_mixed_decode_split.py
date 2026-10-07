"""Real-NPU request partition gate; no native model/GDN edits or user dataset.

Compare the actual FULL implementation with partition disabled/enabled. CPU
mirrors define request boundaries; device guards and real CV/FIA/store calls
must agree. The timing gate covers complete attention forwards on mixed long
shapes, not just the accelerated prefix.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import statistics
import traceback
from types import SimpleNamespace

from . import probe_current_only as shared
from . import probe_history_reuse as reuse
from .phase import atomic_json

ROOT=Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name:str
    lengths:tuple[int,...]
    contexts:tuple[int,...]
    draft:bool=False
    slot_dtype:str='int64'
    timed:bool=False


def case_plan():
    contexts=(20000,23000,27000,30000)*8
    return (Case('mixed_main_small',(4,4,1024),(511,769,1025)),
            Case('mixed_first_mtp_small',(4,4,1024),(511,769,1025),True,'int32'),
            Case('mixed31_decode_continuation4k',(4,)*31+(4096,),contexts[:31]+(15904,),timed=True),
            Case('mixed31_decode_continuation16k',(4,)*31+(16260,),contexts[:31]+(13740,),timed=True))


def _metadata(torch,fixture,tensors,case,enabled):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from oscar_ascend.integration.metadata import from_common
    common=SimpleNamespace(causal=True,query_start_loc=tensors['starts'],
        query_start_loc_cpu=fixture['cpu']['starts'],seq_lens=tensors['lens'],
        block_table_tensor=tensors['table'],slot_mapping=tensors['slots'],
        num_reqs=len(case.lengths),num_actual_tokens=fixture['actual_tokens'],
        num_input_tokens=fixture['tokens'],max_query_len=max(case.lengths),
        max_seq_len=max(a+b for a,b in zip(case.lengths,case.contexts)),
        attn_state=AscendAttentionState.ChunkedPrefill)
    return from_common(common,is_draft=case.draft,first_draft_current_fia=case.draft,
                       mixed_decode_split=enabled)


def _initial_forward(torch,impl,state,tensors,metadata,cut):
    """Observe child metadata/LSE with real delegated forwards; no mock ops."""
    original=impl.forward;records=[]
    lse=torch.full((tensors['q'].shape[0],impl.num_heads),-float('inf'),
                   dtype=torch.float32,device=tensors['q'].device)
    def observed(layer,q,k,v,cache,m,output=None,**kwargs):
        result=original(layer,q,k,v,cache,m,output=output,**kwargs)
        segment=m.mixed_split_segment
        if segment is not None:
            begin=0 if segment=='short_prefix' else cut
            lse[begin:begin+q.shape[0]].copy_(state.workspace.lse[:q.shape[0]])
            records.append({'segment':segment,'tokens':q.shape[0],
                            'reference_tokens':m.cv_shape_tokens})
        elif not m.mixed_decode_split:
            lse[:q.shape[0]].copy_(state.workspace.lse[:q.shape[0]])
        return result
    impl.forward=observed
    try:output=shared._forward(torch,impl,state,tensors,metadata)
    finally:del impl.forward
    torch.npu.synchronize()
    return output,lse,records


def run_case(torch,target,acceptance,device,cores,case):
    mode='identity' if case.draft else 'hadamard'
    fixture=reuse.make_fixture(torch,reuse.Shape(case.name,case.lengths,case.contexts,256,1,1,False),
                               rotation_mode=mode)
    pages=fixture['page_assignments']
    if sum(len(p) for p in pages)!=len({p for group in pages for p in group}):
        raise RuntimeError('partition fixture must have independent physical request pages')
    tensors={k:v.to(device) for k,v in fixture['cpu'].items()}
    tensors['slots']=tensors['slots'].to(getattr(torch,case.slot_dtype))
    cut=sum(case.lengths[:-1]);n=fixture['tokens'];tol=acceptance['fused_attention']
    sides={};snapshots={};outputs={};lses={};witness={}
    for side,enabled in (('old',False),('new',True)):
        config={**target,'experimental_current_only':False,'experimental_mixed_decode_split':enabled}
        impl,state=shared._runtime(torch,fixture,tensors,device,cores,config,False,hadamard=not case.draft)
        metadata=_metadata(torch,fixture,tensors,case,enabled)
        snapshots[side]={key:getattr(state,key).clone() for key in ('raw','window_key','window_value','window_tags')}
        sides[side]=(impl,state,metadata)
        outputs[side],lses[side],witness[side]=_initial_forward(torch,impl,state,tensors,metadata,cut)
        shared._assert_output(torch,outputs[side],fixture['expected'],tol)
    expected=[{'segment':'short_prefix','tokens':cut,'reference_tokens':n},
              {'segment':'long_suffix','tokens':case.lengths[-1],'reference_tokens':n}]
    if witness['old'] or witness['new']!=expected:raise RuntimeError('real request partition route witness differs')
    torch.testing.assert_close(outputs['new'].float(),outputs['old'].float(),**tol)
    torch.testing.assert_close(lses['new'],lses['old'],**tol)
    indexes=sorted(fixture['expected'])
    torch.testing.assert_close(lses['new'][indexes].cpu(),
                              torch.stack([fixture['expected'][i][1] for i in indexes]),**tol)
    shared._assert_stores(torch,sides['old'][1],sides['new'][1])
    if shared._store_bits_equal(torch,snapshots['new']['raw'],sides['new'][1].raw):
        raise RuntimeError('partition writer did not publish compressed bytes')
    timing=None;follow=None
    if case.timed:
        samples={'old':[],'new':[]};policy=acceptance['performance']
        for i in range(policy['warmup']+policy['repeats']):
            order=('old','new') if i<policy['warmup'] or (i-policy['warmup'])%2==0 else ('new','old')
            for side in order:
                impl,state,metadata=sides[side]
                for key,data in snapshots[side].items():getattr(state,key).copy_(data)
                begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                begin.record();shared._forward(torch,impl,state,tensors,metadata,outputs[side]);end.record();end.synchronize()
                elapsed=float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid partition device event')
                if i>=policy['warmup']:samples[side].append(elapsed)
            shared._assert_stores(torch,sides['old'][1],sides['new'][1])
        old_ms,new_ms=(statistics.median(samples[k]) for k in ('old','new'))
        timing={'old_ms':old_ms,'new_ms':new_ms,'ratio':new_ms/old_ms,'samples_ms':samples,
                'gate':'passed' if new_ms/old_ms<=policy['max_latency_ratio'] else 'failed',
                'scope':'complete_real_FULL_forward_with_cache_reset_outside_timing'}
        for output in outputs.values():shared._assert_output(torch,output,fixture['expected'],tol)
    else:
        oi,old,_=sides['old'];ni,new,_=sides['new']
        follow=shared._decode_after_store(torch,fixture,tensors,oi,old,ni,new,device,tol)
    row={'case':case.name,'status':'passed' if timing is None or timing['gate']=='passed' else 'failed',
         'precision':'frozen_output_LSE_and_independent_oracle_passed','store_bitwise':'passed',
         'first_mtp':case.draft,'slot_dtype':case.slot_dtype,'rotation':mode,
         'partition':witness['new'],'original_source_splits_preserved':True,
         'decode_after_store':follow,'timings':timing,'full_model_performance':'not_established'}
    print('[oscar] PERF_MIXED_DECODE_SPLIT '+json.dumps({'case':case.name,'status':row['status'],
        'precision':row['precision'],'old_ms':timing['old_ms'] if timing else None,
        'new_ms':timing['new_ms'] if timing else None,'ratio':timing['ratio'] if timing else None}),flush=True)
    return row


def validate_case_evidence(report,acceptance):
    expected={c.name:c for c in case_plan()};rows=report.get('cases')
    if not isinstance(rows,list) or len(rows)!=len(expected) or any(not isinstance(r,dict) for r in rows):
        raise RuntimeError('mixed partition gate requires every declared case')
    names=[r.get('case') for r in rows]
    if any(not isinstance(n,str) for n in names) or set(names)!=set(expected):raise RuntimeError('partition cases differ')
    for row in rows:
        case=expected[row['case']];cut=sum(case.lengths[:-1]);n=sum(case.lengths)
        if (row.get('status')!='passed' or row.get('precision')!='frozen_output_LSE_and_independent_oracle_passed'
                or row.get('store_bitwise')!='passed' or row.get('first_mtp') is not case.draft
                or row.get('slot_dtype')!=case.slot_dtype or row.get('rotation')!=('identity' if case.draft else 'hadamard')
                or row.get('original_source_splits_preserved') is not True or row.get('partition')!=[
                    {'segment':'short_prefix','tokens':cut,'reference_tokens':n},
                    {'segment':'long_suffix','tokens':case.lengths[-1],'reference_tokens':n}]):
            raise RuntimeError('mixed partition route/numerical evidence differs')
        if case.timed:
            timing=row.get('timings',{});medians={}
            for side in ('old','new'):
                samples=timing.get('samples_ms',{}).get(side)
                if (not isinstance(samples,list) or len(samples)!=acceptance['performance']['repeats'] or
                        any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in samples)):
                    raise RuntimeError('partition timing samples invalid')
                medians[side]=statistics.median(samples)
                if not math.isclose(timing.get(side+'_ms',-1),medians[side],rel_tol=1e-12):raise RuntimeError('partition median differs')
            ratio=medians['new']/medians['old']
            if (timing.get('gate')!='passed' or ratio>acceptance['performance']['max_latency_ratio'] or
                    not math.isclose(timing.get('ratio',-1),ratio,rel_tol=1e-12)):
                raise RuntimeError('partition complete-forward latency gate failed')
        else:
            follow=row.get('decode_after_store',{})
            if any(follow.get(k)!=v for k,v in {'status':'passed','reader':'production_subsequent_q1',
                'output_bitwise':'passed','independent_oracle':'passed','subsequent_store_bitwise':'passed',
                'sampled_queries':len(case.lengths)}.items()):raise RuntimeError('partition next-q1 evidence missing')


def probe(config_path,acceptance_path):
    target=json.loads(config_path.read_text());acceptance=json.loads(acceptance_path.read_text())
    if any(target.get(k) is not True for k in ('experimental_decode_bundle','experimental_mixed_decode_split',
                                              'experimental_first_mtp_current_fia')):
        raise RuntimeError('partition gate requires explicit decoder, partition and first-MTP current flags')
    if acceptance.get('frozen_before_measurement') is not True:raise RuntimeError('frozen acceptance required')
    policy=acceptance['performance']
    if (policy['warmup'],policy['repeats'],policy['max_latency_ratio'])!=(2,5,1.0):
        raise RuntimeError('frozen performance policy changed')
    reuse._active_device(target)
    import torch
    import torch_npu  # noqa: F401
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if not torch.npu.is_available():raise RuntimeError('partition gate requires a real NPU')
    torch.npu.set_device(0)
    from .probe_native_current_fia import _target_geometry
    from .build_ops import normalize_soc
    if _target_geometry(target)!=(6,1,256):raise RuntimeError('partition gate requires target Hq6/Hkv1/D256')
    if normalize_soc(torch.npu.get_device_name(0))!=target['soc_version']:
        raise RuntimeError('selected NPU does not match explicit target SoC')
    import vllm_ascend.ops  # noqa: F401
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    from oscar_ascend.ops.cv_dispatch import DECODE_BUNDLE_OPS
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    require_capabilities(DECODE_BUNDLE_OPS|{'rotate_out','rotate_clip_store_striped_out','prepare_attention_tasks_out','merge_lse_out','status_guard'})
    rows=[];device=torch.device('npu:0');cores=reuse._core_count(torch,target)
    for case in case_plan():
        rows.append(run_case(torch,target,acceptance,device,cores,case));gc.collect();torch.npu.empty_cache()
        if rows[-1]['status']!='passed':raise RuntimeError('partition regression: '+case.name)
    report={'status':'passed','precision':'passed','performance':'passed','store_bitwise':'passed',
            'device_completion':'passed','cases':rows,'artifact_signature':manifest['signature'],
            'artifact_sha256':manifest['sha256'],'graph':'not_run_eager_scope','full_model_performance':'not_established'}
    validate_case_evidence(report,acceptance);return report


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
    print('[oscar] PERF_MIXED_DECODE_SPLIT_RESULT '+json.dumps({'status':report['status'],'report':str(args.output)}),flush=True)
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
