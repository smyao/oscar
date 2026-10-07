"""Real-NPU complete-reader A/B for the integrated decode/window/range bundle.

Archive #126/#129/#140/#145/#148/#151-155/D.4. Both sides read identical
striped bytes. Real q4/q1 graphs change inputs at fixed addresses. Local CPU
and CAModel evidence never substitutes for this gate or whole-model timing.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import statistics
import traceback

from . import probe_striped_cache as shared
from . import probe_history_reuse as reuse
from .striped_fixture import raw_to_striped
from .phase import atomic_json
from oscar_ascend.ops.cv_dispatch import OPTIMIZED_STRIPED_OPS,DECODE_BUNDLE_OPS

ROOT=Path(__file__).resolve().parents[1]


def case_plan():
    # Keep the existing meaningful long/decode/padding/cluster/error domains.
    # The denominator is the CURRENT striped operator, not the older fe0.
    return tuple(replace(case,shape=replace(case.shape,name='bundle_'+case.shape.name),
                         old=case.new,new=OPTIMIZED_STRIPED_OPS[case.new])
                 for case in shared.case_plan())


def run_case(torch,ops,case,device,cores,acceptance):
    mode='identity' if 'q1_' in case.shape.name else 'hadamard'
    fixture=reuse.make_fixture(torch,case.shape,rotation_mode=mode)
    tensors={key:value.to(device) for key,value in fixture['cpu'].items()}
    tensors['raw']=raw_to_striped(fixture).to(device)
    old=shared._allocate(torch,fixture,device,cores,case.cluster)
    new=shared._allocate(torch,fixture,device,cores,case.cluster)
    if case.new in {'attention_cv_bundle_decode_out','attention_cv_bundle_q1_out'}:
        new['workspace']=torch.empty(cores*1146880,dtype=torch.uint8,device=device)
    reuse._prepare(torch,ops,tensors,fixture,old)
    new['tasks'],new['positions']=old['tasks'],old['positions']
    def launch_old():shared._launch(ops,case.old,tensors,fixture,old,cores,case.cluster)
    def launch_new():shared._launch(ops,case.new,tensors,fixture,new,cores,case.cluster)
    for buffers,launch in ((old,launch_old),(new,launch_new)):
        reuse._poison(torch,buffers);launch()
    torch.npu.synchronize()
    invalid=case.shape.corrupt_metadata or case.shape.corrupt_qr
    reference=(shared._check_invalid(torch,old,new) if invalid else
               shared._check_valid(torch,ops,fixture,old,new,acceptance))
    timings=None;policy=acceptance['performance']
    if case.timed:
        samples={'old':[],'new':[]}
        for iteration in range(policy['warmup']+policy['repeats']):
            order=((old,launch_old,'old'),(new,launch_new,'new'))
            if iteration>=policy['warmup'] and (iteration-policy['warmup'])%2:order=order[::-1]
            for buffers,launch,label in order:
                reuse._poison(torch,buffers)
                begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                begin.record();launch();end.record();end.synchronize()
                elapsed=float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid real NPU event')
                reuse._assert_same_bits(torch,reference,buffers,'integrated bundle timed output')
                if iteration>=policy['warmup']:samples[label].append(elapsed)
        old_ms,new_ms=(statistics.median(samples[side]) for side in ('old','new'))
        timings={'old_ms':old_ms,'new_ms':new_ms,'ratio':new_ms/old_ms,'samples_ms':samples,
                 'gate':'passed' if new_ms/old_ms<=policy['max_latency_ratio'] else 'failed',
                 'scope':'complete_CV_device_events_current_source_included'}
    graph=shared._graph(torch,launch_old,launch_new,tensors,tensors,old,new,reference) if case.graph else None
    row={'case':case.shape.name,'old_operator':case.old,'new_operator':case.new,
         'rotation':mode,'same_striped_inputs':True,
         'precision':'error_contract_passed' if invalid else 'bitwise_and_frozen_oracle_passed',
         'invalid_evidence':reference if invalid else None,'timings':timings,'graph':graph,
         'status':'passed' if timings is None or timings['gate']=='passed' else 'failed'}
    print('[oscar] PERF_DECODE_BUNDLE '+json.dumps({'case':row['case'],'status':row['status'],
        'precision':row['precision'],'old_ms':timings['old_ms'] if timings else None,
        'new_ms':timings['new_ms'] if timings else None,'ratio':timings['ratio'] if timings else None,
        'graph':graph['replay'] if graph else 'not_run'},sort_keys=True),flush=True)
    return row


def validate_case_evidence(report,acceptance):
    expected={case.shape.name:case for case in case_plan()};rows=report.get('cases')
    if not isinstance(rows,list) or len(rows)!=len(expected) or any(not isinstance(r,dict) for r in rows):
        raise RuntimeError('decode-bundle gate requires every declared case')
    names=[row.get('case') for row in rows]
    if any(not isinstance(n,str) for n in names) or set(names)!=set(expected):
        raise RuntimeError('decode-bundle cases missing, duplicated or unexpected')
    policy=acceptance['performance']
    for row in rows:
        case=expected[row['case']];invalid=case.shape.corrupt_metadata or case.shape.corrupt_qr
        precision='error_contract_passed' if invalid else 'bitwise_and_frozen_oracle_passed'
        if (row.get('status')!='passed' or row.get('precision')!=precision or
            row.get('old_operator')!=case.old or row.get('new_operator')!=case.new or
            row.get('same_striped_inputs') is not True or
            row.get('rotation')!=('identity' if 'q1_' in row['case'] else 'hadamard')):
            raise RuntimeError('decode-bundle route/format/precision evidence differs')
        if invalid and row.get('invalid_evidence')!={'error_task_parity':'passed','nan_classification':'passed'}:
            raise RuntimeError('decode-bundle error domain evidence missing')
        if case.timed:
            timing=row.get('timings',{});medians={}
            for side in ('old','new'):
                samples=timing.get('samples_ms',{}).get(side)
                if (not isinstance(samples,list) or len(samples)!=policy['repeats'] or
                    any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in samples)):
                    raise RuntimeError('decode-bundle timing samples incomplete')
                medians[side]=statistics.median(samples)
                if not math.isclose(timing.get(side+'_ms',-1),medians[side],rel_tol=1e-12):
                    raise RuntimeError('decode-bundle timing median mismatch')
            ratio=medians['new']/medians['old']
            if (timing.get('gate')!='passed' or ratio>policy['max_latency_ratio'] or
                not math.isclose(timing.get('ratio',-1),ratio,rel_tol=1e-12)):
                raise RuntimeError('decode-bundle device latency gate failed')
        if case.graph:
            graph=row.get('graph',{})
            required={'capture':'passed','replay':'passed','same_address_changed_key_value_alone':True,
                      'same_address_changed_query_key_value_task':True}
            if any(graph.get(k)!=v for k,v in required.items()):
                raise RuntimeError('decode-bundle changed-input graph evidence missing')


def probe(config_path,acceptance_path):
    target=json.loads(config_path.read_text());acceptance=json.loads(acceptance_path.read_text())
    if target.get('experimental_decode_bundle') is not True or target.get('experimental_striped_cache') is not True:
        raise RuntimeError('requires explicit decode and striped bundles')
    if acceptance.get('frozen_before_measurement') is not True:raise RuntimeError('frozen acceptance required')
    policy=acceptance['performance']
    if (policy['warmup'],policy['repeats'],policy['max_latency_ratio'])!=(2,5,1.0):
        raise RuntimeError('frozen performance gate changed')
    reuse._active_device(target)
    import torch
    import torch_npu  # noqa: F401 -- real NPU only
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if not torch.npu.is_available():raise RuntimeError('decode-bundle gate requires NPU')
    torch.npu.set_device(0)
    from .probe_native_current_fia import _target_geometry
    from .build_ops import normalize_soc
    if _target_geometry(target)!=(6,1,256):raise RuntimeError('decode bundle requires target Hq6/Hkv1/D256')
    if normalize_soc(torch.npu.get_device_name(0))!=target['soc_version']:
        raise RuntimeError('selected NPU does not match the explicit target SoC')
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    require_capabilities(DECODE_BUNDLE_OPS|{'prepare_attention_tasks_out','merge_lse_out'})
    device=torch.device('npu:0');cores=reuse._core_count(torch,target);rows=[]
    for case in case_plan():
        rows.append(run_case(torch,torch.ops.oscar_ascend_ops,case,device,cores,acceptance))
        gc.collect();torch.npu.empty_cache()
        if rows[-1]['status']!='passed':raise RuntimeError('integrated bundle regressed '+case.shape.name)
    report={'status':'passed','precision':'passed','performance':'passed','graph_capture':'passed',
            'graph_replay':'passed','device_completion':'passed','cases':rows,
            'artifact_signature':manifest['signature'],'artifact_sha256':manifest['sha256'],
            'full_model_performance':'not_established','model_quality':'not_established'}
    validate_case_evidence(report,acceptance)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/target.json')
    parser.add_argument('--acceptance',type=Path,default=ROOT/'configs/acceptance.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:report=probe(args.config,args.acceptance)
    except Exception as error:
        traceback.print_exc();report={'status':'failed','first_error':str(error)}
    atomic_json(args.output,report)
    print('[oscar] PERF_DECODE_BUNDLE_RESULT '+json.dumps({'status':report['status'],'report':str(args.output)}),flush=True)
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
