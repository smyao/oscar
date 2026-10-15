"""Archive #126/#129/#140/#145/#148/#150/#154-156 and startup D.4: hybrid L1 q4 gate.

D.4: this measures the complete bounded INT2 CV + original merge + BF16 copy,
not a helper or a full-history restoration. Both arms preserve the same tasks,
three sources, FP32 math and error domain. Frozen oracle/bitwise and changed
input graphs precede the declared complete-path >=1.0 non-regression device gates. CPU fixtures and
host orchestration tests never substitute for target NPU completion.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

from . import probe_mdl_decode as mdl
from . import probe_history_reuse as reuse
from . import probe_striped_cache as shared
from .phase import atomic_json
from .striped_fixture import raw_to_striped

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'attention_cv_mdl_decode_out'
CANDIDATE = 'attention_cv_hybrid_l1_decode_out'
OPERATOR_FAMILIES = {'q4': {'baseline': BASELINE, 'candidate': CANDIDATE}}
WORKSPACE = 1146880
# User 2026-10-15 supersedes the historical per-point 1.15 rule.
# This does not change frozen precision or native E2E >30% acceptance.
MIN_SPEEDUP = 1.0
CONTEXTS = (20000, 23000, 27000, 30000)
SCOPE = 'complete_all_sources_CV_merge_and_BF16_copy_device_events'
SOURCE_PATHS = (
    'csrc/kernels/attention_cv_hybrid_l1_decode.cpp',
    'csrc/kernels/attention_hybrid_l1_unpack.h',
    'csrc/include/oscar_hybrid_l1_decode_launch.h',
    'csrc/kernels/attention_cv_mdl_decode.cpp',
    'csrc/kernels/attention_striped_unpack_padded.h',
    'csrc/kernels/attention_range_mask.h',
    'csrc/kernels/attention_tasks.cpp',
    'csrc/kernels/merge_lse.cpp',
)
RV_FORMAT = 'FP32_NZ_C0_8_rows256'
RV_EXTRA_BYTES = 262144



@dataclass(frozen=True)
class Case:
    shape: reuse.Shape
    special_values: bool = False
    timed: bool = False
    graph: bool = False
    minimum_speedup: float = 1.0
    rotation_mode: str = "nontrivial"


def case_plan():
    short = reuse.Shape('short_q4_tail_padding', (4, 4), (1341, 1373),
                        256, 1, 1, False, padded_tokens=12)
    return (
        Case(short, special_values=True, graph=True, timed=True),
        Case(replace(short, name='invalid_live_metadata', corrupt_metadata=True,
                     corrupt_position=80)),
        Case(replace(short, name='invalid_live_qr', corrupt_qr=True, corrupt_qr_token=1)),
        Case(replace(short, name='invalid_dead_tail', corrupt_dead_tail=True)),
        Case(reuse.Shape('q4_R4_20k30k', (4,) * 4, CONTEXTS, 256, 1, 3,
                         False, padded_tokens=32), timed=True, minimum_speedup=MIN_SPEEDUP),
        # Current 32K/64K acceptance contexts, still one bounded synthetic
        # operator pair. This does not request another full AISBench run.
        Case(reuse.Shape('q4_R2_32k64k', (4,4), (32768,65536), 256,1,3,
                         False,padded_tokens=16), timed=True,minimum_speedup=MIN_SPEEDUP),
        Case(reuse.Shape('short_mixed_q1_q2_q3_q4', (1,2,3,4), (511,641,1025,1341),
                         256,1,1,False,padded_tokens=16)),
        Case(replace(short,name='short_q4_identity'),rotation_mode='identity'),
    )


def operators(case):
    if case.shape.slot_context:
        raise ValueError('hybrid L1 gate does not replace later-draft q1')
    return BASELINE, CANDIDATE


def pack_rotation_nz(torch, rotation):
    if rotation.dtype != torch.float32 or tuple(rotation.shape) != (256,256):
        raise ValueError('hybrid L1 Rv requires FP32 [256,256]')
    # CPU diagnostic only. Production packs once while initializing LayerState.
    return rotation.reshape(256,32,8).permute(1,0,2).contiguous().view(256,256)


def source_binding(root=ROOT):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def make_fixture(torch, case):
    def special(request, begin, q, k, v, ck, cv):
        # Store-reachable finite boundary values. Callback executes before the
        # independent encoder/oracle; QR is rebuilt below after changing Q.
        if case.special_values:
            q[0, 0, 0] = -0.0
            ck[0, 0, 0] = 0.0
            cv[0, 0, 0] = -0.0
            k[80, 0, 0] = torch.finfo(torch.bfloat16).tiny
            v[81, 0, 0] = -torch.finfo(torch.bfloat16).tiny
    fixture = reuse.make_fixture(torch, case.shape, on_request=special,
                                 rotation_mode=case.rotation_mode)
    if case.special_values:
        fixture['cpu']['qr'] = (fixture['cpu']['q'].float() @
                                (torch.eye(256) if case.rotation_mode == 'identity' else reuse._hadamard(torch, 256))).contiguous()
    fixture['cpu']['raw'] = raw_to_striped(fixture)
    fixture['cpu']['rv_nz'] = pack_rotation_nz(torch, fixture['cpu']['rv'])
    fixture['input_sha256'] = reuse._fingerprint(fixture['cpu'])
    return fixture


remap_physical_pages = mdl.remap_physical_pages


allocate = mdl.allocate


def launch(ops, name, tensors, fixture, buffers, cores, *, merge=True):
    if name not in (BASELINE,CANDIDATE):
        raise ValueError('unqualified hybrid L1 operator: '+name)
    arm_tensors = tensors if name == BASELINE else dict(tensors,rv=tensors['rv_nz'])
    shared._launch(ops, name, arm_tensors, fixture, buffers, cores, 1)
    if merge:
        n, h, d, s = fixture['tokens'], fixture['spec'].heads, fixture['spec'].dim, 3 * fixture['spec'].splits
        ops.merge_lse_out(buffers['partial'].view(n * h, s, d),
                          buffers['lse'].view(n * h, s), buffers['output'],
                          buffers['output_lse'], buffers['merge_status'])
        buffers['bf16'].copy_(buffers['output'])


poison = mdl.poison


snapshot = mdl.snapshot


same_bits = mdl.same_bits


check = mdl.check


task_evidence = mdl.task_evidence


def graph_gate(torch, ops, case, fixture, tensors, pair, cores, tolerance):
    addresses = {key: value.data_ptr() for key, value in tensors.items()}
    addresses.update({str(i)+key: value.data_ptr() for i,b in enumerate(pair)
                      for key,value in b.items() if value is not None})
    def execute(i):
        launch(ops, operators(case)[i], tensors, fixture, pair[i], cores)
    graphs = [torch.npu.NPUGraph(), torch.npu.NPUGraph()]
    for i, graph in enumerate(graphs):
        with torch.npu.graph(graph, capture_error_mode='thread_local', auto_dispatch_capture=True):
            execute(i)
    states = []
    def replay(label):
        for buffers in pair:poison(torch,buffers)
        for graph in graphs:graph.replay()
        torch.npu.synchronize()
        task = task_evidence(torch,fixture,pair[0])
        detail = check(torch,fixture,*pair,task['task_sha256'],tolerance)
        captured = snapshot(pair[1])
        for i,buffers in enumerate(pair):poison(torch,buffers);execute(i)
        torch.npu.synchronize();check(torch,fixture,*pair,task['task_sha256'],tolerance)
        same_bits(torch,captured,pair[1])
        states.append({'state':label,'contexts':list(fixture['spec'].contexts),**task,**detail,
                       'raw_sha256':reuse._fingerprint({'raw':fixture['cpu']['raw']}),
                       'table_sha256':reuse._fingerprint({'table':fixture['cpu']['table']})})
        return captured
    initial = replay('captured_inputs')
    fixture['cpu']['q'].neg_();fixture['cpu']['qr'].neg_()
    # Rebuild independent oracle for -Q with identical K/V from deterministic
    # fixture construction. No device output enters the oracle.
    def negative_q(request,begin,q,k,v,ck,cv):
        if case.special_values:
            q[0,0,0]=-0.0;ck[0,0,0]=0.0;cv[0,0,0]=-0.0
            k[80,0,0]=torch.finfo(torch.bfloat16).tiny;v[81,0,0]=-torch.finfo(torch.bfloat16).tiny
        q.neg_()
    expected = reuse.make_fixture(torch,case.shape,on_request=negative_q,rotation_mode=case.rotation_mode)
    fixture['expected']=expected['expected']
    tensors['q'].copy_(fixture['cpu']['q']);tensors['qr'].copy_(fixture['cpu']['qr'])
    changed = replay('changed_q_qr')
    if shared._store_bits_equal(torch,initial['output'],changed['output']):
        raise RuntimeError('hybrid L1 graph ignored changed query')
    replacement = make_fixture(torch,replace(case,shape=replace(case.shape,
        contexts=tuple(value+16 for value in case.shape.contexts))))
    remap_physical_pages(torch,replacement)
    for name,tensor in tensors.items():
        value=replacement['cpu'][name]
        if value.shape!=tensor.shape or value.dtype!=tensor.dtype:
            raise RuntimeError('hybrid L1 graph metadata replacement requires allocation: '+name)
        tensor.copy_(value)
    fixture=replacement
    receipt=reuse._prepare(torch,ops,tensors,fixture,pair[0])
    pair[1]['tasks'],pair[1]['positions']=pair[0]['tasks'],pair[0]['positions']
    final=replay('changed_context_cache_table')
    if shared._store_bits_equal(torch,changed['output'],final['output']):
        raise RuntimeError('hybrid L1 graph ignored changed cache/context')
    current={key:value.data_ptr() for key,value in tensors.items()}
    current.update({str(i)+key:value.data_ptr() for i,b in enumerate(pair)
                    for key,value in b.items() if value is not None})
    if current!=addresses:raise RuntimeError('hybrid L1 graph tensor addresses changed')
    if any(states[0][key]==states[2][key] for key in ('task_sha256','raw_sha256','table_sha256')):
        raise RuntimeError('hybrid L1 graph failed to vary actual task/cache/table inputs')
    return {'capture':'passed','replay':'passed','same_addresses':True,
            'q_changed':True,'cache_table_tasks_changed':True,'eager_bitwise':'passed',
            'independent_oracle':'passed','states':states}


def run_case(torch,ops,case,device,cores,tolerance):
    baseline,candidate=operators(case)
    fixture=make_fixture(torch,case);tensors={k:v.to(device) for k,v in fixture['cpu'].items()}
    pair=[allocate(torch,fixture,device,cores) for _ in range(2)]
    receipt=reuse._prepare(torch,ops,tensors,fixture,pair[0])
    pair[1]['tasks'],pair[1]['positions']=pair[0]['tasks'],pair[0]['positions']
    task=task_evidence(torch,fixture,pair[0])
    invalid=case.shape.corrupt_metadata or case.shape.corrupt_qr
    def execute(i):launch(ops,(baseline,candidate)[i],tensors,fixture,pair[i],cores,merge=not invalid)
    for i,b in enumerate(pair):poison(torch,b);execute(i)
    torch.npu.synchronize();detail=check(torch,fixture,*pair,receipt['task_sha256'],tolerance)
    timing=None
    if case.timed:
        reference=snapshot(pair[0]);samples={baseline:[],candidate:[]}
        for iteration in range(7):
            order=(0,1) if iteration<2 or (iteration-2)%2==0 else (1,0)
            for i in order:
                poison(torch,pair[i])
                begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                begin.record();execute(i);end.record();end.synchronize()
                elapsed=float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('hybrid L1 invalid device event time')
                same_bits(torch,reference,pair[i])
                if iteration>=2:samples[(baseline,candidate)[i]].append(elapsed)
        a,b=(statistics.median(samples[name]) for name in (baseline,candidate))
        timing={'baseline_ms':a,'candidate_ms':b,'speedup':a/b,'minimum_speedup':case.minimum_speedup,
                'ratio':b/a,'samples_ms':samples,'warmup':2,'repeats':5,'scope':SCOPE,
                'gate':'passed' if a/b>=case.minimum_speedup else 'failed'}
    graph=graph_gate(torch,ops,case,fixture,tensors,pair,cores,tolerance) if case.graph else None
    return {'case':case.shape.name,'status':'passed' if timing is None or timing['gate']=='passed' else 'failed',
            'baseline_operator':baseline,'candidate_operator':candidate,
            'contexts':list(case.shape.contexts),'qlens':list(case.shape.qlens),
            'input_sha256':fixture['input_sha256'],'special_values':case.special_values,'rotation_mode':case.rotation_mode,
            **task,**detail,'timing':timing,'graph':graph}


def validate_evidence(report):
    expected={'status':'passed','baseline_operator':BASELINE,'candidate_operator':CANDIDATE,
              'operator_families':OPERATOR_FAMILIES,
              'precision':'passed','error_domain':'passed','graph_capture':'passed',
              'graph_replay':'passed','performance':'passed','device_completion':'passed',
              'scope':SCOPE,'workspace_bytes_per_core':WORKSPACE,
              'rotation_format':RV_FORMAT,'rotation_extra_bytes_per_layer':RV_EXTRA_BYTES}
    bad=[key for key,value in expected.items() if not isinstance(report,dict) or report.get(key)!=value]
    rows=report.get('cases',[]) if isinstance(report,dict) else []
    plan=case_plan()
    if not isinstance(rows,list) or len(rows)!=len(plan) or any(not isinstance(row,dict) for row in rows):
        return bad+['cases']
    for case,row in zip(plan,rows):
        baseline,candidate=operators(case)
        shape=case.shape;invalid=shape.corrupt_metadata or shape.corrupt_qr
        expected_case={'case':shape.name,'status':'passed','contexts':list(shape.contexts),'qlens':list(shape.qlens),
            'baseline_operator':baseline,'candidate_operator':candidate,
            'all_tasks':(shape.padded_tokens or sum(shape.qlens))*3*shape.splits,
            'active_queries':sum(shape.qlens),'padded_queries':(shape.padded_tokens or sum(shape.qlens))-sum(shape.qlens),
            'queries':0 if invalid else sum(shape.qlens),'special_values':case.special_values,'rotation_mode':case.rotation_mode,
            'oracle':'expected_error_domain' if invalid else 'all_actual_queries_raw_space_passed',
            'precision':'error_codes_finite_bits_nonfinite_classification' if invalid else 'partial_lse_status_merge_bf16_bitwise'}
        if any(row.get(k)!=v for k,v in expected_case.items()):bad.append(shape.name+'.contract')
        leaders=row.get('source_leaders',{})
        if set(leaders)!= {'0','1','2'} or any(type(v) is not int or v<=0 for v in leaders.values()):
            bad.append(shape.name+'.sources')
        if case.timed:
            timing=row.get('timing') or {};samples=timing.get('samples_ms',{})
            valid=set(samples)=={baseline,candidate} and all(isinstance(samples[n],list) and len(samples[n])==5 and
                all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in samples[n]) for n in samples)
            if valid:
                a,b=(statistics.median(samples[n]) for n in (baseline,candidate))
                valid=(a/b>=case.minimum_speedup and all(timing.get(k)==v for k,v in
                    {'baseline_ms':a,'candidate_ms':b,'ratio':b/a,'speedup':a/b,'minimum_speedup':case.minimum_speedup,
                     'warmup':2,'repeats':5,'gate':'passed','scope':SCOPE}.items()))
            if not valid:bad.append(shape.name+'.device_latency')
        if case.graph:
            graph=row.get('graph') or {}
            if any(graph.get(k)!=v for k,v in {'capture':'passed','replay':'passed','same_addresses':True,
                'q_changed':True,'cache_table_tasks_changed':True,'eager_bitwise':'passed','independent_oracle':'passed'}.items()):
                bad.append(shape.name+'.graph')
            states=graph.get('states',[])
            if len(states)!=3 or any(s.get('oracle')!='all_actual_queries_raw_space_passed' or s.get('queries')!=sum(shape.qlens) for s in states):
                bad.append(shape.name+'.graph_oracle')
            elif any(states[0].get(k)==states[2].get(k) for k in ('task_sha256','raw_sha256','table_sha256')):
                bad.append(shape.name+'.graph_inputs')
    return bad


def probe(config_path,acceptance_path):
    config=json.loads(config_path.read_text());acceptance=json.loads(acceptance_path.read_text())
    if any(config.get(key) is not True for key in ('experimental_hybrid_l1_decode','experimental_mdl_decode','experimental_decode_bundle','experimental_striped_cache')):
        raise RuntimeError('hybrid L1 gate requires explicit MDL/decode/striped candidate')
    if (acceptance.get('frozen_before_measurement') is not True or
        acceptance['fused_attention']!={'atol':0.005,'rtol':0.005} or
        acceptance['performance']!={'max_latency_ratio':1.0,'min_throughput_ratio':1.0,'repeats':5,
        'warmup':2,'statistic':'median','per_case_required':True,'noise_allowance':0.0}):
        raise RuntimeError('hybrid L1 frozen accuracy/performance contract changed')
    reuse._active_device(config)
    import torch
    import torch_npu  # noqa: F401
    if not torch.npu.is_available():raise RuntimeError('hybrid L1 gate requires real NPU; no CPU substitution')
    torch.npu.set_device(0);torch.set_num_threads(min(4,torch.get_num_threads()))
    from .probe_native_current_fia import _target_geometry
    from .build_ops import normalize_soc
    if _target_geometry(config)!=(6,1,256):raise RuntimeError('hybrid L1 gate requires Hq6/Hkv1/D256')
    if normalize_soc(torch.npu.get_device_name(0))!=config['soc_version']:raise RuntimeError('hybrid L1 target SoC differs')
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    require_capabilities({BASELINE,CANDIDATE,
                          'prepare_attention_tasks_out','merge_lse_out'})
    bindings=source_binding();rows=[];cores=reuse._core_count(torch,config)
    for case in case_plan():
        row=run_case(torch,torch.ops.oscar_ascend_ops,case,torch.device('npu:0'),cores,acceptance['fused_attention'])
        rows.append(row)
        print('[oscar] PERF_HYBRID_L1_DECODE '+json.dumps({'case':row['case'],'status':row['status'],
              'operator':row['candidate_operator'],
              'precision':row['precision'],'speedup':row['timing']['speedup'] if row['timing'] else None,
              'minimum_speedup':case.minimum_speedup,'graph':row['graph']['replay'] if row['graph'] else 'not_run'}),flush=True)
        gc.collect();torch.npu.empty_cache()
    passed=all(row['status']=='passed' for row in rows)
    report={'status':'passed' if passed else 'failed','baseline_operator':BASELINE,'candidate_operator':CANDIDATE,
        'operator_families':OPERATOR_FAMILIES,
        'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
        'performance':'passed' if passed else 'failed','device_completion':'passed','scope':SCOPE,
        'workspace_bytes_per_core':WORKSPACE,'source_binding':bindings,'cases':rows,
        'rotation_format':RV_FORMAT,'rotation_extra_bytes_per_layer':RV_EXTRA_BYTES,
        'artifact_signature':manifest['signature'],'artifact_sha256':manifest['sha256'],
        'whole_service_performance':'not_established','model_quality':'not_established'}
    mismatches=validate_evidence(report)
    if mismatches:report.update(status='failed',evidence_mismatches=mismatches)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,default=ROOT/'configs/acceptance.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:report=probe(args.config,args.acceptance)
    except Exception as error:
        traceback.print_exc();report={'status':'failed','first_error':str(error)}
    atomic_json(args.output,report)
    print('[oscar] PERF_HYBRID_L1_RESULT '+json.dumps({'status':report['status'],'report':str(args.output)}),flush=True)
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
