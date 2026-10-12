"""Archive #126/#129/#140/#145/#148/#150/#154-156 and startup D.4: MDL gate.

D.4: this measures the complete bounded INT2 CV + original merge + BF16 copy,
not a helper or a full-history restoration. Both arms preserve the same tasks,
three sources, FP32 math and error domain. Frozen oracle/bitwise and changed
input graphs precede the declared >=1.15 device speed gate. CPU fixtures and
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

from . import probe_history_reuse as reuse
from . import probe_striped_cache as shared
from .phase import atomic_json
from .striped_fixture import raw_to_striped

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'attention_cv_bundle_decode_out'
CANDIDATE = 'attention_cv_mdl_decode_out'
WORKSPACE = 1146880
MIN_SPEEDUP = 1.15
CONTEXTS = (20000, 23000, 27000, 30000)
SCOPE = 'complete_all_sources_CV_merge_and_BF16_copy_device_events'
SOURCE_PATHS = (
    'csrc/kernels/attention_cv_mdl_decode.cpp',
    'csrc/kernels/attention_cv_decode_bundle.cpp',
    'csrc/kernels/attention_striped_unpack_padded.h',
    'csrc/kernels/attention_range_mask.h',
    'csrc/kernels/attention_tasks.cpp',
    'csrc/kernels/merge_lse.cpp',
)


@dataclass(frozen=True)
class Case:
    shape: reuse.Shape
    special_values: bool = False
    timed: bool = False
    graph: bool = False


def case_plan():
    short = reuse.Shape('short_q4_tail_padding', (4, 4), (1341, 1373),
                        256, 1, 1, False, padded_tokens=12)
    return (
        Case(short, special_values=True, graph=True),
        Case(replace(short, name='invalid_live_metadata', corrupt_metadata=True,
                     corrupt_position=80)),
        Case(replace(short, name='invalid_live_qr', corrupt_qr=True, corrupt_qr_token=1)),
        Case(replace(short, name='invalid_dead_tail', corrupt_dead_tail=True)),
        Case(reuse.Shape('q4_R4_20k30k', (4,) * 4, CONTEXTS, 256, 1, 3,
                         False, padded_tokens=32), timed=True),
    )


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
                                 rotation_mode='hadamard')
    if case.special_values:
        fixture['cpu']['qr'] = (fixture['cpu']['q'].float() @
                                reuse._hadamard(torch, 256)).contiguous()
    fixture['cpu']['raw'] = raw_to_striped(fixture)
    fixture['input_sha256'] = reuse._fingerprint(fixture['cpu'])
    return fixture


def remap_physical_pages(torch, fixture):
    """Diagnostic relocation changes every page ID, preserving logical KV."""
    count = fixture['blocks']
    if count < 2:
        raise RuntimeError('graph relocation requires multiple physical pages')
    cpu = fixture['cpu']
    mapping = (torch.arange(count) + 1) % count
    pages = cpu['raw'][reuse.PREFIX:].view(count, fixture['stride']).clone()
    cpu['raw'][reuse.PREFIX:].view(count, fixture['stride'])[mapping] = pages
    for name in ('wk', 'wv', 'tags'):
        value = cpu[name].clone()
        cpu[name][mapping] = value
    table = cpu['table'];live = table >= 0
    subdivisions = reuse.BLOCK_TOKENS // 128
    table[live] = (mapping[table[live].long() // subdivisions] * subdivisions +
                   table[live] % subdivisions).to(table.dtype)
    slots = cpu['slots'];live = slots >= 0
    slots[live] = mapping[slots[live] // reuse.BLOCK_TOKENS] * reuse.BLOCK_TOKENS + slots[live] % reuse.BLOCK_TOKENS
    fixture['page_assignments'] = [tuple(int(mapping[x]) for x in pages)
                                   for pages in fixture['page_assignments']]
    fixture['input_sha256'] = reuse._fingerprint(cpu)


def allocate(torch, fixture, device, cores):
    buffers = reuse._allocate(torch, fixture, device, cores, candidate=False)
    buffers['workspace'] = torch.empty(cores * WORKSPACE, dtype=torch.uint8, device=device)
    n, h, d = fixture['tokens'], fixture['spec'].heads, fixture['spec'].dim
    buffers.update(output=torch.empty((n * h, d), dtype=torch.float32, device=device),
                   output_lse=torch.empty(n * h, dtype=torch.float32, device=device),
                   merge_status=torch.empty(n * h, dtype=torch.int32, device=device),
                   bf16=torch.empty((n * h, d), dtype=torch.bfloat16, device=device))
    return buffers


def launch(ops, name, tensors, fixture, buffers, cores, *, merge=True):
    shared._launch(ops, name, tensors, fixture, buffers, cores, 1)
    if merge:
        n, h, d, s = fixture['tokens'], fixture['spec'].heads, fixture['spec'].dim, 3 * fixture['spec'].splits
        ops.merge_lse_out(buffers['partial'].view(n * h, s, d),
                          buffers['lse'].view(n * h, s), buffers['output'],
                          buffers['output_lse'], buffers['merge_status'])
        buffers['bf16'].copy_(buffers['output'])


def poison(torch, buffers):
    reuse._poison(torch, buffers)
    buffers['output'].fill_(float('nan'));buffers['output_lse'].fill_(float('nan'))
    buffers['merge_status'].fill_(-99);buffers['bf16'].fill_(float('nan'))


def snapshot(buffers, *, merge=True):
    names = ('partial', 'lse', 'status') + (('output', 'output_lse', 'merge_status', 'bf16') if merge else ())
    return {name: buffers[name].clone() for name in names}


def same_bits(torch, reference, buffers):
    for name, tensor in reference.items():
        if not shared._store_bits_equal(torch, tensor, buffers[name]):
            raise RuntimeError('MDL baseline/candidate or repeated result differs bitwise: ' + name)


def check(torch, fixture, old, new, task_hash, tolerance):
    invalid = fixture['spec'].corrupt_metadata or fixture['spec'].corrupt_qr
    for buffers in (old, new):
        reuse._check_status(torch, buffers, invalid=invalid)
        if reuse._fingerprint({'tasks': buffers['tasks'].cpu()}) != task_hash:
            raise RuntimeError('MDL modified prepared tasks')
    if invalid:
        shared._check_invalid(torch, old, new)
        # Same MDL arithmetic and lanes: require full old status codes, not
        # merely any-error parity. Non-finite payload classification is checked
        # separately; NaN payload identity is not a numerical contract.
        if not torch.equal(old['status'], new['status']):
            raise RuntimeError('MDL changed exact error codes/lane ownership')
        return {'oracle': 'expected_error_domain', 'queries': 0,
                'precision': 'error_codes_finite_bits_nonfinite_classification'}
    same_bits(torch, snapshot(old), new)
    indexes = sorted(fixture['expected'])
    if indexes != list(range(fixture['actual_tokens'])):
        raise RuntimeError('MDL independent oracle omitted actual query rows')
    n, h, d = fixture['tokens'], fixture['spec'].heads, fixture['spec'].dim
    expected = torch.stack([fixture['expected'][i][0] for i in indexes])
    expected_lse = torch.stack([fixture['expected'][i][1] for i in indexes])
    for buffers in (old, new):
        if not bool((buffers['merge_status'].cpu() == 0).all()):
            raise RuntimeError('MDL original merge did not complete every padded row')
        torch.testing.assert_close(buffers['output'].view(n,h,d)[indexes].cpu(), expected, **tolerance)
        torch.testing.assert_close(buffers['output_lse'].view(n,h)[indexes].cpu(), expected_lse, **tolerance)
        torch.testing.assert_close(buffers['bf16'].view(n,h,d)[indexes].float().cpu(), expected, **tolerance)
    return {'oracle': 'all_actual_queries_raw_space_passed', 'queries': len(indexes),
            'precision': 'partial_lse_status_merge_bf16_bitwise'}


def task_evidence(torch, fixture, buffers):
    tasks = buffers['tasks'].cpu()
    count = fixture['tokens'] * fixture['spec'].kv_heads * 3 * fixture['spec'].splits
    if tuple(tasks.shape) != (count, 16):
        raise RuntimeError('MDL all-task geometry changed')
    source_counts = {str(source): int(((tasks[:, 7] == source) & (tasks[:, 1] > 0) &
                      (tasks[:, 4] > tasks[:, 3])).sum()) for source in range(3)}
    if any(value == 0 for value in source_counts.values()):
        raise RuntimeError('MDL gate omitted history, exact window or current source')
    return {'all_tasks': count, 'active_queries': fixture['actual_tokens'],
            'padded_queries': fixture['tokens'] - fixture['actual_tokens'],
            'source_leaders': source_counts, 'task_sha256': reuse._fingerprint({'tasks': tasks})}


def graph_gate(torch, ops, case, fixture, tensors, pair, cores, tolerance):
    addresses = {key: value.data_ptr() for key, value in tensors.items()}
    addresses.update({str(i)+key: value.data_ptr() for i,b in enumerate(pair)
                      for key,value in b.items() if value is not None})
    def execute(i):
        launch(ops, (BASELINE,CANDIDATE)[i], tensors, fixture, pair[i], cores)
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
    expected = reuse.make_fixture(torch,case.shape,on_request=negative_q,rotation_mode='hadamard')
    fixture['expected']=expected['expected']
    tensors['q'].copy_(fixture['cpu']['q']);tensors['qr'].copy_(fixture['cpu']['qr'])
    changed = replay('changed_q_qr')
    if shared._store_bits_equal(torch,initial['output'],changed['output']):
        raise RuntimeError('MDL graph ignored changed query')
    replacement = make_fixture(torch,replace(case,shape=replace(case.shape,
        contexts=tuple(value+16 for value in case.shape.contexts))))
    remap_physical_pages(torch,replacement)
    for name,tensor in tensors.items():
        value=replacement['cpu'][name]
        if value.shape!=tensor.shape or value.dtype!=tensor.dtype:
            raise RuntimeError('MDL graph metadata replacement requires allocation: '+name)
        tensor.copy_(value)
    fixture=replacement
    receipt=reuse._prepare(torch,ops,tensors,fixture,pair[0])
    pair[1]['tasks'],pair[1]['positions']=pair[0]['tasks'],pair[0]['positions']
    final=replay('changed_context_cache_table')
    if shared._store_bits_equal(torch,changed['output'],final['output']):
        raise RuntimeError('MDL graph ignored changed cache/context')
    current={key:value.data_ptr() for key,value in tensors.items()}
    current.update({str(i)+key:value.data_ptr() for i,b in enumerate(pair)
                    for key,value in b.items() if value is not None})
    if current!=addresses:raise RuntimeError('MDL graph tensor addresses changed')
    if any(states[0][key]==states[2][key] for key in ('task_sha256','raw_sha256','table_sha256')):
        raise RuntimeError('MDL graph failed to vary actual task/cache/table inputs')
    return {'capture':'passed','replay':'passed','same_addresses':True,
            'q_changed':True,'cache_table_tasks_changed':True,'eager_bitwise':'passed',
            'independent_oracle':'passed','states':states}


def run_case(torch,ops,case,device,cores,tolerance):
    fixture=make_fixture(torch,case);tensors={k:v.to(device) for k,v in fixture['cpu'].items()}
    pair=[allocate(torch,fixture,device,cores) for _ in range(2)]
    receipt=reuse._prepare(torch,ops,tensors,fixture,pair[0])
    pair[1]['tasks'],pair[1]['positions']=pair[0]['tasks'],pair[0]['positions']
    task=task_evidence(torch,fixture,pair[0])
    invalid=case.shape.corrupt_metadata or case.shape.corrupt_qr
    def execute(i):launch(ops,(BASELINE,CANDIDATE)[i],tensors,fixture,pair[i],cores,merge=not invalid)
    for i,b in enumerate(pair):poison(torch,b);execute(i)
    torch.npu.synchronize();detail=check(torch,fixture,*pair,receipt['task_sha256'],tolerance)
    timing=None
    if case.timed:
        reference=snapshot(pair[0]);samples={BASELINE:[],CANDIDATE:[]}
        for iteration in range(7):
            order=(0,1) if iteration<2 or (iteration-2)%2==0 else (1,0)
            for i in order:
                poison(torch,pair[i])
                begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                begin.record();execute(i);end.record();end.synchronize()
                elapsed=float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('MDL invalid device event time')
                same_bits(torch,reference,pair[i])
                if iteration>=2:samples[(BASELINE,CANDIDATE)[i]].append(elapsed)
        a,b=(statistics.median(samples[name]) for name in (BASELINE,CANDIDATE))
        timing={'baseline_ms':a,'candidate_ms':b,'speedup':a/b,'minimum_speedup':MIN_SPEEDUP,
                'ratio':b/a,'samples_ms':samples,'warmup':2,'repeats':5,'scope':SCOPE,
                'gate':'passed' if a/b>=MIN_SPEEDUP else 'failed'}
    graph=graph_gate(torch,ops,case,fixture,tensors,pair,cores,tolerance) if case.graph else None
    return {'case':case.shape.name,'status':'passed' if timing is None or timing['gate']=='passed' else 'failed',
            'contexts':list(case.shape.contexts),'qlens':list(case.shape.qlens),
            'input_sha256':fixture['input_sha256'],'special_values':case.special_values,
            **task,**detail,'timing':timing,'graph':graph}


def validate_evidence(report):
    expected={'status':'passed','baseline_operator':BASELINE,'candidate_operator':CANDIDATE,
              'precision':'passed','error_domain':'passed','graph_capture':'passed',
              'graph_replay':'passed','performance':'passed','device_completion':'passed',
              'scope':SCOPE,'workspace_bytes_per_core':WORKSPACE}
    bad=[key for key,value in expected.items() if not isinstance(report,dict) or report.get(key)!=value]
    rows=report.get('cases',[]) if isinstance(report,dict) else []
    plan=case_plan()
    if not isinstance(rows,list) or len(rows)!=len(plan) or any(not isinstance(row,dict) for row in rows):
        return bad+['cases']
    for case,row in zip(plan,rows):
        shape=case.shape;invalid=shape.corrupt_metadata or shape.corrupt_qr
        expected_case={'case':shape.name,'status':'passed','contexts':list(shape.contexts),'qlens':list(shape.qlens),
            'all_tasks':(shape.padded_tokens or sum(shape.qlens))*3*shape.splits,
            'active_queries':sum(shape.qlens),'padded_queries':(shape.padded_tokens or sum(shape.qlens))-sum(shape.qlens),
            'queries':0 if invalid else sum(shape.qlens),'special_values':case.special_values,
            'oracle':'expected_error_domain' if invalid else 'all_actual_queries_raw_space_passed',
            'precision':'error_codes_finite_bits_nonfinite_classification' if invalid else 'partial_lse_status_merge_bf16_bitwise'}
        if any(row.get(k)!=v for k,v in expected_case.items()):bad.append(shape.name+'.contract')
        leaders=row.get('source_leaders',{})
        if set(leaders)!= {'0','1','2'} or any(type(v) is not int or v<=0 for v in leaders.values()):
            bad.append(shape.name+'.sources')
        if case.timed:
            timing=row.get('timing') or {};samples=timing.get('samples_ms',{})
            valid=set(samples)=={BASELINE,CANDIDATE} and all(isinstance(samples[n],list) and len(samples[n])==5 and
                all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in samples[n]) for n in samples)
            if valid:
                a,b=(statistics.median(samples[n]) for n in (BASELINE,CANDIDATE))
                valid=(a/b>=MIN_SPEEDUP and all(timing.get(k)==v for k,v in
                    {'baseline_ms':a,'candidate_ms':b,'ratio':b/a,'speedup':a/b,'minimum_speedup':MIN_SPEEDUP,
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
    if any(config.get(key) is not True for key in ('experimental_mdl_decode','experimental_decode_bundle','experimental_striped_cache')):
        raise RuntimeError('MDL gate requires explicit MDL/decode/striped candidate')
    if (acceptance.get('frozen_before_measurement') is not True or
        acceptance['fused_attention']!={'atol':0.005,'rtol':0.005} or
        acceptance['performance']!={'max_latency_ratio':1.0,'min_throughput_ratio':1.0,'repeats':5,
        'warmup':2,'statistic':'median','per_case_required':True,'noise_allowance':0.0}):
        raise RuntimeError('MDL frozen accuracy/performance contract changed')
    reuse._active_device(config)
    import torch
    import torch_npu  # noqa: F401
    if not torch.npu.is_available():raise RuntimeError('MDL gate requires real NPU; no CPU substitution')
    torch.npu.set_device(0);torch.set_num_threads(min(4,torch.get_num_threads()))
    from .probe_native_current_fia import _target_geometry
    from .build_ops import normalize_soc
    if _target_geometry(config)!=(6,1,256):raise RuntimeError('MDL gate requires Hq6/Hkv1/D256')
    if normalize_soc(torch.npu.get_device_name(0))!=config['soc_version']:raise RuntimeError('MDL target SoC differs')
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    require_capabilities({BASELINE,CANDIDATE,'prepare_attention_tasks_out','merge_lse_out'})
    bindings=source_binding();rows=[];cores=reuse._core_count(torch,config)
    for case in case_plan():
        row=run_case(torch,torch.ops.oscar_ascend_ops,case,torch.device('npu:0'),cores,acceptance['fused_attention'])
        rows.append(row)
        print('[oscar] PERF_MDL_DECODE '+json.dumps({'case':row['case'],'status':row['status'],
              'precision':row['precision'],'speedup':row['timing']['speedup'] if row['timing'] else None,
              'minimum_speedup':MIN_SPEEDUP,'graph':row['graph']['replay'] if row['graph'] else 'not_run'}),flush=True)
        gc.collect();torch.npu.empty_cache()
    passed=all(row['status']=='passed' for row in rows)
    report={'status':'passed' if passed else 'failed','baseline_operator':BASELINE,'candidate_operator':CANDIDATE,
        'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
        'performance':'passed' if passed else 'failed','device_completion':'passed','scope':SCOPE,
        'workspace_bytes_per_core':WORKSPACE,'source_binding':bindings,'cases':rows,
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
    print('[oscar] PERF_MDL_RESULT '+json.dumps({'status':report['status'],'report':str(args.output)}),flush=True)
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
