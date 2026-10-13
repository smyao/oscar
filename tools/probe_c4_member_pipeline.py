"""Archive #148/#150/#153/#154 and full startup D.4: signed C4 NPU gate.

Compare shipped H16 C4 with H32/member pipeline on Q336/C1341, matching the
complete local composition. Preserve all tasks and old lane16 errors; source2 follows production
prefill suppression. A graph with no mature multi-KV C4 work cannot qualify.
CPU is only the independent oracle, never a device fallback or timing source.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import traceback
from . import probe_history_reuse as reuse
from .probe_striped_cache import _launch, _store_bits_equal
from .striped_fixture import raw_to_striped
from .phase import atomic_json

ROOT=Path(__file__).resolve().parents[1]
BASELINE='attention_cv_window_range_cluster4_out'
CANDIDATE='attention_cv_member_pipeline_cluster4_out'
WORKSPACE={BASELINE:1839104,CANDIDATE:2494464}
CASE_NAMES=('mature_multikv','bad80','bad96','bad336')
# User 2026-10-12: a complete relevant path must improve at least 1.15x
# before joining the combination. This extra gate is stricter than the
# unchanged global non-regression acceptance; never round 1.148704 upward.
MIN_SPEEDUP=1.15
CONTEXT=1341
GRAPH_CONTEXT=1373


def case_plan():
    return tuple(reuse.Shape(name,(336,),(CONTEXT,),256,1,1,True,
                 speed_gate=bad<0,corrupt_metadata=bad>=0,corrupt_position=max(0,bad),
                 padded_tokens=336,required_clusters=1)
                 for name,bad in zip(CASE_NAMES,(-1,80,96,336)))


def _oracle(torch,fixture):
    from oscar_ascend.ops.reference import attention
    spec=fixture['spec'];context=spec.contexts[0]
    k,v,dk,dv=fixture['logical_history'];q=fixture['cpu']['q'];expected={}
    for row in range(336):
        cut=min(context,max(reuse.SINK,context+row+1-spec.recent_tokens))
        kk,vv=k.float().clone(),v.float().clone();kk[reuse.SINK:cut]=dk[reuse.SINK:cut];vv[reuse.SINK:cut]=dv[reuse.SINK:cut]
        value=attention(q[row:row+1],kk,vv,scale=fixture['scale'],causal=False)
        expected[row]=(value.output[0],value.lse[0])
    fixture['expected']=expected


def make_fixture(torch,spec):
    from oscar_ascend.ops.reference import encode_kv,decode_kv
    logical={}
    def retain(request,begin,q,k,v,ck,cv):logical[request]=(k.clone(),v.clone())
    fixture=reuse.make_fixture(torch,spec,on_request=retain)
    k,v=logical[0];rk=reuse._hadamard(torch,256);rv=fixture['cpu']['rv']
    dk,dv=decode_kv(encode_kv(k.float()@rk,v.float()@rv),256)
    fixture['logical_history']=(k,v,dk@rk.T,dv@rv.T)
    fixture['cpu']['raw']=raw_to_striped(fixture)
    if not spec.corrupt_metadata:_oracle(torch,fixture)
    return fixture


def _prepare(torch,ops,tensors,fixture,buffers):
    receipt=reuse._prepare(torch,ops,tensors,fixture,buffers)
    tasks=buffers['tasks'].cpu();mask=(tasks[:,7]==2)&(tasks[:,10]==0)&(tasks[:,4]>=tasks[:,3])
    tasks[mask,4]=tasks[mask,3]  # Exact production prefill source2 suppression.
    buffers['tasks'].copy_(tasks)
    receipt['task_sha256']=reuse._fingerprint({'tasks':tasks})
    return receipt


def expected_fault_status(torch,tasks,bad):
    expected=torch.zeros((len(tasks),2),dtype=torch.int32)
    selected=(tasks[:,1]>0)&(tasks[:,7]==0)&(tasks[:,3]<=bad)&(bad<tasks[:,4])
    ids=selected.nonzero().flatten();lanes=((bad-tasks[ids,3])%32)//16
    if not len(ids):raise RuntimeError('metadata fault hit no source0 leader')
    expected[ids,lanes]=3
    return expected


def _check(torch,ops,tensors,fixture,old,new,receipt,acceptance):
    invalid=fixture['spec'].corrupt_metadata
    for b in (old,new):reuse._check_status(torch,b,invalid=invalid)
    reuse._assert_same_bits(torch,reuse._snapshot(old),new,'H16 versus member pipeline')
    if not torch.equal(old['cluster_stats'],new['cluster_stats']):raise RuntimeError('C4 clusterStats changed')
    stats=reuse._check_cluster_stats(new['cluster_stats'].cpu(),expect_clusters=True,
                                    expected_source0_leaders=receipt['source0_leaders'])
    rounds=math.ceil((fixture['spec'].contexts[0]-reuse.SINK)/256)
    if stats['eligible_clusters']!=1 or rounds<2 or stats['shared_history_kv_tiles']!=rounds*stats['eligible_clusters']:
        raise RuntimeError('gate must execute the declared mature C4 cluster over multiple KV rounds')
    if reuse._fingerprint({'tasks':old['tasks'].cpu()})!=receipt['task_sha256']:
        raise RuntimeError('C4 attention modified prepared tasks')
    if invalid:
        tasks=old['tasks'].cpu();bad=fixture['spec'].corrupt_position
        expected=expected_fault_status(torch,tasks,bad)
        if not torch.equal(old['status'].cpu(),expected):
            raise RuntimeError('metadata fault did not preserve every original lane16 status')
        oracle='expected_invalid_metadata_status'
    else:
        for b in (old,new):
            measured=reuse._merge_and_oracle(torch,ops,fixture,b,acceptance['fused_attention'])
            if measured['sampled_queries']!=336:raise RuntimeError('independent oracle missed query rows')
        oracle='all_336_queries_passed'
    return {'context':fixture['spec'].contexts[0],
            'eligible_clusters':stats['eligible_clusters'],'shared_history_kv_tiles':stats['shared_history_kv_tiles'],
            'rounds_per_cluster':rounds,'oracle':oracle,'task_sha256':receipt['task_sha256']}


def _graph(torch,ops,fixture,tensors,old,new,receipt,cores,acceptance):
    outputs=(old,new);names=(BASELINE,CANDIDATE)
    addresses={k:t.data_ptr() for k,t in tensors.items()}
    addresses.update({str(i)+k:b[k].data_ptr() for i,b in enumerate(outputs)
                      for k in ('tasks','partial','lse','status','workspace','cluster_stats')})
    def launch(i):_launch(ops,names[i],tensors,fixture,outputs[i],cores,4)
    graphs=[torch.npu.NPUGraph(),torch.npu.NPUGraph()]
    for i,graph in enumerate(graphs):
        with torch.npu.graph(graph,capture_error_mode='thread_local',auto_dispatch_capture=True):launch(i)
    states=[]
    def replay_check(label):
        for b in outputs:reuse._poison(torch,b)
        for graph in graphs:graph.replay()
        torch.npu.synchronize()
        state=_check(torch,ops,tensors,fixture,old,new,receipt,acceptance)
        reference=reuse._snapshot(new);counter=new['cluster_stats'].clone()
        for i,b in enumerate(outputs):reuse._poison(torch,b);launch(i)
        torch.npu.synchronize();_check(torch,ops,tensors,fixture,old,new,receipt,acceptance)
        reuse._assert_same_bits(torch,reference,new,'mature graph versus eager')
        if not torch.equal(counter,new['cluster_stats']):raise RuntimeError('graph/eager clusterStats differ')
        states.append({**state,'state':label,'context':fixture['spec'].contexts[0],
                       'packed_sha256':reuse._fingerprint({'raw':fixture['cpu']['raw']})})
        return reference
    initial=replay_check('captured_inputs')
    fixture['cpu']['q'].neg_();fixture['cpu']['qr'].neg_()
    tensors['q'].copy_(fixture['cpu']['q']);tensors['qr'].copy_(fixture['cpu']['qr']);_oracle(torch,fixture)
    changed=replay_check('changed_q_qr')
    if reuse._bitwise_identical(torch,initial['partial'],changed['partial']):raise RuntimeError('graph ignored changed Q/QR')
    replacement=make_fixture(torch,replace(fixture['spec'],contexts=(GRAPH_CONTEXT,)))
    old_task_hash=receipt['task_sha256'];old_raw=fixture['cpu']['raw']
    if _store_bits_equal(torch,old_raw,replacement['cpu']['raw']):raise RuntimeError('graph fixture did not change packed/meta')
    for key,tensor in tensors.items():
        source=replacement['cpu'][key]
        if tensor.shape!=source.shape or tensor.dtype!=source.dtype:raise RuntimeError('changed metadata needs new allocation')
        tensor.copy_(source)
    fixture=replacement;receipt=_prepare(torch,ops,tensors,fixture,old)
    if old_task_hash==receipt['task_sha256']:raise RuntimeError('graph fixture did not change task metadata')
    changed_meta=replay_check('changed_context_tasks_packed_meta')
    if reuse._bitwise_identical(torch,changed['partial'],changed_meta['partial']):raise RuntimeError('graph ignored changed history/metadata')
    current={k:t.data_ptr() for k,t in tensors.items()}
    current.update({str(i)+k:b[k].data_ptr() for i,b in enumerate(outputs)
                    for k in ('tasks','partial','lse','status','workspace','cluster_stats')})
    if current!=addresses:raise RuntimeError('graph inputs/outputs moved addresses')
    return {'capture':'passed','replay':'passed','same_addresses':True,'q_qr_changed':True,
            'task_metadata_changed':True,'packed_meta_changed':True,'independent_oracle':'passed',
            'graph_eager_bitwise':'passed','mature_multikv_in_every_state':True,'states':states}


def run_case(torch,ops,spec,device,cores,acceptance):
    fixture=make_fixture(torch,spec);tensors={k:v.to(device) for k,v in fixture['cpu'].items()}
    pair=[]
    for name in (BASELINE,CANDIDATE):
        b=reuse._allocate(torch,fixture,device,cores,candidate=True)
        b['workspace']=torch.empty(cores*WORKSPACE[name],dtype=torch.uint8,device=device);pair.append(b)
    old,new=pair;receipt=_prepare(torch,ops,tensors,fixture,old)
    new['tasks'],new['positions']=old['tasks'],old['positions']
    for name,b in zip((BASELINE,CANDIDATE),pair):reuse._poison(torch,b);_launch(ops,name,tensors,fixture,b,cores,4)
    torch.npu.synchronize();detail=_check(torch,ops,tensors,fixture,old,new,receipt,acceptance)
    timing=None;graph=None
    if spec.speed_gate:
        samples={BASELINE:[],CANDIDATE:[]};reference=reuse._snapshot(old);reference_stats=old['cluster_stats'].clone()
        for repeat in range(7):
            order=list(zip((BASELINE,CANDIDATE),pair))
            if repeat>=2 and (repeat-2)%2:order.reverse()
            for name,b in order:
                reuse._poison(torch,b);begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                begin.record();_launch(ops,name,tensors,fixture,b,cores,4);end.record();end.synchronize()
                elapsed=float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid C4 device event')
                reuse._assert_same_bits(torch,reference,b,'C4 timed repetition')
                if not torch.equal(reference_stats,b['cluster_stats']):raise RuntimeError('timed C4 clusterStats changed')
                if repeat>=2:samples[name].append(elapsed)
        a,b=(statistics.median(samples[name]) for name in (BASELINE,CANDIDATE))
        timing={'baseline_ms':a,'candidate_ms':b,'ratio':b/a,'speedup':a/b,
                'minimum_speedup':MIN_SPEEDUP,'samples_ms':samples,
                'gate':'passed' if a/b>=MIN_SPEEDUP else 'failed','scope':'complete_H16_vs_H32_pipeline_same_shape_device_events'}
        graph=_graph(torch,ops,fixture,tensors,old,new,receipt,cores,acceptance)
    return {'case':spec.name,'status':'passed' if timing is None or timing['gate']=='passed' else 'failed',
            'precision':'partial_lse_status_clusterStats_bitwise','detail':detail,'timing':timing,'graph':graph}


def validate_evidence(report):
    wanted={'status':'passed','baseline_operator':BASELINE,'candidate_operator':CANDIDATE,
            'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
            'performance':'passed','device_completion':'passed','all_tasks':1008,'source2':'production_prefill_suppressed'}
    bad=[key for key,value in wanted.items() if not isinstance(report,dict) or report.get(key)!=value]
    rows=report.get('cases',[]) if isinstance(report,dict) else []
    if not isinstance(rows,list) or [row.get('case') for row in rows]!=list(CASE_NAMES):return bad+['cases']
    for row in rows:
        detail=row.get('detail',{})
        if row.get('status')!='passed' or row.get('precision')!='partial_lse_status_clusterStats_bitwise':bad.append(row['case']+'.precision')
        if (detail.get('context')!=CONTEXT or detail.get('eligible_clusters')!=1 or
                detail.get('rounds_per_cluster')!=math.ceil((CONTEXT-reuse.SINK)/256) or
                detail.get('shared_history_kv_tiles')!=detail.get('rounds_per_cluster')):bad.append(row['case']+'.mature_multikv')
        if detail.get('oracle')!=('all_336_queries_passed' if row['case']==CASE_NAMES[0] else 'expected_invalid_metadata_status'):
            bad.append(row['case']+'.oracle')
    graph=rows[0].get('graph') or {};timing=rows[0].get('timing') or {}
    samples=timing.get('samples_ms',{})
    sample_ok=(set(samples)=={BASELINE,CANDIDATE} and all(isinstance(samples[n],list) and len(samples[n])==5 and
               all(isinstance(v,(float,int)) and math.isfinite(v) and v>0 for v in samples[n]) for n in samples))
    latency_ok=False
    if sample_ok:
        a,b=(statistics.median(samples[n]) for n in (BASELINE,CANDIDATE))
        latency_ok=(timing.get('gate')=='passed' and a/b>=MIN_SPEEDUP and timing.get('baseline_ms')==a and
                    timing.get('candidate_ms')==b and timing.get('ratio')==b/a and
                    timing.get('speedup')==a/b and timing.get('minimum_speedup')==MIN_SPEEDUP and
                    timing.get('scope')=='complete_H16_vs_H32_pipeline_same_shape_device_events')
    if not latency_ok:bad.append('device_latency')
    expected={'capture':'passed','replay':'passed','same_addresses':True,'q_qr_changed':True,
              'task_metadata_changed':True,'packed_meta_changed':True,'independent_oracle':'passed',
              'graph_eager_bitwise':'passed','mature_multikv_in_every_state':True}
    if any(graph.get(k)!=v for k,v in expected.items()):bad.append('changed_input_graph')
    states=graph.get('states',[])
    if len(states)!=3 or any(s.get('eligible_clusters')!=1 or s.get('rounds_per_cluster',0)<2 or s.get('oracle')!='all_336_queries_passed' for s in states):bad.append('graph_mature_states')
    if len(states)==3 and tuple(s.get('context') for s in states)!=(CONTEXT,CONTEXT,GRAPH_CONTEXT):bad.append('graph_contexts')
    if len(states)==3 and any(s.get('rounds_per_cluster')!=math.ceil((c-reuse.SINK)/256) or
            s.get('shared_history_kv_tiles')!=s.get('rounds_per_cluster')
            for s,c in zip(states,(CONTEXT,CONTEXT,GRAPH_CONTEXT))):bad.append('graph_full_kv_rounds')
    if len(states)==3 and (states[0].get('task_sha256')==states[2].get('task_sha256') or states[0].get('packed_sha256')==states[2].get('packed_sha256')):bad.append('metadata_not_changed')
    return bad


def probe(config_path,acceptance_path):
    target=json.loads(config_path.read_text());acceptance=json.loads(acceptance_path.read_text())
    reuse._active_device(target)
    if target.get('experimental_c4_member_pipeline') is not True:raise RuntimeError('explicit C4 pipeline candidate config required')
    if acceptance['fused_attention']!={'atol':0.005,'rtol':0.005} or acceptance['performance']!={'max_latency_ratio':1.0,'min_throughput_ratio':1.0,'repeats':5,'warmup':2,'statistic':'median','per_case_required':True,'noise_allowance':0.0}:raise RuntimeError('frozen acceptance changed')
    import torch
    import torch_npu  # noqa: F401
    if not torch.npu.is_available():raise RuntimeError('target NPU required; no CPU substitution')
    torch.npu.set_device(0);torch.set_num_threads(min(4,torch.get_num_threads()))
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    require_capabilities({BASELINE,CANDIDATE,'prepare_attention_tasks_out','merge_lse_out'},ROOT/'build/ascendc/build_manifest.json')
    from .probe_native_current_fia import _target_geometry
    if _target_geometry(target)!=(6,1,256):raise RuntimeError('C4 pipeline target requires Hq6/Hkv1/D256')
    cores=reuse._core_count(torch,target);rows=[]
    for spec in case_plan():
        row=run_case(torch,torch.ops.oscar_ascend_ops,spec,torch.device('npu:0'),cores,acceptance);rows.append(row)
        print('[oscar] PERF_C4_PIPELINE '+json.dumps({'case':spec.name,'status':row['status'],'precision':row['precision'],
              'clusters':row['detail']['eligible_clusters'],'kv_rounds':row['detail']['rounds_per_cluster'],
              'ratio':row['timing']['ratio'] if row['timing'] else None,
              'speedup':row['timing']['speedup'] if row['timing'] else None,
              'minimum_speedup':MIN_SPEEDUP,
              'graph':row['graph']['replay'] if row['graph'] else 'not_run'}),flush=True)
    result={'status':'passed' if all(r['status']=='passed' for r in rows) else 'failed',
            'baseline_operator':BASELINE,'candidate_operator':CANDIDATE,'cases':rows,
            'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
            'performance':'passed' if all(r['status']=='passed' for r in rows) else 'failed','device_completion':'passed',
            'all_tasks':1008,'source2':'production_prefill_suppressed','artifact_signature':manifest['signature'],
            'artifact_sha256':manifest['sha256'],'whole_service_performance':'not_established'}
    mismatches=validate_evidence(result)
    if mismatches:
        result['status']='failed';result['evidence_mismatches']=mismatches
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,default=ROOT/'configs/acceptance.json');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:result=probe(args.config,args.acceptance)
    except Exception as error:
        traceback.print_exc();result={'status':'failed','first_error':str(error)}
    atomic_json(args.output,result)
    print('[oscar] PERF_C4_PIPELINE_RESULT '+json.dumps({'status':result['status'],'report':str(args.output)}),flush=True)
    raise SystemExit(0 if result['status']=='passed' else 2)
