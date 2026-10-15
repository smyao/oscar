"""Archive #126/#145/#148/#154/#157 and startup D.4: stable-C16/member-pipeline NPU gate.

Predeclared fixtures: Q672/C337 plus C32768/C65536, real recent256/sink64,
all2016 tasks, one mature16-member cluster and all672 oracle queries. Each speed
gate compares the stable C16 reader directly with the member pipeline. A
different candidate or recent1 mechanism fixture cannot replace this shape.
CV+merge+BF16copy must not regress; frozen error/precision gates are unchanged.
Only the original short case replays changedQ, C337->C593->C337 and packed/task
metadata at the same addresses; the two long cases do not repeat graph capture.

D.4: no full-history dequantization (historical6499.8-6655.1ms), host725ms
prepare or209-215ms store pattern is introduced. Only bounded per-core KV/member
reuse is measured. Preparing this entrypoint is not NPU, graph, or model
performance evidence. Missing production source/symbols fail explicitly.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback
from . import probe_history_reuse as reuse
from .probe_striped_cache import _launch as _cv_launch, _store_bits_equal
from .striped_fixture import raw_to_striped
from .phase import atomic_json
from .c16_probe_oracle import LONG_CONTEXTS, _shared_kv_attention, _long_oracle, _oracle

ROOT=Path(__file__).resolve().parents[1]
BASELINE='attention_cv_window_range_cluster16_out'
CANDIDATE='attention_cv_member_pipeline_cluster16_out'
WORKSPACE={BASELINE:4997120,CANDIDATE:5652480}
SHORT_CASE_NAMES=('mature_multikv','bad80','bad96','bad336','last_member_qr')
QR_CASE_NAME='last_member_qr'
CASE_NAMES=SHORT_CASE_NAMES+tuple(LONG_CONTEXTS)
# User 2026-10-15 accepts incremental net gains on the B1 base and supersedes
# the historical 1.15 per-point rule. Keep complete-path non-regression;
# frozen precision and the native E2E >30% target are unchanged.
MIN_SPEEDUP=1.0
CONTEXT=337
GRAPH_CONTEXT=593
SOURCE_PATHS=(
    'csrc/kernels/attention_cv_member_pipeline_cluster16.cpp',
    'csrc/kernels/attention_cv_window_range_cluster16.cpp',
    'csrc/include/oscar_c16_member_pipeline_launch.h',
    'csrc/include/oscar_attention_launch.h',
    'csrc/include/oscar_cv_schedule.h',
    'csrc/kernels/attention_fast_unpack.h',
    'csrc/kernels/attention_striped_unpack.h',
    'csrc/kernels/attention_striped_unpack32.h',
    'csrc/kernels/attention_range_mask.h',
    'csrc/kernels/oscar_common.h',
    'csrc/kernels/attention_tasks.cpp',
    'csrc/kernels/merge_lse.cpp',
    'csrc/striped_attention_bindings.cpp',
    'oscar_ascend/ops/reference.py',
    'tools/probe_history_reuse.py',
    'tools/striped_fixture.py',
    'tools/probe_c16_member_pipeline.py',
    'tools/c16_probe_oracle.py',
)


def source_binding(root=ROOT):
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def case_plan():
    cases=tuple(reuse.Shape(name,(672,),(CONTEXT,),256,1,1,True,
                 speed_gate=bad<0,corrupt_metadata=bad>=0,corrupt_position=max(0,bad),
                 padded_tokens=672,required_clusters=1)
                 for name,bad in zip(SHORT_CASE_NAMES[:4],(-1,80,96,336)))
    qr=(reuse.Shape(QR_CASE_NAME,(672,),(CONTEXT,),256,1,1,True,
                  speed_gate=False,corrupt_qr=True,corrupt_qr_token=651,
                  padded_tokens=672,required_clusters=1),)
    long_cases=tuple(reuse.Shape(name,(672,),(context,),256,1,1,True,
                     padded_tokens=672,required_clusters=1)
                     for name,context in LONG_CONTEXTS.items())
    return cases+qr+long_cases


def make_fixture(torch,spec):
    if (spec.qlens!=(672,) or spec.recent_tokens!=256 or spec.dim!=256
            or spec.kv_heads!=1 or spec.splits!=1 or spec.padded_tokens!=672):
        raise RuntimeError('member-pipeline target gate requires Q672/Hq6/Hkv1/D256/S1/recent256, not a mechanism fixture')
    if spec.corrupt_qr and (spec.corrupt_qr_token!=651 or spec.corrupt_metadata):
        raise RuntimeError('member-pipeline QR isolation requires only target member15 query651/head0')
    from oscar_ascend.ops.reference import encode_kv,decode_kv
    logical={}
    def retain(request,begin,q,k,v,ck,cv):logical[request]=(k.clone(),v.clone())
    fixture=reuse.make_fixture(torch,spec,on_request=retain)
    if spec.contexts[0] in LONG_CONTEXTS.values():
        fixture['generic_reference_anchors']=fixture['expected']
    k,v=logical[0];rk=reuse._hadamard(torch,256);rv=fixture['cpu']['rv']
    if spec.corrupt_qr:
        fixture['clean_qr']=(fixture['cpu']['q'].float()@rk).contiguous()
        fixture['cpu']['qr'].view(torch.int32)[651,0,0]=0x7fc10001
    dk,dv=decode_kv(encode_kv(k.float()@rk,v.float()@rv),256)
    fixture['logical_history']=(k,v,dk@rk.T,dv@rv.T)
    fixture['cpu']['raw']=raw_to_striped(fixture)
    _fixed_graph_capacity(torch,fixture)
    if not spec.corrupt_metadata:_oracle(torch,fixture)
    return fixture


def _fixed_graph_capacity(torch,fixture):
    """Reserve diagnostic-only tail capacity for changed-context replay.

    BLOCK512 pages change from two to three at C337 -> C593. Extra raw/window
    bytes stay poisoned, extra virtual columns stay -1, and original active
    values/slots remain intact. Long cases retain capacity for their actual
    context; they are never squeezed into this short graph's three pages.
    This changes no serving allocation or layout.
    """
    max_tokens=max(max(fixture['spec'].contexts),GRAPH_CONTEXT)+672
    blocks=math.ceil(max_tokens/reuse.BLOCK_TOKENS)
    columns=math.ceil(max_tokens/128)
    if fixture['blocks']>blocks or fixture['cpu']['table'].shape[1]>columns:
        raise RuntimeError('C16 graph fixture exceeds its fixed diagnostic capacity')
    cpu=fixture['cpu']
    shapes={'raw':(reuse.PREFIX+blocks*fixture['stride'],),
            'table':(1,columns),
            'wk':(blocks,*cpu['wk'].shape[1:]),
            'wv':(blocks,*cpu['wv'].shape[1:]),
            'tags':(blocks,*cpu['tags'].shape[1:])}
    poisons={'raw':0xA5,'table':-1,'wk':float('nan'),'wv':float('nan'),'tags':-1}
    for name,shape in shapes.items():
        old=cpu[name]
        if tuple(old.shape)!=shape:
            expanded=torch.full(shape,poisons[name],dtype=old.dtype)
            expanded[tuple(slice(0,size) for size in old.shape)].copy_(old)
            cpu[name]=expanded
    fixture['blocks']=blocks
    fixture['input_sha256']=reuse._fingerprint(cpu)


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


def qr_fault_domain(torch,tasks,fixture):
    """Derive fault ownership from actual prepared groups, not the Q336 fixture.

    Task ABI columns0/1 are query begin/count,2 KV head,5 request,6 split,
    7 source,8 context and9 request begin. PublishRows owns at most64 rows per
    AIV. The target's second336-query bucket must contain16 complete21-query
    leaders with the same terminal history domain before it can be qualified.
    """
    spec=fixture['spec'];heads=spec.heads;ratio=heads//spec.kv_heads
    token=spec.corrupt_qr_token;head=0;tile=128//ratio
    selected=((tasks[:,1]>0)&(tasks[:,7]==0)&(tasks[:,2]==head//ratio)
              &(tasks[:,0]<=token)&(token<tasks[:,0]+tasks[:,1]))
    ids=selected.nonzero().flatten()
    if len(ids)!=1:raise RuntimeError('QR poison must resolve to exactly one source0 leader')
    task_id=int(ids[0]);row=tasks[task_id].tolist();begin,count=row[:2]
    anchor=row[9]+((begin-row[9])//(16*tile))*(16*tile)
    for member in range(16):
        found=((tasks[:,0]==anchor+member*tile)&(tasks[:,1]==tile)
               &(tasks[:,2]==row[2])&(tasks[:,5]==row[5])&(tasks[:,6]==row[6])
               &(tasks[:,7]==0)&(tasks[:,8]==row[8])&(tasks[:,9]==row[9])
               &(tasks[:,3]==row[3])&(tasks[:,4]==row[4])&(tasks[:,10]==0))
        if int(found.sum())!=1 or row[4]!=row[8]:
            raise RuntimeError('QR poison is not in a complete mature16-member terminal group')
    member=(begin-anchor)//tile;lane=((token-begin)*ratio+head%ratio)//64
    if (member!=15 or begin!=651 or count!=21 or anchor!=336 or lane!=0):
        raise RuntimeError('QR poison is not target member15 query651/head0 lane0')
    expected=torch.zeros((len(tasks),2),dtype=torch.int32)
    expected[task_id,lane]=2
    affected=torch.zeros((fixture['tokens']*heads,3*spec.splits),dtype=torch.bool)
    rows=[]
    for local in range(lane*64,min((lane+1)*64,count*ratio)):
        flat=(begin+local//ratio)*heads+row[2]*ratio+local%ratio
        affected[flat,row[6]]=True;rows.append(flat)
    return expected,affected,{
        'query':token,'head':head,'leader':begin,'cluster_anchor':anchor,'member':member,
        'lane':lane,'code':2,'history_rows':len(rows),'flatten_row_first':min(rows),
        'flatten_row_last':max(rows),'unaffected_rows':fixture['tokens']*heads-len(rows),
        'nonzero_task_lanes':1,
    }


def check_qr_isolation(torch,fixture,buffers,clean,tolerance):
    tasks=buffers['tasks'].cpu()
    expected_status,affected,domain=qr_fault_domain(torch,tasks,fixture)
    if not torch.equal(buffers['status'].cpu(),expected_status):
        raise RuntimeError('QR error escaped its one source0 member/lane')
    spec=fixture['spec'];rows=fixture['tokens']*spec.heads
    partial=buffers['partial'].cpu().view(rows,3*spec.splits,spec.dim)
    lse=buffers['lse'].cpu().view(rows,3*spec.splits)
    if not bool(torch.isnan(partial[affected]).all()) or not bool(torch.isnan(lse[affected]).all()):
        raise RuntimeError('QR affected history rows did not fail closed with NaN')
    clean_partial=clean['partial'].view_as(partial);clean_lse=clean['lse'].view_as(lse)
    if (not _store_bits_equal(torch,partial[~affected],clean_partial[~affected])
            or not _store_bits_equal(torch,lse[~affected],clean_lse[~affected])):
        raise RuntimeError('QR poison changed another member, window or current partial')
    bad_rows=affected.any(-1)
    merge_status=torch.zeros(rows,dtype=torch.int32);merge_status[bad_rows]=2
    if not torch.equal(buffers['merge_status'].cpu().reshape(-1),merge_status):
        raise RuntimeError('QR merge error domain changed')
    merged=buffers['merged'].cpu().view(rows,spec.dim)
    merged_lse=buffers['merged_lse'].cpu().view(rows)
    copied=buffers['bf16_output'].cpu().view(rows,spec.dim)
    # Frozen merge_lse.cpp rejects NaN LSE with zero output/-inf LSE/code2.
    if (not bool((merged[bad_rows]==0).all()) or not bool((copied[bad_rows]==0).all())
            or not bool(torch.isneginf(merged_lse[bad_rows]).all())):
        raise RuntimeError('QR merge must preserve frozen zero/-inf/code2 failure behavior')
    indexes=sorted(fixture['expected'])
    expected=torch.stack([fixture['expected'][i][0] for i in indexes]).view(rows,spec.dim)
    expected_lse=torch.stack([fixture['expected'][i][1] for i in indexes]).view(rows)
    torch.testing.assert_close(merged[~bad_rows],expected[~bad_rows],**tolerance)
    torch.testing.assert_close(merged_lse[~bad_rows],expected_lse[~bad_rows],**tolerance)
    return {**domain,'window_current_preserved':True,'unaffected_oracle':'passed',
            'merge_failure':'zero_output_negative_infinity_lse_code2'}


def _launch(ops,name,tensors,fixture,buffers,cores,cluster):
    """Measure the full changed reader plus its mandatory merge/output copy.

    Source2 current FIA is common unchanged work outside this diagnostic.
    It is explicitly suppressed in both arms, not silently timed as empty
    work and represented as a whole attention-layer or model speedup.
    """
    _cv_launch(ops,name,tensors,fixture,buffers,cores,cluster)
    spec=fixture['spec'];n=fixture['tokens'];h=spec.heads;d=spec.dim
    ops.merge_lse_out(buffers['partial'].view(n*h,3*spec.splits,d),
        buffers['lse'].view(n*h,3*spec.splits),buffers['merged'].view(n*h,d),
        buffers['merged_lse'].view(n*h),buffers['merge_status'].view(n*h))
    buffers['bf16_output'].copy_(buffers['merged'])


def _poison(torch,buffers):
    reuse._poison(torch,buffers)
    for name in ('merged','merged_lse','bf16_output'):buffers[name].fill_(float('nan'))
    buffers['merge_status'].fill_(-99)


def _snapshot(buffers):
    return {**reuse._snapshot(buffers),**{name:buffers[name].clone() for name in
        ('merged','merged_lse','merge_status','bf16_output')}}


def _assert_same_bits(torch,expected,actual,label):
    reuse._assert_same_bits(torch,expected,actual,label)
    for name in ('merged','merged_lse','merge_status','bf16_output'):
        if not _store_bits_equal(torch,expected[name],actual[name]):
            raise RuntimeError(f'{label} differs bitwise in {name}')


def _check(torch,ops,tensors,fixture,old,new,receipt,acceptance):
    invalid=fixture['spec'].corrupt_metadata or fixture['spec'].corrupt_qr
    for b in (old,new):reuse._check_status(torch,b,invalid=invalid)
    _assert_same_bits(torch,_snapshot(old),new,'stable C16 versus member pipeline')
    if not torch.equal(old['cluster_stats'],new['cluster_stats']):raise RuntimeError('C16 clusterStats changed')
    from .probe_mixed_optimization import _check_c16_stats
    stats=_check_c16_stats(torch,new,receipt['source0_leaders'],require_clusters=True)
    rounds=math.ceil((fixture['spec'].contexts[0]-reuse.SINK)/256)
    if (receipt['source0_leaders']!=32 or stats['eligible_clusters']!=1
            or stats['grouped_leaders']!=16 or stats['original_schedule_skips']!=16
            or rounds<2 or stats['shared_history_kv_tiles']!=rounds):
        raise RuntimeError('gate must execute the declared mature C16 cluster over multiple KV rounds')
    if reuse._fingerprint({'tasks':old['tasks'].cpu()})!=receipt['task_sha256']:
        raise RuntimeError('C16 attention modified prepared tasks')
    qr_detail=None
    if fixture['spec'].corrupt_qr:
        clean=fixture.get('clean_baseline')
        if clean is None:raise RuntimeError('QR isolation requires a completed clean stable baseline')
        for b in (old,new):
            qr_detail=check_qr_isolation(torch,fixture,b,clean,acceptance['fused_attention'])
        oracle='late_member_qr_isolation_passed'
    elif invalid:
        tasks=old['tasks'].cpu();bad=fixture['spec'].corrupt_position
        expected=expected_fault_status(torch,tasks,bad)
        if not torch.equal(old['status'].cpu(),expected):
            raise RuntimeError('metadata fault did not preserve every original lane16 status')
        oracle='expected_invalid_metadata_status'
    else:
        for b in (old,new):
            if not bool((b['merge_status']==0).all()):raise RuntimeError('C16 merge status failed')
            measured=reuse._merge_and_oracle(torch,ops,fixture,b,acceptance['fused_attention'])
            if measured['sampled_queries']!=672:raise RuntimeError('independent oracle missed query rows')
        oracle='all_672_queries_passed'
    return {'context':fixture['spec'].contexts[0],
            'query_tokens':672,'recent_tokens':fixture['spec'].recent_tokens,'sink_tokens':reuse.SINK,
            'head_dim':fixture['spec'].dim,
            'eligible_clusters':stats['eligible_clusters'],'shared_history_kv_tiles':stats['shared_history_kv_tiles'],
            'grouped_leaders':stats['grouped_leaders'],
            'original_schedule_skips':stats['original_schedule_skips'],
            'source0_leaders':receipt['source0_leaders'],
            'rounds_per_cluster':rounds,'oracle':oracle,'task_sha256':receipt['task_sha256'],
            **({'qr_fault':qr_detail} if qr_detail is not None else {}),
            **({'reference_anchor_check':fixture.get('reference_anchor_check')}
               if fixture['spec'].name in LONG_CONTEXTS else {})}


def _graph(torch,ops,fixture,tensors,old,new,receipt,cores,acceptance):
    outputs=(old,new);names=(BASELINE,CANDIDATE)
    addresses={k:t.data_ptr() for k,t in tensors.items()}
    addresses.update({str(i)+k:b[k].data_ptr() for i,b in enumerate(outputs)
                      for k in ('tasks','partial','lse','status','workspace','cluster_stats',
                                'merged','merged_lse','merge_status','bf16_output')})
    def launch(i):_launch(ops,names[i],tensors,fixture,outputs[i],cores,16)
    graphs=[torch.npu.NPUGraph(),torch.npu.NPUGraph()]
    for i,graph in enumerate(graphs):
        with torch.npu.graph(graph,capture_error_mode='thread_local',auto_dispatch_capture=True):launch(i)
    states=[]
    def replay_check(label):
        for b in outputs:_poison(torch,b)
        for graph in graphs:graph.replay()
        torch.npu.synchronize()
        state=_check(torch,ops,tensors,fixture,old,new,receipt,acceptance)
        reference=_snapshot(new);counter=new['cluster_stats'].clone()
        for i,b in enumerate(outputs):_poison(torch,b);launch(i)
        torch.npu.synchronize();_check(torch,ops,tensors,fixture,old,new,receipt,acceptance)
        _assert_same_bits(torch,reference,new,'mature graph versus eager')
        if not torch.equal(counter,new['cluster_stats']):raise RuntimeError('graph/eager clusterStats differ')
        states.append({**state,'state':label,'context':fixture['spec'].contexts[0],
                       'packed_sha256':reuse._fingerprint({'raw':fixture['cpu']['raw']})})
        return reference
    initial_spec=fixture['spec']
    initial_task_hash=receipt['task_sha256']
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
    restored=make_fixture(torch,initial_spec)
    for key,tensor in tensors.items():
        source=restored['cpu'][key]
        if tensor.shape!=source.shape or tensor.dtype!=source.dtype:raise RuntimeError('restored metadata changed allocation')
        tensor.copy_(source)
    fixture=restored;receipt=_prepare(torch,ops,tensors,fixture,old)
    if receipt['task_sha256']!=initial_task_hash:raise RuntimeError('restored context did not restore tasks')
    replay_check('restored_context_tasks_packed_meta')
    _assert_same_bits(torch,initial,new,'restored initial graph inputs')
    current={k:t.data_ptr() for k,t in tensors.items()}
    current.update({str(i)+k:b[k].data_ptr() for i,b in enumerate(outputs)
                    for k in ('tasks','partial','lse','status','workspace','cluster_stats',
                                'merged','merged_lse','merge_status','bf16_output')})
    if current!=addresses:raise RuntimeError('graph inputs/outputs moved addresses')
    return {'capture':'passed','replay':'passed','same_addresses':True,'q_qr_changed':True,
            'task_metadata_changed':True,'packed_meta_changed':True,'independent_oracle':'passed',
            'graph_eager_bitwise':'passed','mature_multikv_in_every_state':True,
            'initial_inputs_restored':True,'initial_output_restored_bitwise':True,'states':states}


def run_case(torch,ops,spec,device,cores,acceptance):
    fixture=make_fixture(torch,spec);tensors={k:v.to(device) for k,v in fixture['cpu'].items()}
    pair=[]
    for name in (BASELINE,CANDIDATE):
        b=reuse._allocate(torch,fixture,device,cores,candidate=True)
        b['workspace']=torch.empty(cores*WORKSPACE[name],dtype=torch.uint8,device=device)
        b['merged']=torch.empty((fixture['tokens'],spec.heads,spec.dim),dtype=torch.float32,device=device)
        b['merged_lse']=torch.empty((fixture['tokens'],spec.heads),dtype=torch.float32,device=device)
        b['merge_status']=torch.empty((fixture['tokens'],spec.heads),dtype=torch.int32,device=device)
        b['bf16_output']=torch.empty_like(b['merged'],dtype=torch.bfloat16)
        pair.append(b)
    old,new=pair;receipt=_prepare(torch,ops,tensors,fixture,old)
    new['tasks'],new['positions']=old['tasks'],old['positions']
    if spec.corrupt_qr:
        # One clean stable launch for this fault case establishes that every
        # unaffected partial, including window/current on the faulted rows,
        # remains byte-identical. This launch is outside all speed samples.
        tensors['qr'].copy_(fixture['clean_qr'])
        _poison(torch,old);_launch(ops,BASELINE,tensors,fixture,old,cores,16)
        torch.npu.synchronize();reuse._check_status(torch,old,invalid=False)
        reuse._merge_and_oracle(torch,ops,fixture,old,acceptance['fused_attention'])
        fixture['clean_baseline']={name:old[name].cpu().clone() for name in ('partial','lse')}
        tensors['qr'].copy_(fixture['cpu']['qr'])
    for name,b in zip((BASELINE,CANDIDATE),pair):_poison(torch,b);_launch(ops,name,tensors,fixture,b,cores,16)
    torch.npu.synchronize();detail=_check(torch,ops,tensors,fixture,old,new,receipt,acceptance)
    timing=None;graph=None
    if spec.speed_gate:
        samples={BASELINE:[],CANDIDATE:[]};reference=_snapshot(old);reference_stats=old['cluster_stats'].clone()
        for repeat in range(7):
            order=list(zip((BASELINE,CANDIDATE),pair))
            if repeat>=2 and (repeat-2)%2:order.reverse()
            for name,b in order:
                _poison(torch,b);begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                begin.record();_launch(ops,name,tensors,fixture,b,cores,16);end.record();end.synchronize()
                elapsed=float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid C16 device event')
                _assert_same_bits(torch,reference,b,'C16 timed repetition')
                if not torch.equal(reference_stats,b['cluster_stats']):raise RuntimeError('timed C16 clusterStats changed')
                if repeat>=2:samples[name].append(elapsed)
        a,b=(statistics.median(samples[name]) for name in (BASELINE,CANDIDATE))
        timing={'baseline_ms':a,'candidate_ms':b,'ratio':b/a,'speedup':a/b,
                'minimum_speedup':MIN_SPEEDUP,'samples_ms':samples,
                'gate':'passed' if a/b>=MIN_SPEEDUP else 'failed','scope':'complete_stable_C16_vs_member_pipeline_history_window_CV_merge_BF16_copy_device_events'}
        if spec.name==SHORT_CASE_NAMES[0]:
            graph=_graph(torch,ops,fixture,tensors,old,new,receipt,cores,acceptance)
    return {'case':spec.name,'status':'passed' if timing is None or timing['gate']=='passed' else 'failed',
            'precision':'partial_lse_status_clusterStats_bitwise','detail':detail,'timing':timing,'graph':graph}


def validate_evidence(report):
    wanted={'status':'passed','baseline_operator':BASELINE,'candidate_operator':CANDIDATE,
            'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
            'performance':'passed','device_completion':'passed','all_tasks':2016,'source2':'production_prefill_suppressed'}
    bad=[key for key,value in wanted.items() if not isinstance(report,dict) or report.get(key)!=value]
    rows=report.get('cases',[]) if isinstance(report,dict) else []
    if not isinstance(rows,list) or [row.get('case') for row in rows]!=list(CASE_NAMES):return bad+['cases']
    for row in rows:
        detail=row.get('detail',{})
        if row.get('status')!='passed' or row.get('precision')!='partial_lse_status_clusterStats_bitwise':bad.append(row['case']+'.precision')
        expected_context=LONG_CONTEXTS.get(row['case'],CONTEXT)
        if (detail.get('context')!=expected_context or detail.get('eligible_clusters')!=1 or
                detail.get('query_tokens')!=672 or detail.get('recent_tokens')!=256 or
                detail.get('sink_tokens')!=64 or detail.get('head_dim')!=256 or
                detail.get('grouped_leaders')!=16 or detail.get('original_schedule_skips')!=16 or
                detail.get('source0_leaders')!=32 or
                detail.get('rounds_per_cluster')!=math.ceil((expected_context-reuse.SINK)/256) or
                detail.get('shared_history_kv_tiles')!=detail.get('rounds_per_cluster')):bad.append(row['case']+'.mature_multikv')
        expected_oracle=('all_672_queries_passed' if row['case']==SHORT_CASE_NAMES[0] or row['case'] in LONG_CONTEXTS else
                         'late_member_qr_isolation_passed' if row['case']==QR_CASE_NAME else
                         'expected_invalid_metadata_status')
        if detail.get('oracle')!=expected_oracle:
            bad.append(row['case']+'.oracle')
        if row['case']==QR_CASE_NAME:
            wanted_qr={'query':651,'head':0,'leader':651,'cluster_anchor':336,'member':15,
                'lane':0,'code':2,'history_rows':64,'flatten_row_first':3906,'flatten_row_last':3969,
                'unaffected_rows':3968,'nonzero_task_lanes':1,'window_current_preserved':True,
                'unaffected_oracle':'passed','merge_failure':'zero_output_negative_infinity_lse_code2'}
            if any(detail.get('qr_fault',{}).get(k)!=v for k,v in wanted_qr.items()):
                bad.append(row['case']+'.isolation')
    for row in rows:
        if row['case'] in LONG_CONTEXTS:
            if not _valid_timing(row.get('timing') or {}):bad.append(row['case']+'.device_latency')
            if row.get('graph') is not None:bad.append(row['case']+'.unexpected_graph')
            anchors=row.get('detail',{}).get('reference_anchor_check') or {}
            queries=list(reuse._samples(672))
            if (anchors.get('status')!='bitwise_passed' or anchors.get('domain')!='history_window_current'
                    or anchors.get('queries')!=queries or anchors.get('expected_count')!=len(queries)
                    or anchors.get('checked_count')!=len(queries) or anchors.get('bit_differences')!=[]):
                bad.append(row['case']+'.reference_anchors')
    graph=rows[0].get('graph') or {};timing=rows[0].get('timing') or {}
    if not _valid_timing(timing):bad.append('device_latency')
    expected={'capture':'passed','replay':'passed','same_addresses':True,'q_qr_changed':True,
              'task_metadata_changed':True,'packed_meta_changed':True,'independent_oracle':'passed',
              'graph_eager_bitwise':'passed','mature_multikv_in_every_state':True,
              'initial_inputs_restored':True,'initial_output_restored_bitwise':True}
    if any(graph.get(k)!=v for k,v in expected.items()):bad.append('changed_input_graph')
    states=graph.get('states',[])
    if len(states)!=4 or any(s.get('eligible_clusters')!=1 or s.get('grouped_leaders')!=16 or
            s.get('query_tokens')!=672 or s.get('recent_tokens')!=256 or
            s.get('sink_tokens')!=64 or s.get('head_dim')!=256 or
            s.get('original_schedule_skips')!=16 or s.get('source0_leaders')!=32 or
            s.get('rounds_per_cluster',0)<2 or s.get('oracle')!='all_672_queries_passed' for s in states):bad.append('graph_mature_states')
    if len(states)==4 and tuple(s.get('context') for s in states)!=(CONTEXT,CONTEXT,GRAPH_CONTEXT,CONTEXT):bad.append('graph_contexts')
    if len(states)==4 and any(s.get('rounds_per_cluster')!=math.ceil((c-reuse.SINK)/256) or
            s.get('shared_history_kv_tiles')!=s.get('rounds_per_cluster')
            for s,c in zip(states,(CONTEXT,CONTEXT,GRAPH_CONTEXT,CONTEXT))):bad.append('graph_full_kv_rounds')
    if len(states)==4 and (states[0].get('task_sha256')==states[2].get('task_sha256') or states[0].get('packed_sha256')==states[2].get('packed_sha256')):bad.append('metadata_not_changed')
    if len(states)==4 and (states[0].get('task_sha256')!=states[3].get('task_sha256') or states[0].get('packed_sha256')!=states[3].get('packed_sha256')):bad.append('metadata_not_restored')
    return bad


def _valid_timing(timing):
    samples=timing.get('samples_ms',{})
    sample_ok=(set(samples)=={BASELINE,CANDIDATE} and all(isinstance(samples[n],list) and len(samples[n])==5 and
               all(isinstance(v,(float,int)) and math.isfinite(v) and v>0 for v in samples[n]) for n in samples))
    latency_ok=False
    if sample_ok:
        a,b=(statistics.median(samples[n]) for n in (BASELINE,CANDIDATE))
        latency_ok=(timing.get('gate')=='passed' and a/b>=MIN_SPEEDUP and timing.get('baseline_ms')==a and
                    timing.get('candidate_ms')==b and timing.get('ratio')==b/a and
                    timing.get('speedup')==a/b and timing.get('minimum_speedup')==MIN_SPEEDUP and
                    timing.get('scope')=='complete_stable_C16_vs_member_pipeline_history_window_CV_merge_BF16_copy_device_events')
    return latency_ok


def probe(config_path,acceptance_path):
    target=json.loads(config_path.read_text());acceptance=json.loads(acceptance_path.read_text())
    reuse._active_device(target)
    if target.get('experimental_c16_member_pipeline') is not True:raise RuntimeError('explicit C16 member-pipeline candidate config required')
    if acceptance['fused_attention']!={'atol':0.005,'rtol':0.005} or acceptance['performance']!={'max_latency_ratio':1.0,'min_throughput_ratio':1.0,'repeats':5,'warmup':2,'statistic':'median','per_case_required':True,'noise_allowance':0.0}:raise RuntimeError('frozen acceptance changed')
    import torch
    import torch_npu  # noqa: F401
    if not torch.npu.is_available():raise RuntimeError('target NPU required; no CPU substitution')
    torch.npu.set_device(0);torch.set_num_threads(min(4,torch.get_num_threads()))
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    require_capabilities({BASELINE,CANDIDATE,'prepare_attention_tasks_out','merge_lse_out'},ROOT/'build/ascendc/build_manifest.json')
    from .probe_native_current_fia import _target_geometry
    if _target_geometry(target)!=(6,1,256):raise RuntimeError('C16 member-pipeline target requires Hq6/Hkv1/D256')
    cores=reuse._core_count(torch,target);rows=[]
    for spec in case_plan():
        row=run_case(torch,torch.ops.oscar_ascend_ops,spec,torch.device('npu:0'),cores,acceptance);rows.append(row)
        print('[oscar] PERF_C16_PIPELINE '+json.dumps({'case':spec.name,'status':row['status'],'precision':row['precision'],
              'clusters':row['detail']['eligible_clusters'],'kv_rounds':row['detail']['rounds_per_cluster'],
              'ratio':row['timing']['ratio'] if row['timing'] else None,
              'speedup':row['timing']['speedup'] if row['timing'] else None,
              'minimum_speedup':MIN_SPEEDUP,
              'graph':row['graph']['replay'] if row['graph'] else 'not_run'}),flush=True)
    result={'status':'passed' if all(r['status']=='passed' for r in rows) else 'failed',
            'baseline_operator':BASELINE,'candidate_operator':CANDIDATE,'cases':rows,
            'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
            'performance':'passed' if all(r['status']=='passed' for r in rows) else 'failed','device_completion':'passed',
            'all_tasks':2016,'source2':'production_prefill_suppressed','artifact_signature':manifest['signature'],
            'artifact_sha256':manifest['sha256'],'source_binding':source_binding(),
            'workspace_bytes_per_core':WORKSPACE,'whole_service_performance':'not_established'}
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
    print('[oscar] PERF_C16_PIPELINE_RESULT '+json.dumps({'status':result['status'],'report':str(args.output)}),flush=True)
    raise SystemExit(0 if result['status']=='passed' else 2)
