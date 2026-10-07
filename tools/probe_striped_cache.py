"""Archive #126/#129/#148-154: paired writer/reader NPU gate for striped INT2.

D.4: full-history materialization is forbidden in serving. CPU fixture conversion
is outside timing; production writes striped bytes directly. Precision, graph,
real device timing, and whole-service acceptance remain separate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import json
import math
from pathlib import Path
import statistics
import traceback

from . import probe_history_reuse as reuse
from . import probe_fast_unpack as fast
from .phase import atomic_json
from .striped_fixture import FORMAT, raw_to_striped, striped_slots_to_canonical

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    shape: reuse.Shape
    old: str
    new: str
    cluster: int = 1
    graph: bool = False
    timed: bool = False


def case_plan():
    contexts = (20000, 23000, 27000, 30000) * 8
    cases = [
        Case(reuse.Shape('q4_32_20k30k', (4,)*32, contexts, 256, 1, 3, False),
             'attention_cv_fast_out', 'attention_cv_striped_decode_out', graph=True, timed=True),
        Case(reuse.Shape('q1_32_20k30k', (1,)*32, contexts, 256, 1, 3, False,
                         padded_tokens=128, slot_context=True),
             'attention_cv_fast_q1_out', 'attention_cv_striped_q1_out', graph=True, timed=True),
        Case(reuse.Shape('q4_32_65k', (4,)*32, (65100,)*32, 256, 1, 3, False),
             'attention_cv_fast_out', 'attention_cv_striped_decode_out', timed=True),
        Case(reuse.Shape('q1_32_65k', (1,)*32, (65100,)*32, 256, 1, 3, False,
                         padded_tokens=128, slot_context=True),
             'attention_cv_fast_q1_out', 'attention_cv_striped_q1_out', timed=True),
        Case(reuse.Shape('q4_padded16k', (4,)*32, contexts, 256, 1, 1, False, padded_tokens=16384),
             'attention_cv_fast_balanced_out', 'attention_cv_striped_decode_out', timed=True),
        Case(reuse.Shape('general_frontier', (3, 33), (511, 1025), 256, 1, 3, False),
             'attention_cv_fast_out', 'attention_cv_striped_out'),
        Case(reuse.Shape('balanced_frontier', (4,)*4, (127,511,641,1025), 256, 1, 1, False,
                         padded_tokens=512),
             'attention_cv_fast_balanced_out', 'attention_cv_striped_balanced_out'),
        Case(reuse.Shape('c4_unaligned', (3,385), (641,1025), 256, 1, 3, True),
             'attention_cv_fast_cluster4_out', 'attention_cv_striped_cluster4_out', 4),
        Case(reuse.Shape('c16_active', (1024,), (641,), 256, 1, 1, True, recent_tokens=32),
             'attention_cv_fast_cluster16_out', 'attention_cv_striped_cluster16_out', 16),
        Case(reuse.Shape('c16_mature20k', (8192,), (20000,), 256, 1, 1, True),
             'attention_cv_fast_cluster16_out', 'attention_cv_striped_cluster16_out', 16, timed=True),
        Case(reuse.Shape('invalid_live_metadata', (4,), (511,), 256, 1, 1, False,
                         corrupt_metadata=True, corrupt_position=64),
             'attention_cv_fast_out', 'attention_cv_striped_decode_out'),
        Case(reuse.Shape('invalid_dead_tail', (4,), (511,), 256, 1, 1, False,
                         corrupt_dead_tail=True),
             'attention_cv_fast_out', 'attention_cv_striped_decode_out'),
    ]
    return tuple(cases)


def _allocate(torch, fixture, device, cores, cluster):
    buffers = reuse._allocate(torch, fixture, device, cores, candidate=cluster > 1)
    if cluster == 16:
        from .probe_mixed_optimization import _c16_workspace_per_core
        buffers['workspace'] = torch.empty(cores*_c16_workspace_per_core(256),
                                            dtype=torch.uint8, device=device)
    return buffers


def _launch(ops, name, tensors, fixture, buffers, cores, cluster):
    spec = fixture['spec']
    args = [tensors[k] for k in ('q','qr','ck','cv','rv','raw','table','wk','wv','tags')]
    args += [buffers[k] for k in ('tasks','partial','lse','status','workspace')]
    if cluster > 1:
        args.append(buffers['cluster_stats'])
    args += [reuse.BLOCK_TOKENS,fixture['blocks'],reuse.PREFIX,fixture['stride'],
             reuse.SINK,spec.recent_tokens,reuse.SPECULATIVE,spec.splits,fixture['scale'],cores]
    getattr(ops, name)(*args)


def _paired_status(torch, old, new, invalid=False):
    reuse._check_status(torch, old, invalid=invalid)
    reuse._check_status(torch, new, invalid=invalid)
    if not invalid:
        return
    # M32 and KV64 change AIV row ownership. Errors must remain on the same
    # task; their lane number is not part of status_guard's any-error contract.
    a, b = old['status'].cpu().ne(0).any(-1), new['status'].cpu().ne(0).any(-1)
    if not torch.equal(a, b) or not bool(b.any()):
        raise RuntimeError('striped reader changed task-level invalid-input detection')


def _store_bits_equal(torch, left, right):
    """Compare exact BF16 payload bits, including untouched NaN snapshots."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype in (torch.bfloat16, torch.float16):
        return bool(torch.equal(left.contiguous().view(torch.int16),
                                right.contiguous().view(torch.int16)))
    if left.dtype == torch.float32:
        return bool(torch.equal(left.contiguous().view(torch.int32),
                                right.contiguous().view(torch.int32)))
    return bool(torch.equal(left,right))


def _check_valid(torch, ops, fixture, old, new, acceptance):
    _paired_status(torch, old, new)
    reference = reuse._snapshot(old)
    reuse._assert_same_bits(torch, reference, new, 'striped valid partial/LSE/status')
    old_merge = reuse._merge_and_oracle(torch, ops, fixture, old, acceptance['fused_attention'])
    new_merge = reuse._merge_and_oracle(torch, ops, fixture, new, acceptance['fused_attention'])
    fast._same_merge(torch, old_merge, new_merge, 'striped valid merge')
    if old['cluster_stats'] is not None and not torch.equal(old['cluster_stats'], new['cluster_stats']):
        raise RuntimeError('unchanged striped prefill schedule changed cluster counters')
    return reference


def _check_invalid(torch, old, new):
    """Compare the public error domain without depending on NaN payload bits."""
    _paired_status(torch, old, new, invalid=True)
    for name in ('partial', 'lse'):
        a, b = old[name].cpu(), new[name].cpu()
        if not torch.equal(torch.isnan(a), torch.isnan(b)):
            raise RuntimeError(f'striped invalid input changed {name} NaN locations')
        if (not torch.equal(torch.isinf(a), torch.isinf(b)) or
                not torch.equal(torch.signbit(a[torch.isinf(a)]),
                                torch.signbit(b[torch.isinf(b)]))):
            raise RuntimeError(f'striped invalid input changed {name} infinity locations')
        finite = torch.isfinite(a) & torch.isfinite(b)
        if not torch.equal(a[finite].view(torch.int32), b[finite].view(torch.int32)):
            raise RuntimeError(f'striped invalid input changed finite {name} bits')
    return {'error_task_parity': 'passed', 'nan_classification': 'passed'}


def _graph(torch, launch_old, launch_new, old_t, new_t, old, new, original):
    graphs = [torch.npu.NPUGraph(), torch.npu.NPUGraph()]
    for graph, launch in zip(graphs, (launch_old, launch_new)):
        with torch.npu.graph(graph, capture_error_mode='thread_local', auto_dispatch_capture=True):
            launch()
    torch.npu.synchronize()
    # First vary only current K/V. This proves their graph inputs were not
    # frozen at capture; a later Q/task change cannot mask an ignored K/V.
    old_t['ck'].neg_(); old_t['cv'].neg_()
    for buffers in (old,new): reuse._poison(torch,buffers)
    for graph in graphs: graph.replay()
    torch.npu.synchronize()
    _paired_status(torch,old,new)
    kv_changed=reuse._snapshot(old)
    reuse._assert_same_bits(torch,kv_changed,new,'striped changed-KV graph')
    if reuse._bitwise_identical(torch,original['partial'],kv_changed['partial']):
        raise RuntimeError('graph ignored same-address changed current K/V')
    # Same captured addresses, fresh poison, changed Q values and history end.
    old_t['q'].neg_(); old_t['qr'].neg_()
    changed = old['tasks'].cpu()
    row = next((r for r in changed if int(r[7]) == 0 and int(r[1]) > 0 and int(r[4]-r[3]) > 32), None)
    if row is None:
        raise RuntimeError('graph case has no mutable live historical task')
    row[4] -= 16
    old['tasks'].copy_(changed)
    for buffers in (old,new): reuse._poison(torch,buffers)
    for graph in graphs: graph.replay()
    torch.npu.synchronize()
    _paired_status(torch,old,new)
    changed_bits = reuse._snapshot(old)
    reuse._assert_same_bits(torch,changed_bits,new,'striped changed-input graph')
    if all(reuse._bitwise_identical(torch,original[k],changed_bits[k]) for k in ('partial','lse')):
        raise RuntimeError('graph ignored changed inputs')
    for buffers, launch in ((old,launch_old),(new,launch_new)):
        reuse._poison(torch,buffers);launch();torch.npu.synchronize()
        reuse._assert_same_bits(torch,changed_bits,buffers,'striped graph/eager')
    return {'capture':'passed','replay':'passed',
            'same_address_changed_key_value_alone':True,
            'same_address_changed_query_key_value_task':True}


def run_case(torch, ops, case, device, cores, acceptance):
    fixture = reuse.make_fixture(torch,case.shape)
    old_t = {k:v.to(device) for k,v in fixture['cpu'].items()}
    new_t = {**old_t,'raw':raw_to_striped(fixture).to(device)}
    old = _allocate(torch,fixture,device,cores,case.cluster)
    new = _allocate(torch,fixture,device,cores,case.cluster)
    reuse._prepare(torch,ops,old_t,fixture,old)
    new['tasks'],new['positions'] = old['tasks'],old['positions']
    def old_launch(): _launch(ops,case.old,old_t,fixture,old,cores,case.cluster)
    def new_launch(): _launch(ops,case.new,new_t,fixture,new,cores,case.cluster)
    for b,launch in ((old,old_launch),(new,new_launch)):
        reuse._poison(torch,b);launch()
    torch.npu.synchronize()
    invalid = case.shape.corrupt_metadata or case.shape.corrupt_qr
    original = (_check_invalid(torch,old,new) if invalid else
                _check_valid(torch,ops,fixture,old,new,acceptance))
    timings = None
    if case.timed:
        samples = {'old':[],'new':[]}
        for i in range(7):
            order = ((old,old_launch,'old'),(new,new_launch,'new'))
            if i >= 2 and (i-2)%2: order=order[::-1]
            for b,launch,label in order:
                reuse._poison(torch,b)
                begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                begin.record();launch();end.record();end.synchronize()
                elapsed=float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed<=0: raise RuntimeError('invalid NPU event')
                reuse._assert_same_bits(torch,original,b,'striped timed repeat')
                if i>=2:samples[label].append(elapsed)
        a,b=(statistics.median(samples[k]) for k in ('old','new'))
        timings={'old_ms':a,'new_ms':b,'ratio':b/a,'samples_ms':samples,
                 'gate':'passed' if b<=a else 'failed','scope':'CV_device_events_only'}
    graph=_graph(torch,old_launch,new_launch,old_t,new_t,old,new,original) if case.graph else None
    row={'case':case.shape.name,'old_operator':case.old,'new_operator':case.new,
         'precision':('error_contract_passed' if invalid else
                      'bitwise_and_frozen_oracle_passed'),
         'invalid_evidence':original if invalid else None,
         'timings':timings,'graph':graph,
         'status':'passed' if timings is None or timings['gate']=='passed' else 'failed'}
    print('[oscar] PERF_STRIPED '+json.dumps({
        'case':case.shape.name,'status':row['status'],'precision':row['precision'],
        'old_ms':timings['old_ms'] if timings else None,
        'new_ms':timings['new_ms'] if timings else None,
        'ratio':timings['ratio'] if timings else None,
        'graph':graph['replay'] if graph else 'not_run'},sort_keys=True),flush=True)
    return row


def store_gate(torch,ops,device,cores,acceptance):
    # Current rows cross virtual128/page boundaries and exact-window wrap;
    # the actual new writer is read by old and striped readers after commit.
    spec=reuse.Shape('striped_store', (385,), (641,),256,1,1,True)
    fixture=reuse.make_fixture(torch,spec)
    tensors={k:v.to(device) for k,v in fixture['cpu'].items()}
    buffers=reuse._allocate(torch,fixture,device,cores,candidate=False)
    reuse._prepare(torch,ops,tensors,fixture,buffers)
    rk=reuse._hadamard(torch,256).T.contiguous().to(device)
    rv=tensors['rv'].T.contiguous()
    results=[]
    for name,raw in (('rotate_clip_store_out',fixture['cpu']['raw']),
                     ('rotate_clip_store_striped_out',raw_to_striped(fixture))):
        state={k:tensors[k].clone() for k in ('wk','wv','tags')}
        state['raw']=raw.to(device)
        status=torch.full((fixture['tokens'],1),-99,dtype=torch.int32,device=device)
        getattr(ops,name)(tensors['ck'],tensors['cv'],rk,rv,tensors['slots'],buffers['positions'],
            state['raw'],state['wk'],state['wv'],state['tags'],status,
            reuse.BLOCK_TOKENS,fixture['blocks'],reuse.PREFIX,fixture['stride'],
            reuse.SINK,spec.recent_tokens+reuse.SPECULATIVE,0.,0.,False)
        torch.npu.synchronize()
        if not bool((status.cpu()==0).all()):raise RuntimeError('striped store status failure')
        state={k:v.cpu() for k,v in state.items()};state['status']=status.cpu();results.append(state)
    legacy,striped=results
    for k in ('wk','wv','tags','status'):
        if not _store_bits_equal(torch,legacy[k],striped[k]):
            raise RuntimeError('striped store changed '+k+' bits')
    # Decode every slot, including untouched slots; preserve all raw prefix and padding.
    restored=striped['raw'].clone()
    payload=reuse.BLOCK_TOKENS*136
    for page in range(fixture['blocks']):
        begin=reuse.PREFIX+page*fixture['stride']
        restored[begin:begin+payload]=striped_slots_to_canonical(
            striped['raw'][begin:begin+payload].view(-1,136)).flatten()
    if not torch.equal(restored,legacy['raw']):raise RuntimeError('striped store packed bytes differ')
    # The two writers must have actually committed the current chunk; equal
    # unchanged poison would make a byte-layout comparison meaningless.
    if torch.equal(legacy['raw'],fixture['cpu']['raw']):
        raise RuntimeError('store gate did not change any compressed cache byte')
    reader=_reader_after_writer(torch,ops,device,cores,fixture,tensors,legacy,striped)
    graph=_store_changed_input_graph(torch,ops,device,fixture,tensors,buffers,rk,rv)
    return {'status':'passed','raw_inverse_bitwise':'passed',
            'precise_windows_bitwise':'passed','reader_after_writer':reader,
            'graph_changed_key_value':graph}


def _reader_after_writer(torch,ops,device,cores,fixture,tensors,legacy,striped):
    """Use newly stored current K/V as history in the next synthetic task.

    The final seven-query group is an existing unique owner, so replacing only
    its source0 interval creates no second output writer. Other source windows
    can be overwritten by this store; this checks one specified source0 task,
    not a complete next-step model attention result.
    """
    old_t={**tensors, **{k:legacy[k].to(device) for k in ('raw','wk','wv','tags')}}
    new_t={**tensors, **{k:striped[k].to(device) for k in ('raw','wk','wv','tags')}}
    old=_allocate(torch,fixture,device,cores,1)
    new=_allocate(torch,fixture,device,cores,1)
    reuse._prepare(torch,ops,old_t,fixture,old)
    tasks=old['tasks'].cpu()
    task_id,begin,end,owner,count=_retarget_reader_task(tasks,fixture)
    old['tasks'].copy_(tasks)
    new['tasks'],new['positions']=old['tasks'],old['positions']
    for name,inputs,buffers in (
            ('attention_cv_fast_out',old_t,old),
            ('attention_cv_striped_out',new_t,new)):
        reuse._poison(torch,buffers)
        _launch(ops,name,inputs,fixture,buffers,cores,1)
    torch.npu.synchronize()
    for buffers in (old,new):
        if not bool((buffers['status'][task_id].cpu()==0).all()):
            raise RuntimeError('newly stored history failed source0 reader status')
        if not bool(torch.isfinite(buffers['partial'][owner:owner+count,:,0]).all()) or not bool(
                torch.isfinite(buffers['lse'][owner:owner+count,:,0]).all()):
            raise RuntimeError('newly stored history reader produced nonfinite output')
    for name in ('partial','lse'):
        if not reuse._bitwise_identical(
                torch,old[name][owner:owner+count,:,0],
                new[name][owner:owner+count,:,0]):
            raise RuntimeError(f'new writer→reader history {name} differs bitwise')
    return {'status':'passed','source0_task_id':task_id,
            'source0_history_range':[begin,end],
            'canonical_vs_striped_reader':'bitwise_passed',
            'scope':'single_source0_task_after_store_not_full_next_step'}


def _retarget_reader_task(tasks,fixture):
    """Retarget one existing leader to the just-written cached interval."""
    spec=fixture['spec']
    if (spec.qlens!=(385,) or spec.contexts!=(641,) or spec.kv_heads!=1 or
            spec.heads!=6 or spec.splits!=1):
        raise RuntimeError('reader-after-writer fixture geometry changed')
    qtile=128//(spec.heads//spec.kv_heads)
    owner=((fixture['actual_tokens']-1)//qtile)*qtile
    count=fixture['actual_tokens']-owner
    row=tasks.view(fixture['tokens'],1,3,1,16)[owner,0,0,0]
    if (int(row[0]),int(row[1]),int(row[7]))!=(owner,count,0):
        raise RuntimeError('reader-after-writer source0 has no unique final owner')
    begin,end=spec.contexts[0],spec.contexts[0]+spec.qlens[0]
    row[3],row[4],row[8]=begin,end,end
    return owner*3,begin,end,owner,count


def _store_changed_input_graph(torch,ops,device,fixture,tensors,buffers,rk,rv):
    """Capture the actual striped writer, then mutate K/V at fixed addresses."""
    graph_k=tensors['ck'].clone()
    graph_v=tensors['cv'].clone()
    initial={k:tensors[k].clone() for k in ('wk','wv','tags')}
    initial['raw']=raw_to_striped(fixture).to(device)
    state={k:v.clone() for k,v in initial.items()}
    status=torch.full((fixture['tokens'],1),-99,dtype=torch.int32,device=device)
    def launch():
        ops.rotate_clip_store_striped_out(
            graph_k,graph_v,rk,rv,tensors['slots'],buffers['positions'],
            state['raw'],state['wk'],state['wv'],state['tags'],status,
            reuse.BLOCK_TOKENS,fixture['blocks'],reuse.PREFIX,fixture['stride'],
            reuse.SINK,fixture['spec'].recent_tokens+reuse.SPECULATIVE,0.,0.,False)
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph,capture_error_mode='thread_local',auto_dispatch_capture=True):
        launch()
    # Capture only records operations on some runtimes. A first replay must
    # produce the original-input reference before any K/V mutation.
    graph.replay()
    torch.npu.synchronize()
    captured={k:v.clone() for k,v in state.items()}
    captured_status=status.clone()
    if not bool((captured_status.cpu()==0).all()):
        raise RuntimeError('striped store graph capture produced invalid status')
    before_k,before_v=graph_k.data_ptr(),graph_v.data_ptr()
    graph_k.neg_();graph_v.neg_()
    if graph_k.data_ptr()!=before_k or graph_v.data_ptr()!=before_v:
        raise RuntimeError('store graph mutation changed K/V addresses')
    def reset():
        for key in state:state[key].copy_(initial[key])
        status.fill_(-99)
    reset();graph.replay();torch.npu.synchronize()
    changed={k:v.clone() for k,v in state.items()}
    changed_status=status.clone()
    if not bool((changed_status.cpu()==0).all()):
        raise RuntimeError('striped store graph replay produced invalid status')
    if reuse._bitwise_identical(torch,captured['raw'],changed['raw']):
        raise RuntimeError('striped store graph ignored changed K/V inputs')
    reset();launch();torch.npu.synchronize()
    for key in state:
        if not _store_bits_equal(torch,changed[key],state[key]):
            raise RuntimeError(f'striped store changed-input graph/eager {key} differs')
    if not torch.equal(changed_status,status):
        raise RuntimeError('striped store changed-input graph/eager status differs')
    return {'capture':'passed','replay':'passed','same_address_changed_key_value':True,
            'eager_parity':'bitwise_passed'}


def probe(config_path,acceptance_path):
    target=json.loads(config_path.read_text());acceptance=json.loads(acceptance_path.read_text())
    reuse._active_device(target)
    policy=acceptance['performance']
    if (policy['warmup'],policy['repeats'],policy['max_latency_ratio'])!=(2,5,1.0):
        raise RuntimeError('frozen performance gate changed')
    import torch
    import torch_npu  # noqa: F401
    torch.set_num_threads(min(4,torch.get_num_threads()));torch.npu.set_device(0)
    if not torch.npu.is_available():raise RuntimeError('target NPU required')
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    from .probe_native_current_fia import _target_geometry
    if _target_geometry(target)!=(6,1,256):raise RuntimeError('striped target requires Hq6/Hkv1/D256')
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    cases=case_plan()
    require_capabilities({c.new for c in cases}|{'rotate_clip_store_striped_out'},ROOT/'build/ascendc/build_manifest.json')
    ops=torch.ops.oscar_ascend_ops;device=torch.device('npu:0');cores=reuse._core_count(torch,target)
    writer=store_gate(torch,ops,device,cores,acceptance)
    rows=[]
    for case in cases:
        rows.append(run_case(torch,ops,case,device,cores,acceptance))
        gc.collect();torch.npu.empty_cache()
    passed=all(row['status']=='passed' for row in rows)
    return {'status':'passed' if passed else 'failed','precision':'passed',
            'graph_capture':'passed','graph_replay':'passed',
            'performance':'passed' if passed else 'failed','format':FORMAT,'writer':writer,'cases':rows,
            'artifact_signature':manifest['signature'],'artifact_sha256':manifest['sha256'],
            'full_model_performance':'not_established'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/target.json')
    parser.add_argument('--acceptance',type=Path,default=ROOT/'configs/acceptance.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try: report=probe(args.config,args.acceptance)
    except Exception as exc:
        traceback.print_exc();report={'status':'failed','first_error':str(exc)}
    atomic_json(args.output,report)
    print('[oscar] PERF_STRIPED_RESULT '+json.dumps({k:v for k,v in report.items()
          if k not in ('cases','artifact_sha256')}|{'report':str(args.output)}),flush=True)
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
