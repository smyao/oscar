"""Archive #37-49/#126/#145/#164: v4 probe routing/domain checks on the host.

CPU spies verify wiring only. The real gate always requires NPU and its graph
replay checks use the independent v4 oracle, never an old mixed-domain golden.
"""
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools import probe_official_attention as probe
from tools.official_fixture import make_fixture
from tools.official_reference import fixture_tasks_reference, pack_kv, rotate_bf16
from tools.probe_history_reuse import Shape


class FinalCopyOps:
    """CPU wiring spy for the frozen copy primitive, never a device gate."""
    def copy_validate_current_out(self,value,lse,out,out_lse,status):
        if hasattr(self,'events'):self.events.append('final_copy')
        source_lse=lse.squeeze(-1)
        live=torch.isfinite(source_lse)
        invalid=(~live)&(~torch.isneginf(source_lse))
        invalid|=live&(~torch.isfinite(value.float().sum(-1)))
        out.copy_(torch.where(live[...,None],value,torch.zeros_like(value)))
        out_lse.copy_(torch.where((~live)&(~torch.isneginf(source_lse)),
                                  -torch.inf,source_lse+0.0))
        status.copy_(invalid.int()*2)

    def status_guard(self,*statuses):
        assert len(statuses)==4
        if hasattr(self,'events'):self.events.append('guard')
        if hasattr(self,'guard_states'):
            self.guard_states.append([tensor.clone() for tensor in statuses])
        if any(bool(value.ne(0).any()) for value in statuses):
            raise RuntimeError('device status guard trap (host wiring spy)')


def fixture_and_tasks(length=2,context=65,recent=1,wide_history=False):
    fixture = make_fixture(torch,Shape('official_host',(length,),(context,),256,1,1,False,
                                      recent_tokens=recent,padded_tokens=length+1))
    tasks = torch.zeros((fixture['tokens']*3,16),dtype=torch.int64)
    sink=min(64,context);cut=fixture['cached_quant_ends'][0]
    for token in range(length):
        for kind,(low,high) in enumerate(((sink,context if wide_history else cut),
                                        (0,sink+context-cut),(context,context+length))):
            tasks[token*3+kind,:11] = torch.tensor([token,1,0,low,high,0,kind,kind,context,0,0])
    for kind in range(3):
        tasks[length*3+kind,:11] = torch.tensor([length,-1,0,0,0,0,kind,kind,0,0,0])
    return fixture,tasks


@pytest.mark.parametrize('family',('q4','q1','c4','c16'))
@pytest.mark.parametrize('mode',(0,1))
def test_all_four_v4_reader_abis_receive_rotated_inputs_and_explicit_mode(family,mode):
    calls=[]
    class Ops:
        def __getattr__(self,name):return lambda *args:calls.append((name,args))
    tensors={name:torch.empty(1) for name in ('q','qr','ck','cv','raw','table','wk','wv','tags')}
    tensors['rv']=torch.eye(256,dtype=torch.bfloat16)
    buffers={name:torch.empty(1) for name in ('tasks','partial','lse','status','workspace','cluster_stats')}
    fixture={'spec':SimpleNamespace(recent_tokens=256,splits=1),'blocks':2,'stride':512*144,'scale':1/16}
    probe.launch(Ops(),family,tensors,fixture,buffers,7,dequant_mode=mode,merge=False)
    name,args=calls[0]
    assert name==probe.OPERATORS[family] and '_official_' in name
    assert args[0] is tensors['q'] and args[1] is tensors['qr'] and args[4] is tensors['rv']
    assert args[-3:]==(1/16,mode,7)
    assert len(args)==(27 if probe.CLUSTER[family]>1 else 26)


@pytest.mark.parametrize('mode',(0,1))
def test_fixture_all_sources_share_rotated_frame_and_preserve_original_runtime_inputs(mode):
    fixture,tasks=fixture_and_tasks()
    data,original=fixture['cpu'],fixture['original_inputs']
    assert fixture['cache_format']==probe.CACHE_FORMAT=='striped_rotated_fp32_v3'
    assert fixture['cached_quant_ends']==[65] and fixture['window_ring_tokens']==4
    assert torch.equal(data['q'],data['qr'])
    assert torch.equal(data['q'],rotate_bf16(original['q'],fixture['rotation_k']))
    assert torch.equal(data['ck'],rotate_bf16(original['ck'],fixture['rotation_k']))
    assert torch.equal(data['cv'],rotate_bf16(original['cv'],data['rv']))
    keys,values=fixture['original_histories'][0]
    page=fixture['page_assignments'][0][0]
    torch.testing.assert_close(data['wk'][page,:64],rotate_bf16(keys[:64],fixture['rotation_k']),atol=0,rtol=0)
    torch.testing.assert_close(data['wv'][page,:64],rotate_bf16(values[:64],data['rv']),atol=0,rtol=0)
    result=fixture_tasks_reference(fixture,tasks,mode)
    assert result['contract']==probe.CONTRACT
    assert result['merged_rotated'].dtype==torch.float32 and result['output'].dtype==torch.bfloat16
    assert result['output'][-1].eq(0).all()
    assert not torch.equal(result['pre_inverse_bf16'][0],result['output'][0])


def test_prepare_uses_new_op_and_checks_complete_fixed_map_without_old_prepare():
    fixture,tasks=fixture_and_tasks()
    calls=[]
    class Ops:
        def prepare_official_attention_tasks_out(self,starts,lens,slots,out,positions,*args):
            calls.append(probe.PREPARE);out.copy_(tasks);positions.copy_(torch.tensor([65,66,-1]))
        def prepare_attention_tasks_out(self,*args):pytest.fail('old prepare cannot qualify v4')
    class TorchProxy:
        npu=SimpleNamespace(synchronize=lambda:None)
        def __getattr__(self,name):return getattr(torch,name)
    buffers={'tasks':torch.empty_like(tasks),'positions':torch.empty(3,dtype=torch.int64)}
    result=probe.prepare(TorchProxy(),Ops(),fixture['cpu'],fixture,buffers)
    assert calls==[probe.PREPARE] and result['fixed_cached_map']=='exact'
    broken=tasks.clone();broken[3,4]=64
    with pytest.raises(RuntimeError,match='omitted'):
        probe.check_task_coverage(fixture,broken)


def test_wide_history_tasks_are_checked_after_fixed_cut_mask_but_window_coverage_is_exact():
    fixture,tasks=fixture_and_tasks(context=80,recent=8,wide_history=True)
    assert fixture['cached_quant_ends']==[73]
    assert tasks[0,4]==80 and tasks[1,4]==71
    probe.check_task_coverage(fixture,tasks)
    result=fixture_tasks_reference(fixture,tasks,0)
    assert torch.isfinite(result['output'][:2]).all()
    broken=tasks.clone();broken[0,4]=72
    with pytest.raises(RuntimeError,match='omitted'):
        probe.check_task_coverage(fixture,broken)
    broken=tasks.clone();broken[1,4]=70
    with pytest.raises(RuntimeError,match='omitted'):
        probe.check_task_coverage(fixture,broken)


def test_launch_merges_fp32_rotated_partials_then_casts_once_before_inverse():
    fixture,tasks=fixture_and_tasks()
    golden=fixture_tasks_reference(fixture,tasks,0)
    buffers=probe.allocate(torch,fixture,torch.device('cpu'),1,'q4')
    calls=[]
    class Ops(FinalCopyOps):
        events=calls
        def attention_cv_official_decode_out(self,*args):
            calls.append('cv');buffers['partial'].copy_(golden['partial']);buffers['lse'].copy_(golden['lse'])
            buffers['status'].zero_()
        def merge_lse_out(self,partial,lse,out,out_lse,status):
            calls.append('merge');out.copy_(golden['merged_rotated'].view_as(out))
            out_lse.copy_(golden['output_lse'].view_as(out_lse));status.zero_()
    probe.launch(Ops(),'q4',fixture['cpu'],fixture,buffers,1,dequant_mode=0)
    assert calls==['cv','merge','final_copy','guard','guard']
    assert torch.equal(buffers['pre_inverse'],golden['pre_inverse_bf16'].reshape_as(buffers['pre_inverse']))
    assert torch.equal(buffers['bf16'],golden['output'].reshape_as(buffers['bf16']))
    assert not torch.equal(buffers['pre_inverse'],buffers['bf16'])
    assert len({buffers[name].data_ptr() for name in
                ('output','pre_inverse','inverse_result','bf16','output_lse','final_lse','final_status')})==7


def test_output_gate_checks_fp32_merge_and_final_inverse_independently():
    fixture,tasks=fixture_and_tasks()
    golden=fixture_tasks_reference(fixture,tasks,0)
    buffers=probe.allocate(torch,fixture,torch.device('cpu'),1,'q4')
    buffers['tasks'].copy_(tasks);buffers['status'].zero_();buffers['merge_status'].zero_();buffers['final_status'].zero_()
    for name,field in (('partial','partial'),('lse','lse'),('output','merged_rotated'),
                       ('output_lse','output_lse'),('pre_inverse','pre_inverse_bf16'),
                       ('inverse_result','output'),('final_lse','output_lse'),('bf16','output')):
        buffers[name].copy_(golden[field].reshape_as(buffers[name]))
    assert probe.check_outputs(torch,fixture,buffers,golden,tasks,'q4')['status']=='passed'
    buffers['bf16'].copy_(buffers['pre_inverse'])
    with pytest.raises(RuntimeError,match='bf16'):
        probe.check_outputs(torch,fixture,buffers,golden,tasks,'q4')


@pytest.mark.parametrize('failure',('inverse_nan','copy_lse_nan','source_error_priority'))
def test_final_copy_fault_is_guarded_after_original_source_and_merge_status(failure):
    fixture,tasks=fixture_and_tasks()
    golden=fixture_tasks_reference(fixture,tasks,0)
    buffers=probe.allocate(torch,fixture,torch.device('cpu'),1,'q4')
    events=[]
    class Ops(FinalCopyOps):
        guard_states=[]
        def attention_cv_official_decode_out(self,*args):
            buffers['partial'].copy_(golden['partial']);buffers['lse'].copy_(golden['lse'])
            buffers['status'].zero_()
            if failure=='source_error_priority':buffers['status'][0,0]=3
        def merge_lse_out(self,partial,lse,out,out_lse,status):
            out.copy_(golden['merged_rotated'].view_as(out))
            out_lse.copy_(golden['output_lse'].view_as(out_lse));status.zero_()
            if failure=='copy_lse_nan':out_lse[0]=float('nan')
    ops=Ops();ops.events=events
    if failure in ('inverse_nan','source_error_priority'):
        fixture['cpu']['rv'].fill_(float('nan'))
    with pytest.raises(RuntimeError,match='status guard trap'):
        probe.launch(ops,'q4',fixture['cpu'],fixture,buffers,1,dequant_mode=0)
    assert bool(buffers['final_status'].ne(0).any())
    if failure=='source_error_priority':
        assert events==['final_copy','guard']
        assert ops.guard_states[0][0][0,0]==3
    else:
        assert events==['final_copy','guard','guard']
        assert all(not bool(value.ne(0).any()) for value in ops.guard_states[0])
        assert bool(ops.guard_states[1][0].ne(0).any())


def test_no_npu_cannot_produce_a_cpu_success_receipt(tmp_path,monkeypatch):
    # The CLI owns its process environment. This host double runs in pytest,
    # so keep _active_device's writes out of later readiness tests.
    import os
    monkeypatch.setattr(os,'environ',dict(os.environ))
    config=json.loads((probe.ROOT/'configs/target.json').read_text())
    config.update(experimental_official_math=True,experimental_aligned_math=True,experimental_bf16_attention=True)
    path=tmp_path/'config.json';path.write_text(json.dumps(config));out=tmp_path/'report.json'
    def unavailable():raise RuntimeError('requires real NPU; no CPU substitution')
    monkeypatch.setattr(probe,'require_real_npu',unavailable)
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES',raising=False)
    assert probe.main(['npu','--config',str(path),'--output',str(out)])==2
    report=json.loads(out.read_text())
    assert report['status']=='failed' and report['NPU']=='not_passed'
    assert report['contract']==probe.CONTRACT


def test_v3_configuration_is_rejected_before_device_access(tmp_path,monkeypatch):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'experimental_aligned_math':True,'experimental_bf16_attention':True}))
    monkeypatch.setattr(probe,'require_real_npu',lambda:pytest.fail('old configuration reached the device'))
    with pytest.raises(RuntimeError,match='experimental_official_math=True'):
        probe.run_npu(path)


@pytest.mark.parametrize('family,members', (('c4', 4), ('c16', 16)))
@pytest.mark.parametrize('mode', (0, 1))
def test_single_history_tile_gate_keeps_cluster_and_graph_allocation(family,members,mode):
    from dataclasses import replace
    case = next(row for row in probe.case_plan() if row['shape'].name ==
                f'official_{family}_single_history_tile_mode{mode}')
    shape = case['shape']
    assert case['family'] == family and case['dequant_mode'] == mode
    assert case['graph'] and not case['timed'] and not shape.speed_gate
    assert shape.expect_clusters and shape.qlens == (members * (128 // 6),)
    first = make_fixture(torch,shape)
    changed = make_fixture(torch,replace(shape,contexts=(shape.contexts[0]+32,)))
    assert first['blocks'] == changed['blocks']
    for key,value in first['cpu'].items():
        assert value.shape == changed['cpu'][key].shape
    for fixture in (first,changed):
        assert fixture['cached_quant_ends'] == list(fixture['spec'].contexts)
        assert 0 < fixture['spec'].contexts[0] - fixture['sink_tokens'] <= 256


@pytest.mark.parametrize('mode', (0, 1))
def test_c4_l1_ordinary_gate_has_three_sources_and_stable_replay_shapes(mode):
    from dataclasses import replace
    case = next(row for row in probe.case_plan() if row['shape'].name ==
                f'official_c4_l1_ordinary_sources_mode{mode}')
    shape = case['shape']
    assert case['family'] == 'c4' and case['dequant_mode'] == mode
    assert case['graph'] and not case['timed'] and not shape.speed_gate
    assert not shape.expect_clusters
    assert 128 // 6 < shape.qlens[0] < 4 * (128 // 6)
    first = make_fixture(torch, shape)
    changed = make_fixture(torch, replace(shape, contexts=(shape.contexts[0]+32,)))
    for name, value in first['cpu'].items():
        assert value.shape == changed['cpu'][name].shape
    for fixture in (first, changed):
        # More than one history and exact-window KV256 block, with a
        # nonempty partial current source, all handled by the C1 wrapper.
        context = fixture['spec'].contexts[0]
        cut = fixture['cached_quant_ends'][0]
        assert cut - fixture['sink_tokens'] > 256
        assert fixture['sink_tokens'] + context - cut > 256
        assert 0 < shape.qlens[0] < 256


def complete_receipt():
    rows=[]
    for case in probe.case_plan():
        shape=case['shape'];graph=None
        if case['graph']:
            graph={'capture':'passed','replay':'passed','same_addresses':True,'final_copy_and_guard':'passed','states':[
                {'state':name,'v4_oracle':'passed','eager_replay_bitwise':'passed'}
                for name in ('original','changed_query','changed_context','changed_page_map')]}
        rows.append({'case':shape.name,'family':case['family'],'operator':probe.OPERATORS[case['family']],
            'prepare_operator':probe.PREPARE,'dequant_mode':case['dequant_mode'],'status':'passed',
            'all_tasks':(shape.padded_tokens or sum(shape.qlens))*shape.kv_heads*3*shape.splits,
            'check':{'status':'passed'},'timing':None,'graph':graph})
    return {'status':'passed','contract':probe.CONTRACT,'cache_format':probe.CACHE_FORMAT,'slot_bytes':144,
        **{key:'passed' for key in ('device_completion','precision','error_domain','graph_capture','graph_replay')},
        'performance':'not_run_user_requested','acceptance_sha256':probe.digest(probe.ROOT/'configs/acceptance_official_bf16.json'),
        'oracle_sha256':probe.digest(probe.ROOT/'tools/official_reference.py'),'probe_sha256':probe.digest(probe.__file__),
        'source_binding':probe.source_binding(),'operator_families':probe.OPERATORS,'cases':rows,
        'writer_rotation':{'status':'passed','writer':probe.WRITER,'query_copy':probe.QUERY_COPY,
            'codec_bytes':'exact','rotated_windows':'exact','canaries':'exact','query_status':'exact','writer_status':'exact',
            'graph_capture':'passed','graph_replay':'passed','graph_same_addresses':True,
            'active_rows':{'status':'passed','codec_window_status':'exact','same_addresses':True,
                'eager_and_graph':'passed','timing':None,
                'states':['interleaved','all_padding','changed_active_rows','all_live']}}}


@pytest.mark.parametrize('failure',('v3','old_prepare','old_reader','old_window','missing_case','timing','old_graph_oracle','missing_copy_graph','missing_padding_gate','incomplete_padding_gate','missing_page_map_graph','unchecked_page_map_graph'))
def test_v3_or_incomplete_success_receipts_cannot_certify_v4(failure):
    report=complete_receipt();assert probe.validate_evidence(report)==[]
    if failure=='v3':report['contract']='oscar_bf16_attention_v3'
    elif failure=='old_prepare':report['cases'][0]['prepare_operator']='prepare_attention_tasks_out'
    elif failure=='old_reader':report['cases'][0]['operator']='attention_cv_aligned_decode_out'
    elif failure=='old_window':report['writer_rotation'].pop('rotated_windows')
    elif failure=='missing_case':report['cases'].pop()
    elif failure=='timing':report['cases'][0]['timing']={'passed':True}
    elif failure=='old_graph_oracle':report['cases'][0]['graph']['states'][0].pop('v4_oracle')
    elif failure=='missing_copy_graph':report['cases'][0]['graph'].pop('final_copy_and_guard')
    elif failure=='missing_padding_gate':report['writer_rotation'].pop('active_rows')
    elif failure=='incomplete_padding_gate':report['writer_rotation']['active_rows']['states'].remove('all_padding')
    elif failure=='missing_page_map_graph':report['cases'][0]['graph']['states'].pop()
    elif failure=='unchecked_page_map_graph':report['cases'][0]['graph']['states'][-1].pop('v4_oracle')
    assert probe.validate_evidence(report)


def tensor_bits(tensor):
    return tensor.contiguous().view(torch.uint8)


def test_page_permutation_preserves_all_logical_cache_bytes_padding_and_pointers():
    from tools import probe_history_reuse as reuse
    fixture = make_fixture(torch, Shape('page_map_host', (2, 1), (641, 65), 256, 2, 1, False,
                                       recent_tokens=320, padded_tokens=5))
    data, blocks, stride = fixture['cpu'], fixture['blocks'], fixture['stride']
    before = {name: value.clone() for name, value in data.items()}
    assignments = fixture['page_assignments']
    pointers = {name: value.data_ptr() for name, value in data.items()}
    fingerprint = fixture['input_sha256']
    assert bool(data['table'].lt(0).any()) and bool(data['slots'].lt(0).any())
    probe.permute_physical_pages(torch, fixture)
    assert pointers == {name: value.data_ptr() for name, value in data.items()}
    assert fixture['input_sha256'] != fingerprint
    assert not torch.equal(before['table'], data['table'])
    assert not torch.equal(before['slots'], data['slots'])
    for name in ('table', 'slots'):
        mask = before[name] < 0
        assert torch.equal(data[name][mask], before[name][mask])
    assert torch.equal(data['raw'][:reuse.PREFIX], before['raw'][:reuse.PREFIX])
    old_pages = before['raw'][reuse.PREFIX:].view(blocks, stride)
    new_pages = data['raw'][reuse.PREFIX:].view(blocks, stride)
    for old in range(blocks):
        new = (old + 1) % blocks
        assert torch.equal(old_pages[old], new_pages[new])
        for name in ('wk', 'wv', 'tags'):
            assert torch.equal(tensor_bits(before[name][old]), tensor_bits(data[name][new]))
    for request, pages in enumerate(assignments):
        assert fixture['page_assignments'][request] == tuple((page + 1) % blocks for page in pages)
    # Resolve every virtual128 entry and active current slot independently.
    ratio = reuse.BLOCK_TOKENS // 128
    valid = before['table'] >= 0
    old, new = before['table'][valid].long(), data['table'][valid].long()
    assert bool((new != old).all())
    assert torch.equal(new // ratio, (old // ratio + 1) % blocks)
    assert torch.equal(new % ratio, old % ratio)
    valid = before['slots'] >= 0
    old, new = before['slots'][valid], data['slots'][valid]
    assert torch.equal(new // reuse.BLOCK_TOKENS, (old // reuse.BLOCK_TOKENS + 1) % blocks)
    assert torch.equal(new % reuse.BLOCK_TOKENS, old % reuse.BLOCK_TOKENS)
    for name in set(data) - {'raw', 'table', 'slots', 'wk', 'wv', 'tags'}:
        assert torch.equal(tensor_bits(data[name]), tensor_bits(before[name]))


@pytest.mark.parametrize('context,recent', ((65, 1), (641, 320)))
@pytest.mark.parametrize('mode', (0, 1))
def test_page_map_relocation_keeps_independent_oracle_but_stale_table_does_not(context,recent,mode):
    fixture, tasks = fixture_and_tasks(context=context, recent=recent)
    want = fixture_tasks_reference(fixture, tasks, mode)
    probe.reserve_graph_remap_page(torch, fixture)
    before_table = fixture['cpu']['table'].clone()
    addresses = {name: value.data_ptr() for name, value in fixture['cpu'].items()}
    probe.permute_physical_pages(torch, fixture)
    assert addresses == {name: value.data_ptr() for name, value in fixture['cpu'].items()}
    got = fixture_tasks_reference(fixture, tasks, mode)
    for name, value in want.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(tensor_bits(value), tensor_bits(got[name])), name
    # Negative control: relocated payloads with the OLD table are wrong.
    # With the poisoned spare this addresses dead bytes/tags; with two live
    # pages it addresses another logical page. It must not certify the oracle.
    fixture['cpu']['table'].copy_(before_table)
    try:
        stale = fixture_tasks_reference(fixture, tasks, mode)
    except ValueError:
        return
    assert not torch.equal(tensor_bits(want['output']), tensor_bits(stale['output']))


@pytest.mark.parametrize('family', ('c4', 'c16'))
@pytest.mark.parametrize('mode', (0, 1))
def test_single_history_graph_cases_reserve_spare_before_allocation_and_replay(family,mode):
    from dataclasses import replace
    case = next(row for row in probe.case_plan() if row['shape'].name ==
                f'official_{family}_single_history_tile_mode{mode}')
    first = make_fixture(torch, case['shape'])
    changed = make_fixture(torch, replace(case['shape'], contexts=(160,)))
    assert first['blocks'] == changed['blocks'] == 1
    for fixture in (first, changed):
        original = {name: value.clone() for name, value in fixture['cpu'].items()}
        probe.reserve_graph_remap_page(torch, fixture)
        assert fixture['blocks'] == 2 and fixture['page_assignments'] == [(0,)]
        assert torch.equal(fixture['cpu']['raw'][:original['raw'].numel()], original['raw'])
        assert fixture['cpu']['raw'][original['raw'].numel():].eq(0xA5).all()
        assert fixture['cpu']['tags'][1].eq(-1).all()
        assert torch.isnan(fixture['cpu']['wk'][1]).all() and torch.isnan(fixture['cpu']['wv'][1]).all()
        pointers = {name: value.data_ptr() for name, value in fixture['cpu'].items()}
        probe.reserve_graph_remap_page(torch, fixture)
        assert pointers == {name: value.data_ptr() for name, value in fixture['cpu'].items()}
        probe.permute_physical_pages(torch, fixture)
        assert pointers == {name: value.data_ptr() for name, value in fixture['cpu'].items()}
        assert fixture['page_assignments'] == [(1,)]
    assert {name: value.shape for name, value in first['cpu'].items()} == {
        name: value.shape for name, value in changed['cpu'].items()}


def test_single_page_remap_cannot_silently_certify_unchanged_map():
    fixture, _ = fixture_and_tasks()
    before = {name: tensor_bits(value).clone() for name, value in fixture['cpu'].items()}
    with pytest.raises(ValueError, match='at least two physical pages'):
        probe.permute_physical_pages(torch, fixture)
    assert all(torch.equal(before[name], tensor_bits(value)) for name, value in fixture['cpu'].items())


def test_remapped_decode_slot_context_prepare_resolves_original_logical_position():
    from dataclasses import replace
    fixture, tasks = fixture_and_tasks(length=1, context=641, recent=320)
    fixture['spec'] = replace(fixture['spec'], slot_context=True)
    original_table = fixture['cpu']['table'].clone()
    probe.permute_physical_pages(torch, fixture)
    calls = []

    class Ops:
        def prepare_official_attention_tasks_out(self, starts, lens, slots, out, positions,
                                                 heads, kv_heads, sink, recent, splits, table, slot_context):
            assert slot_context is True and table is fixture['cpu']['table']
            # Host spelling of the native q1 slot-context address rule,
            # independent of page_assignments and declared fixture context.
            physical_slot = int(slots[0])
            columns = min(table.shape[1], (int(lens[0]) + 127) // 128)
            matching = torch.nonzero(table[0, :columns] == physical_slot // 128).flatten()
            assert matching.numel() == 1, 'stale table no longer identifies the current slot'
            logical = int(matching[0]) * 128 + physical_slot % 128
            assert 0 <= logical < int(lens[0])
            calls.append(logical)
            out.copy_(tasks)
            positions.copy_(torch.tensor([logical, -1], dtype=torch.int64))

    class TorchProxy:
        npu = SimpleNamespace(synchronize=lambda: None)
        def __getattr__(self, name):
            return getattr(torch, name)

    buffers = {'tasks': torch.empty_like(tasks), 'positions': torch.empty(2, dtype=torch.int64)}
    result = probe.prepare(TorchProxy(), Ops(), fixture['cpu'], fixture, buffers)
    assert calls == [641] and result['fixed_cached_map'] == 'exact'
    fixture['cpu']['table'].copy_(original_table)
    with pytest.raises((AssertionError, RuntimeError)):
        probe.prepare(TorchProxy(), Ops(), fixture['cpu'], fixture, buffers)


def test_writer_gate_stores_rotated_exact_windows_and_replays_fresh_inputs():
    class Ops:
        def store_aligned_int2_out(self,rk,rv,k,v,slots,positions,raw,wk,wv,tags,status,block,blocks,prefix,stride,sink,recent,kc,vc):
            self.last=(rk,rv,k,v,slots,positions,raw,wk,wv,tags,status,block,blocks,prefix,stride,sink,recent,kc,vc)
            status.zero_()
            for token in range(rk.shape[0]):
                slot=int(slots[token]);position=int(positions[token])
                for head in range(rk.shape[1]):
                    if slot<0:continue
                    if slot>=blocks*block:status[token,head]=1;continue
                    if position<0:status[token,head]=4;continue
                    if not bool(torch.isfinite(rk[token,head]).all() and torch.isfinite(rv[token,head]).all()):
                        status[token,head]=2;continue
                    assert torch.equal(rk[token,head],k[token,head]) and torch.equal(rv[token,head],v[token,head])
                    encoded=pack_kv(rk[token,head],rv[token,head],kc,vc)
                    page,within=divmod(slot,block);offset=prefix+page*stride+(within*rk.shape[1]+head)*144
                    raw[offset:offset+144]=encoded
                    row=position if position<sink else sink+within%recent
                    wk[page,row,head]=k[token,head];wv[page,row,head]=v[token,head];tags[page,row]=within
        def copy_validate_bf16_query_out(self,x,slots,out,status,cores):
            out.copy_(x);out[slots<0]=0
            status.copy_((~torch.isfinite(x).all(-1)).int()*2);status[slots<0]=0
    ops=Ops()
    class Graph:
        def replay(self):ops.store_aligned_int2_out(*ops.last)
    class TorchProxy:
        npu=SimpleNamespace(synchronize=lambda:None,NPUGraph=Graph,graph=lambda *a,**k:nullcontext())
        def __getattr__(self,name):return getattr(torch,name)
    report=probe.writer_rotation_gate(TorchProxy(),ops,torch.device('cpu'),1)
    assert report['rotated_windows']==report['codec_bytes']=='exact'
    assert report['graph_same_addresses'] is True
    assert report['active_rows']['states']==['interleaved','all_padding','changed_active_rows','all_live']
    assert report['active_rows']['codec_window_status']=='exact'
