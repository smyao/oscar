"""Archive #126/#145/#148/#153/#154: CPU diagnostics, not NPU qualification."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from oscar_ascend.ops.reference import AttentionResult, merge_attention
from tools import probe_bf16_history as probe
from tools import bf16_history_pipeline as complete
from tools.serving_variants import research_variant_config
from tools.experiments.bf16_history_screen import Case as ScreenCase, task_ranges

ACCEPTANCE=json.loads((probe.ROOT/'configs/acceptance.json').read_text())
CONFIG=research_variant_config(dict(json.loads((probe.ROOT/'configs/target.json').read_text()),experimental_bf16_history=True),'candidate')
EXECUTION=complete.execution_contract(CONFIG,20)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def task_fixture(fixture):
    """Independent CPU descriptor fixture, never a device substitute."""
    spec=fixture['spec'];n=fixture['tokens'];rows=torch.zeros((n,spec.kv_heads,3,spec.splits,16),dtype=torch.int64)
    begin=0
    for request,(length,context) in enumerate(zip(spec.qlens,spec.contexts)):
        for local in range(length):
            token=begin+local;count=min(21,length-local) if local%21==0 else 0
            ranges=list(task_ranges(ScreenCase('partition',length,context,splits=spec.splits,recent=spec.recent_tokens),local,count))
            for hk in range(spec.kv_heads):
                for kind,split,positions in ranges:
                    # Source 1 indexes the compact sink+tail sequence.
                    if kind==1:
                        first=split*((min(probe.reuse.SINK,context)+context-min(context,max(probe.reuse.SINK,context+local+1-spec.recent_tokens))+spec.splits-1)//spec.splits)
                        end=first+len(positions)
                        first=min(first,end)
                    elif len(positions):first,end=int(positions[0]),int(positions[-1])+1
                    else:first=end=0
                    rows[token,hk,kind,split,:11]=torch.tensor([token,count,hk,first,end,request,kind*spec.splits+split,kind,context,begin,0])
        begin+=length
    for token in range(begin,n):
        for hk in range(spec.kv_heads):
            for kind in range(3):
                for split in range(spec.splits):rows[token,hk,kind,split,:8]=torch.tensor([token,-1,hk,0,0,0,kind*spec.splits+split,kind])
    return rows.view(-1,16)


def oracle_buffers(fixture,tasks):
    gold=probe.partial_oracle(torch,fixture,tasks)
    n,h=fixture['tokens'],fixture['spec'].heads;s=3*fixture['spec'].splits
    merged=merge_attention([AttentionResult(gold['partial'][:,:,i],gold['lse'][:,:,i]) for i in range(s)])
    return {**gold,'tasks':tasks,'positions':torch.empty(n,dtype=torch.int64),
            'status':torch.zeros((len(tasks),2),dtype=torch.int32),'cluster_stats':None,
            'output':merged.output.view(n*h,256),'output_lse':merged.lse.view(n*h),
            'merge_status':torch.zeros(n*h,dtype=torch.int32),'bf16':merged.output.bfloat16().view(n*h,256)}


@pytest.mark.parametrize('bad_arms',[('baseline',),('candidate',),('baseline','candidate')])
def test_failed_pair_reports_both_arms_and_source_without_swallowing_error(bad_arms,capsys):
    case=probe.case_plan()[0]
    fixture={'spec':case.shape}
    def buffers():
        return {'partial':torch.zeros(1,6,9,256),'lse':torch.full((1,6,9),float('-inf')),
                'status':torch.zeros(9,2,dtype=torch.int32),
                'merge_status':torch.zeros(6,dtype=torch.int32),
                'tasks':torch.zeros(9,16,dtype=torch.int64)}
    old,new=buffers(),buffers()
    for name,arm in (('baseline',old),('candidate',new)):
        if name in bad_arms:
            arm['partial'][0,2,4,17]=float('inf')  # sink/recent split 1
            arm['lse'][0,1,0]=float('nan')        # history split 0
            arm['merge_status'][2]=2
    task_hash=probe.reuse._fingerprint({'tasks':old['tasks']})
    with pytest.raises(probe.reuse.HistoryReuseProbeError,match='NaN LSE') as caught:
        probe.compare_pair(torch,case,fixture,old,new,task_hash,ACCEPTANCE)
    report=caught.value.bf16_failure_diagnostics
    assert set(report['arms'])=={'baseline','candidate'}
    for name,arm in report['arms'].items():
        count=int(name in bad_arms)
        assert arm['sources']['history']['invalid_lse_elements']==count
        assert arm['sources']['sink_recent']['nonfinite_partial_elements']==count
        assert arm['sources']['current']['invalid_lse_elements']==0
        assert arm['merge_status']['nonzero_elements']==count
        if count:assert arm['partial_first_bad_coordinates']==[[0,2,4,17]]
    assert 'BF16_HISTORY_FAILURE' in capsys.readouterr().out
    json.dumps(report,allow_nan=False)


def test_failure_diagnostics_accepts_empty_lse_but_counts_positive_infinity():
    case=probe.case_plan()[0]
    buffers={'partial':torch.zeros(1,6,9,256),'lse':torch.full((1,6,9),float('-inf'))}
    buffers['lse'][0,0,8]=float('inf')
    detail=probe.failure_diagnostics(torch,case,{'spec':case.shape},buffers,buffers)
    for arm in detail['arms'].values():
        assert arm['sources']['history']['invalid_lse_elements']==0
        assert arm['sources']['current']['invalid_lse_elements']==1
        assert arm['sources']['current']['empty_lse_elements']==17


@pytest.mark.parametrize('route,count,begin',[('mdl',4,4),('mdl_q1',1,7)])
@pytest.mark.parametrize('norm_bad',[False,True])
def test_retained_norm_diagnostic_only_uses_last_history_writer(route,count,begin,norm_bad):
    case=replace(probe.case_plan()[0],route=route)
    spec=SimpleNamespace(heads=6,kv_heads=1,splits=3,qlens=(4,4))
    tasks=torch.zeros(8*9,16,dtype=torch.int64)
    tasks[0,:11]=torch.tensor([0,count,0,64,128,0,0,0,1341,0,0])
    last=begin*9+2
    tasks[last,:11]=torch.tensor([begin,count,0,128,192,1,2,0,1341,4,0])
    # A later precise task must not become the retained Norm owner.
    tasks[begin*9+8,:11]=torch.tensor([begin,count,0,1341,1342,1,8,2,1341,4,0])
    workspace=torch.empty(probe.candidate_workspace_per_core(route),dtype=torch.uint8)
    workspace.view(torch.float32).fill_(float('nan'))
    norm=workspace[:32*256*4].view(torch.float32).reshape(32,256)
    norm[:count*6].fill_(1)
    if norm_bad:norm[0,0]=float('nan')
    partial=torch.full((8,6,9,256),float('nan'))
    buffers={'workspace':workspace,'tasks':tasks,'partial':partial}
    report=probe.mdl_history_stage_diagnostics(
        torch,case,{'tokens':8,'spec':spec},buffers,torch.eye(256))
    assert len(report['tasks'])==1
    row=report['tasks'][0]
    assert row['task_id']==last and row['qbegin']==begin and row['split']==2
    assert row['norm_nonfinite_elements']==int(norm_bad)
    assert row['partial_nonfinite_elements']==count*6*256
    assert row['cpu_rotation_nonfinite_elements']==(256 if norm_bad else 0)
    assert report['rotation_finite'] is True
    assert row['retained_operands']['status']=='overwritten_by_later_task'


@pytest.mark.parametrize('tiles',[1,2,3,4])
def test_retained_history_value_planes_exclude_unpublished_slot(tiles):
    case=probe.case_plan()[0]
    spec=SimpleNamespace(heads=6,kv_heads=1,splits=1,qlens=(4,))
    tasks=torch.zeros(12,16,dtype=torch.int64)
    tasks[0,:11]=torch.tensor([0,4,0,64,64+tiles*256,0,0,0,4096,0,0])
    workspace=torch.zeros(probe.candidate_workspace_per_core('mdl'),dtype=torch.uint8)
    operand=workspace[3*32*256*4:].view(torch.bfloat16)
    for slot in range(2):
        offset=32*256+256*256+slot*2*256*256
        operand[offset+17]=float('nan')
    norm=workspace[:32*256*4].view(torch.float32).reshape(32,256)
    norm[:24,17]=float('nan')
    result=probe.mdl_history_stage_diagnostics(torch,case,{'tokens':4,'spec':spec},
        {'tasks':tasks,'workspace':workspace,'partial':torch.zeros(4,6,3,256)},torch.eye(256))
    row=result['tasks'][0]
    assert row['norm_bad_columns']==[17]
    values=row['retained_operands']['value_tiles']
    assert len(values)==min(2,tiles)
    for entry in values:
        assert entry['nonfinite_elements']==1 and entry['bad_columns']==[17]
        assert entry['tile_index']==max(t for t in range(tiles) if t%2==entry['slot'])
    assert row['retained_operands']['probability_nonfinite_elements']==0


@pytest.mark.parametrize('index',[0,1,4,5,6,8,9,10,11])
def test_independent_partial_merge_matches_every_dense_query(index):
    case=probe.case_plan()[index];fixture=probe.make_fixture(torch,case);tasks=task_fixture(fixture)
    buffers=oracle_buffers(fixture,tasks)
    result=probe.compare_pair(torch,case,fixture,buffers,buffers,probe.reuse._fingerprint({'tasks':tasks}),ACCEPTANCE)
    assert result['passed'],result
    assert result['queries']==sum(case.shape.qlens)
    if case.cancellation:
        assert fixture['expected'][0][0][0,0].item()==pytest.approx(-.021484375,abs=1e-7)


def test_large_fixture_does_not_reuse_old_sampled_query_oracle():
    for index in (2,3):
        case=probe.case_plan()[index];fixture=probe.make_fixture(torch,case)
        assert sorted(fixture['expected'])==list(range(sum(case.shape.qlens)))


def test_dropped_or_duplicate_task_owner_is_not_hidden_by_zero_filling():
    case=probe.case_plan()[0];fixture=probe.make_fixture(torch,case);tasks=task_fixture(fixture)
    missing=tasks.clone();missing[0,1]=0
    with pytest.raises(RuntimeError,match='missing or duplicate'):probe.partial_oracle(torch,fixture,missing)
    with pytest.raises(RuntimeError,match='missing or duplicate'):probe.partial_oracle(torch,fixture,torch.cat((tasks,tasks[:1])))


def test_all_bf16_sources_are_checked_against_oracle_without_old_partial_bitwise():
    case=probe.case_plan()[0];fixture=probe.make_fixture(torch,case);tasks=task_fixture(fixture)
    old=oracle_buffers(fixture,tasks);new={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in old.items()}
    task_hash=probe.reuse._fingerprint({'tasks':tasks})
    new['partial'][:fixture['actual_tokens'],:,0]+=0.1
    result=probe.compare_pair(torch,case,fixture,old,new,task_hash,ACCEPTANCE)
    assert not result['passed'] and not result['comparisons']['candidate']['partial']['passed']
    new['partial'][:fixture['actual_tokens'],:,case.shape.splits]+=0.01
    changed=probe.compare_pair(torch,case,fixture,old,new,task_hash,ACCEPTANCE)
    assert not changed['passed'] and changed['operand_contract']==probe.CONTRACT
    assert changed['empty_partial_bits'] is True


@pytest.mark.parametrize('cluster',[4,16])
def test_cluster_counters_use_actual_family_and_keep_lane_ownership(cluster):
    row=torch.tensor([[1,cluster,2,5,(cluster-1)*5,3,cluster,1]])
    result=probe.check_cluster_stats(row,cluster,expected_leaders=cluster+2,require_cluster=True)
    assert result['eligible_clusters']==1
    with pytest.raises(RuntimeError,match='ownership'):probe.check_cluster_stats(row,16 if cluster==4 else 4,expected_leaders=cluster+2,require_cluster=True)
    with pytest.raises(RuntimeError,match='duplicated'):probe.check_cluster_stats(row,cluster,expected_leaders=cluster+1,require_cluster=True)


def synthetic_evidence():
    """Only parser fixtures. This object is never written as device evidence."""
    rows=[]
    for case in probe.case_plan():
        names=probe.ROUTES[case.route][:2];invalid=probe._invalid(case)
        detail={'passed':True,'queries':0 if invalid else sum(case.shape.qlens),
                'oracle':'expected_error_domain' if invalid else 'all_actual_queries_raw_space',
                'status_lane_bits':True,'empty_partial_bits':True,'operand_contract':probe.CONTRACT,
                'comparisons':{} if invalid else {arm:{key:{'passed':True} for key in ('partial','lse','output','output_lse','bf16','relative_l2')} for arm in ('baseline','candidate')}}
        row={'case':case.shape.name,'route':case.route,'baseline_operator':names[0],'candidate_operator':names[1],
             'status':'passed','device_completion':'passed','queries':sum(case.shape.qlens),
             'padding':(case.shape.padded_tokens or sum(case.shape.qlens))-sum(case.shape.qlens),
             'old_repeat_bitwise':True,'candidate_repeat_bitwise':True,'input_bytes_unchanged':True,
             'precision':detail,'timing':None,'graph':None,'input_sha256':'operator-input','task_sha256':'operator-tasks'}
        if case.graph:
            labels=['captured_inputs','changed_q_qr','changed_cache_table_tasks','smaller_live_rows','all_padding','restored_live_rows']
            states=[{'state':label,'precision':detail,'q_sha256':'q1' if i==1 else 'q0',
                     'task_sha256':'t1' if i==2 else 't0','raw_sha256':'r1' if i==2 else 'r0',
                     'table_sha256':'b1' if i==2 else 'b0'} for i,label in enumerate(labels)]
            row['graph']={'capture':'passed','replay':'passed','precision':'passed','same_addresses':True,'eager_bitwise':True,'states':states}
        if case.timed:
            row['timing']={scope:{'samples_ms':{names[0]:[12.]*5,names[1]:[10.]*5},
                'baseline_ms':12.,'candidate_ms':10.,'speedup':1.2,'minimum_speedup':1.15,'warmup':2,'repeats':5,'gate':'passed'} for scope in probe.SCOPES}
            full_case=complete.production_case(case,EXECUTION)
            full_detail=copy.deepcopy(detail);full_detail['queries']=sum(full_case.shape.qlens)
            row['full_attention']={'scope':complete.SCOPE,'policy':complete.case_policy(full_case),
                'settings':dict(EXECUTION['settings']),'execution':copy.deepcopy(EXECUTION),
                'shape':json.loads(json.dumps(asdict(full_case.shape))),'physical_page_stride_bytes':512*136,
                'input_sha256':'full-input','initial_task_sha256':'full-tasks',
                'operator_input_sha256':row['input_sha256'],'operator_task_sha256':row['task_sha256'],
                'reset_scope':'same_address_raw_window_tags_and_poison_outside_events',
                'whole_model_performance':'not_established','model_quality':'not_established',
                'numerical':{key:True for key in ('passed','writer_bytes_equal','rotation_bf16_operands_equal','rotation_trailer_valid','immutable_inputs','all_status_zero','same_arm_repeat_bitwise')},
                'timing':dict(row['timing'][probe.SCOPES[0]],scope=complete.SCOPE)}
            row['full_attention']['numerical']['precision']=full_detail
            row['full_attention']['numerical']['native_current_bitwise']=True if complete.case_policy(full_case)['native_current'] else None
        rows.append(row)
    return {'schema':probe.SCHEMA,'operand_contract':probe.CONTRACT,'status':'passed','device_completion':'passed','precision':'passed','error_domain':'passed',
            'graph_capture':'passed','graph_replay':'passed','cv_performance':'passed',
            'full_attention_performance':'passed','model_quality':'not_established','production_qualified':False,'cases':rows,'cpu_screen':synthetic_cpu_pass(),
            'configuration':{'execution':copy.deepcopy(EXECUTION),'model_config_sha256':'0'*64,'target_geometry':[6,1,256],
                'soc_version':'ascend910b4','devices':[0,1,2,3],'probe_physical_device':0}}


def test_report_separates_operator_evidence_from_full_attention_and_quality():
    report=synthetic_evidence();assert probe.validate_evidence(report)==[]
    for field in ('model_quality','production_qualified'):
        broken=copy.deepcopy(report);broken[field]='passed'
        assert field in probe.validate_evidence(broken)


@pytest.mark.parametrize('mutation',['missing_case','wrong_route','source_bitwise','partial_oracle','final_bf16','graph_oracle','same_address','metadata','all_padding','timing_scope','under_threshold','missing_sample'])
def test_validator_rejects_incomplete_or_weak_operator_evidence(mutation):
    report=synthetic_evidence();row=report['cases'][0]
    if mutation=='missing_case':report['cases'].pop()
    elif mutation=='wrong_route':row['candidate_operator']='attention_cv_mdl_decode_out'
    elif mutation=='source_bitwise':row['precision']['empty_partial_bits']=False
    elif mutation=='partial_oracle':row['precision']['comparisons']['candidate']['partial']['passed']=False
    elif mutation=='final_bf16':row['precision']['comparisons']['candidate'].pop('bf16')
    elif mutation=='graph_oracle':row['graph']['states'][0]['precision']['passed']=False
    elif mutation=='same_address':row['graph']['same_addresses']=False
    elif mutation=='metadata':row['graph']['states'][2]['raw_sha256']=row['graph']['states'][0]['raw_sha256']
    elif mutation=='all_padding':row['graph']['states'].pop(4)
    elif mutation=='timing_scope':row['timing'].pop(probe.SCOPES[0])
    elif mutation=='missing_sample':row['timing'][probe.SCOPES[0]]['samples_ms'][probe.ROUTES['mdl'][0]].pop()
    else:
        entry=row['timing'][probe.SCOPES[0]]
        entry.update(baseline_ms=11.48704,speedup=1.148704)
        entry['samples_ms'][probe.ROUTES['mdl'][0]]=[11.48704]*5
    assert probe.validate_evidence(report)


def test_no_npu_is_explicit_failure_and_preserves_real_returncode(monkeypatch,tmp_path):
    monkeypatch.setattr(probe.reuse,'_active_device',lambda _:None)
    monkeypatch.setitem(sys.modules,'torch_npu',ModuleType('torch_npu'))
    monkeypatch.setattr(torch,'npu',SimpleNamespace(is_available=lambda:False),raising=False)
    config=tmp_path/'config.json';config.write_text(json.dumps({'experimental_bf16_history':True}))
    output=tmp_path/'result.json';output.write_text(json.dumps({'status':'passed','old_evidence':'stale'}))
    assert probe.main(['--config',str(config),'--output',str(output)])==2
    report=json.loads(output.read_text())
    assert report['status']=='failed' and 'real NPU' in report['first_error']
    assert report['production_qualified'] is False and 'old_evidence' not in report


def test_failure_report_retains_completed_cases(monkeypatch,tmp_path):
    def failed(*a,**kw):
        error=RuntimeError('target synchronization failed')
        error.bf16_report={'schema':probe.SCHEMA,'status':'failed','first_error':'fixture: target synchronization failed',
                           'cases':[{'case':'completed','status':'failed'}]}
        raise error
    monkeypatch.setattr(probe,'probe',failed)
    out=tmp_path/'report.json'
    assert probe.main(['--config',str(tmp_path/'unused'),'--output',str(out)])==2
    assert json.loads(out.read_text())['cases']==[{'case':'completed','status':'failed'}]


def test_observe_bf16_gate_requires_matching_source_and_artifact_and_releases_resources(monkeypatch,tmp_path):
    from tools import observe_serve as observe
    from oscar_ascend.ops import loader
    manifest={'signature':'bf16-build','sha256':{'extension':'artifact'}}
    monkeypatch.setattr(loader,'validate_build_artifacts',lambda *a:manifest)
    monkeypatch.setattr(probe,'source_binding',lambda:{'source':'sha'})
    monkeypatch.setattr(observe,'read_npu_resources',lambda *a,**k:{})
    releases=[]
    monkeypatch.setattr(observe,'wait_for_release',lambda *a,**k:releases.append('released') or {'status':'passed'})
    report=synthetic_evidence()
    expected_configuration=copy.deepcopy(report['configuration'])
    monkeypatch.setattr(probe,'configuration_binding',lambda *a:expected_configuration)
    report.update(artifact_signature=manifest['signature'],artifact_sha256=manifest['sha256'],source_binding={'source':'sha'})
    def phase(name,argv,**kw):
        assert name=='bf16-history-npu' and 'tools.probe_bf16_history' in argv
        Path(argv[argv.index('--output')+1]).write_text(json.dumps(report))
    monkeypatch.setattr(observe,'_phase',phase)
    state={};observe._bf16_history_gate(tmp_path/'config',{}, {},tmp_path,state)
    assert state['bf16_history_gate']['status']=='passed' and releases==['released']
    report['source_binding']={'source':'stale'}
    with pytest.raises(RuntimeError,match='source_binding'):observe._bf16_history_gate(tmp_path/'config',{}, {},tmp_path,{})
    assert releases==['released','released']
    report['source_binding']={'source':'sha'}
    report['configuration']['execution']['effective_config_sha256']='stale'
    with pytest.raises(RuntimeError,match='configuration'):
        observe._bf16_history_gate(tmp_path/'config',{}, {},tmp_path,{})
    assert releases==['released']*3


@pytest.mark.parametrize('index',range(6))
def test_every_graph_family_changes_metadata_with_fixed_tensor_geometry(index):
    case=probe.case_plan()[index];first=probe.make_fixture(torch,case)
    changed=probe.make_fixture(torch,replace(case,shape=replace(case.shape,contexts=tuple(c+16 for c in case.shape.contexts))))
    small=probe.make_fixture(torch,replace(case,shape=replace(case.shape,qlens=tuple(max(1,q-1) for q in case.shape.qlens))))
    shape=lambda f:{k:(v.shape,v.dtype) for k,v in f['cpu'].items()}
    assert shape(first)==shape(changed)==shape(small)
    assert first['blocks']==changed['blocks']==small['blocks']
    assert not torch.equal(first['cpu']['raw'],changed['cpu']['raw'])
    probe.mdl.remap_physical_pages(torch,changed)
    assert not torch.equal(first['cpu']['table'],changed['cpu']['table'])
    assert len(first['expected'])==sum(case.shape.qlens)


def test_all_padding_oracle_has_only_zero_output_and_negative_infinite_lse():
    fixture=probe.make_fixture(torch,probe.case_plan()[0])
    fixture['actual_tokens']=0;fixture['expected']={}
    tasks=task_fixture(fixture);tasks[:,1]=-1
    gold=probe.partial_oracle(torch,fixture,tasks)
    assert bool((gold['partial']==0).all()) and bool(torch.isneginf(gold['lse']).all())


def synthetic_cpu_pass():
    """Deliberately fabricated parser fixture, never saved as numerical evidence."""
    from dataclasses import asdict
    from tools.experiments import bf16_all_sources_screen as screen
    import hashlib
    cases=[asdict(case) for case in screen.CASES]
    direct={'contract':probe.CONTRACT,'status':'cpu_direct_decode_passed','returncode':0,'packed_words':65536,
        'packed_code_positions_checked':524288,'valid_scale_patterns':31743,'finite_zero_patterns':63488,
        'metadata_pairs':634872,'levels_checked':2539488,'exact_halfway_values_checked':142980,
        'metadata_cartesian_product_exhaustive':False,'negative_zero_code0_bits':0,
        'device_completion':'not_run','production_qualified':False}
    return {'schema_version':2,'operand_contract':probe.CONTRACT,'direct_decode':direct,'status':'cpu_screen_passed_not_qualified','returncode':0,'required_cases_complete':True,
            'required_cases':cases,'frozen_attention_gate':ACCEPTANCE['fused_attention'],
            'frozen_prefill_window_gate':ACCEPTANCE['prefill_window'],'next_stage':'device_design_allowed',
            'source_sha256':{name:hashlib.sha256((probe.ROOT/name).read_bytes()).hexdigest() for name in probe.CPU_SOURCE_PATHS},
            'cases':[{'case':case,'baseline_passed':True,'candidate_passed':True,'input_bytes_unchanged':True,
                      'codec_oracle_identity':True,'all_queries_and_heads_checked':True} for case in cases]}


@pytest.mark.parametrize('failure',['cpu_failed','cpu_subset','cpu_stale','last_device_case','late_cancellation','graph_failure'])
def test_performance_barrier_never_times_before_entire_numerical_matrix(monkeypatch,failure):
    report=synthetic_evidence();rows=report['cases'];cpu=report['cpu_screen'];calls=[]
    # Numerical receipts in the first pass have no timing at all.
    for row in rows:row['timing']=None
    if failure=='cpu_failed':cpu['cases'][6]['candidate_passed']=False
    elif failure=='cpu_subset':cpu['cases'].pop()
    elif failure=='cpu_stale':cpu['source_sha256']['configs/acceptance.json']='stale'
    elif failure=='last_device_case':rows[-1]['precision']['passed']=False
    elif failure=='late_cancellation':rows[6]['precision']['passed']=False
    else:rows[3]['graph']['states'][0]['precision']['passed']=False
    monkeypatch.setattr(probe,'time_case',lambda *a,**k:calls.append('timed'))
    result=probe.performance_phase(None,None,rows,cpu,None,None,ACCEPTANCE,CONFIG)
    assert result['status']=='not_run_numerical_gate_failed' and result['blocking_errors']
    assert calls==[] and all(row['timing'] is None for row in rows)


def test_performance_code_is_reachable_only_after_all_mandatory_receipts(monkeypatch):
    report=synthetic_evidence();rows=report['cases'];expected={row['case']:row['timing'] for row in rows}
    original_graphs={row['case']:copy.deepcopy(row['graph']) for row in rows};calls=[]
    for row in rows:row['timing']=None
    def timed(torch,ops,case,device,cores,acceptance,receipt):
        calls.append(case.shape.name)
        assert receipt['graph']==original_graphs[case.shape.name]
        return expected[case.shape.name]
    monkeypatch.setattr(probe,'time_case',timed)
    full={r['case']:copy.deepcopy(r.get('full_attention')) for r in rows}
    monkeypatch.setattr(complete,'numerical_case',lambda t,o,c,d,k,conf,a,r:dict(full[c.shape.name],timing=None))
    monkeypatch.setattr(complete,'timing_case',lambda t,o,c,d,k,conf,a,r:full[c.shape.name]['timing'])
    backend=SimpleNamespace(npu=SimpleNamespace(empty_cache=lambda:None))
    result=probe.performance_phase(backend,None,rows,report['cpu_screen'],None,20,ACCEPTANCE,CONFIG)
    assert result['status']=='passed'
    assert calls==[case.shape.name for case in probe.case_plan() if case.timed]
    assert all(row['graph']==original_graphs[row['case']] for row in rows)


def test_cpu_performance_prerequisite_runs_the_fixed_complete_screen(monkeypatch,tmp_path):
    from tools.experiments import bf16_all_sources_screen as screen
    called=[];sentinel=object()
    def run(*,cases,progress):
        called.append(cases)
        assert callable(progress)
        return sentinel
    monkeypatch.setattr(screen,'run',run)
    monkeypatch.setattr(probe,'CPU_SCREEN_REPORT',tmp_path/'missing.json')
    assert probe.run_cpu_screen() is sentinel
    assert called==[screen.CASES] and len(called[0])==11


def test_first_device_pass_contains_no_timing_call():
    import ast,inspect
    tree=ast.parse(inspect.getsource(probe.run_case))
    calls={node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    assert not {'_timing','time_case','performance_phase'} & calls


def test_completed_source_bound_cpu_failure_is_reused_and_still_blocks_performance(monkeypatch,tmp_path):
    from tools.experiments import bf16_all_sources_screen as screen
    cpu=synthetic_cpu_pass()
    cpu['cases'][0]['candidate_passed']=False
    cpu.update(status='cpu_screen_failed',returncode=2,next_stage='numerical_gate_failed_no_performance_qualification')
    report=tmp_path/'cpu.json';report.write_text(json.dumps(cpu))
    monkeypatch.setattr(probe,'CPU_SCREEN_REPORT',report)
    monkeypatch.setattr(screen,'run',lambda **kw:pytest.fail('matching completed CPU screen must not rerun'))
    assert probe.run_cpu_screen()==cpu
    assert probe.cpu_screen_errors(cpu)
    assert report.read_text()==json.dumps(cpu)


def test_stale_cpu_report_cannot_authorize_timing_and_requires_full_rerun(monkeypatch,tmp_path):
    from tools.experiments import bf16_all_sources_screen as screen
    cpu=synthetic_cpu_pass();cpu['source_sha256']['configs/acceptance.json']='stale'
    path=tmp_path/'cpu.json';path.write_text(json.dumps(cpu))
    monkeypatch.setattr(probe,'CPU_SCREEN_REPORT',path)
    seen=[]
    monkeypatch.setattr(screen,'run',lambda **kwargs:seen.append(kwargs['cases']) or cpu)
    probe.run_cpu_screen()
    assert seen==[screen.CASES]
    assert probe.cpu_screen_errors(cpu)


def test_empty_protocol_remains_bitwise_even_with_all_source_bf16_arithmetic():
    case=probe.case_plan()[0];fixture=probe.make_fixture(torch,case);tasks=task_fixture(fixture)
    old=oracle_buffers(fixture,tasks);new={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in old.items()}
    new['partial'][-1,0,0,0]=0.001  # below frozen atol, but invalid empty protocol
    with pytest.raises(RuntimeError,match='empty partial zero'):
        probe.compare_pair(torch,case,fixture,old,new,probe.reuse._fingerprint({'tasks':tasks}),ACCEPTANCE)


@pytest.mark.parametrize('route,cluster,old_length',[('c4',4,336),('c16',16,672)])
def test_mature_fixture_retains_complete_cluster_after_graph_shrink(route,cluster,old_length):
    case=next(c for c in probe.case_plan() if c.route==route)
    def eligible(length):
        spec=replace(case.shape,qlens=(length,))
        # CPU descriptors also used by independent per-source oracle tests.
        tasks=task_fixture({'spec':spec,'tokens':spec.padded_tokens})
        width=cluster*21;found=0
        for anchor in range(0,length,width):
            members=[tasks[(anchor+j*21)*3] for j in range(cluster)
                     if anchor+j*21<length]
            if len(members)!=cluster:continue
            if all(int(r[1])==21 and int(r[4])==int(r[8]) and int(r[3])<int(r[4])
                   and torch.equal(r[3:5],members[0][3:5]) for r in members):found+=1
        return found
    assert eligible(old_length)==1
    assert eligible(old_length-1)==0  # reproduces the old graph-only failure
    assert eligible(case.shape.qlens[0])==1
    assert eligible(case.shape.qlens[0]-1)==1
    assert case.shape.expect_clusters  # coverage gate must not be disabled
    assert case.shape.qlens[0]<case.shape.padded_tokens
