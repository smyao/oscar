"""Archive #126/#145/#150/#151/#154-156: host gate contracts only, not NPU evidence."""
import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools import probe_mdl_decode as probe


def synthetic_report():
    """Deliberately synthetic report to exercise fail-closed validation."""
    rows=[]
    for case in probe.case_plan():
        spec=case.shape;invalid=spec.corrupt_metadata or spec.corrupt_qr
        row={'case':spec.name,'status':'passed','contexts':list(spec.contexts),'qlens':list(spec.qlens),
            'operators':list(case.operators),
            'all_tasks':spec.padded_tokens*3*spec.splits,'active_queries':sum(spec.qlens),
            'padded_queries':spec.padded_tokens-sum(spec.qlens),'queries':0 if invalid else sum(spec.qlens),
            'special_values':case.special_values,'source_leaders':{'0':2,'1':2,'2':2},
            'oracle':'expected_error_domain' if invalid else 'all_actual_queries_raw_space_passed',
            'precision':'error_codes_finite_bits_nonfinite_classification' if invalid else 'partial_lse_status_merge_bf16_bitwise',
            'graph':None,'timing':None}
        if case.timed:
            a,b=12.,10.
            row['timing']={'baseline_ms':a,'candidate_ms':b,'ratio':b/a,'speedup':a/b,'minimum_speedup':1.15,
                'samples_ms':{case.operators[0]:[a]*5,case.operators[1]:[b]*5},'warmup':2,'repeats':5,
                'scope':probe.SCOPE,'gate':'passed'}
        if case.graph:
            states=[dict(row,state=name,task_sha256=task,raw_sha256=raw,table_sha256=table)
                    for name,task,raw,table in [('captured_inputs','t0','r0','b0'),
                    ('changed_q_qr','t0','r0','b0'),('changed_context_cache_table','t1','r1','b1')]]
            row['graph']={'capture':'passed','replay':'passed','same_addresses':True,'q_changed':True,
                'cache_table_tasks_changed':True,'eager_bitwise':'passed','independent_oracle':'passed','states':states}
        rows.append(row)
    return {'status':'passed','baseline_operator':probe.BASELINE,'candidate_operator':probe.CANDIDATE,
        'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
        'performance':'passed','device_completion':'passed','scope':probe.SCOPE,
        'workspace_bytes_per_core':probe.WORKSPACE,'cases':rows}


def test_report_contract_requires_real_complete_shapes_and_sources():
    report=synthetic_report();assert probe.validate_evidence(report)==[]
    for name in ('missing_case','wrong_operator','source2','padding','query_oracle','no_merge'):
        broken=copy.deepcopy(report)
        if name=='missing_case':broken['cases'].pop()
        elif name=='wrong_operator':broken['candidate_operator']='attention_cv_bundle_decode_out'
        elif name=='source2':broken['cases'][-1]['source_leaders']['2']=0
        elif name=='padding':broken['cases'][-1]['all_tasks']-=1
        elif name=='query_oracle':broken['cases'][-1]['queries']-=1
        else:broken['scope']='helper_only'
        assert probe.validate_evidence(broken),name


@pytest.mark.parametrize('base,new,allowed',[(11.5,10.,True),(11.49,10.,False),(10.,10.,False)])
def test_full_cv_merge_threshold_is_15_percent_without_rounding(base,new,allowed):
    report=synthetic_report();timing=report['cases'][-1]['timing']
    timing.update(baseline_ms=base,candidate_ms=new,ratio=new/base,speedup=base/new,
                  samples_ms={probe.BASELINE:[base]*5,probe.CANDIDATE:[new]*5})
    errors=probe.validate_evidence(report)
    assert (not any('device_latency' in e for e in errors)) is allowed
    timing['samples_ms'][probe.BASELINE].pop()
    assert any('device_latency' in e for e in probe.validate_evidence(report))


def test_q4_success_cannot_qualify_q1_operator_or_speed():
    report = synthetic_report()
    q1 = next(row for row in report['cases'] if row['case'] == 'q1_R4_20k30k')
    q1['operators'] = [probe.BASELINE, probe.CANDIDATE]
    assert 'q1_R4_20k30k.contract' in probe.validate_evidence(report)
    q1['operators'] = [probe.Q1_BASELINE, probe.Q1_CANDIDATE]
    q1['timing']['samples_ms'][probe.Q1_CANDIDATE] = [11] * 5
    assert 'q1_R4_20k30k.device_latency' in probe.validate_evidence(report)


def test_changed_input_graph_needs_cache_table_tasks_and_oracle():
    report=synthetic_report()
    for key in ('task_sha256','raw_sha256','table_sha256'):
        broken=copy.deepcopy(report);states=broken['cases'][0]['graph']['states']
        states[-1][key]=states[0][key]
        assert any('graph_inputs' in e for e in probe.validate_evidence(broken))
    report['cases'][0]['graph']['states'][-1]['queries']=0
    assert any('graph_oracle' in e for e in probe.validate_evidence(report))


def test_short_boundary_fixture_and_graph_relocation_keep_logical_payload():
    case=probe.case_plan()[0];first=probe.make_fixture(torch,case)
    second=probe.make_fixture(torch,probe.replace(case,shape=probe.replace(case.shape,
        contexts=tuple(c+16 for c in case.shape.contexts))))
    assert len(first['expected'])==len(second['expected'])==sum(case.shape.qlens)
    assert {k:(v.shape,v.dtype) for k,v in first['cpu'].items()}=={k:(v.shape,v.dtype) for k,v in second['cpu'].items()}
    raw=second['cpu']['raw'][probe.reuse.PREFIX:].view(second['blocks'],second['stride']).clone()
    assignments=copy.deepcopy(second['page_assignments'])
    table=second['cpu']['table'].clone();windows=second['cpu']['wk'].clone()
    probe.remap_physical_pages(torch,second)
    after=second['cpu']['raw'][probe.reuse.PREFIX:].view(second['blocks'],second['stride'])
    assert not torch.equal(table,second['cpu']['table'])
    for previous,current in zip(assignments,second['page_assignments']):
        for old,new in zip(previous,current):
            assert torch.equal(raw[old],after[new])
            assert probe.shared._store_bits_equal(torch,windows[old],second['cpu']['wk'][new])
    assert bool((second['cpu']['slots'][8:]==-1).all())
    assert torch.signbit(first['cpu']['q'][0,0,0])


def test_source_binding_changes_on_changed_source(tmp_path):
    for name in probe.SOURCE_PATHS:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
    old=probe.source_binding(tmp_path)
    (tmp_path/probe.SOURCE_PATHS[0]).write_text('changed kernel')
    assert probe.source_binding(tmp_path)!=old


def test_no_npu_is_an_explicit_failure(monkeypatch,tmp_path):
    monkeypatch.setattr(probe.reuse,'_active_device',lambda _:None)
    monkeypatch.setitem(sys.modules,'torch_npu',ModuleType('torch_npu'))
    monkeypatch.setattr(torch,'npu',SimpleNamespace(is_available=lambda:False),raising=False)
    path=tmp_path/'config.json';path.write_text(json.dumps({key:True for key in
        ('experimental_mdl_decode','experimental_decode_bundle','experimental_striped_cache')}))
    with pytest.raises(RuntimeError,match='real NPU'):
        probe.probe(path,probe.ROOT/'configs/acceptance.json')


def test_cli_preserves_failure_and_report(monkeypatch,tmp_path,capsys):
    def fail(*args):raise RuntimeError('intentional device failure')
    monkeypatch.setattr(probe,'probe',fail)
    output=tmp_path/'result.json'
    assert probe.main(['--config',str(tmp_path/'config.json'),'--output',str(output)])==2
    assert json.loads(output.read_text())['first_error']=='intentional device failure'
    captured=capsys.readouterr();assert 'intentional device failure' in captured.err
    assert 'PERF_MDL_RESULT' in captured.out


def test_observe_rejects_stale_kernel_signature_and_incomplete_graph(monkeypatch,tmp_path):
    from tools import observe_serve as observe
    from oscar_ascend.ops import loader
    manifest={'signature':'current-build','sha256':{'extension':'current-artifact'}}
    monkeypatch.setattr(loader,'validate_build_artifacts',lambda *a:manifest)
    monkeypatch.setattr(observe,'read_npu_resources',lambda *a,**k:{})
    monkeypatch.setattr(observe,'wait_for_release',lambda *a,**k:{'status':'passed'})
    monkeypatch.setattr(probe,'source_binding',lambda:{'candidate':'current-source'})
    report=synthetic_report();report.update(artifact_signature=manifest['signature'],
        artifact_sha256=manifest['sha256'],source_binding=probe.source_binding())
    def phase(name,argv,**kwargs):
        assert name=='mdl-decode-npu' and 'tools.probe_mdl_decode' in argv
        out=Path(argv[argv.index('--output')+1]);assert out.name!=name+'.json'
        out.write_text(json.dumps(report))
    monkeypatch.setattr(observe,'_phase',phase)
    observe._mdl_decode_gate(tmp_path/'config.json',{}, {},tmp_path,{})
    for field in ('artifact_signature','source_binding'):
        old=report[field];report[field]='stale'
        with pytest.raises(RuntimeError,match=field):observe._mdl_decode_gate(tmp_path/'config.json',{}, {},tmp_path,{})
        report[field]=old
    report['cases'][0]['graph']=None
    with pytest.raises(RuntimeError,match='graph'):observe._mdl_decode_gate(tmp_path/'config.json',{}, {},tmp_path,{})


def test_phase_failure_still_checks_resource_release_and_keeps_original(monkeypatch,tmp_path):
    from tools import observe_serve as observe
    monkeypatch.setattr(observe,'read_npu_resources',lambda *a,**k:{})
    calls=[]
    def release(*args,**kwargs):calls.append('release');return {'status':'passed'}
    def phase(*args,**kwargs):raise RuntimeError('NPU operator rc2')
    monkeypatch.setattr(observe,'wait_for_release',release);monkeypatch.setattr(observe,'_phase',phase)
    with pytest.raises(RuntimeError,match='NPU operator rc2'):
        observe._mdl_decode_gate(tmp_path/'config.json',{}, {},tmp_path,{})
    assert calls==['release']


def test_both_candidate_entrypoints_enable_mdl_and_failed_gate_blocks_service(monkeypatch,tmp_path,capsys):
    from tools import observe_serve as observe
    from tools import install_serve
    assert observe.main(['--variant','candidate','--plan','--probe-only'])==0
    plan=json.loads(capsys.readouterr().out)
    assert 'mdl_decode_gate' in plan['phases']
    assert plan['runtime_selection']['optimizations']['mdl_decode'] is True
    assert install_serve.main(['--variant','candidate','--plan'])==0
    direct=json.loads(capsys.readouterr().out)
    assert direct['runtime_selection']==plan['runtime_selection']
    assert direct['runtime_selection']['prefix_caching'] is True
    for name in ('_preflight','_candidate_gate','_fast_unpack_gate','_mixed_optimization_gate',
                 '_striped_cache_gate','_decode_bundle_gate','_c4_member_pipeline_gate',
                 '_current_only_gate','_first_mtp_current_gate','_mixed_decode_split_gate','_request_window_gate'):
        monkeypatch.setattr(observe,name,lambda *a,**k:None)
    def fail(path,config,*args):
        assert config['experimental_mdl_decode'] is True
        assert json.loads(Path(path).read_text())['experimental_mdl_decode'] is True
        error=RuntimeError('MDL speedup below 1.15');error.phase='mdl-decode-evidence';raise error
    monkeypatch.setattr(observe,'_mdl_decode_gate',fail)
    monkeypatch.setattr(observe,'managed_server',lambda *a,**k:pytest.fail('failed MDL gate must block service'))
    monkeypatch.setattr(observe,'_terminal',lambda *a,**k:None)
    log=tmp_path/'run'
    assert observe.run(observe.ROOT/'configs/target.json',log,'candidate')==1
    result=json.loads((log/'status.json').read_text())
    assert result['status']=='failed' and result['failed_phase']=='mdl-decode-evidence'
