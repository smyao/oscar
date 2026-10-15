"""Archive #126/#145/#150/#151/#154-156: host gate contracts only, not NPU evidence."""
import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools import probe_hybrid_l1_decode as probe


def synthetic_report():
    """Deliberately synthetic report to exercise fail-closed validation."""
    rows=[]
    for case in probe.case_plan():
        baseline,candidate=probe.operators(case)
        spec=case.shape;invalid=spec.corrupt_metadata or spec.corrupt_qr
        row={'case':spec.name,'status':'passed','contexts':list(spec.contexts),'qlens':list(spec.qlens),
            'baseline_operator':baseline,'candidate_operator':candidate,
            'all_tasks':spec.padded_tokens*3*spec.splits,'active_queries':sum(spec.qlens),
            'padded_queries':spec.padded_tokens-sum(spec.qlens),'queries':0 if invalid else sum(spec.qlens),
            'special_values':case.special_values,'rotation_mode':case.rotation_mode,'source_leaders':{'0':2,'1':2,'2':2},
            'oracle':'expected_error_domain' if invalid else 'all_actual_queries_raw_space_passed',
            'precision':'error_codes_finite_bits_nonfinite_classification' if invalid else 'partial_lse_status_merge_bf16_bitwise',
            'graph':None,'timing':None}
        if case.timed:
            a,b=12.,10.
            row['timing']={'baseline_ms':a,'candidate_ms':b,'ratio':b/a,'speedup':a/b,'minimum_speedup':case.minimum_speedup,
                'samples_ms':{baseline:[a]*5,candidate:[b]*5},'warmup':2,'repeats':5,
                'scope':probe.SCOPE,'gate':'passed'}
        if case.graph:
            states=[dict(row,state=name,task_sha256=task,raw_sha256=raw,table_sha256=table)
                    for name,task,raw,table in [('captured_inputs','t0','r0','b0'),
                    ('changed_q_qr','t0','r0','b0'),('changed_context_cache_table','t1','r1','b1')]]
            row['graph']={'capture':'passed','replay':'passed','same_addresses':True,'q_changed':True,
                'cache_table_tasks_changed':True,'eager_bitwise':'passed','independent_oracle':'passed','states':states}
        rows.append(row)
    return {'status':'passed','baseline_operator':probe.BASELINE,'candidate_operator':probe.CANDIDATE,
        'operator_families':probe.OPERATOR_FAMILIES,
        'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
        'performance':'passed','device_completion':'passed','scope':probe.SCOPE,
        'workspace_bytes_per_core':probe.WORKSPACE,'cases':rows,
        'rotation_format':probe.RV_FORMAT,'rotation_extra_bytes_per_layer':probe.RV_EXTRA_BYTES}


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


@pytest.mark.parametrize('index', [4,5])
@pytest.mark.parametrize('base,new,allowed',[(11.5,10.,True),(10.41,10.,True),(10.,10.,True),(9.999,10.,False)])
def test_full_cv_merge_requires_non_regression(base,new,allowed,index):
    baseline,candidate=probe.operators(probe.case_plan()[index])
    report=synthetic_report();timing=report['cases'][index]['timing']
    timing.update(baseline_ms=base,candidate_ms=new,ratio=new/base,speedup=base/new,
                  samples_ms={baseline:[base]*5,candidate:[new]*5})
    errors=probe.validate_evidence(report)
    assert (not any('device_latency' in e for e in errors)) is allowed
    timing['samples_ms'][baseline].pop()
    assert any('device_latency' in e for e in probe.validate_evidence(report))


def test_changed_input_graph_needs_cache_table_tasks_and_oracle():
    report=synthetic_report()
    for key in ('task_sha256','raw_sha256','table_sha256'):
        broken=copy.deepcopy(report);states=broken['cases'][0]['graph']['states']
        states[-1][key]=states[0][key]
        assert any('graph_inputs' in e for e in probe.validate_evidence(broken))
    report['cases'][0]['graph']['states'][-1]['queries']=0
    assert any('graph_oracle' in e for e in probe.validate_evidence(report))


@pytest.mark.parametrize('index,actual', [(0,8)])
def test_short_boundary_fixture_and_graph_relocation_keep_logical_payload(index,actual):
    case=probe.case_plan()[index];first=probe.make_fixture(torch,case)
    second=probe.make_fixture(torch,probe.replace(case,shape=probe.replace(case.shape,
        contexts=tuple(c+16 for c in case.shape.contexts))))
    assert len(first['expected'])==len(second['expected'])==actual
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
    assert bool((second['cpu']['slots'][actual:]==-1).all())
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
        ('experimental_hybrid_l1_decode','experimental_mdl_decode','experimental_decode_bundle','experimental_striped_cache')}))
    with pytest.raises(RuntimeError,match='real NPU'):
        probe.probe(path,probe.ROOT/'configs/acceptance.json')


def test_cli_preserves_failure_and_report(monkeypatch,tmp_path,capsys):
    def fail(*args):raise RuntimeError('intentional device failure')
    monkeypatch.setattr(probe,'probe',fail)
    output=tmp_path/'result.json'
    assert probe.main(['--config',str(tmp_path/'config.json'),'--output',str(output)])==2
    assert json.loads(output.read_text())['first_error']=='intentional device failure'
    captured=capsys.readouterr();assert 'intentional device failure' in captured.err
    assert 'PERF_HYBRID_L1_RESULT' in captured.out



def test_short_and_long_gates_both_require_non_regression():
    report=synthetic_report();timing=report['cases'][0]['timing']
    timing.update(baseline_ms=10.,candidate_ms=10.,ratio=1.,speedup=1.,
                  samples_ms={probe.BASELINE:[10.]*5,probe.CANDIDATE:[10.]*5})
    assert probe.validate_evidence(report)==[]
    timing['candidate_ms']=10.01
    timing['ratio']=10.01/10.;timing['speedup']=10./10.01
    timing['samples_ms'][probe.CANDIDATE]=[10.01]*5
    assert any('device_latency' in e for e in probe.validate_evidence(report))
    assert probe.case_plan()[4].minimum_speedup==1.0


def test_rv_layout_is_exact_bits_for_every_coordinate_and_asymmetric_values():
    # Includes zeros, NaN payloads, infinities and subnormals; this is a layout
    # assertion, not a numerical finite-input attention acceptance claim.
    words=torch.arange(256*256,dtype=torch.int32).reshape(256,256)
    words[0,:7]=torch.tensor([0,-2147483648,0x7fc00001,0x7f800000,1,0x3f800000,-8388608])
    rotation=words.view(torch.float32)
    original=words.clone()
    packed=probe.pack_rotation_nz(torch,rotation).view(torch.int32).flatten()
    rows=torch.arange(256)[:,None];cols=torch.arange(256)[None,:]
    offset=(cols//8)*256*8+rows*8+cols%8
    assert torch.equal(packed[offset],original)
    assert torch.equal(words,original)
    assert not torch.equal(packed.view(256,256),words)
    with pytest.raises(ValueError):probe.pack_rotation_nz(torch,rotation.to(torch.bfloat16))


def test_launch_only_candidate_receives_nz_and_q1_is_not_registered(monkeypatch):
    natural=object();nz=object();calls=[]
    monkeypatch.setattr(probe.shared,'_launch',lambda ops,name,tensors,*args:calls.append((name,tensors['rv'])))
    tensors={'rv':natural,'rv_nz':nz}
    for name in (probe.BASELINE,probe.CANDIDATE):
        probe.launch(None,name,tensors,None,None,1,merge=False)
    assert calls==[(probe.BASELINE,natural),(probe.CANDIDATE,nz)]
    assert tensors['rv'] is natural
    assert set(probe.OPERATOR_FAMILIES)=={'q4'}
    with pytest.raises(ValueError):
        probe.operators(probe.replace(probe.case_plan()[0],shape=probe.replace(probe.case_plan()[0].shape,slot_context=True)))


def test_rv_format_and_memory_accounting_are_required():
    for field in ('rotation_format','rotation_extra_bytes_per_layer'):
        report=synthetic_report();report.pop(field)
        assert field in probe.validate_evidence(report)


def test_packaging_restores_qualified_source_and_rejects_drift(tmp_path):
    from tools import package_hybrid_l1_decode as package
    report=package.package(check=True)
    assert report['inverse_full_source_bytes']
    assert report['source_sha256']=='dd732015252328efef1befc66895d27a99d9efd516935841dd775f17329eb5c1'
    assert report['workspace_bytes_per_core_D256']==probe.WORKSPACE
    source=(package.ROOT/package.TARGET).read_text()
    assert 'OSCAR_CV_RUN(1)' not in source
    assert 'OSCAR_CV_RUN(4)' in source
    for name in (package.SOURCE,package.HELPER,package.BASELINE):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((package.ROOT/name).read_bytes())
    (tmp_path/package.SOURCE).write_text('source changed')
    with pytest.raises(ValueError,match='source drift'):package.package(tmp_path)


def test_real_32k64k_contexts_have_complete_timed_operator_gate():
    case=next(c for c in probe.case_plan() if c.shape.name=='q4_R2_32k64k')
    assert case.shape.contexts==(32768,65536)
    assert case.shape.qlens==(4,4) and case.shape.splits==3
    assert case.timed and case.minimum_speedup==1.0
