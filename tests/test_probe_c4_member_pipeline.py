"""Host gates only; archive #148/#150/#153/#154. No NPU/graph claims."""
import copy
import json
from pathlib import Path
from types import ModuleType,SimpleNamespace
import sys
import pytest
import torch
from tools import probe_c4_member_pipeline as probe


def evidence():
    detail={'eligible_clusters':1,'shared_history_kv_tiles':3,'rounds_per_cluster':3,
            'oracle':'all_336_queries_passed','task_sha256':'initial-tasks'}
    states=[dict(detail,state=name,context=context,task_sha256=task,packed_sha256=raw)
            for name,context,task,raw in [('captured_inputs',769,'initial-tasks','initial-raw'),
              ('changed_q_qr',769,'initial-tasks','initial-raw'),('changed_context_tasks_packed_meta',801,'new-tasks','new-raw')]]
    graph={'capture':'passed','replay':'passed','same_addresses':True,'q_qr_changed':True,
           'task_metadata_changed':True,'packed_meta_changed':True,'independent_oracle':'passed',
           'graph_eager_bitwise':'passed','mature_multikv_in_every_state':True,'states':states}
    timing={'baseline_ms':10.,'candidate_ms':8.,'ratio':.8,'speedup':1.25,'minimum_speedup':1.15,'gate':'passed',
            'samples_ms':{probe.BASELINE:[10.]*5,probe.CANDIDATE:[8.]*5},
            'scope':'complete_H16_vs_H32_pipeline_same_shape_device_events'}
    rows=[{'case':name,'status':'passed','precision':'partial_lse_status_clusterStats_bitwise',
           'detail':dict(detail,oracle='all_336_queries_passed' if i==0 else 'expected_invalid_metadata_status'),
           'graph':graph if i==0 else None,'timing':timing if i==0 else None}
          for i,name in enumerate(probe.CASE_NAMES)]
    return {'status':'passed','baseline_operator':probe.BASELINE,'candidate_operator':probe.CANDIDATE,
            'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed',
            'performance':'passed','device_completion':'passed','all_tasks':1008,
            'source2':'production_prefill_suppressed','cases':rows}


def test_missing_or_c1_only_graph_cannot_qualify_pipeline():
    complete=evidence();assert probe.validate_evidence(complete)==[]
    for mutation in ('missing','c1','one_round','no_task_change','no_raw_change','missing_oracle','timing_sample'):
        report=copy.deepcopy(complete);row=report['cases'][0];graph=row['graph']
        if mutation=='missing':row['graph']=None
        elif mutation=='c1':graph['states'][0]['eligible_clusters']=0
        elif mutation=='one_round':graph['states'][0]['rounds_per_cluster']=1
        elif mutation=='no_task_change':graph['states'][2]['task_sha256']=graph['states'][0]['task_sha256']
        elif mutation=='no_raw_change':graph['states'][2]['packed_sha256']=graph['states'][0]['packed_sha256']
        elif mutation=='missing_oracle':report['cases'][2]['detail'].pop('oracle')
        else:row['timing']['samples_ms'][probe.BASELINE].pop()
        assert probe.validate_evidence(report),mutation


@pytest.mark.parametrize('position,lane',[(80,1),(96,0),(336,1)])
def test_all_task_status_oracle_preserves_old_16_row_lanes(position,lane):
    tasks=torch.zeros((5,16),dtype=torch.int64)
    tasks[:,1]=21;tasks[:,3]=64;tasks[:,4]=769
    tasks[1,1]=0;tasks[2,7]=1;tasks[3,4]=position;tasks[4,3]=position+1
    expected=probe.expected_fault_status(torch,tasks,position)
    assert torch.nonzero(expected).tolist()==[[0,lane]]
    assert expected[0,lane].item()==3


def test_graph_replacement_is_same_geometry_and_changes_real_packed_metadata():
    first=probe.make_fixture(torch,probe.case_plan()[0])
    second=probe.make_fixture(torch,probe.replace(probe.case_plan()[0],contexts=(801,)))
    assert len(first['expected'])==len(second['expected'])==336
    assert first['blocks']==second['blocks']
    assert {k:(v.shape,v.dtype) for k,v in first['cpu'].items()}=={k:(v.shape,v.dtype) for k,v in second['cpu'].items()}
    assert not torch.equal(first['cpu']['raw'],second['cpu']['raw'])
    assert not torch.equal(first['cpu']['lens'],second['cpu']['lens'])
    assert torch.equal(first['cpu']['starts'],second['cpu']['starts'])


def test_requested_probe_has_no_cpu_fallback(monkeypatch,tmp_path):
    monkeypatch.setattr(probe.reuse,'_active_device',lambda target:None)
    monkeypatch.setitem(sys.modules,'torch_npu',ModuleType('torch_npu'))
    monkeypatch.setattr(torch,'npu',SimpleNamespace(is_available=lambda:False),raising=False)
    path=tmp_path/'config.json';path.write_text(json.dumps({'experimental_c4_member_pipeline':True}))
    with pytest.raises(RuntimeError,match='target NPU required'):
        probe.probe(path,probe.ROOT/'configs/acceptance.json')


def test_observe_requires_complete_signed_pipeline_evidence(monkeypatch,tmp_path):
    from tools import observe_serve as observe
    from oscar_ascend.ops import loader
    manifest={'signature':'new-C4-build','sha256':{'extension':'same-loaded-artifact'}}
    monkeypatch.setattr(loader,'validate_build_artifacts',lambda *a:manifest)
    monkeypatch.setattr(observe,'read_npu_resources',lambda *a,**k:{})
    monkeypatch.setattr(observe,'wait_for_release',lambda *a,**k:{'status':'passed'})
    report=evidence();report.update(artifact_signature=manifest['signature'],artifact_sha256=manifest['sha256'])
    def phase(name,argv,**kwargs):
        assert name=='c4-member-pipeline' and 'tools.probe_c4_member_pipeline' in argv
        out=Path(argv[argv.index('--output')+1]);assert out.name!='c4-member-pipeline.json'
        out.write_text(json.dumps(report))
    monkeypatch.setattr(observe,'_phase',phase)
    observe._c4_member_pipeline_gate(tmp_path/'config.json',{}, {},tmp_path,{})
    report['cases'][0]['graph']['states'][0]['eligible_clusters']=0
    with pytest.raises(RuntimeError,match='graph_mature_states'):
        observe._c4_member_pipeline_gate(tmp_path/'config.json',{}, {},tmp_path,{})
    report['cases'][0]['graph']['states'][0]['eligible_clusters']=1
    report['artifact_signature']='old-build'
    with pytest.raises(RuntimeError,match='artifact_signature'):
        observe._c4_member_pipeline_gate(tmp_path/'config.json',{}, {},tmp_path,{})


def test_explicit_research_plan_and_failed_pipeline_gate_block_model(monkeypatch,tmp_path,capsys):
    from tools import observe_serve as observe
    from tools import install_serve
    target=json.loads((observe.ROOT/'configs/target.json').read_text())
    target['experimental_c4_member_pipeline']=True
    config_path=tmp_path/'explicit-research.json';config_path.write_text(json.dumps(target))
    assert observe.main(['--config',str(config_path),'--variant','candidate','--plan','--probe-only'])==0
    plan=json.loads(capsys.readouterr().out)
    assert 'c4_member_pipeline_gate' in plan['phases']
    assert plan['runtime_selection']['optimizations']['c4_member_pipeline'] is True
    assert install_serve.main(['--config',str(config_path),'--variant','candidate','--plan'])==0
    direct=json.loads(capsys.readouterr().out)
    assert direct['runtime_selection']==plan['runtime_selection']
    assert direct['runtime_selection']['prefix_caching'] is True
    for name in ('_preflight','_candidate_gate','_fast_unpack_gate','_mixed_optimization_gate',
                 '_striped_cache_gate','_decode_bundle_gate','_current_only_gate',
                 '_first_mtp_current_gate','_mixed_decode_split_gate','_request_window_gate'):
        monkeypatch.setattr(observe,name,lambda *a,**k:None)
    seen=[]
    def fail(config_path,config,*args):
        assert config['experimental_c4_member_pipeline'] is True
        assert json.loads(Path(config_path).read_text())['experimental_c4_member_pipeline'] is True
        seen.append('pipeline_gate')
        error=RuntimeError('pipeline NPU graph rejected');error.phase='c4-member-pipeline-evidence';raise error
    monkeypatch.setattr(observe,'_c4_member_pipeline_gate',fail)
    monkeypatch.setattr(observe,'managed_server',lambda *a,**k:pytest.fail('failed C4 gate must block model startup'))
    monkeypatch.setattr(observe,'_terminal',lambda *a,**k:None)
    logs=tmp_path/'observe'
    assert observe.run(config_path,logs,'candidate')==1
    assert seen==['pipeline_gate']
    status=json.loads((logs/'status.json').read_text())
    assert status['status']=='failed' and status['failed_phase']=='c4-member-pipeline-evidence'


@pytest.mark.parametrize('baseline,candidate,passed',[(1.148704,1.0,False),(11.5,10.0,True),(10.0,10.0,False)])
def test_complete_path_threshold_does_not_round_measured_subthreshold_gain(baseline,candidate,passed):
    report=evidence();timing=report['cases'][0]['timing']
    timing.update(baseline_ms=baseline,candidate_ms=candidate,ratio=candidate/baseline,
                  speedup=baseline/candidate,samples_ms={probe.BASELINE:[baseline]*5,probe.CANDIDATE:[candidate]*5})
    assert ('device_latency' not in probe.validate_evidence(report)) is passed
    timing.pop('minimum_speedup')
    assert 'device_latency' in probe.validate_evidence(report)
