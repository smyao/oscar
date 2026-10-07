"""Mock-report rejection tests; no NPU acceptance is generated here."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from tools import probe_mixed_decode_split as probe


def evidence():
    acceptance=json.loads(Path('configs/acceptance.json').read_text());rows=[]
    for case in probe.case_plan():
        n=sum(case.lengths);cut=sum(case.lengths[:-1])
        rows.append({'case':case.name,'status':'passed',
            'precision':'frozen_output_LSE_and_independent_oracle_passed','store_bitwise':'passed',
            'first_mtp':case.draft,'slot_dtype':case.slot_dtype,'rotation':'identity' if case.draft else 'hadamard',
            'original_source_splits_preserved':True,
            'partition':[{'segment':'short_prefix','tokens':cut,'reference_tokens':n},
                         {'segment':'long_suffix','tokens':case.lengths[-1],'reference_tokens':n}],
            'timings':{'old_ms':2.,'new_ms':1.,'ratio':.5,'gate':'passed',
                       'samples_ms':{'old':[2.]*5,'new':[1.]*5}} if case.timed else None,
            'decode_after_store':{'status':'passed','reader':'production_subsequent_q1',
                'output_bitwise':'passed','independent_oracle':'passed','subsequent_store_bitwise':'passed',
                'sampled_queries':len(case.lengths)} if not case.timed else None})
    return {'cases':rows},acceptance


def test_complete_mock_schema():
    report,acceptance=evidence();probe.validate_case_evidence(report,acceptance)


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(cases=[]),
    lambda r:r['cases'].__setitem__(1,deepcopy(r['cases'][0])),
    lambda r:r['cases'][0]['partition'][0].update(reference_tokens=8),
    lambda r:r['cases'][0].update(original_source_splits_preserved=False),
    lambda r:r['cases'][0]['decode_after_store'].update(independent_oracle='not_run'),
    lambda r:r['cases'][1].update(first_mtp=False),
    lambda r:r['cases'][2]['timings']['samples_ms']['new'].pop(),
    lambda r:r['cases'][2]['timings'].update(ratio=.1),
    lambda r:r['cases'][2]['timings']['samples_ms']['new'].__setitem__(0,float('nan')),
])
def test_incomplete_or_wrong_partition_fails(mutate):
    report,acceptance=evidence();mutate(report)
    with pytest.raises(RuntimeError):probe.validate_case_evidence(report,acceptance)


def test_latency_is_complete_forward_and_cannot_be_relaxed():
    report,acceptance=evidence();row=report['cases'][2]['timings']
    row.update(new_ms=2.01,ratio=1.005);row['samples_ms']['new']=[2.01]*5
    with pytest.raises(RuntimeError,match='complete-forward'):
        probe.validate_case_evidence(report,acceptance)


@pytest.mark.parametrize('failure',[None,'phase','release','empty','artifact'])
def test_observer_gate_blocks_failures_and_preserves_cleanup(monkeypatch,tmp_path,failure):
    from tools import observe_serve
    from oscar_ascend.ops import loader
    report,_=evidence()
    report.update({k:'passed' for k in ('status','precision','performance','store_bitwise','device_completion')})
    report.update(artifact_signature='mock-signature',artifact_sha256='mock-sha')
    if failure=='empty':report['cases']=[]
    if failure=='artifact':report['artifact_signature']='stale-signature'
    monkeypatch.setattr(loader,'validate_build_artifacts',lambda p:{'signature':'mock-signature','sha256':'mock-sha'})
    monkeypatch.setattr(observe_serve,'read_npu_resources',lambda *a,**k:{})
    released=[]
    def release(*a,**k):
        released.append(True);return {'status':'failed' if failure=='release' else 'passed'}
    monkeypatch.setattr(observe_serve,'wait_for_release',release)
    def phase(*a,**k):
        if failure=='phase':raise RuntimeError('mock NPU failure')
        (tmp_path/'mixed-decode-split-report.json').write_text(json.dumps(report))
    monkeypatch.setattr(observe_serve,'_phase',phase)
    status={}
    if failure:
        with pytest.raises(RuntimeError):observe_serve._mixed_decode_split_gate(
            tmp_path/'config',{'experimental_mixed_decode_split':True},{},tmp_path,status)
        assert 'mixed_decode_split_gate' not in status
    else:
        observe_serve._mixed_decode_split_gate(tmp_path/'config',{'experimental_mixed_decode_split':True},{},tmp_path,status)
        assert status['mixed_decode_split_gate']['full_service_performance']=='not_established'
    assert released==[True]


def test_lse_witness_saves_prefix_before_shared_workspace_is_reused(monkeypatch):
    import torch
    from dataclasses import replace
    from types import SimpleNamespace
    from oscar_ascend.integration.metadata import OscarMetadata
    state=SimpleNamespace(packed=object(),workspace=SimpleNamespace(lse=torch.empty(8,1)))
    starts=torch.tensor([0,2,6],dtype=torch.int32)
    meta=OscarMetadata(starts,torch.tensor([4,9]),torch.zeros(2,1,dtype=torch.int32),
                       torch.arange(6),2,6,4,9,num_input_tokens=6,mixed_decode_split=True)
    class Impl:
        num_heads=1
        def forward(self,layer,q,k,v,cache,m,output=None,**kwargs):
            if m.mixed_decode_split:
                self.forward(layer,q[:2],k[:2],v[:2],cache,
                    replace(m,mixed_decode_split=False,mixed_split_segment='short_prefix',cv_shape_tokens=6),output=output[:2])
                self.forward(layer,q[2:],k[2:],v[2:],cache,
                    replace(m,mixed_decode_split=False,mixed_split_segment='long_suffix',cv_shape_tokens=6),output=output[2:])
            else:
                state.workspace.lse[:q.shape[0]].fill_(3 if m.mixed_split_segment=='short_prefix' else 7)
                output.copy_(q)
            return output
    proxy=SimpleNamespace(empty_like=torch.empty_like,full=torch.full,float32=torch.float32,
                          npu=SimpleNamespace(synchronize=lambda:None))
    q=torch.arange(6,dtype=torch.float32).view(6,1,1)
    impl=Impl();out,lse,records=probe._initial_forward(proxy,impl,state,{'q':q,'ck':q,'cv':q},meta,2)
    assert torch.equal(out,q)
    assert lse[:,0].tolist()==[3,3,7,7,7,7]
    assert len(records)==2 and 'forward' not in vars(impl)
