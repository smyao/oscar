"""Archive #126/#145/#151: BF16 public ABI and fail-closed probe orchestration."""
from types import SimpleNamespace
import json
import pytest
import torch
from tools import probe_bf16_attention as probe

@pytest.mark.parametrize('family',('q4','q1','c4','c16'))
def test_each_public_family_receives_bf16_natural_rotation_and_original_abi(family):
    calls=[]
    class Ops:
        def __getattr__(self,name):return lambda *args:calls.append((name,args))
    tensors={name:torch.empty(1) for name in ('q','qr','ck','cv','raw','table','wk','wv','tags')}
    tensors['rv']=torch.eye(256);tensors['rv_nz']=torch.zeros(256,256);tensors['rv_bf16']=torch.eye(256,dtype=torch.bfloat16)
    buffers={name:torch.empty(1) for name in ('tasks','partial','lse','status','workspace','cluster_stats')}
    shape=SimpleNamespace(recent_tokens=256,splits=1)
    fixture={'spec':shape,'blocks':2,'stride':69632,'scale':1/16}
    probe.launch(Ops(),family,tensors,fixture,buffers,3,merge=False)
    name,args=calls[0]
    assert name==probe.OPERATORS[family]
    assert args[4] is tensors['rv_bf16'] and args[4].dtype==torch.bfloat16
    assert len(args)==(26 if probe.CLUSTER[family]>1 else 25)
    assert args[-1]==3 and args[-2]==1/16
    if probe.CLUSTER[family]>1:assert args[15] is buffers['cluster_stats']
    calls.clear();probe.launch(Ops(),family,tensors,fixture,buffers,3,baseline=True,merge=False)
    assert calls[0][1][4] is tensors['rv_nz' if family=='q4' else 'rv']

def test_case_matrix_covers_all_families_generic_and_faults():
    plan=probe.case_plan()
    for family in probe.OPERATORS:
        rows=[x for x in plan if x['family']==family]
        assert any(x['graph'] for x in rows)
        assert not any(x['timed'] for x in rows)
        assert any(x['shape'].corrupt_metadata for x in rows)
        assert any(x['shape'].corrupt_qr for x in rows)
    generic=[x for x in plan if x['shape'].name=='bf16_generic_C1_fallback']
    assert len(generic)==1 and generic[0]['family']=='c4' and not generic[0]['shape'].expect_clusters

def test_no_npu_fails_and_never_reports_cpu_substitute(tmp_path,monkeypatch):
    def unavailable():raise RuntimeError('BF16 gate requires real NPU; no CPU substitution')
    monkeypatch.setattr(probe,'require_real_npu',unavailable)
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES',raising=False)
    output=tmp_path/'probe.json'
    config=json.loads((probe.ROOT/'configs/target.json').read_text());config['experimental_bf16_attention']=True
    path=tmp_path/'effective.json';path.write_text(json.dumps(config))
    assert probe.main(['npu','--config',str(path),'--output',str(output)])==2
    report=json.loads(output.read_text())
    assert report['status']=='failed' and report['NPU']=='not_passed'
    assert 'no CPU substitution' in report['error']

def test_success_word_without_complete_receipts_is_rejected():
    bad=probe.validate_evidence({'status':'passed'})
    assert 'complete_case_plan' in bad and 'oracle_sha256' in bad

def test_config_must_enable_same_serving_branch_before_any_device_access(tmp_path,monkeypatch):
    def forbidden():raise AssertionError('device must not be touched when effective flag is off')
    monkeypatch.setattr(probe,'require_real_npu',forbidden)
    path=tmp_path/'effective.json';path.write_text(json.dumps({'experimental_bf16_attention':False}))
    with pytest.raises(RuntimeError,match='experimental_bf16_attention=True'):
        probe.run_npu(path)

def test_source_binding_covers_kernel_abi_oracle_and_actual_route():
    binding=probe.source_binding()
    required={'csrc/kernels/attention_cv_bf16_decode.cpp','csrc/kernels/attention_cv_bf16_cluster4.cpp',
              'csrc/kernels/attention_cv_bf16_cluster16.cpp','tools/bf16_reference.py',
              'configs/acceptance_bf16.json','oscar_ascend/runtime.py',
              'oscar_ascend/integration/impl.py','oscar_ascend/ops/cv_dispatch.py',
              'tools/probe_bf16_attention.py'}
    assert required<=binding.keys()
    assert all(len(value)==64 for value in binding.values())
