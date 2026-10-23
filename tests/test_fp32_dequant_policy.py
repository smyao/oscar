"""Archive #13-20/#148: a read-side precision policy cannot change PT or argv."""
import json
from pathlib import Path

from oscar_ascend import rotations
from oscar_ascend.rotations import calibration_cache_identity
from tools.prepare_rotations import prepare
from tools.serving_variants import variant_config, runtime_feature_summary
from tools.target_cli import serve_argv
from tools import probe_official_attention as probe
from test_center_prepare import environment, publish_old, forbid_capture, frozen_files
from test_probe_official_attention import complete_receipt


def test_candidate_uses_fp32_reconstruction_without_new_calibration_or_native_arguments(environment):
    old,center,fingerprint=environment
    path,_=publish_old(center,fingerprint);before=frozen_files(path)
    candidate=variant_config(center,'candidate')
    assert candidate['oscar_dequant_policy']=='fp32_metadata'
    assert 'rotation_k_clip' not in candidate
    assert calibration_cache_identity(candidate,fingerprint)==calibration_cache_identity(center,fingerprint)
    assert rotations.rotation_artifact_path(candidate,fingerprint)==path
    result=prepare(candidate,capture_runner=forbid_capture,testing=True)
    assert result['reused'] and Path(result['artifact'])==path
    assert all(p.read_bytes()==b for p,b in before.items())
    # Compare only the read-side flag; no native serving option changes.
    serving=variant_config(json.loads((probe.ROOT/'configs/target.json').read_text()),'candidate')
    native_phase={**serving,'oscar_dequant_policy':'native_phase'}
    assert serve_argv(serving)==serve_argv(native_phase)
    summary=runtime_feature_summary(candidate,'candidate')
    assert summary['dequant_arithmetic']=='float32_then_bfloat16'
    assert summary['runtime_precision_contract']=='oscar_bf16_runtime_fp32_dequant_v1'
    for variant in ('baseline','native'):
        assert 'oscar_dequant_policy' not in variant_config(candidate,variant)


def test_operator_gate_cannot_use_old_runtime_precision_receipt():
    config=variant_config(json.loads((probe.ROOT/'configs/target.json').read_text()),'candidate')
    report=complete_receipt()
    assert 'runtime_precision' in probe.validate_evidence(report,config)
    report['runtime_precision']=probe.runtime_precision(config)
    assert report['runtime_precision']['dequant_modes']==dict(prefill=0,decode=0,mtp=0,capture=0)
    assert probe.validate_evidence(report,config)==[]
    report['runtime_precision']['dequant_modes']['capture']=1
    assert 'runtime_precision' in probe.validate_evidence(report,config)


def test_existing_v4_graph_matrix_exercises_mode0_for_each_reader_family():
    plan=probe.case_plan()
    assert {case['family'] for case in plan if case['graph'] and case['dequant_mode']==0}==set(probe.OPERATORS)


def test_compiler_cache_identity_cannot_replay_a_native_phase_graph(tmp_path):
    from tools.compile_cache import select_compile_cache
    package=tmp_path/'package';package.mkdir();(package/'__init__.py').write_text('')
    config=variant_config(json.loads((probe.ROOT/'configs/target.json').read_text()),'candidate')
    base={'VLLM_CACHE_ROOT':str(tmp_path/'compile-cache')}
    old=select_compile_cache({**config,'oscar_dequant_policy':'native_phase'},native=False,env=dict(base),source_root=package)
    new=select_compile_cache(config,native=False,env=dict(base),source_root=package)
    assert old['root']!=new['root']
