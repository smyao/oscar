"""Archive #86/#97/#173: read-only audit binds the selected sample/PT, never an NPU."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from oscar_ascend.rotation_calibration import OfflineLayerAccumulator, fit_checkpoint_pair, checkpoint_pair_artifact
from oscar_ascend.rotations import calibration_cache_identity


@pytest.fixture
def diagnostic_case(tmp_path):
    name = 'model.layers.3.self_attn.attn'
    gen = torch.Generator().manual_seed(21)
    q = torch.randn(128, 4, 64, generator=gen).bfloat16()
    k = (torch.randn(128, 2, 64, generator=gen) + torch.arange(64) * 2).bfloat16()
    v = torch.randn(128, 2, 64, generator=gen).bfloat16()
    acc = OfflineLayerAccumulator(head_dim=64, local_kv_heads=2, local_q_heads=4, token_budget=128, testing=True)
    acc.observe(q, k, v)
    rank = dict(rank=0, tp_size=1, global_q_heads=4, global_kv_heads=2,
                prompt_sha256='prompt', model_fingerprint='model', layers={3:acc.export_payload()})
    path = tmp_path/'rank0.pt'; torch.save(rank, path)
    config = dict(tensor_parallel_size=1, k_clip_ratio=.96,
                  mamba_cache_dtype='bfloat16', mamba_ssm_cache_dtype='bfloat16')
    profile = calibration_cache_identity(config, 'model')
    ks, vs = fit_checkpoint_pair([rank], layer_names=[name], head_dim=64,
        model_path='/model', model_fingerprint='model', global_q_heads=4, global_kv_heads=2,
        tp_size=1, prompt_sha256='prompt', max_tokens=profile['max_tokens'], calibration_profile=profile, testing=True)
    artifact = checkpoint_pair_artifact(ks, vs, layer_names=[name], head_dim=64,
        model_fingerprint='model', calibration_profile=profile, testing=True)
    artifact['capture_manifest'] = dict(status='native_post_rope_capture_complete',
        model_fingerprint='model', tp_size=1, global_q_heads=4, global_kv_heads=2, layer_ids=[3], token_count=128, prompt_sha256='prompt',
        calibration_profile=profile, rank_paths=[str(path)], rank_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
    return artifact, config, path


def test_comparison_is_read_only_and_has_all_three_recipes(diagnostic_case):
    from tools.rotation_recipe_diagnostic import compare
    artifact, config, path = diagnostic_case
    before = path.read_bytes()
    result = compare(artifact, config, max_samples=32)
    assert len(result['heads']) == 2
    assert result['model_or_npu_loaded'] is False and result['production_changed'] is False
    for head in result['heads']:
        assert head['sample_count'] == 32
        assert set(head['variants']) == {'shared_v1','perhead','perhead_center'}
        assert head['variants']['perhead_center']['prefill']['score_error_proxy'] < head['variants']['perhead']['prefill']['score_error_proxy']
    assert path.read_bytes() == before


def test_stale_sample_hash_and_wrong_model_rejected(diagnostic_case):
    from tools.rotation_recipe_diagnostic import compare
    artifact, config, path = diagnostic_case
    artifact['capture_manifest']['model_fingerprint'] = 'wrong'
    with pytest.raises(ValueError,match='fingerprint'): compare(artifact, config)
    artifact['capture_manifest']['model_fingerprint'] = 'model'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA256'): compare(artifact, config)


def test_proc_binding_groups_only_service_and_allowlisted_env(tmp_path):
    from tools.audit_precision_identity import find_active_configs, read_allowed_env
    project = tmp_path/'project'; project.mkdir()
    proc = tmp_path/'proc'; proc.mkdir()
    for pid, command in ((111,'vllm\0serve'),(112,'vllm\0serve'),(113,'python\0unrelated')):
        folder=proc/str(pid);folder.mkdir();(folder/'cwd').symlink_to(project,target_is_directory=True)
        (folder/'cmdline').write_bytes(command.encode())
        (folder/'environ').write_bytes(b'OSCAR_TARGET_CONFIG=config.json\0OSCAR_ENABLED=1\0SECRET=neverprint\0')
    rows=find_active_configs(proc,project)
    assert len(rows)==1 and rows[0]['pids']==[111,112]
    assert 'SECRET' not in read_allowed_env(proc/'111'/'environ')
    (proc/'112'/'environ').write_bytes(b'OSCAR_TARGET_CONFIG=config.json\0OSCAR_ENABLED=1\0OSCAR_ROTATION_ARTIFACT=other.pt\0')
    assert len(find_active_configs(proc,project))==2


def test_explicit_artifact_does_not_inherit_diagnostic_shell(monkeypatch,tmp_path):
    from tools.audit_precision_identity import artifact_path, main
    monkeypatch.setenv('OSCAR_ROTATION_ARTIFACT','/wrong/shell.pt')
    config={'rotation_method':'hadamard'}
    assert artifact_path(config,'fp',root=tmp_path)==tmp_path/'artifacts/rotations/fp/hadamard.pt'
    assert artifact_path(config,'fp',{'OSCAR_ROTATION_ARTIFACT':str(tmp_path/'actual.pt')},root=tmp_path)==tmp_path/'actual.pt'
    monkeypatch.setattr('tools.audit_precision_identity.find_active_configs',lambda:[])
    assert main([])==2


@pytest.mark.parametrize("refinement", [None, "optr"])
def test_calibrated_audit_matches_runtime_path_and_preserves_overrides(monkeypatch, tmp_path, refinement):
    from oscar_ascend.rotations import rotation_artifact_path
    from tools.audit_precision_identity import artifact_path
    config = dict(rotation_method="qqt_sst", rotation_variant="perhead_center",
                  experimental_official_math=True, oscar_dequant_policy="fp32_metadata",
                  max_num_batched_tokens=16384, tensor_parallel_size=4)
    if refinement is not None:
        config["rotation_refinement"] = refinement
    monkeypatch.delenv("OSCAR_ROTATION_ARTIFACT", raising=False)
    runtime_path = rotation_artifact_path(config, "fp", root=tmp_path)
    assert runtime_path.name == ("refined.pt" if refinement == "optr" else "calibrated.pt")
    before = dict(config)
    monkeypatch.setenv("OSCAR_ROTATION_ARTIFACT", "/wrong/shell.pt")
    assert artifact_path(config, "fp", root=tmp_path) == runtime_path
    process_env = {"OSCAR_ROTATION_ARTIFACT": "service.pt"}
    assert artifact_path(config, "fp", process_env, root=tmp_path) == tmp_path / "service.pt"
    assert artifact_path({**config, "rotation_artifact": "explicit.pt"}, "fp", process_env,
                         root=tmp_path) == tmp_path / "explicit.pt"
    assert config == before and process_env == {"OSCAR_ROTATION_ARTIFACT": "service.pt"}
    assert os.environ["OSCAR_ROTATION_ARTIFACT"] == "/wrong/shell.pt"


def test_score_proxy_removes_common_logit_offset():
    from tools.rotation_recipe_diagnostic import _metrics
    original=torch.arange(12,dtype=torch.float64).view(3,4)
    result=_metrics(torch,original,original+2,original,torch.eye(4,dtype=torch.float64))
    assert result['raw_score_error_proxy']==4 and result['score_error_proxy']==0
