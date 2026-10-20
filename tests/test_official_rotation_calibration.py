"""Archive #86/#97/#165/#166: calibration provenance must fail closed.

Offline CPU calibration semantics; no model/NPU/quality acceptance claim.
"""
import copy, hashlib, json
from pathlib import Path
import pytest
import torch
from oscar_ascend.rotation_calibration import (OfflineLayerAccumulator,local_covariance_sums,
    fit_checkpoint_pair,checkpoint_pair_artifact,publish_checkpoint_pair,verified_pair_files,OFFICIAL_COMMIT)
from oscar_ascend.rotations import (rotation_artifact_path,model_artifact_fingerprint,
    build_hadamard_artifact,save_artifact,load_active_artifact,checkpoint_identity,
    calibration_cache_identity,CALIBRATION_RECIPE_VERSION)
from tools.prepare_rotations import model_geometry,prepare

NAME='model.layers.3.self_attn.attn'


def sample(seed=7,n=615,hq=4,hk=2,d=8):
    g=torch.Generator().manual_seed(seed)
    return tuple(torch.randn(n,h,d,generator=g).to(torch.bfloat16) for h in (hq,hk,hk))


def payload(q,k,v):
    acc=OfflineLayerAccumulator(head_dim=q.shape[-1],local_q_heads=q.shape[1],
        local_kv_heads=k.shape[1],token_budget=len(q),testing=True)
    acc.observe(q[:513],k[:513],v[:513]);acc.observe(q[513:],k[513:],v[513:])
    return acc.export_payload()


def test_fp64_statistics_match_independent_all_head_qqt_sst_equations():
    q,k,v=sample(n=2111);data=payload(q,k,v);actual=local_covariance_sums(data,testing=True)
    rk=torch.zeros(8,8,dtype=torch.float64);rv=torch.zeros_like(rk)
    for head in range(2):
        query=q[:,head*2:(head+1)*2].reshape(-1,8).double()
        key,val=k[:,head].double(),v[:,head].double()
        cov=query.T@query/len(query);energy=((key@cov)*key).sum(-1)
        rk+=cov;rv+=val.T@(val*energy[:,None])/energy.sum().clamp_min(1e-12)
    assert actual.dtype==torch.float64 and data['k'].dtype==torch.bfloat16
    torch.testing.assert_close(actual,torch.stack((rk,rv)),atol=2e-14,rtol=2e-14)


def test_one_pass_collector_rejects_cpu_production_misalignment_and_incomplete_rows():
    q,k,v=sample(n=3)
    acc=OfflineLayerAccumulator(head_dim=8,local_q_heads=4,local_kv_heads=2,token_budget=3)
    with pytest.raises(RuntimeError,match='NPU'):acc.observe(q,k,v)
    acc=OfflineLayerAccumulator(head_dim=8,local_q_heads=4,local_kv_heads=2,token_budget=3,testing=True)
    with pytest.raises(ValueError,match='token counts'):acc.observe(q,k[:-1],v)
    with pytest.raises(ValueError,match='budget'):acc.export_payload()
    q[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='non-finite'):acc.observe(q,k,v)


def test_zero_key_energy_matches_official_clamp_without_nan_or_replay():
    q,k,v=sample(n=17);k.zero_()
    sums=local_covariance_sums(payload(q,k,v),testing=True)
    assert torch.isfinite(sums).all() and torch.count_nonzero(sums[1])==0


def test_offline_statistics_do_not_inherit_a_workers_default_tensor_device():
    q,k,v=sample(n=17)
    with torch.device('meta'):
        data=payload(q,k,v)
        sums=local_covariance_sums(data,testing=True)
    assert all(data[key].device.type=='cpu' for key in ('q_gram','k','v'))
    assert sums.device.type=='cpu' and sums.dtype==torch.float64


def rank_states(fingerprint='model',prompt='prompt',n=615):
    result=[]
    for rank in range(2):
        q,k,v=sample(seed=rank+3,n=n,hq=2,hk=1)
        result.append({'rank':rank,'tp_size':2,'global_q_heads':4,'global_kv_heads':2,
            'prompt_sha256':prompt,'model_fingerprint':fingerprint,'layers':{3:payload(q,k,v)}})
    return result


def profile(fingerprint='model'):
    # Standalone statistic tests have no model files. Declare both concrete
    # native state dtypes instead of inventing metadata for an absent model.
    return calibration_cache_identity({'mamba_cache_dtype':'bfloat16',
        'mamba_ssm_cache_dtype':'bfloat16'},fingerprint)


def fitted(states=None,fingerprint='model',calibration_profile=None):
    states=rank_states(fingerprint) if states is None else states
    expected_profile=profile(fingerprint) if calibration_profile is None else calibration_profile
    return fit_checkpoint_pair(states,layer_names=[NAME],head_dim=8,model_path='/model',
        model_fingerprint=fingerprint,global_q_heads=4,global_kv_heads=2,tp_size=2,
        prompt_sha256='prompt',max_tokens=expected_profile['max_tokens'],
        calibration_profile=expected_profile,testing=True)


def test_tp_global_layer_rotation_uses_fp64_eigh_hadamard_then_official_pbr():
    states=rank_states();k,v=fitted(states)
    cov=sum((local_covariance_sums(r['layers'][3],testing=True) for r in states))/2
    cov=(cov+cov.transpose(-1,-2))/2
    values,vectors=torch.linalg.eigh(cov)
    h=torch.tensor([[(-1.)**((i&j).bit_count()) for j in range(8)] for i in range(8)],dtype=torch.float64)/(8**.5)
    br=torch.tensor([0,4,2,6,1,5,3,7])
    for index,state in enumerate((k,v)):
        permutation=torch.empty(8,dtype=torch.long);permutation[br]=torch.argsort(values[index],descending=True)
        expected=(vectors[index]@h)[:,permutation].float()
        torch.testing.assert_close(state['layers'][3]['rotation'],expected,atol=1e-7,rtol=1e-7)
        assert state['calibration']['accumulator_dtype']=='float64'
        assert state['source_grouping']=='layer' and state['layers'][3]['rotation'].shape==(8,8)
    assert not torch.equal(k['layers'][3]['rotation'],v['layers'][3]['rotation'])


@pytest.mark.parametrize('mutation,match',[
    (lambda x:x.pop(),'complete unique'),
    (lambda x:x[1].update(prompt_sha256='other'),'identity'),
    (lambda x:x[1]['layers'].clear(),'coverage'),
    (lambda x:x[1]['layers'][3].update(local_q_heads=4),'geometry'),
])
def test_rank_identity_and_complete_coverage_fail_closed(mutation,match):
    states=rank_states();mutation(states)
    with pytest.raises(ValueError,match=match):fitted(states)


def test_official_pair_import_publication_and_corruption_do_not_fall_back(tmp_path):
    k,v=fitted()
    artifact=checkpoint_pair_artifact(k,v,layer_names=[NAME],head_dim=8,model_fingerprint='model',
        calibration_profile=profile(),testing=True)
    assert artifact['pr_fingerprint']==OFFICIAL_COMMIT and artifact['calibrated']
    publish_checkpoint_pair(k,v,tmp_path,artifact=artifact)
    gotk,gotv,got=verified_pair_files(tmp_path)
    assert gotk['objective']=='qqt_r_h_pbr' and gotv['objective']=='sst_r_h_pbr'
    assert torch.equal(got['layers'][NAME]['Rk'],k['layers'][3]['rotation'])
    (tmp_path/'.oscar.pending.json').write_text('{}')
    with pytest.raises(ValueError,match='incomplete'):verified_pair_files(tmp_path)
    (tmp_path/'.oscar.pending.json').unlink()
    (tmp_path/'calibrated.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash mismatch'):verified_pair_files(tmp_path)


def test_pair_generation_and_objective_mismatch_is_rejected():
    k,v=fitted();v['calibration']['generation_id']='other'
    with pytest.raises(ValueError,match='generation mismatch'):
        checkpoint_pair_artifact(k,v,layer_names=[NAME],head_dim=8,model_fingerprint='model',
            calibration_profile=profile(),testing=True)
    v['objective']='hadamard'
    with pytest.raises(ValueError,match='objective'):
        checkpoint_pair_artifact(k,v,layer_names=[NAME],head_dim=8,model_fingerprint='model',
            calibration_profile=profile(),testing=True)


def model(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'num_hidden_layers':4,
        'layer_types':['linear_attention']*3+['full_attention'],'head_dim':8,
        'num_attention_heads':4,'num_key_value_heads':2,'mamba_ssm_dtype':'float32'}))
    (tmp_path/'model.safetensors').write_bytes(b'test-weight-identity')
    return {'model':str(tmp_path),'rotation_method':'qqt_sst','tensor_parallel_size':2,
            'rotation_calibration_tokens':1000}


def test_cache_changes_with_weight_identity_prompt_bytes_and_budget(tmp_path):
    config=model(tmp_path);_,_,fp=model_geometry(tmp_path)
    path=rotation_artifact_path(config,fp,root=tmp_path)
    assert path.name=='calibrated.pt' and 'hadamard' not in str(path)
    assert rotation_artifact_path({**config,'rotation_calibration_tokens':2000},fp,root=tmp_path)!=path
    (tmp_path/'model.safetensors').write_bytes(b'different-weight-identity')
    assert model_artifact_fingerprint(tmp_path)!=fp
    prompt=tmp_path/'prompt.jsonl';prompt.write_text('"one"\n')
    config['rotation_calibration_prompts']=str(prompt);first=rotation_artifact_path(config,fp,root=tmp_path)
    prompt.write_text('"two"\n');assert rotation_artifact_path(config,fp,root=tmp_path)!=first


def test_checkpoint_identity_accepts_hf_snapshot_weight_symlinks(tmp_path):
    snapshot=tmp_path/'snapshot';snapshot.mkdir();model(snapshot)
    blob=tmp_path/'weight-blob';blob.write_bytes(b'actual-weights')
    (snapshot/'model.safetensors').unlink();(snapshot/'model.safetensors').symlink_to(blob)
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'layer.weight':'model.safetensors'}}))
    first=checkpoint_identity(snapshot)[2];blob.write_bytes(b'changed-weights')
    assert checkpoint_identity(snapshot)[2]!=first


def test_prepare_calls_native_capture_then_offline_fit_and_reuses_verified_pair(tmp_path):
    config=model(tmp_path);_,_,fp=model_geometry(tmp_path);calls=[]
    def capture(_config,directory):
        calls.append('native-capture');paths=[]
        for rank in rank_states(fp,n=13):
            path=directory/f'rank{rank["rank"]}.pt';torch.save(rank,path);paths.append(str(path))
        return {'status':'native_post_rope_capture_complete','model_fingerprint':fp,
            'layer_ids':[3],'prompt_sha256':'prompt','token_count':13,'rank_paths':paths,
            'calibration_profile':calibration_cache_identity(config,fp)}
    output=tmp_path/'generated/calibrated.pt'
    first=prepare(config,output=output,capture_runner=capture,testing=True)
    assert first['calibrated'] and first['accumulator_dtype']=='float64' and calls==['native-capture']
    second=prepare(config,output=output,capture_runner=lambda *_:pytest.fail('cache resampled'),testing=True)
    assert second['reused'] is True
    assert (output.parent/'k_rotation_qqt_r_h_pbr.pt').exists()
    # An explicit path cannot make unsupported native state storage valid,
    # even though the weights and all file hashes still match this checkpoint.
    with pytest.raises(ValueError,match='unsupported.*Ascend910B4'):
        load_active_artifact({**config,'mamba_ssm_cache_dtype':'float32'},
            path=output,device='cpu',testing=True)


def test_old_explicit_hadamard_is_rejected_without_overwriting(tmp_path):
    config=model(tmp_path);_,_,fp=model_geometry(tmp_path);path=tmp_path/'old.pt'
    h=build_hadamard_artifact(layer_names=[NAME],head_dim=8,model_fingerprint=fp,device='cpu',testing=True)
    save_artifact(h,path);before=path.read_bytes()
    with pytest.raises(ValueError,match='PR fingerprint|calibrated'):
        prepare(config,output=path,capture_runner=lambda *_:pytest.fail('invalid explicit cache resampled'),testing=True)
    assert path.read_bytes()==before


def test_interrupted_managed_pair_is_recalibrated_but_never_reused(tmp_path,monkeypatch):
    from oscar_ascend import rotations
    config=model(tmp_path);_,_,fp=model_geometry(tmp_path);calls=[]
    output=tmp_path/'managed/calibrated.pt'
    monkeypatch.setattr(rotations,'rotation_artifact_path',lambda *_a,**_k:output)
    def capture(_config,directory):
        calls.append(1);paths=[]
        for rank in rank_states(fp,n=13):
            path=directory/f'rank{rank["rank"]}.pt';torch.save(rank,path);paths.append(str(path))
        return {'status':'native_post_rope_capture_complete','model_fingerprint':fp,
            'layer_ids':[3],'prompt_sha256':'prompt','token_count':13,'rank_paths':paths,
            'calibration_profile':calibration_cache_identity(config,fp)}
    prepare(config,capture_runner=capture,testing=True)
    (output.parent/'.oscar.pending.json').write_text('{}')
    result=prepare(config,capture_runner=capture,testing=True)
    assert len(calls)==2 and result['reused'] is False
    assert not (output.parent/'.oscar.pending.json').exists()
    load_active_artifact(config,device='cpu',testing=True)


def test_official_pair_import_and_active_loader_bind_complete_generation(tmp_path):
    config=model(tmp_path);_,_,fp=model_geometry(tmp_path)
    assert checkpoint_identity(tmp_path)==model_geometry(tmp_path)
    k,v=fitted(rank_states(fp),fp,calibration_cache_identity(config,fp));source=tmp_path/'source';source.mkdir()
    kp,vp=source/'k.pt',source/'v.pt';torch.save(k,kp);torch.save(v,vp)
    originals={p:p.read_bytes() for p in (kp,vp)}
    output=tmp_path/'runtime/calibrated.pt'
    result=prepare(config,output=output,import_pair=(kp,vp),testing=True)
    assert result['source']=='official_qqt_sst_import'
    loaded=load_active_artifact(config,path=output,device='cpu',testing=True)
    assert loaded['calibrated'] and torch.equal(loaded['layers'][NAME]['Rv'],v['layers'][3]['rotation'])
    assert all(p.read_bytes()==b for p,b in originals.items())
    # Even a freshly recalculated manifest cannot turn unrelated combined
    # constants into the official K/V pair merely by setting calibrated=True.
    loaded['layers'][NAME]['Rk']=loaded['layers'][NAME]['Rv'].clone();torch.save(loaded,output)
    marker=output.parent/'.oscar.complete.json';manifest=json.loads(marker.read_text())
    manifest['files'][output.name]=hashlib.sha256(output.read_bytes()).hexdigest();marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='differ from official pair'):
        load_active_artifact(config,path=output,device='cpu',testing=True)


def test_model_identity_changes_with_tokenizer_and_quantization_descriptors(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    before=model_artifact_fingerprint(tmp_path)
    for name in ('tokenizer.json','chat_template.jinja','quant_model_description.json'):
        path=tmp_path/name;path.write_text('first')
        after=model_artifact_fingerprint(tmp_path);assert after!=before
        path.write_text('second')
        changed=model_artifact_fingerprint(tmp_path);assert changed!=after
        before=changed


def test_calibration_path_binds_native_bf16_state_and_rejects_unsupported_storage(tmp_path):
    config=model(tmp_path);_,_,fp=model_geometry(tmp_path)
    current=calibration_cache_identity(config,fp)
    assert current['recipe_version']==CALIBRATION_RECIPE_VERSION
    assert current['model_execution']['mamba_cache_dtype']=='bfloat16'
    # This fixture's HF JSON declares float32, but the pinned 910B4 chunked
    # prefill route supports BF16 state; model metadata cannot override that.
    assert current['model_execution']['mamba_ssm_cache_dtype']=='bfloat16'
    current_path=rotation_artifact_path(config,fp,root=tmp_path)
    for choice in ('auto','bfloat16'):
        explicit={**config,'mamba_cache_dtype':choice,'mamba_ssm_cache_dtype':choice}
        assert rotation_artifact_path(explicit,fp,root=tmp_path)==current_path
    for name in ('mamba_cache_dtype','mamba_ssm_cache_dtype'):
        for dtype in ('float16','float32'):
            with pytest.raises(ValueError,match='unsupported.*Ascend910B4'):
                rotation_artifact_path({**config,name:dtype},fp,root=tmp_path)


@pytest.mark.parametrize('field',['mamba_cache_dtype','mamba_ssm_cache_dtype'])
@pytest.mark.parametrize('dtype',['float16','float32'])
def test_current_recipe_cannot_fit_a_forged_non_native_state_profile(field,dtype):
    altered=profile()
    altered['model_execution'][field]=dtype
    with pytest.raises(ValueError,match='requires the current native BF16 state'):
        fitted(calibration_profile=altered)


@pytest.mark.parametrize('mutation,match',[
    (lambda meta:meta.pop('model_fingerprint'),'fingerprint'),
    (lambda meta:meta.update(model_fingerprint='different-weights'),'fingerprint'),
    (lambda meta:meta.pop('calibration_profile'),'execution/profile'),
    (lambda meta:meta['calibration_profile']['model_execution'].update(
        mamba_ssm_cache_dtype='float32'),'execution/profile'),
    (lambda meta:meta['calibration_profile']['model_execution'].update(
        quantization='ascend'),'execution/profile'),
    (lambda meta:meta['calibration_profile'].pop('recipe_version'),'execution/profile'),
])
def test_import_rejects_missing_or_different_model_and_execution_identity(tmp_path,mutation,match):
    config=model(tmp_path);_,_,fp=model_geometry(tmp_path)
    expected=calibration_cache_identity(config,fp)
    k,v=fitted(rank_states(fp,n=13),fp,expected)
    for state in (k,v):
        # Official external PTs may omit our fingerprint. That omission is
        # unknown provenance, never permission to relabel another model's PT.
        state['calibration']=copy.deepcopy(state['calibration'])
        mutation(state['calibration'])
    source=tmp_path/'foreign';source.mkdir()
    paths=source/'k.pt',source/'v.pt'
    for state,path in zip((k,v),paths):torch.save(state,path)
    before={path:path.read_bytes() for path in paths}
    output=tmp_path/'selected/calibrated.pt'
    with pytest.raises(ValueError,match=match):
        prepare(config,output=output,import_pair=paths,testing=True)
    assert not output.exists() and not (output.parent/'.oscar.complete.json').exists()
    assert all(path.read_bytes()==data for path,data in before.items())


def test_old_automatic_recipe_moves_path_and_generates_once_old_explicit_pair_fails(tmp_path,monkeypatch):
    from oscar_ascend import rotations
    config=model(tmp_path);_,_,fp=model_geometry(tmp_path)
    expected=calibration_cache_identity(config,fp)
    k,v=fitted(rank_states(fp,n=13),fp,expected)
    artifact=checkpoint_pair_artifact(k,v,layer_names=[NAME],head_dim=8,
        model_fingerprint=fp,calibration_profile=expected,testing=True)
    # Recreate the pre-fix cache schema and directory, including a valid pair
    # manifest. A valid hash does not bind the old pair's native execution recipe.
    legacy=copy.deepcopy(expected);legacy.pop('recipe_version')
    for name in ('mamba_cache_dtype','mamba_ssm_cache_dtype'):
        legacy['model_execution'].pop(name)
    artifact['calibration_profile']=legacy
    for state in (k,v,artifact):
        state['calibration']=copy.deepcopy(state['calibration'])
        state['calibration'].pop('calibration_profile')
    digest=hashlib.sha256(json.dumps(legacy,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
    old=tmp_path/'artifacts/rotations'/fp/('qqt_sst_'+digest)/'calibrated.pt'
    publish_checkpoint_pair(k,v,old.parent,artifact=artifact)
    old_files={path:path.read_bytes() for path in old.parent.iterdir() if path.suffix in {'.pt','.json'}}
    original_path=rotations.rotation_artifact_path
    monkeypatch.setattr(rotations,'rotation_artifact_path',
        lambda config,fingerprint:original_path(config,fingerprint,root=tmp_path))
    new=rotations.rotation_artifact_path(config,fp)
    assert new!=old
    calls=[]
    def capture(_config,directory):
        calls.append(1);paths=[]
        for rank in rank_states(fp,n=13):
            path=directory/f'rank{rank["rank"]}.pt';torch.save(rank,path);paths.append(str(path))
        return {'status':'native_post_rope_capture_complete','model_fingerprint':fp,
            'layer_ids':[3],'prompt_sha256':'prompt','token_count':13,'rank_paths':paths,
            'calibration_profile':expected}
    first=prepare(config,capture_runner=capture,testing=True)
    second=prepare(config,capture_runner=lambda *_:pytest.fail('new cache resampled'),testing=True)
    assert first['reused'] is False and second['reused'] is True and calls==[1]
    assert Path(first['artifact'])==new
    loaded=load_active_artifact(config,device='cpu',testing=True)
    assert loaded['calibration']['calibration_profile']==expected
    with pytest.raises(ValueError,match='execution/profile'):
        prepare(config,output=old,capture_runner=lambda *_:pytest.fail('explicit old cache replaced'),testing=True)
    assert all(path.read_bytes()==data for path,data in old_files.items())


@pytest.mark.parametrize('missing',[True,False])
def test_prepare_rejects_capture_from_unbound_or_different_state_precision(tmp_path,missing):
    config=model(tmp_path);_,_,fp=model_geometry(tmp_path)
    expected=calibration_cache_identity(config,fp)
    captured=copy.deepcopy(expected)
    captured['model_execution']['mamba_ssm_cache_dtype']='float32'
    def capture(_config,_directory):
        result={'status':'native_post_rope_capture_complete','model_fingerprint':fp,'layer_ids':[3]}
        if not missing:result['calibration_profile']=captured
        return result
    output=tmp_path/'capture/calibrated.pt'
    with pytest.raises(ValueError,match='native sample calibration execution/profile'):
        prepare(config,output=output,capture_runner=capture,testing=True)
    assert not output.exists() and not (output.parent/'.oscar.complete.json').exists()
