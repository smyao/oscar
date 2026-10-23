"""Archive #86/#97/#165/#166: recipe-4 migration authenticates saved samples."""
import copy
from pathlib import Path

import pytest
import torch

from oscar_ascend import rotations
from oscar_ascend.rotation_calibration import publish_checkpoint_pair, verified_pair_files
from tools.prepare_rotations import prepare
from test_center_prepare import environment, publish_old, frozen_files, forbid_capture, capture_rows, NAME


@pytest.mark.parametrize('source', ['recipe2', 'recipe3', 'recipe3_from_recipe2'])
def test_automatic_clip_upgrade_retains_rotations_and_uses_authenticated_samples(environment, source):
    old, center, fingerprint = environment
    if source == 'recipe3':
        old_path, _ = publish_old(center, fingerprint)
    else:
        old_path, _ = publish_old(old, fingerprint)
        if source == 'recipe3_from_recipe2':
            old_path = Path(prepare(center,capture_runner=forbid_capture,testing=True)['artifact'])
    before=frozen_files(old_path)
    _,_,source_artifact=verified_pair_files(old_path.parent)
    config={**center,'rotation_k_clip':'per_head_calibrated'}
    result=prepare(config,capture_runner=forbid_capture,testing=True)
    target=Path(result['artifact'])
    assert target!=old_path and result['capture_reused'] and not result['model_forward_executed']
    _,_,artifact=verified_pair_files(target.parent)
    assert artifact['calibration_profile']['recipe_version']==4
    assert artifact['capture_manifest']==source_artifact['capture_manifest']
    assert artifact['refit_provenance']['source_artifact']==str(old_path.resolve())
    entry=artifact['layers'][NAME]
    assert entry['k_clip_ratio'].dtype==torch.float64 and entry['k_clip_ratio'].shape==(2,)
    if source!='recipe2':
        for key in ('Rk','Rv','k_mean'):
            assert torch.equal(entry[key],source_artifact['layers'][NAME][key])
    assert all(path.read_bytes()==data for path,data in before.items())
    assert prepare(config,capture_runner=forbid_capture,testing=True)['reused']
    loaded=rotations.load_active_artifact(config,device='cpu',testing=True)
    assert torch.equal(loaded['layers'][NAME]['k_clip_ratio'],entry['k_clip_ratio'])


def test_clip_upgrade_recaptures_if_no_authentic_retained_rows(environment):
    _,center,fingerprint=environment
    config={**center,'rotation_k_clip':'per_head_calibrated'};calls=[]
    def capture(_,directory):
        calls.append(directory);return capture_rows(config,fingerprint,directory)
    result=prepare(config,capture_runner=capture,testing=True)
    assert len(calls)==1 and result['model_forward_executed']
    artifact=verified_pair_files(Path(result['artifact']).parent)[2]
    assert artifact['capture_manifest']['calibration_profile']['recipe_version']==4


def test_combined_clip_mutation_cannot_pass_even_with_updated_pair_manifest(environment):
    _,center,fingerprint=environment
    config={**center,'rotation_k_clip':'per_head_calibrated'}
    path,_=publish_old(config,fingerprint)
    k,v,artifact=verified_pair_files(path.parent)
    original=copy.deepcopy(artifact)
    artifact['layers'][NAME]['k_clip_ratio'][0]=.5
    publish_checkpoint_pair(k,v,path.parent,artifact=artifact)
    with pytest.raises(ValueError,match='constants differ'):
        rotations.load_active_artifact(config,device='cpu',testing=True)
    original['layers'][NAME]['clip_selection']={'forged':True}
    publish_checkpoint_pair(k,v,path.parent,artifact=original)
    with pytest.raises(ValueError,match='provenance differs'):
        rotations.load_active_artifact(config,device='cpu',testing=True)


def test_profile_binds_scalar_baseline_and_local_selection_policy(environment):
    _,center,fingerprint=environment
    config={**center,'rotation_k_clip':'per_head_calibrated'}
    first=rotations.calibration_cache_identity(config,fingerprint)
    assert first['recipe_version']==4
    assert first['clip_calibration']['baseline_ratio']==.96
    assert 'raw_and_centered' in first['clip_calibration']['criterion']
    assert rotations.rotation_artifact_path(config,fingerprint)!=rotations.rotation_artifact_path({**config,'k_clip_ratio':.98},fingerprint)
