# Archive G19/G20/G23/#23-25/#86/#96/#97/#111/#165/#166: official one-pass
# calibration; only reuse/import the exact model and native execution recipe.
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def model_geometry(model: Path) -> tuple[list[str], int, str]:
    from oscar_ascend.rotations import checkpoint_identity
    return checkpoint_identity(model)


def _capture(config_path, directory):
    from tools.calibrate_model_rotations import calibration_process_env
    # The child owns the native model and exits before offline FP64 fitting.
    # Inherit stdout/stderr so both one-key launchers preserve live failures.
    subprocess.run([sys.executable,"-m","tools.calibrate_model_rotations",
                    "--config",str(config_path),"--output-dir",str(directory)],
                   env=calibration_process_env(),check=True)
    return json.loads((directory/"capture-manifest.json").read_text())



def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retained_capture(config, profile, fingerprint, layers, dim, *, testing):
    """Authenticate one exact recipe-2 cache; never scan arbitrary old runs.

    A missing retained capture is recoverable by normal native calibration.
    Corrupt complete artifacts or changed retained files are errors, not a
    reason to trust or relabel unknown samples. No NPU context is initialized.
    """
    from oscar_ascend.rotations import calibration_cache_identity, rotation_artifact_path
    from oscar_ascend.rotation_calibration import verified_pair_files, checkpoint_pair_artifact
    import torch
    old_config = dict(config)
    old_config.pop("rotation_variant", None)
    old_profile = calibration_cache_identity(old_config, fingerprint)
    old_path = rotation_artifact_path(old_config, fingerprint)
    if old_profile.get("recipe_version") != 2:
        raise ValueError("retained capture migration requires the fixed recipe-2 source")
    if not old_path.is_file():
        return None, None
    kstate, vstate, artifact = verified_pair_files(old_path.parent, artifact_name=old_path.name)
    expected = checkpoint_pair_artifact(kstate, vstate, layer_names=layers, head_dim=dim,
        model_fingerprint=fingerprint, calibration_profile=old_profile, testing=testing)
    if (artifact.get("format_version") != 1 or artifact.get("model_fingerprint") != fingerprint
            or artifact.get("calibrated") is not True
            or artifact.get("calibration_profile") != old_profile
            or artifact.get("calibration") != expected["calibration"]
            or (artifact.get("test_only") and not testing)):
        raise ValueError("retained recipe-2 artifact model/profile identity differs")
    for name in layers:
        for key in ("Rk", "Rv"):
            matrix = artifact.get("layers", {}).get(name, {}).get(key)
            if not torch.is_tensor(matrix) or not torch.equal(matrix, expected["layers"][name][key]):
                raise ValueError("retained combined rotation constants differ from authenticated pair")
    sampling = lambda item: {k: v for k, v in item.items() if k not in {"recipe_version", "rotation_variant"}}
    if sampling(old_profile) != sampling(profile):
        raise ValueError("recipe migration would change the native sampling profile")
    capture = artifact.get("capture_manifest")
    if not isinstance(capture, dict) or not capture.get("rank_paths") or not capture.get("rank_sha256"):
        return None, None
    if capture.get("calibration_profile") != old_profile:
        raise ValueError("retained native capture profile differs from its source artifact")
    paths = capture["rank_paths"]
    if not isinstance(paths, list) or not isinstance(capture["rank_sha256"], dict):
        raise ValueError("retained capture path/hash manifest is malformed")
    if any(not Path(path).is_file() for path in paths):
        return None, None
    for path in paths:
        if capture["rank_sha256"].get(path) != _sha256(path):
            raise ValueError("retained rank payload changed after native capture: " + str(path))
    provenance = {"source_artifact": str(old_path.resolve()), "source_artifact_sha256": _sha256(old_path),
        "source_calibration_profile": old_profile, "fit_calibration_profile": profile,
        "model_forward_executed": False, "operation": "refit_authenticated_retained_native_samples"}
    return capture, provenance


def prepare(config, *, config_path=None, output=None, import_pair=None, capture_runner=None,
            testing=False):
    from oscar_ascend.rotations import (rotation_artifact_path,calibration_cache_identity,
        build_hadamard_artifact,load_artifact,save_artifact,validate_artifact,load_active_artifact)
    from oscar_ascend.rotation_calibration import (OFFICIAL_COMMIT,K_FILENAME,V_FILENAME,
        fit_checkpoint_pair,checkpoint_pair_artifact,publish_checkpoint_pair,verified_pair_files)
    import torch
    layers,dim,fingerprint=model_geometry(Path(config["model"]))
    method=config.get("rotation_method","qqt_sst")
    if method not in {"qqt_sst","hadamard"}:raise ValueError("rotation_method must be qqt_sst or explicit data-free hadamard")
    destination=Path(output).expanduser().resolve() if output else rotation_artifact_path(config,fingerprint)
    device="cpu" if testing else "npu:0"
    def validate(artifact):
        return validate_artifact(artifact,layer_names=layers,head_dim=dim,
            model_fingerprint=fingerprint,pr_fingerprint=OFFICIAL_COMMIT,
            require_calibrated=True,device=device,testing=testing)
    if method=="hadamard":
        options=dict(layer_names=layers,head_dim=dim,model_fingerprint=fingerprint,device=device,testing=testing)
        artifact=load_artifact(destination,**options) if destination.exists() else build_hadamard_artifact(**options)
        if not destination.exists():save_artifact(artifact,destination)
        return {"artifact":str(destination),"source":"hadamard_data_free","calibrated":False,
            "official_calibration_aligned":False,"model_fingerprint":fingerprint,"quality_acceptance":"not_run"}
    profile=calibration_cache_identity(config,fingerprint)
    print(json.dumps({'rotation_prepare':'checking','artifact':str(destination),
        'prompt_source':profile['prompt_source'],'recipe_version':profile['recipe_version'],
        'rotation_variant':profile.get('rotation_variant','shared_v1')}),flush=True)
    if import_pair is None:
        k=config.get("rotation_k_path");v=config.get("rotation_v_path")
        if bool(k)!=bool(v):raise ValueError("both rotation_k_path and rotation_v_path are required")
        if k:import_pair=(Path(k),Path(v))
    automatic_cache=(output is None and not config.get('rotation_artifact') and
                     not os.environ.get('OSCAR_ROTATION_ARTIFACT'))
    interrupted_pair=automatic_cache and (destination.parent/'.oscar.pending.json').exists()
    if destination.exists() and not interrupted_pair:
        # Never replace a malformed explicit artifact or reuse old Hadamard.
        if not (destination.parent/'.oscar.complete.json').exists():
            raise ValueError('official calibrated K/V pair manifest required; a data-free or standalone cache is not accepted')
        load_active_artifact(config,path=destination,device=device,testing=testing,
            layer_names=layers,head_dim=dim,model_fingerprint=fingerprint)
        return {"artifact":str(destination),"source":"official_qqt_sst_cache","calibrated":True,
            "model_fingerprint":fingerprint,"quality_acceptance":"not_run","reused":True,
            "rotation_variant":profile.get("rotation_variant","shared_v1"),
            "recipe_version":profile["recipe_version"],"model_forward_executed":False}
    destination.parent.mkdir(parents=True,exist_ok=True)
    if import_pair is not None:
        kpath,vpath=map(lambda p:Path(p).expanduser().resolve(),import_pair)
        if kpath==vpath or os.path.samefile(kpath,vpath):raise ValueError('K/V import must be distinct files')
        if kpath in {destination.parent/K_FILENAME,destination.parent/V_FILENAME} or vpath in {
                destination.parent/K_FILENAME,destination.parent/V_FILENAME}:
            raise ValueError('import destination must not overwrite the supplied K/V files')
        kstate=torch.load(kpath,map_location='cpu',weights_only=True)
        vstate=torch.load(vpath,map_location='cpu',weights_only=True)
        artifact=checkpoint_pair_artifact(kstate,vstate,layer_names=layers,head_dim=dim,
            model_fingerprint=fingerprint,calibration_profile=profile,testing=testing)
        artifact['import_provenance']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (kpath,vpath)}
        source='official_qqt_sst_import'
    else:
        capture, provenance = (None, None)
        if automatic_cache and config.get("rotation_variant") == "perhead_center":
            capture, provenance = _retained_capture(config, profile, fingerprint, layers, dim, testing=testing)
        if capture is None:
            if config_path is None and capture_runner is None:raise ValueError('native calibration requires a config path')
            print(json.dumps({'rotation_prepare':'generating','artifact':str(destination),
                'reason':'interrupted_publication' if interrupted_pair else 'matching_cache_missing'}),flush=True)
            directory=Path(tempfile.mkdtemp(prefix='capture-',dir=destination.parent))
            capture=(capture_runner or _capture)(config_path,directory)
        else:
            print(json.dumps({'rotation_prepare':'refitting_retained_capture','artifact':str(destination),
                'source_artifact':provenance['source_artifact'],'model_forward_executed':False}),flush=True)
        if capture.get('status')!='native_post_rope_capture_complete':raise ValueError('native sampling did not complete')
        if capture.get('model_fingerprint')!=fingerprint or capture.get('layer_ids')!=[int(x.split('.')[2]) for x in layers]:
            raise ValueError('native sample model/layer identity differs')
        capture_profile = provenance['source_calibration_profile'] if provenance else profile
        if capture.get('calibration_profile')!=capture_profile:
            raise ValueError('native sample calibration execution/profile differs')
        if capture.get('rank_sha256') is not None and any(
                capture['rank_sha256'].get(p)!=_sha256(p) for p in capture['rank_paths']):
            raise ValueError('captured rank payload changed after native completion')
        ranks=[torch.load(Path(p),map_location='cpu',weights_only=True) for p in capture['rank_paths']]
        model=json.loads((Path(config['model'])/'config.json').read_text());model=model.get('text_config',model)
        kstate,vstate=fit_checkpoint_pair(ranks,layer_names=layers,head_dim=dim,
            model_path=config['model'],model_fingerprint=fingerprint,
            global_q_heads=model['num_attention_heads'],global_kv_heads=model['num_key_value_heads'],
            tp_size=config['tensor_parallel_size'],prompt_sha256=capture['prompt_sha256'],
            max_tokens=profile['max_tokens'],calibration_profile=profile,
            model_revision=config.get('revision'),testing=testing)
        if kstate['calibration']['tokens']!=capture['token_count']:raise ValueError('captured prompt row budget differs')
        artifact=checkpoint_pair_artifact(kstate,vstate,layer_names=layers,head_dim=dim,
            model_fingerprint=fingerprint,calibration_profile=profile,testing=testing)
        artifact['capture_manifest']=capture
        if provenance:
            artifact['capture_fit_profile']=profile
            artifact['refit_provenance']=provenance
        source=('official_qqt_sst_refit_retained_samples' if provenance else
                'official_qqt_sst_native_post_rope_calibration')
    artifact['calibration_profile']=profile
    validate(artifact)
    publish_checkpoint_pair(kstate,vstate,destination.parent,artifact=artifact,artifact_name=destination.name)
    # Read under the same pair lock and validate the published generation.
    _,_,published=verified_pair_files(destination.parent,artifact_name=destination.name);validate(published)
    return {"artifact":str(destination),"k_rotation":str(destination.parent/K_FILENAME),
        "v_rotation":str(destination.parent/V_FILENAME),"source":source,"calibrated":True,
        "model_fingerprint":fingerprint,"full_layers":layers,"accumulator_dtype":"float64",
        "statistics_device":"cpu_offline","model_forward_device":"npu" if not testing else "cpu_test",
        "quality_acceptance":"not_run","draft_layer":"explicit_identity_not_sampled","reused":False,
        "rotation_variant":config.get("rotation_variant","shared_v1"),
        "capture_reused":source=="official_qqt_sst_refit_retained_samples",
        "model_forward_executed":source=="official_qqt_sst_native_post_rope_calibration"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--import-k", type=Path)
    parser.add_argument("--import-v", type=Path)
    args = parser.parse_args(argv)
    from .model_config import apply_model_options
    # The standalone prepare command can receive raw "auto" configuration;
    # bind the same resolved weights mode that the capture child will execute.
    config = apply_model_options(json.loads(args.config.read_text()), require_model=True)
    from .target_cli import target_env
    os.environ.update(target_env(config))
    if bool(args.import_k)!=bool(args.import_v):parser.error('both --import-k and --import-v are required')
    # Do not initialize a parent NPU context before the native model child.
    # Matrix validation lazily imports torch_npu after that child has exited.
    pair=(args.import_k,args.import_v) if args.import_k else None
    print(json.dumps(prepare(config,config_path=args.config,output=args.output,import_pair=pair)),flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
