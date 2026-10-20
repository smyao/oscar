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
        'prompt_source':profile['prompt_source']}),flush=True)
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
            "model_fingerprint":fingerprint,"quality_acceptance":"not_run","reused":True}
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
        if config_path is None and capture_runner is None:raise ValueError('native calibration requires a config path')
        print(json.dumps({'rotation_prepare':'generating','artifact':str(destination),
            'reason':'interrupted_publication' if interrupted_pair else 'matching_cache_missing'}),flush=True)
        directory=Path(tempfile.mkdtemp(prefix='capture-',dir=destination.parent))
        capture=(capture_runner or _capture)(config_path,directory)
        if capture.get('status')!='native_post_rope_capture_complete':raise ValueError('native sampling did not complete')
        if capture.get('model_fingerprint')!=fingerprint or capture.get('layer_ids')!=[int(x.split('.')[2]) for x in layers]:
            raise ValueError('native sample model/layer identity differs')
        if capture.get('calibration_profile')!=profile:
            raise ValueError('native sample calibration execution/profile differs')
        if capture.get('rank_sha256') is not None and any(
                capture['rank_sha256'].get(p)!=hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in capture['rank_paths']):
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
        source='official_qqt_sst_native_post_rope_calibration'
    artifact['calibration_profile']=profile
    validate(artifact)
    publish_checkpoint_pair(kstate,vstate,destination.parent,artifact=artifact,artifact_name=destination.name)
    # Read under the same pair lock and validate the published generation.
    _,_,published=verified_pair_files(destination.parent,artifact_name=destination.name);validate(published)
    return {"artifact":str(destination),"k_rotation":str(destination.parent/K_FILENAME),
        "v_rotation":str(destination.parent/V_FILENAME),"source":source,"calibrated":True,
        "model_fingerprint":fingerprint,"full_layers":layers,"accumulator_dtype":"float64",
        "statistics_device":"cpu_offline","model_forward_device":"npu" if not testing else "cpu_test",
        "quality_acceptance":"not_run","draft_layer":"explicit_identity_not_sampled","reused":False}


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
