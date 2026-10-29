"""Archive #123/#166/#172/#174: one persistent offline worker per selected NPU."""
from __future__ import annotations

import os


def fit_worker(spec, index, cpu_threads, pipe):
    # Spawn imports no accelerator state from the parent. Limit native pools
    # before importing Torch; these settings never reach the service process.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cpu_threads)
    import torch
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    from oscar_ascend.optr_artifacts import file_sha256
    from oscar_ascend.optr_checkpoint import FitCheckpoints
    from oscar_ascend.optr_policy import optr_policy
    from oscar_ascend.optr_sequential import fitting_device
    from oscar_ascend.optr_traces import load_trace_manifest, load_layer_samples
    from tools.prepare_optr_seed import load_seed
    from tools.prepare_optr_rotations import refine_layer, _layer_artifact, _validate_layer_checkpoint

    name = None
    def emit(**fields):
        pipe.send({"kind": "progress", "layer": name, "fields": fields})

    try:
        emit(phase="fit_worker_initializing", cpu_threads=cpu_threads)
        config, testing = spec["config"], spec["testing"]
        if file_sha256(spec["base_config"]["rotation_artifact"]) != spec["certificate"]["base_artifact_sha256"]:
            raise ValueError("OptR worker seed file changed")
        baseline = load_seed(spec["base_config"], testing=testing)
        if baseline.get("test_only") and not testing:
            raise ValueError("test-only seed cannot enter a production OptR worker")
        manifest = load_trace_manifest(spec["trace_manifest"], config, spec["names"],
            spec["template"]["head_dim"], spec["template"]["model_fingerprint"], testing=testing)
        if manifest["_manifest_sha256"] != spec["certificate"]["trace_manifest_sha256"]:
            raise ValueError("OptR worker trace manifest changed")
        checkpoints = FitCheckpoints(spec["checkpoint_directory"], spec["checkpoint_identity"])
        policy = optr_policy(config)
        device = fitting_device(config, testing=testing, logical_index=index)
        emit(phase="fit_worker_ready", fitting_device=device, cpu_threads=cpu_threads)
        pipe.send({"kind": "ready"})
        while True:
            name = pipe.recv()
            if name is None:
                return
            if name not in spec["names"]:
                raise ValueError("OptR worker received an unknown layer")
            seed = baseline["layers"][name]
            owner = f"layer-{seed['layer_id']}"
            emit(layer=name, phase="layer_start")
            saved = checkpoints.load(owner)
            if saved is None:
                samples = load_layer_samples(manifest, seed["layer_id"])
                emit(layer=name, phase="layer_traces_verified")
                selected, record = refine_layer(name, seed, samples, policy,
                    checkpoints=checkpoints, progress=emit, fit_device=device)
                saved = {"artifact": _layer_artifact(spec["template"], spec["certificate"], name, selected, record)}
                _validate_layer_checkpoint(saved, spec["template"], spec["certificate"], name, seed)
                checkpoints.save(owner, saved)
                del samples
                emit(layer=name, phase="layer_checked_saved")
            else:
                _validate_layer_checkpoint(saved, spec["template"], spec["certificate"], name, seed)
                emit(layer=name, phase="layer_checked_resumed")
            pipe.send({"kind": "done", "layer": name})
    finally:
        pipe.close()
