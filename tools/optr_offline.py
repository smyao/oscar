"""Archive #97/#166/#170/#175-177: portable, authenticated CPU OptR fitting.

Transport retains original file bytes, model identity, partitions and strict
publication gates. It never samples a model or imports torch_npu. Import on
the source host rechecks its actual model/base/trace before atomic publication.
Synthetic fixtures remain test-only; the CLI has no testing switch.
Use the same project revision for export, fit and import; publication and
source-host import require the current fitting implementation digest.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import posixpath
import shutil

from oscar_ascend.optr_artifacts import file_sha256


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("offline OptR metadata must be an object")
    return value


def _requirements(config):
    from oscar_ascend.optr_policy import optr_policy, validate_refinement_config
    validate_refinement_config(config)
    policy = optr_policy(config)
    if (config.get("optr_baseline") != "recipe3" or config.get("optr_recipe3_repair") is not None or
            policy.get("execution") != "cpu_native_residual_constrained_selection"):
        raise ValueError("offline OptR requires the current CPU native-residual configuration")
    return policy


class PortableInputs:
    """Resolve original logical paths without rewriting sample/PT bytes.

    No model fingerprint is recomputed from copied files. The original source
    host checks it again on import; hashes alone are integrity, not signatures.
    Baseline authentication opens/creates base/.oscar.lock, so its directory
    must be writable even though authenticated input data is never rewritten.
    """
    @staticmethod
    def logical_path(value, root=None):
        """Normalize a source-host UNIX identity without consulting this host.

        A Linux capture may name /tmp while the fitting Mac maps /tmp to
        /private/tmp. Source identities must not follow that local symlink.
        Physical bundle paths are resolved and containment-checked separately.
        """
        try:
            value = os.fspath(value)
        except TypeError as error:
            raise ValueError("offline OptR logical path must be a UNIX path") from error
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("offline OptR logical path must be a nonempty UNIX path")
        if not posixpath.isabs(value):
            if root is None:
                raise ValueError("offline OptR logical path must be absolute")
            value = posixpath.join(str(PortableInputs.logical_path(root)), value)
        normalized = posixpath.normpath(value)
        if not normalized.startswith("/") or normalized.startswith("//"):
            raise ValueError("offline OptR logical path requires one absolute UNIX root")
        return Path(normalized)

    def __init__(self, directory, *, testing=False):
        if type(testing) is not bool:
            raise ValueError("offline testing flag must be boolean")
        self.root = Path(directory).resolve()
        self.header = _json(self.root / "bundle.json")
        header = self.header
        if (header.get("schema_version") != 1 or header.get("status") != "complete" or
                type(header.get("test_only")) is not bool or header["test_only"] and not testing):
            raise ValueError("incomplete or test-only offline OptR bundle")
        self.testing = testing
        self.config = header.get("config")
        identity = header.get("identity")
        if (not isinstance(self.config, dict) or not isinstance(header.get("base_config"), dict) or
                not isinstance(identity, list) or len(identity) != 3 or
                not isinstance(identity[0], list) or not identity[0] or
                any(not isinstance(name, str) or not name for name in identity[0]) or
                len(set(identity[0])) != len(identity[0]) or type(identity[1]) is not int or identity[1] < 2 or
                not isinstance(identity[2], str) or not identity[2]):
            raise ValueError("offline OptR configuration/model identity is invalid")
        _requirements(self.config)
        self.names, self.dim, self.fingerprint = identity
        files = header.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("offline OptR file inventory is missing")
        physical = set()
        for key, row in files.items():
            if not isinstance(key, str) or str(self.logical_path(key)) != key:
                raise ValueError("offline OptR logical paths must be canonical and absolute")
            if not isinstance(row, dict):
                raise ValueError("offline OptR file mapping must be an object")
            relative = row.get("path")
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("offline OptR path escapes its bundle")
            path = (self.root / relative).resolve()
            digest = row.get("sha256")
            if (path in physical or not path.is_relative_to(self.root) or not path.is_file() or
                    type(row.get("size")) is not int or row["size"] < 0 or
                    not isinstance(digest, str) or len(digest) != 64 or
                    any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("offline OptR duplicate, missing or invalid file mapping")
            physical.add(path)
        from oscar_ascend.rotations import calibration_cache_identity
        if calibration_cache_identity(self.config, self.fingerprint, files=self) != header["profile"]:
            raise ValueError("offline OptR configuration/profile differs")

    def resolve(self, logical):
        key = str(self.logical_path(logical))
        row = self.header["files"].get(key)
        if row is None:
            raise ValueError("offline OptR file is absent from its inventory: " + key)
        path = (self.root / row["path"]).resolve()
        if (not path.is_relative_to(self.root) or not path.is_file() or
                path.stat().st_size != row["size"] or file_sha256(path) != row["sha256"]):
            raise ValueError("offline OptR file escaped, changed or is missing: " + key)
        return path

    def load_base(self):
        import torch
        from oscar_ascend.rotation_calibration import verified_pair_files, checkpoint_pair_artifact, OFFICIAL_COMMIT
        from oscar_ascend.rotations import validate_artifact
        from tools.prepare_optr_recipe3_base import capture_source_metadata
        logical = Path(self.header["base_config"]["rotation_artifact"])
        path = self.resolve(logical)
        # Also authenticate the original pair completion metadata, never a
        # newly serialized/relabelled approximation of the original baseline.
        for name in (".oscar.complete.json", "k_rotation_qqt_r_h_pbr.pt", "v_rotation_sst_r_h_pbr.pt"):
            if self.resolve(logical.parent / name).parent != path.parent:
                raise ValueError("offline baseline pair must share one directory")
        k, v, artifact = verified_pair_files(path.parent, artifact_name=path.name)
        expected = checkpoint_pair_artifact(k, v, layer_names=self.names, head_dim=self.dim,
            model_fingerprint=self.fingerprint, calibration_profile=self.header["profile"]["base_calibration_profile"],
            testing=self.testing)
        for key, value in expected.items():
            if key == "layers":
                for name, constants in value.items():
                    actual = artifact.get("layers", {}).get(name, {})
                    for field, tensor in constants.items():
                        observed = actual.get(field)
                        differs = (not torch.is_tensor(observed) or not torch.equal(observed, tensor)
                                   if torch.is_tensor(tensor) else observed != tensor)
                        if differs:
                            raise ValueError("offline combined baseline differs from its pair")
            elif artifact.get(key) != value:
                raise ValueError("offline combined baseline metadata differs from its pair: " + key)
        if not self.testing and (artifact.get("test_only") or artifact["calibration"].get("test_only") or
                                artifact["calibration"].get("model_forward_device") != "npu"):
            raise ValueError("offline production fitting requires a real NPU-captured baseline")
        validate_artifact(artifact, layer_names=self.names, head_dim=self.dim,
            model_fingerprint=self.fingerprint, pr_fingerprint=OFFICIAL_COMMIT,
            require_calibrated=True, device="cpu", testing=True)
        metadata = {**capture_source_metadata(artifact), "baseline_artifact_sha256": file_sha256(path)}
        if self.config.get("optr_seed_capture") != metadata:
            raise ValueError("offline trace seed binding differs from the original baseline")
        return artifact

    def load_trace(self):
        from oscar_ascend.optr_traces import load_trace_manifest
        return load_trace_manifest(self.header["trace_manifest"], self.config, self.names,
            self.dim, self.fingerprint, testing=self.testing, files=self)


def export_bundle(config, output, *, trace_manifest=None, testing=False):
    """Read/validate existing captures only; missing input fails without sampling."""
    from oscar_ascend.rotations import checkpoint_identity, calibration_cache_identity, rotation_artifact_path
    from oscar_ascend.optr_traces import load_trace_manifest, load_layer_samples
    from oscar_ascend.rotation_calibration import K_FILENAME, V_FILENAME
    from tools.prepare_optr_recipe3_base import base_config, load_base, capture_source_metadata
    from tools.prepare_optr_rotations import _existing_trace_manifest
    from tools.phase import atomic_json
    _requirements(config)
    names, dim, fingerprint = checkpoint_identity(config["model"])
    base = base_config(config, fingerprint)
    baseline = load_base(base, testing=testing)
    selected = copy.deepcopy(config)
    # Logical identities must remain absolute when the bundle is opened from
    # another host/directory. These fields already resolve this way in the
    # original fingerprint/profile, so this does not relabel their identity.
    selected["model"] = str(Path(selected["model"]).expanduser().resolve())
    base["model"] = selected["model"]
    if selected.get("rotation_calibration_prompts"):
        selected["rotation_calibration_prompts"] = str(Path(
            selected["rotation_calibration_prompts"]).expanduser().resolve())
        base["rotation_calibration_prompts"] = selected["rotation_calibration_prompts"]
    selected["optr_seed_capture"] = {**capture_source_metadata(baseline),
        "baseline_artifact_sha256": file_sha256(base["rotation_artifact"])}
    profile = calibration_cache_identity(selected, fingerprint)
    manifest_path = Path(trace_manifest).resolve() if trace_manifest else _existing_trace_manifest(
        rotation_artifact_path(selected, fingerprint), profile, selected)
    manifest = load_trace_manifest(manifest_path, selected, names, dim, fingerprint, testing=testing)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    inventory = {}

    def add(path, relative=None):
        source = Path(path).resolve()
        key = str(source)
        if key in inventory:
            return
        target = root / (relative or f"files/{len(inventory):06d}-{source.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        wanted = file_sha256(source)
        shutil.copyfile(source, target)
        if file_sha256(target) != wanted or file_sha256(source) != wanted:
            raise ValueError("offline export source changed during copy")
        inventory[key] = {"path": str(target.relative_to(root)), "sha256": wanted, "size": target.stat().st_size}

    add(manifest_path)
    add(Path(config["model"]) / "config.json")
    add(manifest["prompt_source"])
    if selected.get("rotation_calibration_prompts"):
        add(selected["rotation_calibration_prompts"])
    logical_base = Path(base["rotation_artifact"])
    for name in (logical_base.name, K_FILENAME, V_FILENAME, ".oscar.complete.json"):
        add(logical_base.parent / name, "base/" + name)
    for path, rank in zip(manifest["rank_paths"], manifest["_rank_manifests"], strict=True):
        add(Path(manifest["_root"]) / path)
        for layer in rank["layers"].values():
            add(Path(manifest["_root"]) / layer["wo_path"])
            for sequence in layer["sequences"]:
                add(Path(manifest["_root"]) / sequence["path"])
    header = {"schema_version": 1, "status": "complete", "test_only": testing,
        "config": selected, "base_config": base, "identity": [names, dim, fingerprint],
        "profile": profile, "trace_manifest": str(manifest_path.resolve()), "files": inventory}
    # Publish completion only after production payload provenance, tensor
    # shapes, source partitions and TP alignment all pass; no replay/fit here.
    for name in names:
        samples = load_layer_samples(manifest, baseline["layers"][name]["layer_id"])
        del samples
    atomic_json(root / "bundle.json", header)
    try:
        copied = PortableInputs(root, testing=testing)
        copied.load_base()
        copied.load_trace()
    except BaseException:
        (root / "bundle.json").unlink(missing_ok=True)
        raise
    return {"bundle": str(root), "files": len(inventory), "bytes": sum(r["size"] for r in inventory.values()),
            "model_forward": "not_run", "npu": "not_used",
            "checkout_requirement": "same_project_revision_for_export_fit_import"}


def _validate_result(artifact, files, baseline):
    import torch
    from oscar_ascend.optr_artifacts import validate_refinement_metadata, matrix_digest
    from oscar_ascend.optr_recipe3_certificate import require_publishable
    from oscar_ascend.rotations import validate_artifact
    from oscar_ascend.rotation_calibration import OFFICIAL_COMMIT
    if type(artifact.get("test_only")) is not bool or artifact["test_only"] and not files.testing:
        raise ValueError("test-only offline result cannot enter production")
    certificate = validate_refinement_metadata(artifact, expected_profile=files.header["profile"])
    require_publishable(artifact)
    if (certificate["base_artifact_sha256"] != files.config["optr_seed_capture"]["baseline_artifact_sha256"] or
            certificate["trace_manifest_sha256"] != file_sha256(files.resolve(files.header["trace_manifest"])) or
            set(artifact["layers"]) != set(baseline["layers"])):
        raise ValueError("offline result source binding differs")
    for name, seed in baseline["layers"].items():
        if (certificate["layers"][name]["base_constants_sha256"] != matrix_digest({name: seed}) or
                not torch.equal(artifact["layers"][name]["k_mean"], seed["k_mean"])):
            raise ValueError("offline result changed its baseline identity or fixed means")
    validate_artifact(artifact, layer_names=files.names, head_dim=files.dim,
        model_fingerprint=files.fingerprint, pr_fingerprint=OFFICIAL_COMMIT,
        require_calibrated=True, device="cpu", testing=True)


def fit_bundle(bundle, output, *, workers=1, threads=1, testing=False):
    """Run the real optimizer, all validation and independent test on CPU."""
    from oscar_ascend.optr_artifacts import (matrix_digest, publish_artifact,
        read_published_artifact, artifact_test_status)
    from oscar_ascend.optr_checkpoint import FitCheckpoints, fitting_implementation_digest
    from oscar_ascend.optr_policy import OBJECTIVE
    from oscar_ascend.optr_recipe3_certificate import strict_certificate
    from oscar_ascend.rotation_calibration import OFFICIAL_COMMIT
    from tools.prepare_optr_rotations import _parallel_layers, _progress
    from tools.optr_progress import ProgressWatchdog
    from tools.rotation_preparation_limits import preparation_progress_timeout
    files = PortableInputs(bundle, testing=testing)
    if any(type(n) is not int or not 1 <= n <= 64 for n in (workers, threads)):
        raise ValueError("offline CPU workers/threads must be integers in [1,64]")
    destination = Path(output).resolve() / "refined.pt"
    if destination.is_relative_to(files.root):
        raise ValueError("offline fitting output must be outside its input bundle")
    baseline, manifest = files.load_base(), files.load_trace()
    if destination.is_file() and not (destination.parent / ".optr.pending.json").exists():
        from oscar_ascend.optr_recipe3_certificate import RefitRequired
        artifact = read_published_artifact(destination, files.fingerprint)
        try:
            _validate_result(artifact, files, baseline)
        except RefitRequired as error:
            _progress(phase="refit_required", reason=str(error), artifact=str(destination))
        else:
            return {"artifact": str(destination), "reused": True,
                    "checkout_requirement": "same_project_revision_for_export_fit_import",
                    **artifact_test_status(artifact)}
    config, profile = files.config, files.header["profile"]
    implementation = fitting_implementation_digest()
    base_hash = config["optr_seed_capture"]["baseline_artifact_sha256"]
    artifact = {"format_version": 2, "source_grouping": "layer_kv_head", "rotation_variant": "perhead_center",
        "global_kv_heads": baseline["global_kv_heads"], "head_dim": files.dim,
        "model_fingerprint": files.fingerprint, "pr_fingerprint": OFFICIAL_COMMIT,
        "objective": OBJECTIVE, "calibrated": True, "accumulation_dtype": "float32",
        "test_only": testing, "calibration_profile": profile, "layers": {}}
    certificate = {"schema_version": 1, "policy": _requirements(config),
        "metric": "all_tp_heads_bf16_gate_fp32_wo_output",
        "arithmetic_evidence": "cpu_deployment_reference_not_npu_bitwise",
        "model_quality": "not_measured", "performance": "not_measured",
        "base_artifact_sha256": base_hash, "trace_manifest_sha256": manifest["_manifest_sha256"],
        "full_layer_names": files.names, "fitting_implementation_sha256": implementation, "layers": {}}
    checkpoints = FitCheckpoints(destination.parent / "fit-checkpoints", {
        "profile": profile, "base_artifact_sha256": base_hash, "trace_manifest_sha256": manifest["_manifest_sha256"],
        "full_layers": files.names, "global_kv_heads": baseline["global_kv_heads"], "head_dim": files.dim,
        "fitting_implementation_sha256": implementation, "test_only": testing})
    with ProgressWatchdog(preparation_progress_timeout(config), output=_progress) as watchdog:
        _parallel_layers(config, files.names, baseline, artifact, certificate, files.header["base_config"],
            files.header["trace_manifest"], checkpoints, watchdog, testing=testing,
            offline_bundle=files.root, cpu_workers=workers, cpu_threads=threads)
    certificate["accepted_layers"] = sorted(n for n, row in certificate["layers"].items() if row["accepted"])
    certificate["matrices_sha256"] = matrix_digest(artifact["layers"])
    artifact["refinement_certificate"] = strict_certificate(certificate)
    _validate_result(artifact, files, baseline)
    publish_artifact(artifact, destination)
    _validate_result(read_published_artifact(destination, files.fingerprint), files, baseline)
    return {"artifact": str(destination), "reused": False, "optimization_device": "cpu",
            "model_forward": "not_run", "quality_acceptance": "not_run", "performance": "not_measured",
            "checkout_requirement": "same_project_revision_for_export_fit_import",
            "fitting_implementation_sha256": implementation,
            **artifact_test_status(artifact)}


def import_result(config, result, *, trace_manifest=None):
    """Rebind to the actual source-host model and original captures; no NPU."""
    from oscar_ascend.rotations import checkpoint_identity, rotation_artifact_path, calibration_cache_identity
    from oscar_ascend.optr_artifacts import load_refined_artifact, publish_artifact, artifact_test_status
    from oscar_ascend.optr_recipe3_certificate import validate_baseline_binding
    from tools.prepare_optr_rotations import _existing_trace_manifest
    from tools.prepare_optr_recipe3_base import base_config, load_base, capture_source_metadata
    from oscar_ascend.optr_traces import load_trace_manifest
    _requirements(config)
    names, dim, fingerprint = checkpoint_identity(config["model"])
    source = Path(result).resolve() / "refined.pt"
    artifact = load_refined_artifact(config, source, device="cpu", testing=True,
        layer_names=names, head_dim=dim, model_fingerprint=fingerprint)
    if type(artifact.get("test_only")) is not bool or artifact["test_only"]:
        raise ValueError("test-only offline result cannot be imported")
    validate_baseline_binding(artifact, config, testing=False)
    base = base_config(config, fingerprint)
    baseline = load_base(base, testing=False)
    selected = {**config, "optr_seed_capture": {**capture_source_metadata(baseline),
        "baseline_artifact_sha256": file_sha256(base["rotation_artifact"])}}
    profile = calibration_cache_identity(config, fingerprint)
    destination = rotation_artifact_path(config, fingerprint)
    trace = (Path(trace_manifest).expanduser().resolve() if trace_manifest is not None else
             _existing_trace_manifest(destination, profile, selected))
    manifest = load_trace_manifest(trace, selected, names, dim, fingerprint)
    if manifest["_manifest_sha256"] != artifact["refinement_certificate"]["trace_manifest_sha256"]:
        raise ValueError("offline result differs from the original host trace")
    if source != destination:
        publish_artifact(artifact, destination)
    load_refined_artifact(config, destination, device="cpu", testing=True,
        layer_names=names, head_dim=dim, model_fingerprint=fingerprint)
    return {"artifact": str(destination), "source": "authenticated_offline_cpu_fit",
            "model_forward": "not_run", "npu": "not_used",
            "checkout_requirement": "same_project_revision_for_export_fit_import",
            **artifact_test_status(artifact)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export existing authenticated native samples; never capture")
    export.add_argument("--config", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--trace-manifest")
    export.add_argument("--variant", default="candidate")
    fit = sub.add_parser("fit", help="fit on CPU without original paths, model weights or NPU")
    fit.add_argument("--bundle", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--workers", type=int, default=1)
    fit.add_argument("--threads", type=int, default=1)
    install = sub.add_parser("import", help="validate against source host and atomically publish")
    install.add_argument("--config", required=True)
    install.add_argument("--result", required=True)
    install.add_argument("--trace-manifest")
    install.add_argument("--variant", default="candidate")
    args = parser.parse_args()
    if args.command == "export":
        from tools.serving_variants import variant_config
        result = export_bundle(variant_config(_json(args.config), args.variant), args.output,
                               trace_manifest=args.trace_manifest)
    elif args.command == "fit":
        result = fit_bundle(args.bundle, args.output, workers=args.workers, threads=args.threads)
    else:
        from tools.serving_variants import variant_config
        result = import_result(variant_config(_json(args.config), args.variant), args.result,
                               trace_manifest=args.trace_manifest)
    print("[oscar] OPTR_OFFLINE " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
