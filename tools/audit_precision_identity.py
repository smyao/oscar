"""Archive #86/#96/#97/#165/#166: read-only serving config/PT identity audit.

Reads only allowlisted /proc environment values, never prints command lines,
never starts a model/NPU or regenerates rotations. Checkout HEAD is not proof
of code already loaded by a worker. Explicit --config is configuration-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV_KEYS = ("OSCAR_TARGET_CONFIG", "OSCAR_ROTATION_ARTIFACT", "OSCAR_ENABLED")


def read_allowed_env(path):
    result = {}
    for item in Path(path).read_bytes().split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator and key in {name.encode() for name in ENV_KEYS}:
            result[key.decode()] = value.decode("utf-8", errors="replace")
    return result


def _absolute(value, cwd):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(cwd) / path).resolve()


def find_active_configs(proc_root=Path("/proc"), project_root=ROOT):
    """Group live service PIDs by config, explicit PT override and enablement."""
    proc_root, project_root = Path(proc_root), Path(project_root).resolve()
    if not proc_root.is_dir():
        raise ValueError("Linux /proc is unavailable; use --config for a configuration-only audit")
    groups = {}
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            cwd = (process / "cwd").resolve(strict=True)
            if cwd != project_root:
                continue
            argv = (process / "cmdline").read_bytes().split(b"\0")
            service = (b"tools.target_cli" in argv or b"vllm.entrypoints.openai.api_server" in argv
                       or any(arg.startswith(b"VLLM::APIServer") for arg in argv)
                       or (b"serve" in argv and any(arg.rsplit(b"/", 1)[-1] == b"vllm" for arg in argv)))
            if not service:
                continue
            env = read_allowed_env(process / "environ")
            if not env.get("OSCAR_TARGET_CONFIG"):
                continue
            config = str(_absolute(env["OSCAR_TARGET_CONFIG"], cwd))
            override = str(_absolute(env["OSCAR_ROTATION_ARTIFACT"], cwd)) if env.get("OSCAR_ROTATION_ARTIFACT") else None
            identity = config, override, env.get("OSCAR_ENABLED")
            group = groups.setdefault(identity, {"config_path": config, "rotation_override": override,
                "oscar_enabled": env.get("OSCAR_ENABLED"), "pids": []})
            group["pids"].append(int(process.name))
        except (OSError, RuntimeError):
            continue  # Inaccessible/exited PIDs are not evidence of this service.
    return [dict(group, pids=sorted(group["pids"])) for _, group in sorted(groups.items(), key=lambda row: repr(row[0]))]


def artifact_path(config, fingerprint, process_env=None, root=ROOT):
    """Use this process's override, never the diagnostic shell's environment."""
    from oscar_ascend.rotations import calibration_cache_identity, rotation_artifact_path
    explicit = config.get("rotation_artifact") or (process_env or {}).get("OSCAR_ROTATION_ARTIFACT")
    if explicit:
        selected = _absolute(explicit, root)
    elif config.get("rotation_method", "qqt_sst") == "hadamard":
        selected = Path(root) / "artifacts/rotations" / fingerprint / "hadamard.pt"
    else:
        profile = calibration_cache_identity(config, fingerprint)
        digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
        selected = Path(root) / "artifacts/rotations" / fingerprint / ("qqt_sst_" + digest) / "calibrated.pt"
    # Explicit copy prevents rotation_artifact_path from consulting our shell's
    # unrelated OSCAR_ROTATION_ARTIFACT. Neither config nor environ is mutated.
    return rotation_artifact_path({**config, "rotation_artifact": str(selected)}, fingerprint)


def _matrix_summary(artifact):
    shapes, layouts, has_mean = {}, set(), "k_mean" in artifact
    for name, entry in artifact.get("layers", {}).items():
        if not isinstance(entry, dict):
            continue
        has_mean |= "k_mean" in entry
        layer = {}
        for key in ("Rk", "Rv", "rotation", "q_rotation", "o_rotation"):
            value = entry.get(key)
            if hasattr(value, "shape"):
                shape = list(value.shape)
                layer[key] = shape
                layouts.add("shared" if len(shape) == 2 else "per_head" if len(shape) == 3 else "unknown")
        if layer:
            shapes[str(name)] = layer
    return {"layout": next(iter(layouts)) if len(layouts) == 1 else "mixed" if layouts else "unknown",
            "shapes": shapes, "k_mean_present": bool(has_mean)}


def inspect_config(config_path, *, scope="configuration_only", process_env=None,
                   pids=(), compare_calibration=False):
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("selected configuration must be a JSON object")
    from oscar_ascend.rotations import checkpoint_identity, calibration_cache_identity
    _, _, fingerprint = checkpoint_identity(config["model"])
    path = artifact_path(config, fingerprint, process_env)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    report = {"scope": scope, "attestation": "env config binding, no in-process kernel attestation" if pids else "configuration only, no running-process attestation",
        "checkout_commit": head, "config_path": str(config_path), "pids": list(pids),
        "official_math_v4_configured": config.get("experimental_official_math") is True,
        "oscar_enabled_environment": (process_env or {}).get("OSCAR_ENABLED"),
        "model": config.get("model"), "quantization_mode": config.get("quantization_mode"),
        "quantization": config.get("quantization"), "rotation_method": config.get("rotation_method", "qqt_sst"),
        "rotation_artifact": str(path), "model_fingerprint": fingerprint,
        "expected_calibration_profile": calibration_cache_identity(config, fingerprint),
        "artifact_status": "present" if path.is_file() else "missing_on_disk"}
    if not path.is_file():
        if compare_calibration:
            raise FileNotFoundError("selected rotation artifact is missing; comparison cannot run")
        return report
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"  # Diagnostic process only; archive #96.
    import torch
    torch.set_num_threads(min(2, torch.get_num_threads()))
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("rotation artifact must be a mapping")
    report.update(artifact_format_version=artifact.get("format_version"),
        artifact_model_fingerprint=artifact.get("model_fingerprint"),
        calibration_profile=artifact.get("calibration_profile"),
        capture_manifest_present=isinstance(artifact.get("capture_manifest"), dict),
        matrices=_matrix_summary(artifact))
    report["model_identity_matches"] = artifact.get("model_fingerprint") == fingerprint
    report["calibration_profile_matches"] = artifact.get("calibration_profile") == report["expected_calibration_profile"]
    if compare_calibration:
        try:
            if not report['model_identity_matches'] or not report['calibration_profile_matches']:
                raise ValueError('selected PT does not match the model/calibration profile; comparison refused')
            from tools.rotation_recipe_diagnostic import compare
            report["calibration_comparison"] = compare(artifact, config)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            report['calibration_comparison_error'] = str(error)
    return report


def compact_report(report):
    summary = {k:v for k,v in report.items() if k not in (
        'expected_calibration_profile','calibration_profile','matrices','calibration_comparison')}
    if 'matrices' in report:
        summary['matrices'] = {k:v for k,v in report['matrices'].items() if k != 'shapes'}
    comparison = report.get('calibration_comparison')
    if comparison:
        rows = comparison['heads']
        scores = {mode:{name:sum(row['variants'][name][mode]['score_error_proxy'] for row in rows)
                       for name in ('shared_v1','perhead','perhead_center')} for mode in ('prefill','decode')}
        summary['comparison'] = {'heads':len(rows),'sum_head_score_proxy':scores,
            'scope':'calibration K proxy only, not model accuracy; fixed scalar clip',
            'k_clip_ratio':comparison['k_clip_ratio']}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="explicit offline configuration; not proof of the running service")
    parser.add_argument("--compare-calibration", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        selected = None
        if args.config is None:
            candidates = find_active_configs()
            if len(candidates) != 1:
                raise ValueError(f"found {len(candidates)} distinct live service config/artifact bindings; use --config explicitly (configuration_only)")
            selected = candidates[0]
        env = ({"OSCAR_ENABLED": selected["oscar_enabled"],
                "OSCAR_ROTATION_ARTIFACT": selected["rotation_override"]} if selected else None)
        report = inspect_config(args.config or selected["config_path"],
            scope="running_service_environment" if selected else "configuration_only",
            process_env=env, pids=selected["pids"] if selected else (),
            compare_calibration=args.compare_calibration)
        code = 2 if 'calibration_comparison_error' in report else 0
    except Exception as error:
        report, code = {"status": "error", "error": str(error)[:2000]}, 2
    text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    protected = {Path(value).resolve() for value in (report.get("config_path"), report.get("rotation_artifact"), args.config) if value}
    if args.output and args.output.resolve() in protected:
        print(json.dumps({"status":"error","error":"--output must not overwrite the inspected config or rotation artifact"}), flush=True)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(json.dumps({**compact_report(report), **({'full_report':str(args.output)} if args.output else {})},
                     ensure_ascii=False, separators=(',',':')), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
