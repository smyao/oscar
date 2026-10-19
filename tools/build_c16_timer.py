# Archive G7/G8/#79-85/#98-109/#133/#164: signed independent diagnostic build;
# generated files never touch csrc or the already-built production library.
"""Build/reuse the separate C16 cycle library without importing torch or an NPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import uuid

from oscar_ascend.c16_timer_schema import CLOCK_HZ, CLOCK_SOURCE, SCHEMA_ID
from .build_ops import cann_root, normalize_soc
from .environment import file_fingerprint
from .phase import atomic_json, run_phase

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE = Path("csrc/experiments/c16_timer_20261019")
CLOCK_FILE = Path(CLOCK_SOURCE.split(":", 1)[0])


class C16BuildError(RuntimeError):
    def __init__(self, message, returncode=1):
        super().__init__(message)
        self.returncode = returncode


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def required_sources():
    return [SOURCE_RELATIVE / name for name in ("CMakeLists.txt", "generate.py", "timer.h")] + [
        Path("csrc/kernels/attention_cv_aligned_cluster16.cpp"), Path("csrc/aligned_attention_bindings.cpp"),
        Path("oscar_ascend/c16_timer_schema.py"), CLOCK_FILE, Path("tools/build_c16_timer.py")]


def source_hashes():
    # Matches the production package's source boundary: changing *any* csrc
    # file invalidates its old signature. This tool writes only under build/.
    for relative in required_sources():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"C16 diagnostic source is missing or symlinked: {path}")
    result = {"csrc/" + key: value for key, value in file_fingerprint(ROOT / "csrc").items()}
    for path in (ROOT / "oscar_ascend/c16_timer_schema.py", ROOT / CLOCK_FILE,
                 ROOT / "tools/build_c16_timer.py"):
        result[str(path.relative_to(ROOT))] = _digest(path)
    return result


def _writable_destination(path):
    path = Path(path).expanduser().resolve()
    for forbidden in (ROOT / "csrc", ROOT / "references", ROOT / "build/ascendc"):
        if path.is_relative_to(forbidden.resolve()):
            raise ValueError(f"diagnostic build output must not touch production/reference paths: {path}")
    if path == ROOT:
        raise ValueError("diagnostic build output must use a dedicated directory")
    return path


def discover_toolchain(soc, env):
    """Inspect package locations, never import torch/torch_npu or query a card."""
    cann = cann_root(env)
    locations = {}
    for name in ("torch", "torch_npu"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"{name} package is required for this diagnostic build")
        locations[name] = str(Path(spec.origin).resolve().parent)
    npu = Path(locations["torch_npu"])
    header = npu / "include/torch_npu/csrc/core/npu/NPUStream.h"
    if not header.is_file():
        raise RuntimeError(f"torch_npu development header missing: {header}")
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("cmake is not installed")
    versions = {str(path): _digest(path) for path in (
        cann / "version.cfg", cann / "version.info", cann / "compiler/version.info") if path.is_file()}
    configuration = {
        "soc": soc, "cann": str(cann), "cann_versions": versions,
        "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "torch_npu", "pybind11")},
        "package_paths": locations, "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version, "python_soabi": sysconfig.get_config_var("SOABI"),
        "cmake": str(Path(cmake).resolve()), "flags": {name: env.get(name) for name in ("CC", "CXX", "CXXFLAGS", "CFLAGS")},
        "build_type": "Release"}
    selected = dict(env)
    for name in ("ASCEND_CANN_PACKAGE_PATH", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        selected[name] = str(cann)
    selected.update(SOC_VERSION=soc, ASCEND_COMPUTE_UNIT=soc, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OSCAR_ENABLED="0")
    for name in ("OSCAR_TIMER_CONTROL", "OSCAR_C16_BREAKDOWN_DIR", "OSCAR_C16_TIMER_LIBRARY", "OSCAR_C16_TIMER_MANIFEST"):
        selected.pop(name, None)
    for directory in (cann / "bin", cann / "compiler/ccec_compiler/bin", cann / "compiler/bin"):
        if directory.is_dir():
            selected["PATH"] = str(directory) + os.pathsep + selected.get("PATH", "")
    for directory in (cann / "lib64", npu / "lib"):
        if directory.is_dir():
            selected["LD_LIBRARY_PATH"] = str(directory) + os.pathsep + selected.get("LD_LIBRARY_PATH", "")
    return configuration, selected


def _signature(configuration):
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reusable_receipt(path, signature, hashes):
    try:
        receipt = json.loads(Path(path).read_text())
        if (receipt.get("status") != "built" or receipt.get("signature") != signature
                or receipt.get("phase_schema_id") != SCHEMA_ID or receipt.get("source_sha256") != hashes
                or _signature(receipt["configuration"]) != signature):
            return None
        for name in ("library", "kernel_library"):
            artifact = Path(receipt[name])
            if (not artifact.is_absolute() or artifact.is_symlink() or not artifact.is_file()
                    or not artifact.resolve().is_relative_to(Path(path).resolve().parent)
                    or artifact.stat().st_size == 0 or _digest(artifact) != receipt[name + "_sha256"]):
                return None
        generated = receipt["generated_sha256"]
        if not isinstance(generated, dict) or len(generated) < 3:
            return None
        for artifact, expected in generated.items():
            if not Path(artifact).is_file() or _digest(artifact) != expected:
                return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _phase(name, command, *, log_dir, env, timeout):
    result = run_phase(name, command, cwd=ROOT, log_dir=log_dir, env=env, timeout=timeout, grace=5, heartbeat=60)
    if result.returncode or not result.cleanup_complete:
        raise C16BuildError(f"{name} failed rc={result.returncode}; log={result.log}", result.returncode or 125)
    return result


def _generated_paths(directory):
    record = json.loads((directory / "generation.json").read_text())
    paths = {}
    for name in ("kernel", "bindings", "header"):
        path = Path(record[name]).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or not path.stat().st_size:
            raise C16BuildError(f"generator did not publish a valid {name} inside its build directory")
        paths[name] = path
    return paths


def build(*, output, log_dir, soc="ascend910b4", output_dir=None, jobs=2, timeout=1800):
    soc = normalize_soc(soc)
    if soc != "ascend910b4":
        raise ValueError("C16 timer clock/source contract is scoped to ascend910b4")
    if type(jobs) is not int or jobs <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("build jobs and timeout must be positive")
    output = _writable_destination(output)
    log_dir = _writable_destination(log_dir)
    base = _writable_destination(output_dir or ROOT / "build/c16_timer")
    hashes = source_hashes()
    toolchain, env = discover_toolchain(soc, os.environ.copy())
    configuration = {"source_sha256": hashes, "phase_schema_id": SCHEMA_ID,
                     "clock_hz": CLOCK_HZ, "clock_source": CLOCK_SOURCE, "toolchain": toolchain}
    signature = _signature(configuration)
    cache = base / signature
    state_path = cache / "completed.json"
    previous = reusable_receipt(state_path, signature, hashes)
    if previous is not None:
        receipt = {**previous, "reused": True}
        atomic_json(output, receipt)
        print(f"[c16-build] reuse signed diagnostic library={receipt['library']}", flush=True)
        return receipt
    # Preserve previous attempts. Never remove an arbitrary --output-dir, and
    # never overwrite a library that may still be loaded by another process.
    attempt = cache / ("attempt-" + uuid.uuid4().hex)
    generated = attempt / "generated"
    cmake_dir = attempt / "cmake"
    attempt.mkdir(parents=True, exist_ok=False)
    source_dir = ROOT / SOURCE_RELATIVE
    _phase("c16-generate", [sys.executable, str(source_dir / "generate.py"), "--output-dir", str(generated)],
           log_dir=log_dir, env=env, timeout=timeout)
    paths = _generated_paths(generated)
    cann, npu = toolchain["cann"], toolchain["package_paths"]["torch_npu"]
    command = [toolchain["cmake"], "-S", str(source_dir), "-B", str(cmake_dir),
        f"-DASCEND_HOME_PATH={cann}", f"-DASCEND_CANN_PACKAGE_PATH={cann}", f"-DASCEND_TOOLKIT_HOME={cann}",
        f"-DSOC_VERSION={soc}", f"-DASCEND_COMPUTE_UNIT={soc}", f"-DTORCH_NPU_PATH={npu}",
        f"-DPython3_EXECUTABLE={sys.executable}", f"-DOSCAR_BUILD_SIGNATURE={signature}",
        f"-DOSCAR_TIMER_KERNEL_SOURCE={paths['kernel']}", f"-DOSCAR_TIMER_BINDINGS_SOURCE={paths['bindings']}",
        f"-DOSCAR_TIMER_GENERATED_DIR={generated}", "-DCMAKE_BUILD_TYPE=Release"]
    _phase("c16-configure", command, log_dir=log_dir, env=env, timeout=timeout)
    _phase("c16-compile", [toolchain["cmake"], "--build", str(cmake_dir), "--parallel", str(jobs)],
           log_dir=log_dir, env=env, timeout=timeout)
    built = json.loads((cmake_dir / "build_manifest.json").read_text())
    artifacts = {}
    for name in ("library", "kernel_library"):
        path = Path(built[name]).resolve()
        if (not path.is_relative_to(cmake_dir) or not path.is_file() or path.is_symlink()
                or not path.stat().st_size or signature[:16] not in path.name):
            raise C16BuildError(f"independent {name} is missing or lacks its versioned build identity: {path}")
        artifacts[name] = str(path)
        artifacts[name + "_sha256"] = _digest(path)
    if source_hashes() != hashes:
        raise C16BuildError("source changed during diagnostic build; refusing to publish stale artifact")
    receipt = {"schema_version": 1, "status": "built", "reused": False, "signature": signature,
        "source_sha256": hashes, "phase_schema_id": SCHEMA_ID, "clock_hz": CLOCK_HZ, "clock_source": CLOCK_SOURCE,
        "configuration": configuration, "soc": soc, "build_dir": str(cmake_dir), **artifacts,
        "generated_sha256": {str(path): _digest(path) for path in paths.values()},
        "generation_manifest": str(generated / "generation.json"), "device_completion": "not_run",
        "production_artifacts_modified": False}
    atomic_json(attempt / "receipt.json", receipt)
    atomic_json(state_path, receipt)
    atomic_json(output, receipt)
    print(f"[c16-build] built signed diagnostic library={receipt['library']}", flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/build-c16-timer")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/c16_timer")
    parser.add_argument("--soc", default="ascend910b4")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if normalize_soc(args.soc) != "ascend910b4":
        raise ValueError("C16 timer clock/source contract is scoped to ascend910b4")
    if args.jobs <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("build jobs and timeout must be positive")
    # Check destinations even in plan mode; plan never probes installed CANN,
    # imports backends, creates directories or generates source.
    output = _writable_destination(args.output)
    logs = _writable_destination(args.log_dir)
    base = _writable_destination(args.output_dir)
    if args.plan:
        print(json.dumps({"status": "plan", "soc": args.soc, "phase_schema_id": SCHEMA_ID,
            "source_directory": str(ROOT / SOURCE_RELATIVE), "build_root": str(base), "output": str(output),
            "log_dir": str(logs), "generated_location": "<build_root>/<source-and-toolchain-signature>/attempt-<id>/generated",
            "stages": ["c16-generate", "c16-configure", "c16-compile"],
            "source_sha256": source_hashes(), "device_access": False}, indent=2))
        return 0
    try:
        build(output=output, log_dir=logs, output_dir=base, soc=args.soc, jobs=args.jobs, timeout=args.timeout)
        return 0
    except Exception as error:
        rc = getattr(error, "returncode", 1)
        atomic_json(output, {"status": "failed", "returncode": rc, "phase_schema_id": SCHEMA_ID,
                             "device_completion": "not_run", "error": f"{type(error).__name__}: {error}"})
        print(f"[c16-build] FAILED rc={rc}: {error}", file=sys.stderr, flush=True)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
