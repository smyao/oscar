# 档案 G7/G8/G10/G11/#79–#85/#100–#109：自动 CANN/SOC、单配置build、一次干净重试、真实产物。
"""Build the standalone direct-launch AscendC package; never modifies CANN/vLLM."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from .environment import file_fingerprint
from .phase import atomic_json, run_phase

ROOT = Path(__file__).resolve().parents[1]


def cann_root(env: dict[str, str]) -> Path:
    candidates = [env[k] for k in ("ASCEND_CANN_PACKAGE_PATH", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if env.get(k)]
    candidates += ["/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/cann/latest", "/usr/local/Ascend/cann-9.1.0"]
    for value in candidates:
        p = Path(value).resolve()
        if (p / "include").is_dir() and any((p / x).is_dir() for x in (
                "tools/tikcpp/ascendc_kernel_cmake", "compiler/tikcpp/ascendc_kernel_cmake", "ascendc_devkit/tikcpp/samples/cmake")):
            return p
    raise RuntimeError("CANN with AscendC CMake support is missing; inspected " + repr(candidates))


def normalize_soc(device_name: str) -> str:
    value = device_name.lower().replace(" ", "")
    if re.fullmatch(r"ascend910b(?:[1-4]|2c|4-1)", value):
        return value
    raise ValueError(f"A2 device name required, got {device_name!r}; generic ascend910b selects the wrong header family (#83)")


def clear_owned_build(path: Path) -> None:
    expected = ROOT / "build" / "ascendc"
    if path.is_symlink() or path.resolve() != expected:
        raise ValueError("refusing to clean a non-project build directory")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def build(log_dir: Path, timeout: float, soc: str | None = None) -> dict:
    env = os.environ.copy()
    cann = cann_root(env)
    if not shutil.which("cmake"):
        raise RuntimeError("cmake is not installed")
    npu_spec = importlib.util.find_spec("torch_npu")
    if npu_spec is None or npu_spec.origin is None:
        raise RuntimeError("torch_npu package with C++ headers is required for the target build")
    npu_path = Path(npu_spec.origin).parent
    header = npu_path / "include/torch_npu/csrc/core/npu/NPUStream.h"
    if not header.is_file():
        raise RuntimeError(f"torch_npu development header missing: {header}")
    if soc is None:
        # Device access is isolated; caller must establish physical-card selection.
        if not env.get("ASCEND_RT_VISIBLE_DEVICES"):
            raise RuntimeError("select physical devices before automatic SOC detection")
        completed = subprocess.run([sys.executable, "-c", "import torch,torch_npu; print(torch.npu.get_device_name(0))"], env=env, text=True, capture_output=True, timeout=30)
        if completed.returncode:
            raise RuntimeError("SOC detection failed: " + completed.stderr)
        soc = normalize_soc(completed.stdout.strip().splitlines()[-1])
    else:
        soc = normalize_soc(soc)
    for key in ("ASCEND_CANN_PACKAGE_PATH", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        env[key] = str(cann)
    env["SOC_VERSION"] = soc
    env["ASCEND_COMPUTE_UNIT"] = soc
    # Archive #96/#107: build-time Torch CMake discovery must not initialize
    # torch_npu or dlopen runtime driver/HCCL libraries on a compiler-only VM.
    # NPU execution tools import torch_npu explicitly in their own process.
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    build_dir = ROOT / "build/ascendc"
    versions = {}
    for path in (cann / "version.cfg", cann / "version.info", cann / "compiler/version.info"):
        if path.is_file():
            versions[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in (cann / "bin", cann / "compiler/ccec_compiler/bin", cann / "compiler/bin"):
        if directory.is_dir():
            env["PATH"] = str(directory) + os.pathsep + env.get("PATH", "")
    for directory in (cann / "lib64", npu_path / "lib"):
        if directory.is_dir():
            env["LD_LIBRARY_PATH"] = str(directory) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    package_versions = {name: importlib.metadata.version(name) for name in ("torch", "torch_npu", "pybind11")}
    signature_data = {"source": file_fingerprint(ROOT / "csrc"), "cann": str(cann), "cann_versions": versions, "soc": soc,
                      "package_versions": package_versions,
                      "python": sys.executable, "torch_npu": str(npu_path), "flags": {k: env.get(k) for k in ("CXX", "CC", "CXXFLAGS")}}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    state = build_dir / "oscar_build_signature.json"
    previous = json.loads(state.read_text()) if state.exists() else {}
    if previous.get("signature") != signature:
        clear_owned_build(build_dir)
    command = ["cmake", "-S", str(ROOT / "csrc"), "-B", str(build_dir),
               f"-DASCEND_HOME_PATH={cann}", f"-DASCEND_CANN_PACKAGE_PATH={cann}", f"-DASCEND_TOOLKIT_HOME={cann}",
               f"-DSOC_VERSION={soc}", f"-DASCEND_COMPUTE_UNIT={soc}", f"-DTORCH_NPU_PATH={npu_path}",
               f"-DOSCAR_BUILD_SIGNATURE={signature}",
               f"-DPython3_EXECUTABLE={sys.executable}", "-DCMAKE_BUILD_TYPE=Release"]
    for attempt in (1, 2):
        configured = run_phase(f"configure-{attempt}", command, cwd=ROOT, log_dir=log_dir, env=env, timeout=timeout)
        result = configured
        if configured.returncode == 0:
            result = run_phase(f"compile-{attempt}", ["cmake", "--build", str(build_dir), "--parallel", str(min(os.cpu_count() or 1, 8))],
                               cwd=ROOT, log_dir=log_dir, env=env, timeout=timeout)
        if result.returncode == 0:
            break
        if result.interrupted or result.timed_out or not result.cleanup_complete or attempt == 2:
            raise RuntimeError(f"build failed: {result.phase}, rc={result.returncode}, log={result.log}")
        print("clean build retry once: archive G7/G8/#85; original failure retained", flush=True)
        clear_owned_build(build_dir)
    extensions = list(build_dir.rglob("_oscar_ascend_ops*.so"))
    kernels = list(build_dir.rglob("liboscar_ascend_kernels*.so"))
    if len(extensions) != 1 or len(kernels) != 1:
        raise RuntimeError(f"expected exactly one extension and kernel library: extensions={extensions}, kernels={kernels}")
    manifest = {"signature": signature, "configuration": signature_data,
                "extension": str(extensions[0].resolve()), "kernel_library": str(kernels[0].resolve()),
                "sha256": {str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in extensions+kernels},
                "build": "passed", "device_completion": "not_run"}
    atomic_json(state, manifest)
    generated_manifest = build_dir / "build_manifest.json"
    if not generated_manifest.is_file():
        raise RuntimeError(f"CMake did not generate the runtime manifest: {generated_manifest}")
    runtime_manifest = json.loads(generated_manifest.read_text())
    runtime_manifest.update({k: manifest[k] for k in ("signature", "sha256", "build", "device_completion")})
    atomic_json(generated_manifest, runtime_manifest)
    atomic_json(ROOT / "reports/build.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/build")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--soc")
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.log_dir, args.timeout, args.soc), indent=2))
        return 0
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        atomic_json(ROOT / "reports/build.json", {"build": "failed", "device_completion": "not_run", "error": str(exc)})
        print(f"FAILED phase=build-ops: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
