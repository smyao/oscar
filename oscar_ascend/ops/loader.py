"""One strict AscendC direct-launch loader. No CPU/torch/ATB fallback route.

Archive G18/#77/#85/#98-110/#122/#125: load the verified dependency by absolute path.
"""
import ctypes
import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from .contracts import ABI_VERSION, PRODUCTION_CAPABILITIES, missing_capabilities


class OperatorUnavailable(RuntimeError):
    pass


_loaded = None
_loaded_path = None
_kernel_library = None


def _manifest_path(path=None):
    if path is not None:
        return Path(path).expanduser().resolve()
    value = os.environ.get("OSCAR_BUILD_MANIFEST")
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "build" / "ascendc" / "build_manifest.json"


def read_build_manifest(path=None):
    manifest_path = _manifest_path(path)
    if not manifest_path.is_file():
        raise OperatorUnavailable(f"AscendC build manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("abi_version") != ABI_VERSION:
        raise OperatorUnavailable(f"AscendC ABI mismatch: expected {ABI_VERSION}")
    if manifest.get("execution_interface") != "direct_launch":
        raise OperatorUnavailable("This package supports only the direct_launch interface")
    return manifest


def validate_build_artifacts(manifest_path=None):
    """Check build evidence and content hashes without importing torch/NPU.

    This establishes artifact/configuration integrity only. It is not a device
    execution, numerical, graph-capture, replay or performance certificate.
    """
    path = _manifest_path(manifest_path)
    manifest = read_build_manifest(path)
    if manifest.get("build") != "passed":
        raise OperatorUnavailable("AscendC compilation has not completed; configure manifest is not build evidence")
    state_path = path.parent / "oscar_build_signature.json"
    if not state_path.is_file():
        raise OperatorUnavailable(f"AscendC build signature evidence missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    signature = manifest.get("signature")
    if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
        raise OperatorUnavailable("AscendC build signature must be a SHA256 digest")
    if state.get("build") != "passed" or state.get("signature") != signature:
        raise OperatorUnavailable("Runtime manifest signature differs from completed build evidence")
    configuration = state.get("configuration")
    if not isinstance(configuration, dict):
        raise OperatorUnavailable("AscendC build configuration is missing")
    actual_signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    if actual_signature != signature:
        raise OperatorUnavailable("AscendC build configuration signature mismatch")
    for kind in ("extension", "kernel_library"):
        value = manifest.get(kind)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise OperatorUnavailable(f"AscendC {kind} must have an absolute artifact path")
        artifact = Path(value).resolve()
        if str(artifact) != state.get(kind):
            raise OperatorUnavailable(f"AscendC {kind} differs from completed build evidence")
        if not artifact.is_file():
            raise OperatorUnavailable(f"AscendC built artifact missing: {artifact}")
        expected = manifest.get("sha256", {}).get(str(artifact))
        if not isinstance(expected, str) or expected != state.get("sha256", {}).get(str(artifact)):
            raise OperatorUnavailable(f"AscendC {kind} content hash evidence is missing/inconsistent")
        hasher = hashlib.sha256()
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != expected:
            raise OperatorUnavailable(f"AscendC artifact SHA256 mismatch: {artifact}")
    return manifest


def load_extension(manifest_path=None):
    global _loaded, _loaded_path, _kernel_library
    manifest = validate_build_artifacts(manifest_path)
    state_path = _manifest_path(manifest_path).parent / "oscar_build_signature.json"
    configuration = json.loads(state_path.read_text())["configuration"]
    source_root = Path(__file__).resolve().parents[2] / "csrc"
    if not source_root.is_dir():
        raise OperatorUnavailable("The source checkout is required to verify the compiled AscendC package")
    from tools.environment import file_fingerprint
    if file_fingerprint(source_root) != configuration.get("source"):
        raise OperatorUnavailable("AscendC source changed since compilation; rebuild before loading")
    extension = Path(manifest["extension"]).resolve()
    kernels = Path(manifest["kernel_library"]).resolve()
    for path in (extension, kernels):
        if not path.is_file():
            raise OperatorUnavailable(f"AscendC built artifact missing: {path}")
    artifact_key = (str(extension),manifest["sha256"][str(extension)],
                    str(kernels),manifest["sha256"][str(kernels)],
                    manifest["signature"],tuple(sorted(manifest["source_capabilities"])))
    if _loaded is not None:
        if artifact_key != _loaded_path:
            raise OperatorUnavailable("AscendC artifacts changed in this process; restart to avoid stale device code")
        return _loaded
    # Registration stays lightweight; backend import occurs only at this seam.
    importlib.import_module("torch_npu")
    # CANN's CMake helpers can suppress RPATH (#125). The signed kernel SONAME
    # must resolve to this already-verified artifact, independent of the cwd or
    # inherited library search path. Keep the handle alive with the extension.
    try:
        _kernel_library = ctypes.CDLL(str(kernels), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    except OSError as exc:
        raise OperatorUnavailable(f"Cannot load verified AscendC kernel library {kernels}: {exc}") from exc
    spec = importlib.util.spec_from_file_location("_oscar_ascend_ops", extension)
    if spec is None or spec.loader is None:
        raise OperatorUnavailable(f"Cannot load native extension: {extension}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.abi_version() != ABI_VERSION:
        raise OperatorUnavailable("Loaded extension ABI differs from manifest")
    if set(module.capabilities()) != set(manifest["source_capabilities"]):
        raise OperatorUnavailable("Extension capabilities differ from build manifest")
    from .meta import register_meta
    register_meta()
    sys.modules["_oscar_ascend_ops"] = module
    _loaded = module
    _loaded_path = artifact_key
    return module


def require_capabilities(required, manifest_path=None):
    manifest = read_build_manifest(manifest_path)
    missing = missing_capabilities(manifest.get("source_capabilities", ()), required)
    if missing:
        raise OperatorUnavailable("Required AscendC operators are not implemented: " + ", ".join(missing))
    return load_extension(manifest_path)


def require_production_ops(manifest_path=None):
    return require_capabilities(PRODUCTION_CAPABILITIES, manifest_path)
