# Archive #28/#77/#96: bind cache before importing torch/vLLM; no device imports.
# #34/#36: preserve native graph/MTP capacity; #85: source/config-bound identity.
# #94/#95/#133: persist observable cache provenance without deleting artifacts.
"""Give native outer AOT and inner compiler caches the same stable namespace.

Native evidence: vllm/envs.py VLLM_CACHE_ROOT and compilation/decorators.py
outer torch_aot_compile path. Native cache keys do not cover the complete
external plugin, so changing only the inner CompilationConfig hash is too late.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile

SCHEMA = "oscar-compile-v1"
# These settings are read by the orchestration helpers, never by model execution.
# tools/service_probe.py: _positive_time, resource release; tools/deploy.py phases.
OBSERVATION_FIELDS = frozenset({"phase_timeout_seconds", "shutdown_timeout_seconds",
    "service_startup_timeout_seconds", "resource_release_tolerance_bytes"})
VERSIONS = ("torch", "torch-npu", "vllm", "vllm-ascend")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _package_root() -> Path:
    # Resolving a top-level package spec does not execute its __init__ or import
    # torch. It hashes the package that the subsequent register() will import.
    spec = importlib.util.find_spec("oscar_ascend")
    if spec is None or not spec.origin:
        raise RuntimeError("cannot resolve OSCAR source for compiler cache identity")
    return Path(spec.origin).resolve().parent


def _sources(package: Path) -> dict:
    files = {str(path.relative_to(package)): _digest(path.read_bytes())
             for path in sorted(package.rglob("*.py")) if path.is_file()}
    if not files:
        raise RuntimeError(f"OSCAR package has no Python sources: {package}")
    return {"sha256": _digest(_json_bytes(files)), "files": files}


def _build_identity(package: Path, env: dict) -> dict:
    explicit = env.get("OSCAR_BUILD_MANIFEST")
    path = Path(explicit).expanduser() if explicit else package.parent / "build/ascendc/build_manifest.json"
    if not path.is_file():
        if explicit:
            raise ValueError(f"explicit OSCAR_BUILD_MANIFEST is unavailable: {path}")
        # require_runtime/loader remains responsible for rejecting missing builds.
        return {"status": "unavailable"}
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"OSCAR build manifest must be an object: {path}")
    hashes = manifest.get("sha256", {})
    if not isinstance(hashes, dict):
        raise ValueError(f"OSCAR build manifest sha256 must be an object: {path}")
    # Ignore installation paths/timestamps but bind the signed build and binary
    # identities checked independently by oscar_ascend.ops.loader at load time.
    return {"status": manifest.get("build"), "signature": manifest.get("signature"),
            "abi_version": manifest.get("abi_version"), "soc": manifest.get("soc"),
            "artifacts": {kind: hashes.get(manifest.get(kind, ""))
                          for kind in ("extension", "kernel_library")}}


def _config_identity(config: dict) -> dict:
    result = deepcopy(config)
    compilation = result.get("compilation_config", {})
    if not isinstance(compilation, dict):
        raise ValueError("compilation_config must be an object for cache isolation")
    for field in ("cache_dir", "local_cache_dir"):
        if compilation.get(field):
            raise ValueError(f"explicit compilation_config.{field} bypasses isolated "
                             "compiler caches; use VLLM_CACHE_ROOT as the cache base instead")
    for field in OBSERVATION_FIELDS:
        result.pop(field, None)
    profiler = result.get("profiler_config")
    if isinstance(profiler, dict):
        # Native ProfilerConfig declares this as export destination only.
        profiler.pop("torch_profiler_dir", None)
    return result


def _versions() -> dict[str, str]:
    values = {}
    for package in VERSIONS:
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = "not-installed"
    return values


def select_compile_cache(config: dict, *, native: bool,
                         env: dict[str, str] | None = None,
                         source_root: Path | None = None) -> dict:
    """Bind env before heavy imports; repeated calls/children never nest roots.

    An explicit VLLM_CACHE_ROOT is the base, not an instruction to erase or use
    unversioned artifacts. Existing caches are retained byte for byte. No graph,
    model or scheduling option is changed. source_root is for isolated host tests.
    """
    env = os.environ if env is None else env
    package = _package_root() if source_root is None else Path(source_root).resolve()
    identity = {"schema": SCHEMA, "mode": "native" if native else "oscar",
                "c16_armed": bool(env.get("OSCAR_C16_BREAKDOWN_DIR")),
                "config": _config_identity(config), "versions": _versions(),
                "source": _sources(package), "build": _build_identity(package, env)}
    key = _digest(_json_bytes(identity))
    current = env.get("VLLM_CACHE_ROOT")
    if current and current == env.get("OSCAR_COMPILE_CACHE_ROOT") and env.get("OSCAR_COMPILE_CACHE_BASE"):
        base = Path(env["OSCAR_COMPILE_CACHE_BASE"])
    else:
        base = Path(current) if current else Path(env.get("XDG_CACHE_HOME", "~/.cache")) / "vllm"
    base = base.expanduser().resolve()
    root = base / SCHEMA / key
    receipt = root / "identity.json"
    reused = receipt.is_file()
    if reused:
        if json.loads(receipt.read_text()) != identity:
            raise RuntimeError(f"compiler cache identity mismatch: {receipt}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        # Atomic publication lets simultaneous launches/TP children observe the
        # complete same identity. Only our temporary receipt is ever removed.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".identity-", delete=False) as output:
                temporary = Path(output.name)
                output.write(_json_bytes(identity) + b"\n")
            os.replace(temporary, receipt)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    env.update(VLLM_CACHE_ROOT=str(root), OSCAR_COMPILE_CACHE_BASE=str(base),
               OSCAR_COMPILE_CACHE_ROOT=str(root), OSCAR_COMPILE_CACHE_KEY=key)
    return {"key": key, "root": str(root), "base": str(base),
            "mode": identity["mode"], "c16_armed": identity["c16_armed"],
            "reused": reused, "identity_file": str(receipt)}
