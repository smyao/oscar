"""Archive #85/#126/#145/#148/#157/#164/#172: checked same-ABI derivatives.

D.4(1): bounded fused FIA task loading, finite checks and current-token store.
D.4(2): the old 6499.8--6655.1ms history restore and 209--215ms stores are
failure examples, not measurements of these kernels. D.4(3): remove inactive
work and redundant finite checks; fixed tiles, live-row math and ownership remain.
D.4(4): no extra history buffer or precision change; generated-source checks
are not CANN, CPU-debug, NPU, graph, model-quality or performance acceptance.
The complete accepted mothers stay immutable, including their public ABI.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "csrc/active_kernels_manifest.json"
KERNELS = (
    {
        "mother": "csrc/kernels/attention_cv_official_decode.cpp",
        "active": "csrc/kernels/attention_cv_official_decode_active.cpp",
        "mother_sha256": "817a5cbb7738e9ad572e39796838cd915a902c98301c452e21caefe6ceaebc22",
        "transform": "tools.active_decode_transform",
    },
    {
        "mother": "csrc/kernels/store_aligned_int2.cpp",
        "active": "csrc/kernels/store_aligned_int2_active.cpp",
        "mother_sha256": "b9c44ddd1cf6c003173c6470d8596e5d1b8d10d8a51862a22694077d4050e134",
        "transform": "tools.active_store_transform",
    },
    {
        "mother": "csrc/kernels/attention_cv_official_cluster4.cpp",
        "active": "csrc/kernels/attention_cv_official_cluster4_active.cpp",
        "mother_sha256": "5069fa48b89a72d78b269b42c50be833007dd24e3bd05eceda0a62ceec722858",
        "transform": "tools.cluster_prefill_transform",
    },
    {
        "mother": "csrc/kernels/attention_cv_official_cluster16.cpp",
        "active": "csrc/kernels/attention_cv_official_cluster16_active.cpp",
        "mother_sha256": "1bcf9ccfe0181f3103b255f67589845a9d83b4e232baa9aa11561b61326d8264",
        "transform": "tools.cluster_prefill_transform",
    },
)
TOOL_PATHS = (
    "tools/generate_active_kernels.py",
    "tools/active_decode_transform.py",
    "tools/active_store_transform.py",
    "tools/cluster_prefill_transform.py",
    "tools/cluster_visibility_transform.py",
    "tools/precise_finite_transform.py",
    "tools/precise_zero_fill_transform.py",
    "tools/cluster_state_transform.py",
)


def _read(root: Path, name: str) -> bytes:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError("active kernel source missing or symlinked: " + name)
    return path.read_bytes()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def outputs(root: Path | None = None) -> dict[str, bytes]:
    """Regenerate in memory and prove exact reversibility before any write."""
    root = ROOT if root is None else Path(root)
    generated = {}
    kernels = []
    for entry in KERNELS:
        original = _read(root, entry["mother"])
        if _sha(original) != entry["mother_sha256"]:
            raise ValueError("accepted active-kernel mother changed: " + entry["mother"])
        module = importlib.import_module(entry["transform"])
        active = module.transform(original.decode("utf-8"))
        if not isinstance(active, str) or active.encode("utf-8") == original:
            raise ValueError("active kernel transform produced no derivative: " + entry["active"])
        if module.restore(active).encode("utf-8") != original:
            raise ValueError("active kernel inverse differs from complete mother: " + entry["active"])
        data = active.encode("utf-8")
        generated[entry["active"]] = data
        kernels.append({**entry, "active_sha256": _sha(data), "public_abi": "unchanged"})
    # Storing tool hashes inside csrc brings their identity into the existing
    # complete-csrc build signature. A changed tool makes --check fail before
    # a cached binary can be reused; build never silently regenerates files.
    manifest = {
        "schema_version": 1,
        "scope": "same-ABI inactive-work and finite-check optimization; not execution acceptance",
        "kernels": kernels,
        "generation_tools_sha256": {name: _sha(_read(root, name)) for name in TOOL_PATHS},
    }
    generated[MANIFEST] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return generated


def check_generated_sources(root: Path | None = None) -> None:
    """Fail closed on any mother, transform, manifest or derivative drift."""
    root = ROOT if root is None else Path(root)
    for name, expected in outputs(root).items():
        if _read(root, name) != expected:
            raise ValueError("active kernel generated source drift: " + name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only; never rewrite")
    args = parser.parse_args(argv)
    if args.check:
        check_generated_sources()
    else:
        generated = outputs()
        for name, data in generated.items():
            (ROOT / name).write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
