"""Patch a disposable vLLM-Ascend source copy for one OSCAR custom op."""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_build_script(path: Path, operator: str) -> None:
    text = path.read_text()
    # Both 910B and 910_93 arrays contain this stable item in the pinned tree.
    marker = '        "kv_quant_sparse_flash_attention"'
    occurrences = text.count(marker)
    if occurrences < 2:
        raise RuntimeError(
            "unexpected vLLM-Ascend build_aclnn.sh layout; refusing to patch"
        )
    needle = f'        "{operator}"\n'
    present = text.count(needle)
    if present == 0:
        text = text.replace(marker, needle + marker)
    elif present != occurrences:
        raise RuntimeError(
            "partially patched build_aclnn.sh; refusing an ambiguous rewrite"
        )
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    args = parser.parse_args()
    build = args.source / "csrc" / "build_aclnn.sh"
    if not build.is_file():
        raise SystemExit(f"missing build script: {build}")
    patch_build_script(build, args.operator)


if __name__ == "__main__":
    main()
