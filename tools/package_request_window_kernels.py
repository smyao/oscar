"""Package optional request-window kernels without changing device math.

Archive #87/#103/#108/#125/#128/#153: one explicit header path, distinct
symbols, matched optional capabilities, and source identity separate from
compilation/device acceptance. D.4 kernel explanations stay verbatim in each
packaged file. Only local project files are read/written; no native deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/request_window_packaging_20261009.json"
SOURCES = (
    ("csrc/experiments/request_windows_attention/decode_bundle.cpp", "csrc/kernels/attention_request_window_decode.cpp"),
    ("csrc/experiments/request_windows_attention/window_range.cpp", "csrc/kernels/attention_request_window.cpp"),
    ("csrc/experiments/request_windows_attention/window_range_balanced.cpp", "csrc/kernels/attention_request_window_balanced.cpp"),
    ("csrc/experiments/request_windows_attention/window_range_cluster4.cpp", "csrc/kernels/attention_request_window_cluster4.cpp"),
    ("csrc/experiments/request_windows_attention/window_range_cluster16.cpp", "csrc/kernels/attention_request_window_cluster16.cpp"),
    ("csrc/experiments/request_windows_store/packed_store.cpp", "csrc/kernels/rotate_clip_store_request_packed.cpp"),
    ("csrc/experiments/request_windows_store/window_store.cpp", "csrc/kernels/rotate_clip_store_request_window.cpp"),
    ("csrc/experiments/request_windows_attention/member_pipeline_cluster4.cpp", "csrc/kernels/attention_request_window_member_pipeline_cluster4.cpp"),
)
HEADER_SOURCE = "csrc/experiments/request_windows_store/request_window_launch.h"
HEADER_DEST = "csrc/include/oscar_request_window_store_launch.h"
SHARED_HELPERS = (
    ("csrc/experiments/striped_store_simd_pack/pack.h", "csrc/kernels/striped_store_pack.h"),
    ("csrc/experiments/hadamard_indices/vector_indices.h", "csrc/kernels/hadamard_indices.h"),
)


def relocated_helper(text: str) -> str:
    """Relocate the qualified helper; reverse every allowed non-math edit."""
    old = '// Research only.'
    new = '// Shared candidate helper; target NPU/graph qualification pending.'
    before = '#include "../../kernels/oscar_common.h"'
    after = '#include "oscar_common.h"'
    if text.count(old) != 1 or text.count(before) != 1:
        raise ValueError("Qualified helper source changed; re-audit relocation")
    result = text.replace(old, new).replace(before, after)
    if result.replace(new, old).replace(after, before) != text:
        raise ValueError("Shared helper relocation is not reversible")
    return result


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def without_includes(text: str) -> str:
    """Keep every other byte, including math, events, macro aliases and comments."""
    return re.sub(r'^#include[^\n]*\n', '', text, flags=re.M)


def relocated(text: str, source: str) -> str:
    result = re.sub(r'(#include ")\.\./\.\./kernels/', r'\1', text)
    result = re.sub(r'(#include ")\.\./\.\./include/', r'\1../include/', result)
    result = result.replace('#include "request_window_launch.h"',
                            '#include "../include/oscar_request_window_store_launch.h"')
    if "request_windows_attention/" in source:
        marker = '#include "../include/oscar_attention_launch.h"\n'
        if result.count(marker) != 1:
            raise ValueError(f"Expected one attention ABI include in {source}")
        result = result.replace(marker, marker + '#include "../include/oscar_request_window_launch.h"\n')
    if without_includes(result) != without_includes(text):
        raise ValueError(f"Packaging changed non-include bytes: {source}")
    return result


def package(root: Path = ROOT, *, check: bool = False) -> dict:
    helper_records = []
    for source, target in SHARED_HELPERS:
        original = (root / source).read_text()
        result = relocated_helper(original)
        if check:
            if not (root / target).is_file() or (root / target).read_text() != result:
                raise ValueError(f"Shared helper differs from qualified source: {target}")
        else:
            (root / target).write_text(result)
        helper_records.append({"experiment": source, "packaged": target,
                               "experiment_sha256": digest(original),
                               "packaged_sha256": digest(result),
                               "changes": ["project include path", "research origin comment"],
                               "reversible_source_identity": True})
    header = (root / HEADER_SOURCE).read_text()
    if not check:
        (root / HEADER_DEST).write_text(header)
    records = []
    for source, target in SOURCES:
        original = (root / source).read_text()
        result = relocated(original, source)
        if check:
            if not (root / target).is_file() or (root / target).read_text() != result:
                raise ValueError(f"Packaged source differs from include-only mapping: {target}")
        else:
            (root / target).write_text(result)
        includes = re.findall(r'^#include "([^"]+)"', result, re.M)
        local_headers = []
        for include in includes:
            resolved = (root / target).parent / include
            # lib/ and adv_api/ are CANN SDK headers; project headers must exist.
            if include.startswith(("lib/", "adv_api/")):
                continue
            if not resolved.is_file():
                raise ValueError(f"Missing packaged project header {include} in {target}")
            local_headers.append(str(resolved.resolve().relative_to(root.resolve())))
        records.append({"experiment": source, "packaged": target,
                        "experiment_sha256": digest(original), "packaged_sha256": digest(result),
                        "non_include_body_sha256": digest(without_includes(original)),
                        "non_include_body_identical": True, "project_headers": sorted(set(local_headers)),
                        "project_header_sha256": {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                                  for name in sorted(set(local_headers))}})
    if check:
        if (root / HEADER_DEST).read_text() != header:
            raise ValueError("Published store launch header differs from experimental ABI")
    else:
        (root / HEADER_DEST).write_text(header)
    return {"schema_version": 1, "status": "source_identity_checked_not_device_acceptance",
            "archive_refs": [87, 103, 108, 125, 128, 153],
            "reader_kernel_count": 7, "store_kernel_count": 2, "torch_operator_count": 8,
            "mapping": records,
            "shared_helpers": helper_records,
            "store_header": {"source": HEADER_SOURCE, "target": HEADER_DEST,
                             "sha256": digest(header), "verbatim": True},
            "reader_header": {"path": "csrc/include/oscar_request_window_launch.h",
                              "sha256": hashlib.sha256((root/"csrc/include/oscar_request_window_launch.h").read_bytes()).hexdigest()},
            "changes": ["project include path relocation", "add authoritative reader launch declaration include"],
            "default_activation": "not_changed; experimental_request_windows remains false",
            "full_build": "not_run_by_this_packaging_step", "NPU": "not_run",
            "graph_capture": "not_run", "graph_replay": "not_run", "performance": "not_run"}


def package_packed_writer(root: Path = ROOT, *, check: bool = False) -> dict:
    """Package only the regenerated writer; never rewrite the reader kernels."""
    source, target = SOURCES[5]
    original = (root / source).read_text()
    result = relocated(original, source)
    for helper_source, helper_target in SHARED_HELPERS:
        if (root / helper_target).read_text() != relocated_helper((root / helper_source).read_text()):
            raise ValueError(f"Shared writer helper differs: {helper_target}")
    if (root / HEADER_DEST).read_text() != (root / HEADER_SOURCE).read_text():
        raise ValueError("Existing request-window ABI header differs")
    if check:
        if (root / target).read_text() != result:
            raise ValueError("Request-packed writer differs from include-only mapping")
    else:
        (root / target).write_text(result)
    return {"source": source, "target": target,
            "source_sha256": digest(original), "target_sha256": digest(result),
            "non_include_body_sha256": digest(without_includes(original)),
            "non_include_body_identical": True, "ABI": "unchanged",
            "full_build": "pending", "NPU": "pending", "graph": "pending"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / REPORT)
    args = parser.parse_args(argv)
    report = package(check=args.check)
    if not args.check:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "source_files": len(report["mapping"]),
                      "check_only": args.check, "report": str(args.output)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
