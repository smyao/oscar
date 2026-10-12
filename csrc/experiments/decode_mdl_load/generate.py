"""Archive G3-G13/#21/#81-85/#126/#129/#140/#145/#148/#154-156; full D.4.

Independent current-bundle Normal/MDL pair. The SDK configuration differs
only in doNorm/doMultiDataLoad; preserve all AIV code, ABI, K128 FP32 math,
workspace and flags. Compiler-derived tiling is asserted, never overwritten.
Source/host checks are not CANN, CPU-debug, device or performance acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.request_window_decode_profile.generate import PINS
from csrc.experiments.decode_phase_profile.generate import function

SOURCE = "csrc/kernels/attention_cv_decode_bundle.cpp"
SDK = ROOT / "artifacts/cann_sdk_matmul_tiling_20261012"
HEADER = """// GENERATED RESEARCH ONLY. Archive G3-G13/#21/#81-85/#126/#129/#140/#145/#148/#154-156.
// Startup D.4 complete original log reread; four questions:
// 1. Fused FIA QK/PV ND-to-L1 delivery, not a new dequant or service path.
// 2. D.4 whole-history recovery6499.8-6655.1ms is forbidden; fixed KV256 only.
// 3. Current d817 AIV publisher, FP32/HF32-off, per-output K128 sequence,
//    actual M6/M24, finite/error order and cross-core flags are unchanged.
//    Only public SDK Normal/MDL dispatch differs. No host prepack or NZ GM.
// 4. Full B DMA/residency and net speed are hypotheses requiring completed
//    same-library all-source A/B + instruction evidence. No busy-rate claim.
// SDK-generated L1/L0 sizes are compile-time checked, not a hand-made tiling.
"""
ASSERTS = """
// Official constant_tiling.h + matmul_constant_tiling_utils.h, CANN9.1 A2.
static_assert(kCubeConfig.baseM==64 && kCubeConfig.baseN==64 && kCubeConfig.baseK==128,
    "research changed the arithmetic base shape");
static_assert(kCubeConfig.stepM==2 && kCubeConfig.stepN==4 &&
    kCubeConfig.stepKa==2 && kCubeConfig.stepKb==2, "SDK L1 shape differs from audited full-load plan");
static_assert(kCubeConfig.depthA1==4 && kCubeConfig.depthB1==8,
    "SDK full A/B single-buffer depths differ");
static_assert(kCubeConfig.shareL1Size==395264 && kCubeConfig.shareL1Size<=512*1024,
    "SDK L1 allocation differs: 128KiB A + 256KiB B + 2KiB quant-vector reserve");
static_assert(kCubeConfig.dbL0A==2 && kCubeConfig.dbL0B==2 &&
    kCubeConfig.shareL0CSize==16384, "L0 double-buffer or result resource changed");
"""


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def verify_inputs():
    for path, expected in PINS.items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise ValueError("pinned source/dependency drift: " + path)
    bindings = {}
    for name in ("provenance.json", "mdl-provenance.json", "mdl-implementation-provenance.json"):
        for item in json.loads((SDK / name).read_text())["files"]:
            path = item["relative"]
            if sha((SDK / path).read_bytes()) != item["sha256"]:
                raise ValueError("official SDK snapshot drift: " + path)
            bindings[path] = item["sha256"]
    return bindings


def generate(candidate):
    original = (ROOT / SOURCE).read_text()
    text, edits = original, []
    def edit(before, after, count=1):
        nonlocal text
        if text.count(before) != count:
            raise ValueError((before, text.count(before), count))
        text = text.replace(before, after)
        edits.append((before, after, count))
    if candidate:
        edit("  auto cfg=GetNormalConfig();", """  // Public GetMDLConfig defaults differ in unitFlag and batchMode too.
  // Restore both original values so the only config differences are the
  // documented Normal/MDL template selection, not extra synchronization.
  auto cfg=GetMDLConfig(false,false,0,false,false,false,true);
  cfg.batchMode=BatchMode::BATCH_LESS_THAN_L1;""")
    anchor = "constexpr auto kCubeConfig=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(CvConfig());"
    edit(anchor, anchor + ASSERTS)
    arm = "candidate" if candidate else "baseline"
    for before, after, count in (
        ("oscar_attention_cv_decode_bundle_kernel", f"oscar_mdl_{arm}_q4_kernel", 2),
        ("oscar_attention_cv_decode_bundle_q1_kernel", f"oscar_mdl_{arm}_q1_kernel", 2),
        ("attention_cv_decode_bundle_launch", f"mdl_{arm}_q4_launch", 1),
        ("attention_cv_decode_bundle_q1_launch", f"mdl_{arm}_q1_launch", 1),
    ):
        edit(before, after, count)
    for header in ("oscar_common.h", "attention_striped_unpack_padded.h", "attention_range_mask.h"):
        edit(f'#include "{header}"', f'#include "../../kernels/{header}"')
    edit('"../include/', '"../../include/', 2)
    restored = text
    for before, after, count in reversed(edits):
        if restored.count(after) != count:
            raise ValueError("inverse edit ambiguity")
        restored = restored.replace(after, before)
    if restored != original:
        raise ValueError("mother byte restoration failed")
    methods = ("LoadQueries", "LoadPackedInto", "PipelineUnpack", "PipelinePublishKv",
               "PrefetchKeys", "PrefetchValues", "CommitPrefetchErrors", "Softmax",
               "HistoryCubeTask", "HistoryVectorTask", "CubeTask", "VectorTask",
               "PipelineConsumePv", "PublishRows", "PublishEmpty", "PublishStatus", "Matmul")
    for name in methods:
        if text[slice(*function(text,name))] != original[slice(*function(original,name))]:
            raise ValueError("math/publisher/error/handshake changed: " + name)
    result = HEADER + text
    return result, {"arm": arm, "source": SOURCE, "source_sha256": PINS[SOURCE],
        "source_bytes_restored": True, "sha256": sha(result),
        "actual_config_differences": ["doNorm", "doMultiDataLoad"] if candidate else [],
        "unchanged_method_sha256": {name: sha(original[slice(*function(original,name))]) for name in methods},
        "workspace_bytes_per_core_D256": 1146880, "explicit_UB_bytes_per_AIV_D256": 158656,
        "ABI_pointer_count": 15, "int64_attributes": 17, "float_attributes": 1,
        "symbols": {mode: f"oscar_mdl_{arm}_{mode}_kernel" for mode in ("q4", "q1")}}


def probe():
    original = (ROOT / "csrc/experiments/window_batch_decode/probe.cpp").read_text()
    text = original
    edits = (("oscar_attention_cv_striped_decode_simd_kernel", "oscar_mdl_baseline_q4_kernel"),
             ("oscar_attention_cv_striped_q1_simd_kernel", "oscar_mdl_baseline_q1_kernel"),
             ("oscar_ascend::attention_workspace_per_core(d)", "oscar_ascend::attention_decode_bundle_workspace_per_core(d)"),
             ("current_production_simd", "current_production_decode_bundle"))
    for before, after in edits:
        if before not in text:
            raise ValueError("shared probe drift")
        text = text.replace(before, after)
    restored = text
    for before, after in reversed(edits):
        restored = restored.replace(after, before)
    if restored != original:
        raise ValueError("probe inverse identity failed")
    return text, {"shared_source_sha256": sha(original), "sha256": sha(text), "restored": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--build-binding", type=Path)
    args = parser.parse_args()
    manifest = {"official_sdk": verify_inputs(), "dependencies": PINS, "arms": {},
                "resource_prediction_requires_compile_assertions": {"stepM": 2, "stepN": 4,
                    "stepKa": 2, "stepKb": 2, "depthA1": 4, "depthB1": 8,
                    "shareL1Size": 395264, "shareL0CSize": 16384, "dbL0A": 2, "dbL0B": 2},
                "validation": {key: "not_run" for key in ("CPU", "CANN", "CAModel", "NPU", "graph", "whole_service")}}
    files = {}
    for candidate in (False, True):
        text, item = generate(candidate)
        files[f"attention_{item['arm']}.cpp"] = text
        manifest["arms"][item["arm"]] = item
    files["probe.cpp"], manifest["probe"] = probe()
    files["manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    for name, text in files.items():
        path = HERE / name
        if args.check:
            if path.read_text() != text:
                raise ValueError("stale generated file: " + name)
        else:
            path.write_text(text)
    if args.build_binding:
        if not args.check:
            raise ValueError("build binding requires --check")
        args.build_binding.write_text(json.dumps({"manifest_sha256": sha(files["manifest.json"]),
            "generator_sha256": sha(Path(__file__).read_bytes()), "sources": {
                name: sha(text) for name,text in files.items()}}, indent=2) + "\n")
    print(json.dumps({"status": "checked" if args.check else "generated", "production_changed": False,
                      "executed": "host_identity_checks_only"}))


if __name__ == "__main__":
    main()
