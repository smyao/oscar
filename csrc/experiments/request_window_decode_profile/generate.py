"""Archive #132-135/#145/#150/#154/#155; startup D.4: bundle-only attribution.

D.4: inspect fused FIA without a history-sized restore or host metadata loop.
Only diagnostic scalar clocks/counters/MOV markers are added. No completion
barriers, mathematical changes, data-path changes, or speed claims. Exact
inverse edits preserve all source bytes; the uninstrumented arm is same-source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PINS = {
    "csrc/kernels/attention_cv_decode_bundle.cpp": "d8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401",
    "csrc/kernels/oscar_common.h": "20574a83f919df5bd6450cd581d1dba5a5a6f169768e7e237ba62f51b7499964",
    "csrc/kernels/attention_striped_unpack_padded.h": "1551af143666d7465352426dbbe7024af8a86d81ee12d7ccfd9840c328e4ceee",
    "csrc/kernels/attention_range_mask.h": "c1608937a57a0d74159f236b823a38682024bb3ea1893d75a9d1f3652788aa84",
    "csrc/include/oscar_attention_launch.h": "d8731742b2d928b3f8d9a9e330767df449ace11af6808917a3b41721dd2707df",
    "csrc/include/oscar_cv_schedule.h": "36c9bb6276e2ac066057d3d30be0f22f83bc65d6b3aef9497f6ceca946313db8",
    "csrc/experiments/decode_phase_profile/generate.py": "49298a943ce3c3c3a54abf9fdd69d07044b5d31cb5ec264ad622b319d23bdc44",
}
SOURCE = next(iter(PINS))


def sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def generate(profile):
    # Check the mature marker helper before importing it; this is not an
    # automatically repinned clone of whatever helper happens to be present.
    for path, expected in PINS.items():
        actual = sha((ROOT / path).read_bytes())
        if actual != expected:
            raise ValueError(f"source drift; re-audit before repinning {path}: {actual}")
    from csrc.experiments.decode_phase_profile.generate import ENUM, HELPERS, FIELDS, function, masked

    original = (ROOT / SOURCE).read_text()
    text = original
    edits = []

    def replace(old, new, count=1, label=""):
        nonlocal text
        if text.count(old) != count:
            raise ValueError((label or old[:60], text.count(old), count))
        text = text.replace(old, new)
        edits.append((old, new, count, label))

    def method(name, change):
        a, b = function(text, name)
        before = text[a:b]
        after = change(before)
        if before == after:
            raise ValueError("unchanged diagnostic edit: " + name)
        replace(before, after, label="method " + name)

    def scope(name, field, task=False):
        insertion = "\n    ProfileScope profileMethod(*this," + field + ");"
        if task:
            insertion += "\n    ProfileAdd(kTasks,1);"
        method(name, lambda body: body[:1] + insertion + body[1:])

    def waits(name, flag, fields):
        token = "CrossCoreWaitFlag(" + flag + ");"

        def change(body):
            if body.count(token) != len(fields):
                raise ValueError((name, flag, body.count(token), fields))
            for field in fields:
                body = body.replace(token, "ProfileWait(" + field + "," + flag + ");", 1)
            return body
        method(name, change)

    if profile:
        replace("template<int32_t D,int32_t OwnerTile> class AttentionCv {",
                ENUM + "\ntemplate<int32_t D,int32_t OwnerTile> class AttentionCv {", label="enum")
        replace("GM_ADDR lse,GM_ADDR status,GM_ADDR workspace,const Geometry& geometry)",
                "GM_ADDR lse,GM_ADDR status,GM_ADDR workspace,GM_ADDR profile,const Geometry& geometry)", label="Init ABI")
        replace("    g=geometry;", "    TraceMarker(kComputeSpan,false);\n    profileStart=GetSystemCycle();\n    g=geometry;", label="actor start")
        replace("    outStatus.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));",
                "    outStatus.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));\n"
                "    profileGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(profile));", label="profile buffer")
        replace("    coreIndex=core;", """    coreIndex=core;
    engineIndex=0;
    if ASCEND_IS_AIV {engineIndex=lane+1;}
    for(int32_t source=0;source<4;++source)
      for(int32_t field=0;field<24;++field)profileCounters[source][field]=0;""", label="actor counters")
        replace("  __aicore__ void Process() {", HELPERS + "  __aicore__ void Process() {", label="mature helpers")
        replace("    if ASCEND_IS_AIC {mm.End();}\n  }", """    if ASCEND_IS_AIC {mm.End();}
    profileCounters[3][kComputeSpan]=static_cast<int64_t>(GetSystemCycle()-profileStart);
    TraceMarker(kComputeSpan,true);
    ProfileFlush();
  }""", label="actor end; diagnostic flush outside span")
        replace("  Geometry g;", """  Geometry g;
  GlobalTensor<int64_t> profileGm;
  int64_t profileCounters[4][24];uint64_t profileStart;int32_t engineIndex=0;""", label="counter storage")
        for name, field in (
            ("LoadTask", "kTaskMetadata"), ("LoadQueries", "kQueryStage"),
            ("LoadPackedInto", "kPackedDma"), ("Unpack", "kUnpack"),
            ("PipelineUnpack", "kUnpack"), ("LoadPrecise", "kPreciseLoad"),
            ("PublishKv", "kKvPublish"), ("PipelinePublishKv", "kKvPublish"),
            ("Softmax", "kSoftmaxP"), ("PipelineConsumePv", "kPvAcc"),
            ("PublishRows", "kFinalWriteback"),
        ):
            scope(name, field)
        scope("Matmul", "rotation?kAicRotation:(transposeB?kAicQk:kAicPv)")
        for name in ("CubeTask", "VectorTask", "HistoryCubeTask", "HistoryVectorTask"):
            scope(name, "kTaskSpan", task=True)
        waits("CubeTask", "kVectorReady", ["kAicKvWait", "kAicPWait", "kAicAccWait", "kRotationWait", "kRotationWait"])
        waits("VectorTask", "kCubeReady", ["kAivQkWait", "kAivPvWait", "kRotationWait"])
        for flag, field in (("kKvReady", "kAicKvWait"), ("kPReady", "kAicPWait"),
                            ("kNormReady", "kRotationWait"), ("kTaskDone", "kRotationWait")):
            waits("HistoryCubeTask", flag, [field])
        for flag, field in (("kScoreReady", "kAivQkWait"), ("kPvReady", "kAivPvWait"), ("kRotReady", "kRotationWait")):
            waits("HistoryVectorTask", flag, [field])
        for loop in ("for(int64_t start=kvbegin;start<kvend;start+=kKvRows) {",
                     "for(int64_t start=kvbegin;start<kvend;start+=kKvRows,buffer^=1) {"):
            replace(loop, loop + "\n      ProfileAdd(kKvUnits,1);\n"
                    "      ProfileAdd(kKvRowsObserved,Min64(kKvRows,kvend-start));", 2, "exact KV counts")

        def acc_scope(body):
            start = body.index("      // P is fully consumed by Cube before this point.")
            end = body.index("      CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);", start)
            return body[:start] + "      { ProfileScope profileAcc(*this,kPvAcc);\n" + body[start:end] + "      }\n" + body[end:]
        method("VectorTask", acc_scope)
        for name in ("VectorTask", "HistoryVectorTask"):
            token = "    Fence<HardEvent::V_S>();\n    for(int32_t r=0;r<kLaneRows;++r) {"
            def final_scope(body):
                if body.count(token) != 1:
                    raise ValueError("finalize anchor drift")
                return body.replace(token, "    ProfileScope profileFinalize(*this,kAivFinalize);\n" + token)
            method(name, final_scope)
        replace("GM_ADDR status, GM_ADDR workspace, \\\n    int64_t tokens", "GM_ADDR status, GM_ADDR workspace, GM_ADDR profile, \\\n    int64_t tokens", label="kernel profile argument")
        replace("tasks,output,lse,status,workspace,g);", "tasks,output,lse,status,workspace,profile,g);", label="Init profile value")
        replace("void* status,void* workspace,int64_t tokens", "void* status,void* workspace,void* profile,int64_t tokens", label="launch profile argument")
        replace("static_cast<uint8_t*>(workspace), \\\n      tokens,hq", "static_cast<uint8_t*>(workspace),static_cast<uint8_t*>(profile), \\\n      tokens,hq", label="launch profile value")

    arm = "profile" if profile else "normal"
    for old, new, count in (
        ("oscar_attention_cv_decode_bundle_kernel", f"oscar_bundle_{arm}_q4_kernel", 2),
        ("oscar_attention_cv_decode_bundle_q1_kernel", f"oscar_bundle_{arm}_q1_kernel", 2),
        ("attention_cv_decode_bundle_launch", f"bundle_{arm}_q4_launch", 1),
        ("attention_cv_decode_bundle_q1_launch", f"bundle_{arm}_q1_launch", 1),
    ):
        replace(old, new, count, "isolated symbol " + old)
    for header in ("oscar_common.h", "attention_striped_unpack_padded.h", "attention_range_mask.h"):
        replace(f'#include "{header}"', f'#include "../../kernels/{header}"', label="relative include")
    replace('"../include/', '"../../include/', 2, "relative launch/schedule includes")

    restored = text
    for old, new, count, label in reversed(edits):
        if restored.count(new) != count:
            raise ValueError(("inverse edit drift", label))
        restored = restored.replace(new, old)
    if restored != original:
        raise ValueError("instrumentation failed byte-exact inverse identity")
    for token in ("Fence<HardEvent::", "PipeBarrier<", "CrossCoreSetFlag<", "SetHF32Mode("):
        if masked(text).count(token) != masked(original).count(token):
            raise ValueError("new math/completion synchronization: " + token)
    if profile:
        # One wrapper preserves every original wait call. No extra wait sites.
        if masked(original).count("CrossCoreWaitFlag(") != masked(text).count("ProfileWait(") - 1:
            raise ValueError("cross-core wait count drift")
    methods = ("Matmul", "Unpack", "PipelineUnpack", "LoadPackedInto", "LoadPrecise", "LoadWindowRuns",
               "PublishKv", "PipelinePublishKv", "Softmax", "PipelineConsumePv", "PublishRows", "PublishStatus",
               "CommitPrefetchErrors", "HistoryCubeTask", "HistoryVectorTask", "CubeTask", "VectorTask")
    header = """// GENERATED ISOLATED DIAGNOSTIC. Archive #132-135/#145/#150/#154/#155; full D.4.
// D.4 four questions: fused FIA residual attribution; forbidden full-history
// restore remains absent; scalar markers/counters add no completion fence or
// data-path change; perturbation requires matched normal/profile measurement.
// This is the shipped per-page striped_v1 bundle, NOT dual-region window_map.
// No production route, new math, NPU result, graph result, or speed claim.
"""
    return header + text, {
        "source": SOURCE, "source_sha256": PINS[SOURCE], "arm": arm,
        "inverse_source_bytes_identical": True, "new_completion_fences": 0,
        "math_method_body_sha256": {name: sha(original[slice(*function(original, name))]) for name in methods},
        "edits": [{"label": label, "count": count, "before_sha256": sha(old), "after_sha256": sha(new)}
                  for old, new, count, label in edits],
        "kernel_symbols": [f"oscar_bundle_{arm}_{mode}_kernel" for mode in ("q4", "q1")],
        "pointer_count": 16 if profile else 15,
    }, FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = {
        "schema_version": 1, "diagnostic_only": True, "cache_format": "striped_v1_per_physical_page_window",
        "not_measured": "new dual-region request-window reader ABI",
        "dependency_sha256": PINS,
        "validation": {key: "not_run" for key in ("cann_compile", "cpu_debug", "camodel", "npu", "graph", "whole_service")},
        "abi": {"profile_shape": ["cube_cores", 3, 4, 24], "dtype": "int64", "bytes_per_core": 2304,
                "profile_pointer_position_0based": 15, "int64_attributes": 17, "float_attributes": 1,
                "workspace_bytes_per_core": 1146880, "profile_schema": 2, "all_profile_slots_written": True},
        "arms": {},
    }
    for profile in (False, True):
        text, entry, fields = generate(profile)
        dest = OUT / (entry["arm"] + ".cpp")
        entry.update(output=str(dest.relative_to(ROOT)), output_sha256=sha(text))
        manifest["arms"][entry["arm"]] = entry
        if args.check:
            if dest.read_text() != text:
                raise ValueError("stale generated clone: " + str(dest))
        else:
            dest.write_text(text)
    manifest["abi"]["fields"] = dict(enumerate(fields))
    encoded = json.dumps(manifest, indent=2) + "\n"
    if args.check:
        if (OUT / "manifest.json").read_text() != encoded:
            raise ValueError("stale generated manifest")
    else:
        (OUT / "manifest.json").write_text(encoded)
    print(json.dumps({"status": "checked" if args.check else "generated", "source_bytes_restored": True,
                      "normal_and_profile": True, "cann_compile": "not_run", "camodel": "not_run"}))


if __name__ == "__main__":
    main()
