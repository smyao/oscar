"""Archive #145/#164 and D.4: deterministic, separately named BF16 C16 clone.

Writes outside csrc only, so production build fingerprints remain stable.
Original source and all mathematical expressions are retained verbatim except
new timer scopes/local completion fences and the appended diagnostic buffer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "csrc/kernels/attention_cv_aligned_cluster16.cpp"


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"C16 timer source anchor drift: {old[:100]!r}, expected {count}")
    return text.replace(old, new)


def function_scope(text, name, phase, drain="None"):
    pattern = r"(__aicore__ (?:inline )?(?:void|bool) " + re.escape(name) + r"\([^{}]*?\)(?: const)?\s*\{)"
    return re.sub(pattern, lambda m: m[1] + f"\n    oscar_c16_timer::Scope timerScope(timer,{phase},oscar_c16_timer::{drain});", text, count=1)


def generate(directory):
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT / "csrc"):
        raise ValueError("generated diagnostics must not modify production csrc fingerprints")
    directory.mkdir(parents=True, exist_ok=True)
    text = SOURCE.read_text()
    source_sha = hashlib.sha256(text.encode()).hexdigest()
    text = text.replace("attention_cv_aligned_cluster16_launch", "oscar_c16_timer_launch")
    text = text.replace("oscar_attention_cv_aligned_cluster16_kernel", "oscar_c16_timer_kernel")
    text = replace(text, '#include "oscar_common.h"', '#include "oscar_common.h"\n#include "timer.h"')
    text = replace(text, "const Geometry& geometry) {", "const Geometry& geometry,GM_ADDR timerTicks) {\n    timer.Start(timerTicks);\n    oscar_c16_timer::Scope initScope(timer,oscar_c16_timer::InitPhase);")
    text = replace(text, "  TPipe pipe;", "  oscar_c16_timer::Recorder timer;\n public:\n  __aicore__ void FinishTimer(){timer.Finish();}\n private:\n  TPipe pipe;")
    text = replace(text, "clusterStats,g);op.Process()", "clusterStats,g,timerTicks);op.Process();op.FinishTimer()")
    text = replace(text, "float scale, int32_t dequantMode\n#define OSCAR_CV_INIT", "float scale, int32_t dequantMode, GM_ADDR timerTicks\n#define OSCAR_CV_INIT")
    text = replace(text, "float scale,int32_t dequantMode,uint32_t cores) {", "float scale,int32_t dequantMode,uint32_t cores,void* timerTicks) {")
    text = replace(text, "sink,recent,speculative,splits,scale,dequantMode);", "sink,recent,speculative,splits,scale,dequantMode,static_cast<uint8_t*>(timerTicks));")
    # Explicit stage argument distinguishes identically shaped QK and final Rv.
    text = replace(text, "bool transposeB=true,uint8_t atomic=0) {", "bool transposeB=true,uint8_t atomic=0,int timerStage=oscar_c16_timer::Qk) {\n    oscar_c16_timer::Scope matmulScope(timer,timerStage,oscar_c16_timer::Fix);")
    text = replace(text, "rotation?true:transposeB);", "rotation?true:transposeB,0,rotation?oscar_c16_timer::FinalRv:(transposeB?oscar_c16_timer::Qk:oscar_c16_timer::Pv));")
    text = replace(text, "false,static_cast<uint8_t>(1));", "false,static_cast<uint8_t>(1),oscar_c16_timer::Pv);", 2)
    text = replace(text, "rows,D,kKvRows,false);", "rows,D,kKvRows,false,0,oscar_c16_timer::Pv);")
    text = replace(text, "MatmulResidentB(qOffset,rotOffset,rows,D,D,true);", "MatmulResidentB(qOffset,rotOffset,rows,D,D,true,0,oscar_c16_timer::FinalRv);")
    scopes = {
        "Process": ("ProcessOther", "None"), "LoadTask": ("TaskMetadata", "None"),
        "IsClusterMember": ("TaskMetadata", "None"), "LoadQueries": ("QueryLoad", "Store"),
        "LoadPacked": ("HistoryLoad", "Load"), "LoadPrecise": ("PreciseLoad", "Vector"),
        "PublishHistoryKv": ("PublishKv", "Store"), "PublishKv": ("PublishKv", "Store"),
        "LoadResidentB": ("B1Load", "Load"), "Softmax": ("SoftmaxMath", "Vector"),
        "RestoreState": ("StateRestore", "Load"), "SpillState": ("StateSpill", "Store"),
        "PublishNormalizedQueries": ("QueryLoad", "Store"), "PublishRows": ("OutputPublish", "Store"),
        "CubeTask": ("TaskOther", "None"), "VectorTask": ("TaskOther", "None"),
        "CubeCluster": ("ClusterOther", "None"), "VectorCluster": ("ClusterOther", "None"),
    }
    for name, (phase, drain) in scopes.items():
        changed = function_scope(text, name, "oscar_c16_timer::" + phase, drain)
        if changed == text:
            raise ValueError(f"missing timer function: {name}")
        text = changed
    changed = function_scope(text, "Unpack", "value?oscar_c16_timer::UnpackV:oscar_c16_timer::UnpackK", "Vector")
    if changed == text:
        raise ValueError("missing timer function: Unpack")
    text = changed
    # Final normalization is arithmetic. Its subsequent BF16 publication uses
    # a separate scope; do not label publication time as the division itself.
    normalization = re.compile(r"(?P<indent> +)Fence<HardEvent::V_S>\(\);\n(?P=indent)for\(int32_t r=0;r<kLaneRows;\+\+r\) \{\n.*?\n(?P=indent)\}", re.S)
    if len(normalization.findall(text)) != 2:
        raise ValueError("C16 final normalization anchor drift")
    text = normalization.sub(lambda m: m['indent'] + "{oscar_c16_timer::Scope normScope(timer,oscar_c16_timer::Normalize,oscar_c16_timer::Vector);\n" + m[0] + "}", text)
    # Retain score copy/Muls order while separating completed load and P store.
    text = replace(text,
        "    DataCopy(scores,work[scoreOffset+rowBase*kKvRows],kHalfRows*kKvRows);\n    Fence<HardEvent::MTE2_V>();Muls(scores,scores,g.scale,kHalfRows*kKvRows);",
        "    {oscar_c16_timer::Scope loadScope(timer,oscar_c16_timer::SoftmaxLoad,oscar_c16_timer::Load);\n    DataCopy(scores,work[scoreOffset+rowBase*kKvRows],kHalfRows*kKvRows);\n    Fence<HardEvent::MTE2_V>();}\n    Muls(scores,scores,g.scale,kHalfRows*kKvRows);")
    text = replace(text, "    auto roundedP=naturalBuf.Get<bfloat16_t>();", "    {oscar_c16_timer::Scope publishScope(timer,oscar_c16_timer::SoftmaxPublish,oscar_c16_timer::Store);\n    auto roundedP=naturalBuf.Get<bfloat16_t>();")
    text = replace(text, "    Fence<HardEvent::MTE3_MTE2>();\n  }\n  __aicore__ void VectorTask", "    Fence<HardEvent::MTE3_MTE2>();}\n  }\n  __aicore__ void VectorTask")
    # BF16 rounding is Vector work, not DMA. Nested intervals are deducted
    # from KV/P/query publication so their category cannot hide Cast cost.
    casts = re.compile(r"oscar_bf16::CastRne\([^;]*\);")
    if not casts.findall(text):
        raise ValueError("C16 BF16 publication cast anchor drift")
    text = casts.sub(lambda m: "{oscar_c16_timer::Scope castScope(timer,oscar_c16_timer::Bf16Cast,oscar_c16_timer::Vector);" + m[0] + "}", text)
    # Each wait has its own exclusive actor-local interval; no collective clock.
    text = re.sub(r"CrossCoreWaitFlag\(([^;]+)\);", r"{oscar_c16_timer::Scope waitScope(timer,oscar_c16_timer::CrossWait);CrossCoreWaitFlag(\1);}", text)
    # AscendC's external precompile does not inherit ordinary CMake directory
    # include paths. The clone lives outside kernels/, so bind its local
    # includes to the actual source files; SDK-owned includes remain unchanged.
    def source_include(match):
        name = match[1]
        if name.startswith(("lib/", "adv_api/")):
            return match[0]
        path = (Path(__file__).parent / name if name == "timer.h" else SOURCE.parent / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"C16 diagnostic source include missing: {path}")
        return '#include "' + str(path) + '"'
    text = re.sub(r'^#include "([^"]+)"', source_include, text, flags=re.M)
    kernel = directory / "c16_timer_kernel.cpp"
    kernel.write_text(f"// Diagnostic clone source SHA256 {source_sha}\n" + text)

    # Clone audited host validation, changing only the added counter buffer ABI.
    binding_source = (ROOT / "csrc/aligned_attention_bindings.cpp").read_text()
    start = binding_source.index("void AttentionStripedCluster(")
    end = binding_source.index("\n}\n}\nTORCH_LIBRARY_FRAGMENT", start) + 2
    function = binding_source[start:end].replace("AttentionStripedCluster", "TimerAttention")
    function = replace(function, "int64_t dequantMode,int64_t cores) {", "int64_t dequantMode,int64_t cores,at::Tensor ticks) {")
    function = replace(function, "  Check(query,query,at::kBFloat16,\"query\");", "  constexpr bool Bf16Rotation=true;constexpr int ClusterSize=16;\n  constexpr int64_t WorkspaceBytes=oscar_ascend::attention_bf16_cluster16_workspace_per_core(256);\n  Check(ticks,query,at::kLong,\"ticks\");\n  TORCH_CHECK(ticks.dim()==3 && ticks.size(0)==3*cores && ticks.size(1)==32 && ticks.size(2)==2 && reinterpret_cast<uintptr_t>(ticks.data_ptr())%64==0,\"ticks must be aligned int64[3*cores,32,2]\");\n  Check(query,query,at::kBFloat16,\"query\");")
    function = replace(function, "  if(!n) {clusterStats.zero_();return;}", "  TORCH_CHECK(n>0,\"diagnostic requires nonempty C16 inputs\");\n  for(const auto& tensor:{query,queryRot,currentKey,currentValue,rotation,raw,table,windowKey,windowValue,windowTags,tasks,partial,lse,status,workspace,clusterStats})Disjoint(ticks,tensor);")
    function = function.replace("  Launch(", "  oscar_ascend::oscar_c16_timer_launch(")
    function = replace(function, "static_cast<uint32_t>(cores));", "static_cast<uint32_t>(cores),ticks.data_ptr());")
    checks = binding_source[binding_source.index("void Check("):binding_source.index("template<auto Launch")]
    launch = text[text.index("void attention_cv_fast_cluster4_launch("):]
    launch = launch[:launch.index(" {")].replace("attention_cv_fast_cluster4_launch", "oscar_c16_timer_launch") + ";"
    (directory / "launch.h").write_text('#pragma once\n#include <cstdint>\nnamespace oscar_ascend {\n' + launch + '\n}\n')
    includes = binding_source[:binding_source.index("namespace {")]
    includes += '#include "launch.h"\nnamespace {\n'
    schema_start = binding_source.index('  m.def("attention_cv_aligned_cluster16_out(')
    schema_end = binding_source.index(' -> ()");', schema_start) + len(' -> ()");')
    schema = binding_source[schema_start:schema_end].replace("attention_cv_aligned_cluster16_out", "attention_out").replace("int cube_cores) ->", "int cube_cores, Tensor(f!) ticks) ->")
    binding = directory / "bindings.cpp"
    binding.write_text(includes + checks + function + '\n}\nTORCH_LIBRARY(oscar_c16_timer,m){\n' + schema + '\n}\nTORCH_LIBRARY_IMPL(oscar_c16_timer,PrivateUse1,m){m.impl("attention_out",&TimerAttention);}\n')
    result = {"source_sha256": source_sha, "kernel": str(kernel), "bindings": str(binding),
              "header": str(directory / "launch.h")}
    (directory / "generation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(json.dumps(generate(parser.parse_args().output_dir)))
