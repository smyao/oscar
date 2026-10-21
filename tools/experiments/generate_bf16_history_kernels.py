"""Archive G4/#81-83/#126/#145/#148/#154/#157/#171/#173 and full startup D.4.

Generate unified BF16 kernels from the exact 6029f2a mothers. Each
replacement is reversible; --check proves both final bytes and inverse identity.
All sources use BF16 operands; FP32 softmax, accumulators and Rv are retained.
No reference tree or existing kernel is written. This is not a CANN validator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOTHER_COMMIT = "6029f2a8c6277ab8136e5a5eeaa2c2c16313ca08"
SPECS = (
    ("mdl", "attention_cv_mdl_decode.cpp", "1d451bd9c0c47e1d61a5531c785ef5b0a72a1978c79ee972dca8f7a1c6cf7b0f"),
    ("member_pipeline_cluster4", "attention_cv_member_pipeline_cluster4.cpp", "0795a07b1db0ba3d12e7ae762929b752d3bfb6d7475a925cf9e25e62ddeeb4bc"),
    ("window_range_cluster16", "attention_cv_window_range_cluster16.cpp", "708d83d87d436550bd82f545d64af19cd5c5af34acc92b47fbe6375a0496f335"),
    ("window_range", "attention_cv_window_range.cpp", "d6d50b5c91b0cdf64398edab2b071f0a5b69490759511f7dd8e4f1585b64ffc4"),
    ("window_range_balanced", "attention_cv_window_range_balanced.cpp", "1bae967a1b9e0126e627b5bf0c29e6b9799f6202d12a3bc373c2455ea00ca013"),
)

HEADER = """// SPDX-License-Identifier: Apache-2.0
// GENERATED UNIFIED BF16 KERNEL. Default routing is not numerical acceptance.
// Mother: {mother} at {commit}; SHA256 {sha}.
// Generate/check: tools/experiments/generate_bf16_history_kernels.py --check.
// Archive G4/#81-83/#126/#145/#148/#154/#157/#171/#173; full startup D.4 reread.
// D.4 four questions: (1) direct INT2-to-BF16 and all-source BF16 QK/PV;
// (2) failed full-history restore cost6499.8-6655.1ms versus FIA18.5-18.9ms;
// (3) fixed KV256 BF16 operands, exact BF16 window/current K/V; FP32
// softmax/LSE/state/Rv, masks, error order and cross-core flags are retained;
// (4) compact typed GM arenas, extra UB16KiB (+512B on H16); BF16-first Init
// removes the unused initial FP32 allocation/reset. Later dtype switches and
// bounded unpack lookup remain. High/low planes double operand bytes; history
// QK/PV use four BF16 products, precise QK one/PV two. No speed claim.
// One active Matmul resource set avoids simultaneous L1/L0 allocations.
// Re-Init/Reset needs actual SDK and device verification; host has neither.
// The mother's inherited historical comments below are not claims about
// this BF16 candidate. The executable delta and independent gates govern it.
"""


def derive(name: str, mother: str, expected_sha: str):
    source = (ROOT / "csrc/kernels" / mother).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha:
        raise ValueError(f"mother changed: {mother}; expected {MOTHER_COMMIT}")
    text = source
    edits = []

    def replace(old: str, new: str, count: int = 1):
        nonlocal text
        found = text.count(old)
        if found != count:
            raise ValueError(f"{mother}: replacement count {found} != {count}: {old[:100]}")
        text = text.replace(old, new)
        edits.append((old, new, count))

    # Rename actual macro aliases, never the legacy names within the body.
    if name == "mdl":
        old, new = "attention_cv_mdl_decode", "attention_bf16_history_mdl"
        replace(old, new, text.count(old))
    else:
        old_launch = f"attention_cv_{name}_launch"
        old_kernel = ("oscar_attention_cv_member_pipeline_cluster4_kernel"
                      if name == "member_pipeline_cluster4" else f"oscar_attention_{name}_kernel")
        replace(old_launch, f"attention_bf16_history_{name}_launch")
        replace(old_kernel, f"oscar_attention_bf16_history_{name}_kernel")

    replace('#include "oscar_common.h"', '#include "oscar_common.h"\n'
            '#include "attention_bf16_history_operands.h"\n'
            '#include "attention_striped_unpack_bf16.h"\n'
            '#include "../include/oscar_bf16_history_launch.h"')
    cube_alias = "using CubeMatmul=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kCubeConfig>;"
    replace(cube_alias, cube_alias + """
using HistoryA=MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,false>;
using HistoryB=MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,true>;
constexpr auto kHistoryCubeConfig=GetMatmulApiTiling<HistoryA,HistoryB,Matrix,Matrix>(CvConfig());
using HistoryMatmul=matmul::MatmulImpl<HistoryA,HistoryB,Matrix,Matrix,kHistoryCubeConfig>;
static_assert(kHistoryCubeConfig.baseM==64 && kHistoryCubeConfig.baseN==64 &&
    kHistoryCubeConfig.baseK==128,"BF16 reduction base geometry changed");
static_assert(kHistoryCubeConfig.shareL1Size<=512*1024 &&
    kHistoryCubeConfig.shareL0CSize<=128*1024,"BF16 single-instance L1/L0C overflow");
static_assert(64*128*2*kHistoryCubeConfig.dbL0A<=64*1024 &&
    64*128*2*kHistoryCubeConfig.dbL0B<=64*1024,"BF16 single-instance L0A/L0B overflow");
""")

    init = source[source.index("__aicore__ void Init("):source.index("__aicore__ void Process()")]
    sizes = re.findall(r"pipe.InitBuffer\(\w+,([^;]*)\);", init)
    if len(sizes) not in (18, 19):
        raise ValueError(f"unreviewed UB allocation count: {mother}: {len(sizes)}")
    plane = "  static constexpr int32_t kPlaneStride=kWords+16;"
    natural_extra = 0 if name in ("mdl", "member_pipeline_cluster4") else 512
    replace(plane, plane + "\n  static constexpr int32_t kOriginalVectorBytes=\n      "
            + "+\n      ".join(f"({s})" for s in sizes) + ";\n"
            f"  static constexpr int32_t kNaturalPaddingBytes={natural_extra};\n"
            "  static_assert(kOriginalVectorBytes+kNaturalPaddingBytes+oscar_bf16_history::kOperandScratchBytes<=184*1024,\n"
            '      "BF16 publisher exceeds A2 usable UB; do not relax this bound");')
    if natural_extra:
        replace("pipe.InitBuffer(naturalBuf,kElements*2);", "pipe.InitBuffer(naturalBuf,kHalfKv*(D+16)*2);")
    replace("    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+core*floatsPerCore);",
            """    // One private region per Cube: unchanged float arena followed by its
    // index-preserving BF16 mirror. Result/state/Rv never alias that mirror.
    auto coreWorkspace=workspace+core*floatsPerCore*6;
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(coreWorkspace));
    operandWork.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
        coreWorkspace+floatsPerCore*4));""")
    replace("      pipe.InitBuffer(addressBuf,128);", "      pipe.InitBuffer(addressBuf,128);\n"
            "      pipe.InitBuffer(operandBuf,oscar_bf16_history::kOperandScratchBytes);")
    if "cluster" in name:
        replace("    if ASCEND_IS_AIC {\n      mm.End();", "    if ASCEND_IS_AIC {\n      if(historyMatmulActive)historyMm.End();else mm.End();")
    else:
        replace("    if ASCEND_IS_AIC {mm.End();}", "    if ASCEND_IS_AIC {if(historyMatmulActive)historyMm.End();else mm.End();}")
    replace("  TPipe pipe;CubeMatmul mm;", "  TPipe pipe;CubeMatmul mm;HistoryMatmul historyMm;\n"
            "  bool historyMatmulActive=false;\n"
            "  int64_t residualOffset;\n"
            "  GlobalTensor<bfloat16_t> operandWork;\n"
            "  TBuf<TPosition::VECCALC> operandBuf;")

    sig = "  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,\n      bool rotation=false,bool transposeB=true) {"
    org = ("historyMm.SetOrgShape(m,transposeB?kKvRows:D,transposeB?D:kKvRows);"
           if name == "mdl" else "historyMm.SetOrgShape(m,n,k);")
    replace(sig, sig + "\n" + """    const bool history=!rotation;
    oscar_bf16_history::SelectMatmul(history,historyMatmulActive,pipe,mm,historyMm);
    if(history) {
      SetHF32Mode(false);

      // History: (Ah+Al)(Bh+Bl); precise QK: AhBh; precise PV: (Ah+Al)Bh.
      const int32_t aTerms=(kind==0 || !transposeB)?2:1;
      const int32_t bTerms=kind==0?2:1;
      for(int32_t ai=0;ai<aTerms;++ai)for(int32_t bi=0;bi<bTerms;++bi) {
        """ + org + """
        historyMm.SetSingleShape(m,n,k);
        historyMm.SetTensorA(operandWork[a+ai*residualOffset],false);
        historyMm.SetTensorB(operandWork[b+bi*residualOffset],transposeB);
        historyMm.template IterateAll<true>(work[c],static_cast<uint8_t>((ai||bi)?1:0));
        historyMm.End();
      }
      return;
    }""")
    if name == "member_pipeline_cluster4":
        atomic_sig = "  __aicore__ void MatmulAtomic(int64_t a,int64_t b,int64_t c,\n      int32_t m,int32_t n,int32_t k) {"
        begin = text.index(atomic_sig)
        end = text.index("  __aicore__ void CubeTask()", begin)
        replace(text[begin:end], atomic_sig + """
    oscar_bf16_history::SelectMatmul(true,historyMatmulActive,pipe,mm,historyMm);
    SetHF32Mode(false);
    // Every term adds to the existing rescaled online accumulator.
    for(int32_t ai=0;ai<2;++ai)for(int32_t bi=0;bi<2;++bi) {
      historyMm.SetOrgShape(m,n,k);historyMm.SetSingleShape(m,n,k);
      historyMm.SetTensorA(operandWork[a+ai*residualOffset],false);
      historyMm.SetTensorB(operandWork[b+bi*residualOffset],false);
      historyMm.template IterateAll<true>(work[c],static_cast<uint8_t>(1));
      historyMm.End();
    }
  }
""")
    replace("if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],transposeB);",
            "mm.SetTensorB(rv,true); // The only FP32 matmul is inverse rotation.")
    if name == "mdl":
        replace("mm.SetOrgShape(m,rotation?D:(transposeB?kKvRows:D),\n"
                "        rotation?D:(transposeB?D:kKvRows));", "mm.SetOrgShape(m,D,D);")
    # These are only input-Q/padding, unpacked K/V and P writes. Acc/state and
    # normalized Q-for-Rv writes deliberately remain the original float copies.
    for old, new in (
        ("DataCopy(work[qOffset+row*D],dst,D)", "PublishOperand(qOffset+row*D,dst,D)"),
        ("DataCopy(work[qOffset+row*D],zeros,count*D)", "PublishOperand(qOffset+row*D,zeros,count*D)"),
        ("DataCopy(work[(value?vOffset:kOffset)+slot*D],src,kElements)", "PublishUnpackedOperand((value?vOffset:kOffset)+slot*D,src,kElements)"),
        ("DataCopy(work[pOffset+rowBase*kKvRows],scores,activeRows*kKvRows)", "PublishOperand(pOffset+rowBase*kKvRows,scores,activeRows*kKvRows)"),
    ):
        replace(old, new)
    if name == "mdl":
        replace("DataCopy(work[(value?vOffset:kOffset)+buffer*(2*kKvRows*D)+slot*D],\n        dequantBuf.Get<float>(),kElements)",
                "PublishUnpackedOperand((value?vOffset:kOffset)+buffer*(2*kKvRows*D)+slot*D,\n        dequantBuf.Get<bfloat16_t>(),kElements)")
    helpers = """  __aicore__ void PublishOperand(int64_t offset,LocalTensor<float> source,int32_t elements) {
    oscar_bf16_history::Publish(operandWork[offset],operandWork[residualOffset+offset],source,operandBuf,elements);
  }
  __aicore__ void PublishBf16Operand(int64_t offset,LocalTensor<bfloat16_t> source,
      int32_t elements) {
    DataCopy(operandWork[offset],source,elements);
  }
  __aicore__ void PublishUnpackedOperand(int64_t offset,LocalTensor<bfloat16_t> source,
      int32_t elements) {
    PublishBf16Operand(offset,source,elements);
    if(kind==0)DataCopy(operandWork[residualOffset+offset],source[kElements],elements);
  }
  __aicore__ void CheckBf16Row(LocalTensor<bfloat16_t> source) {
    // Keep the original FP32 ReduceSum finite/error contract, one row only.
    // Matrix operands retain the original BF16 bits, including signed zero.
    auto row=qFloatBuf.Get<float>();auto check=scratchBuf.Get<float>();
    Cast(row,source,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
    ReduceSum(check,row,check[16],D);Fence<HardEvent::V_S>();
    if(!Finite(check.GetValue(0)))error=2;
    Fence<HardEvent::S_V>();
  }
"""
    if name == "member_pipeline_cluster4":
        helpers = helpers.replace("source[kElements],elements", "source[kHistoryElements],elements")
        replace("DataCopy(work[(value?vOffset:kOffset)+slot*D],src,copy)",
                "PublishHistoryOperand((value?vOffset:kOffset)+slot*D,src,copy)")
        helpers += """  __aicore__ void PublishHistoryOperand(int64_t offset,LocalTensor<bfloat16_t> source,
      const DataCopyParams& originalCopy) {
    // Original two old-16-row lane domains, now measured in BF16 elements.
    PublishBf16Operand(offset,source,kElements);
    PublishBf16Operand(offset+2*kElements,source[kElements],kElements);
    DataCopy(operandWork[residualOffset+offset],source[kHistoryHalfKv*D],kElements);
    DataCopy(operandWork[residualOffset+offset+2*kElements],source[kHistoryHalfKv*D+kElements],kElements);
  }
"""
    replace("  __aicore__ void LoadQueries() {", helpers + "  __aicore__ void LoadQueries() {")

    # Direct packed INT2 -> BF16 high/low. FP32 reconstruction is the four
    # scalar levels per token, never an expanded FP32 [KV,D] tile.
    half = "kHistoryHalfKv" if name == "member_pipeline_cluster4" else "kHalfKv"
    old_ns = ("oscar_ascend_unpack32" if name == "member_pipeline_cluster4" else
              "oscar_ascend_padded_natural" if name == "mdl" else "oscar_ascend_fast")
    old_template = f"D,{half}" if name in ("mdl", "member_pipeline_cluster4") else "D"
    for method in ("InitIndices", "Unpack"):
        old = f"{old_ns}::{method}<{old_template}>"
        replace(old, f"oscar_ascend_striped_bf16::{method}<D,{half}>", text.count(old))
    for args in ("        metadataIndexBuf,metadataHalfBuf,dequantBuf,scratchBuf);",
                 "        metadataHalfBuf,dequantBuf,scratchBuf);"):
        if args in text:
            replace(args, args[:-2] + ",operandBuf);", text.count(args))

    # A small row conversion is retained for the existing finite check only.
    # Do not round-trip exact Q/K/V through FP32 for operand publication.
    replace("PublishOperand(qOffset+row*D,dst,D);",
            "if(kind==0)PublishOperand(qOffset+row*D,dst,D);\n"
            "      else PublishBf16Operand(qOffset+row*D,src,D);")
    replace("auto zeros=dequantBuf.Get<float>();", "auto zeros=dequantBuf.Get<bfloat16_t>();")
    replace("Duplicate(zeros,0.0F,count*D);", "Duplicate(zeros.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),count*D);")
    replace("PublishOperand(qOffset+row*D,zeros,count*D)", "PublishBf16Operand(qOffset+row*D,zeros,count*D);\n"
            "      DataCopy(operandWork[residualOffset+qOffset+row*D],zeros,count*D)")

    def function_body(method):
        start = text.index(f"  __aicore__ void {method}(")
        end = text.index("  __aicore__ ", start + 10)
        return text[start:end]

    old = function_body("LoadWindowRuns")
    new = old.replace("auto dst=dequantBuf.Get<float>();\n    auto batch=naturalBuf.Get<bfloat16_t>();",
                      "auto dst=dequantBuf.Get<bfloat16_t>();")
    new = new.replace("Duplicate(batch.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),rows*D);",
                      "Duplicate(dst[j*D].ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),rows*D);")
    new = new.replace("DataCopyPad(batch[begin*D]", "DataCopyPad(dst[(j+begin)*D]")
    new = new.replace("      Cast(dst[j*D],batch,RoundMode::CAST_NONE,rows*D);PipeBarrier<PIPE_V>();\n", "")
    new = re.sub(r"      auto check=scratchBuf.Get<float>\((?:128)?\);\n", "", new)
    new = new.replace("        ReduceSum(check,dst[(j+row)*D],check[16],D);Fence<HardEvent::V_S>();\n"
                      "        if(!Finite(check.GetValue(0)))error=2;", "        CheckBf16Row(dst[(j+row)*D]);")
    replace(old, new)

    old = function_body("LoadPrecise")
    new = old.replace("auto dst=dequantBuf.Get<float>();auto src=qBf16Buf.Get<bfloat16_t>();",
                      "auto dst=dequantBuf.Get<bfloat16_t>();")
    new = new.replace("Duplicate(dst,0.0F,kElements);", "Duplicate(dst.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),kElements);")
    start = new.index("      // The natural unpack buffer")
    end = new.index("      Fence<HardEvent::V_MTE2>();", start)
    new = new[:start] + "      // Load exact BF16 directly into the bounded operand tile.\n" + new[end:]
    new = new.replace("DataCopyPad(batch,", "DataCopyPad(dst,")
    new = new.replace("Fence<HardEvent::MTE2_V>();Cast(dst,batch,RoundMode::CAST_NONE,kElements);\n      PipeBarrier<PIPE_V>();", "Fence<HardEvent::MTE2_V>();")
    new = re.sub(r"      auto check=scratchBuf.Get<float>\((?:128)?\);\n", "", new)
    new = new.replace("        ReduceSum(check,dst[row*D],check[16],D);Fence<HardEvent::V_S>();\n"
                      "        if(!Finite(check.GetValue(0)))error=2;", "        CheckBf16Row(dst[row*D]);")
    # Callers enter only for source1/2. Preserve address/tag/finite order in
    # the reachable paths, and reject an accidental unsupported source.
    end = new.index("    if(kind==1) {LoadWindowRuns(start,value);return;}")
    new = new[:end] + "    if(kind==1) {LoadWindowRuns(start,value);return;}\n    error=1;\n  }\n"
    replace(old, new)

    for method in (("PublishKv", "PublishHistoryKv") if name == "member_pipeline_cluster4" else ("PublishKv",)):
        old = function_body(method)
        new = old.replace("dequantBuf.Get<float>()", "dequantBuf.Get<bfloat16_t>()")
        new = new.replace("natural [KV,D] FP32", "natural [KV,D] BF16")
        replace(old, new)

    # Archive #126/#145/#148/#154 and D.4: typed offsets are independent.
    # Keep all existing producers, barriers, atomics and state lifetimes;
    # only final serialized normalization shares a single FP32 MxD input.
    members = 4 if name == "member_pipeline_cluster4" else 16 if name == "window_range_cluster16" else 1
    kv_buffers = 2 if name in ("mdl", "member_pipeline_cluster4") else 1
    score_slots = 2 if name == "member_pipeline_cluster4" else 1
    begin = text.index("    const int64_t " + ("baseFloats=" if members == 4 else "floatsPerCore="))
    end = text.index("    if ASCEND_IS_AIC", begin)
    compact = f"""    constexpr oscar_ascend::Bf16AttentionWorkspace layout{{D,kQueryRows,{members},{kv_buffers},{score_slots}}};
    // Byte stride is per Cube; offsets below are elements in their own type.
    // No unused FP32 K/V or BF16 score/accumulator mirror is reserved.
    static_assert(layout.FloatElements()*4%32==0 && layout.Bytes()%32==0,
        "typed workspace must preserve DMA alignment");
    auto coreWorkspace=workspace+core*layout.Bytes();
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(coreWorkspace));
    operandWork.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
        coreWorkspace+layout.FloatElements()*4));
    residualOffset=layout.OperandElements();
    qOffset=layout.Query();kOffset=layout.Key();vOffset=layout.Value();
    normOffset=layout.Norm();scoreOffset=layout.Score();pOffset=layout.Probability();
    pvOffset=layout.Pv();rotOffset=pvOffset;
"""
    if members > 1:
        compact += "    accStateOffset=layout.Acc();statStateOffset=layout.Stat();\n"
    if members == 4:
        compact += "    baseKOffset=kOffset;baseScoreOffset=scoreOffset;basePOffset=pOffset;\n"
        compact += "    extraKvOffset=layout.ExtraKv();extraScoreOffset=layout.ExtraScore();\n"
        compact += "    extraPOffset=layout.ExtraProbability();\n"
    if members > 1:
        compact += "    for(int32_t i=0;i<8;++i)counters[i]=0;\n"
    replace(text[begin:end], compact)
    replace("int64_t qOffset,kOffset,vOffset,scoreOffset,pOffset,pvOffset,rotOffset;",
            "int64_t qOffset,kOffset,vOffset,scoreOffset,pOffset,pvOffset,rotOffset,normOffset;")
    if members == 4:
        replace("scoreOffset=slot?extraScoreOffset:baseScoreOffset;pOffset=scoreOffset;",
                "scoreOffset=slot?extraScoreOffset:baseScoreOffset;\n"
                "    pOffset=slot?extraPOffset:basePOffset;")
        replace("int64_t baseKOffset,baseScoreOffset,extraKvOffset,extraScoreOffset;",
                "int64_t baseKOffset,baseScoreOffset,extraKvOffset,extraScoreOffset,basePOffset,extraPOffset;")
    for old, new in (
        ("DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D)",
         "DataCopy(work[normOffset+lane*kLaneRows*D],acc,kLaneRows*D)"),
        ("Matmul(qOffset,0,rotOffset,rows,D,D,true)",
         "Matmul(normOffset,0,rotOffset,rows,D,D,true)"),
    ):
        replace(old, new, text.count(old))
    replace("mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);",
            "oscar_bf16_history::InitMatmul(historyMatmulActive,pipe,historyMm);")

    # Archive #173: retain FP32 rotation output until the bounded Q publisher
    # splits high/low BF16. QK itself never takes an FP32 operand.

    # Exact rows keep the same per-row reduction and tag/error order, while
    # four adjacent rows share one Cast into otherwise-dead natural scratch.
    replace('#include "attention_bf16_history_operands.h"',
            '#include "attention_bf16_history_operands.h"\n'
            '#include "attention_bf16_precise_check.h"')
    natural_rows = "kHistoryHalfKv" if name == "member_pipeline_cluster4" else "kHalfKv"
    replace("      for(int32_t row=0;row<rows;++row) {\n"
            "        CheckBf16Row(dst[row*D]);\n      }",
            f"      oscar_bf16_precise_check::CheckRows<D,{natural_rows}>(\n"
            "          dst,rows,naturalBuf,scratchBuf,error);")
    replace("      for(int32_t row=0;row<rows;++row) {\n"
            "        if(tags.GetValue(row)!=inpage+row) {error=4;continue;}\n"
            "        CheckBf16Row(dst[(j+row)*D]);\n      }",
            f"      oscar_bf16_precise_check::CheckTaggedRows<D,{natural_rows}>(\n"
            "          dst[j*D],tags,inpage,rows,naturalBuf,scratchBuf,error);")
    replace(function_body("CheckBf16Row"),
            "  // Precise row validation uses the shared bounded four-row Cast helper.\n")

    if name == "mdl":
        replace("  __aicore__ void PipelineConsumePv() {\n",
                """  __aicore__ void PipelineConsumePv() {
    // Archive #147/#171, D.4: bounded KV/PV handoff, no full-history restore.
    // PrefetchValues publishes next V from dequantBuf on MTE3. This task's
    // current PVReady does not depend on that NEXT tile's KVReady, so MTE2
    // must not overwrite dequantBuf until its last outgoing read completes.
    // MTE3_V alone gates Vector, not this incoming DMA. No arithmetic,
    // allocation or host readback is added; one local dependency per KV tile.
    Fence<HardEvent::MTE3_MTE2>();
""")

    # Reverse the transformations rather than trusting copied output alone.
    inverse = text
    for old, new, count in reversed(edits):
        if inverse.count(new) != count:
            raise ValueError(f"inverse replacement is ambiguous in {mother}: {new[:100]}")
        inverse = inverse.replace(new, old)
    if inverse != source:
        raise ValueError(f"inverse identity failed: {mother}")
    prefix = HEADER.format(mother=mother, commit=MOTHER_COMMIT, sha=expected_sha)
    if name == "mdl":
        prefix += ("// Archive #147/#171 fix: next-V MTE3 read must retire before\n"
                   "// current-PV MTE2 overwrites the shared bounded UB tile.\n")
    return prefix + text, len(edits)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rows = []
    for name, mother, sha in SPECS:
        result, edits = derive(name, mother, sha)
        path = ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp"
        if args.check:
            if not path.is_file() or path.read_text() != result:
                raise ValueError(f"derived source differs: {path}")
        else:
            path.write_text(result)
        rows.append({"kernel": str(path.relative_to(ROOT)), "mother_sha256": sha,
                     "derived_sha256": hashlib.sha256(result.encode()).hexdigest(),
                     "inverse_identity": True, "transformations": edits})
    print(json.dumps({"mother_commit": MOTHER_COMMIT, "check": args.check,
                      "kernels": rows}, indent=2))


if __name__ == "__main__":
    main()
