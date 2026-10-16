"""Archive G4/#81-83/#126/#145/#148/#154/#157 and full startup D.4.

Generate independent research kernels from the exact 6029f2a mothers. Each
replacement is reversible; --check proves both final bytes and inverse identity.
Only source0 operand publication/matmul and private workspace/resources change.
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
// GENERATED RESEARCH KERNEL. CPU numerical gate FAILED; never production-qualified.
// Mother: {mother} at {commit}; SHA256 {sha}.
// Generate/check: tools/experiments/generate_bf16_history_kernels.py --check.
// Archive G4/#81-83/#126/#145/#148/#154/#157; full startup D.4 reread.
// D.4 four questions: (1) source0 fused INT2 dequant + QK/PV;
// (2) failed full-history restore cost6499.8-6655.1ms versus FIA18.5-18.9ms;
// (3) only fixed KV256 BF16 operand mirrors; FP32 softmax/LSE/state/Rv,
// exact source1/2, masks, errors, ownership and cross-core flags are retained;
// (4) GM capacity=1.5x mother, extra UB16KiB; casts and Matmul dtype-switch
// resets add cost, so no speed or target-numerical claim is made.
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
    replace(plane, plane + "\n  static constexpr int32_t kOriginalVectorBytes=\n      "
            + "+\n      ".join(f"({s})" for s in sizes) + ";\n"
            "  static_assert(kOriginalVectorBytes+oscar_bf16_history::kOperandScratchBytes<=184*1024,\n"
            '      "BF16 publisher exceeds A2 usable UB; do not relax this bound");')
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
            "  GlobalTensor<bfloat16_t> operandWork;\n"
            "  TBuf<TPosition::VECCALC> operandBuf;")

    sig = "  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,\n      bool rotation=false,bool transposeB=true) {"
    org = ("historyMm.SetOrgShape(m,transposeB?kKvRows:D,transposeB?D:kKvRows);"
           if name == "mdl" else "historyMm.SetOrgShape(m,n,k);")
    replace(sig, sig + "\n" + """    const bool history=kind==0 && !rotation;
    oscar_bf16_history::SelectMatmul(history,historyMatmulActive,pipe,mm,historyMm);
    if(history) {
      SetHF32Mode(false);
      """ + org + """
      historyMm.SetSingleShape(m,n,k);
      historyMm.SetTensorA(operandWork[a],false);
      historyMm.SetTensorB(operandWork[b],transposeB);
      historyMm.template IterateAll<true>(work[c]);
      historyMm.End();
      return;
    }""")
    if name == "member_pipeline_cluster4":
        atomic_sig = "  __aicore__ void MatmulAtomic(int64_t a,int64_t b,int64_t c,\n      int32_t m,int32_t n,int32_t k) {"
        replace(atomic_sig, atomic_sig + """
    oscar_bf16_history::SelectMatmul(kind==0,historyMatmulActive,pipe,mm,historyMm);
    if(kind==0) {
      SetHF32Mode(false);
      historyMm.SetOrgShape(m,n,k);historyMm.SetSingleShape(m,n,k);
      historyMm.SetTensorA(operandWork[a],false);historyMm.SetTensorB(operandWork[b],false);
      historyMm.template IterateAll<true>(work[c],static_cast<uint8_t>(1));
      historyMm.End();return;
    }""")
    # These are only input-Q/padding, unpacked K/V and P writes. Acc/state and
    # normalized Q-for-Rv writes deliberately remain the original float copies.
    for old, new in (
        ("DataCopy(work[qOffset+row*D],dst,D)", "PublishOperand(qOffset+row*D,dst,D)"),
        ("DataCopy(work[qOffset+row*D],zeros,count*D)", "PublishOperand(qOffset+row*D,zeros,count*D)"),
        ("DataCopy(work[(value?vOffset:kOffset)+slot*D],src,kElements)", "PublishOperand((value?vOffset:kOffset)+slot*D,src,kElements)"),
        ("DataCopy(work[pOffset+rowBase*kKvRows],scores,activeRows*kKvRows)", "PublishOperand(pOffset+rowBase*kKvRows,scores,activeRows*kKvRows)"),
    ):
        replace(old, new)
    if name == "mdl":
        replace("DataCopy(work[(value?vOffset:kOffset)+buffer*(2*kKvRows*D)+slot*D],\n        dequantBuf.Get<float>(),kElements)",
                "PublishOperand((value?vOffset:kOffset)+buffer*(2*kKvRows*D)+slot*D,\n        dequantBuf.Get<float>(),kElements)")
    helpers = """  __aicore__ void PublishOperand(int64_t offset,LocalTensor<float> source,int32_t elements) {
    if(kind==0)oscar_bf16_history::Publish(operandWork[offset],source,operandBuf,elements);
    else DataCopy(work[offset],source,elements);
  }
"""
    if name == "member_pipeline_cluster4":
        replace("DataCopy(work[(value?vOffset:kOffset)+slot*D],src,copy)",
                "PublishHistoryOperand((value?vOffset:kOffset)+slot*D,src,copy)")
        helpers += """  __aicore__ void PublishHistoryOperand(int64_t offset,LocalTensor<float> source,
      const DataCopyParams& originalCopy) {
    if(kind==0) {
      // Original two old-16-row lane domains, now measured in BF16 elements.
      PublishOperand(offset,source,kElements);
      PublishOperand(offset+2*kElements,source[kElements],kElements);
    } else DataCopy(work[offset],source,originalCopy);
  }
"""
    replace("  __aicore__ void LoadQueries() {", helpers + "  __aicore__ void LoadQueries() {")

    # Reverse the transformations rather than trusting copied output alone.
    inverse = text
    for old, new, count in reversed(edits):
        if inverse.count(new) != count:
            raise ValueError(f"inverse replacement is ambiguous in {mother}: {new[:100]}")
        inverse = inverse.replace(new, old)
    if inverse != source:
        raise ValueError(f"inverse identity failed: {mother}")
    prefix = HEADER.format(mother=mother, commit=MOTHER_COMMIT, sha=expected_sha)
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
