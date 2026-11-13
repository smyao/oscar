"""Archive #145/#150/#151/#154/#164/#172: bounded C4 B1 reuse.

D.4(1): fused QK/PV operand reuse inside the existing KV256 tile.
D.4(2): no full-history dequant (old6499.8--6655.1ms), no writer expansion.
D.4(3): copy the protected C16 single-B1 implementation exactly; retain C4
math, Vector methods, independent P/state, two score slots and two-AIV flags.
D.4(4): one128KiB B1 plus the SAME C16 SDK reserve assertion; no second SDK
instance or extra GM/UB. Work reduction is not measured HBM/NPU performance.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from tools.cluster_state_transform import method

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = "f6f44d4aa0c5420a3cfaa6b0b394340a66dc0b06d22fdf29a7bde28cd3cbea44"
C16_SHA256 = "1bcf9ccfe0181f3103b255f67589845a9d83b4e232baa9aa11561b61326d8264"
HEADER = """// C4_SINGLE_B1_REUSE_V1
// Archive #145/#150/#151/#154/#164/#172; complete startup D.4 reviewed.
// D.4(1): bounded C4 QK/PV operand reuse; two score slots, four P/state slots.
// D.4(2): no full-history restoration (old6499.8--6655.1ms) or writer change.
// D.4(3): exact protected C16 B1 implementation and SDK tiling. Wait both
// AIV PReady owners before score reuse or switching K to V; keep round/final
// completion and all Vector math. No BF16/FP32 arithmetic or error change.
// D.4(4): one128KiB B1 and one SDK Matmul instance; the existing C16 reserve
// assertion remains. No extra GM/UB, no measured speed/HBM claim.
// C1 CubeTask/VectorTask are unchanged, but their Matmul B transport now uses
// that same C16 B1 wrapper. C1 still loads one B per QK/PV; it needs device QA.
"""
OLD_INSTANCE = "  TPipe pipe;CubeMatmul mm;"
NEW_INSTANCE = OLD_INSTANCE + "TBuf<TPosition::B1> residentB;"
OLD_INIT = "      mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);"
NEW_INIT = "      pipe.InitBuffer(residentB,128*1024);\n" + OLD_INIT
CLUSTER = """  __aicore__ void CubeCluster(const TaskState* members) {
    const int64_t first=members[0].kvbegin,last=members[0].kvend;
    const int32_t rows=static_cast<int32_t>(members[0].qcount*(g.hq/g.hk));
    int32_t buffer=0;
    for(int64_t start=first;start<last;start+=kKvRows,buffer^=1) {
      CrossCoreWaitFlag(kClusterKvReady);
      SelectClusterSlots(buffer,0);LoadResidentB(kOffset);
      for(int32_t member=0;member<4;++member) {
        const int32_t slot=member%2;
        // P(i-2) is published only after BOTH AIVs consumed Score(i-2).
        // Each member's P lives in its own Q-slot second half, not score.
        if(member>=2)CrossCoreWaitFlag(kClusterP0+slot);
        SelectClusterSlots(buffer,slot);
        MatmulResidentB(member*kQueryRows*D,scoreOffset,rows,kKvRows,D);
        CrossCoreSetFlag<2,PIPE_FIX>(kClusterScore0+slot);
      }
      for(int32_t slot=0;slot<2;++slot)CrossCoreWaitFlag(kClusterP0+slot);
      // All four P/alpha states are published. The copied C16 B1 loader
      // waits for every old MTE1 read before MTE2 may replace K with V.
      SelectClusterSlots(buffer,0);LoadResidentB(vOffset);
      for(int32_t member=0;member<4;++member) {
        MatmulResidentB(ProbabilityOffset(member*kQueryRows*D),
            accStateOffset+member*kQueryRows*D,rows,D,kKvRows,false,static_cast<uint8_t>(1));
      }
      CrossCoreSetFlag<2,PIPE_FIX>(kClusterRoundDone);
    }
    // Acknowledge each completed publication before this flag can be reused.
    for(int32_t member=0;member<4;++member) {
      CrossCoreWaitFlag(kVectorReady);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
    }
  }"""


def _config(text: str) -> str:
    begin = text.index("using Matrix=MatmulType<")
    end = text.index("using CubeMatmul=", begin)
    return text[begin:text.index(";", end) + 1]


def _c16() -> str:
    source = (ROOT / "csrc/kernels/attention_cv_official_cluster16.cpp").read_text()
    if sha256(source.encode()).hexdigest() != C16_SHA256:
        raise ValueError("C4 L1 requires the exact protected C16 B1 source")
    return source


def _baseline(text: str) -> None:
    if sha256(text.encode()).hexdigest() != BASELINE_SHA256:
        raise ValueError("C4 L1 requires the exact b49a689 C4 active source")


def _replacements(original: str) -> tuple[tuple[str, str], ...]:
    c16 = _c16()
    # The complete C16 config carries its audited M64/N64/K128 and single-B1
    # L1 capacity assertions. All transport functions are copied byte-for-byte.
    resident = "\n".join(method(c16, name) for name in
                         ("LoadResidentB", "MatmulResidentB", "Matmul"))
    return ((_config(original), _config(c16)),
            (method(original, "Matmul"), resident),
            (method(original, "MatmulAtomic"), method(c16, "MatmulAtomic")),
            (method(original, "CubeCluster"), CLUSTER),
            (OLD_INSTANCE, NEW_INSTANCE), (OLD_INIT, NEW_INIT))


def transform(text: str) -> str:
    _baseline(text)
    for old, new in _replacements(text):
        if text.count(old) != 1:
            raise ValueError("C4 L1 source replacement anchor drift")
        text = text.replace(old, new, 1)
    return HEADER + text


def restore(text: str) -> str:
    if not text.startswith(HEADER):
        raise ValueError("C4 L1 derived header drift")
    # Reconstruct the pinned baseline independently of the mutable generated
    # active file, which the owner generator replaces with this derivative.
    from tools import cluster_visibility_transform as visible
    from tools import precise_finite_transform as finite
    from tools import precise_zero_fill_transform as zero
    from tools import cluster_state_transform as state
    mother = (ROOT / "csrc/kernels/attention_cv_official_cluster4.cpp").read_text()
    original = state.transform(zero.transform(finite.transform(visible.transform(mother))))
    _baseline(original)
    restored = text.removeprefix(HEADER)
    for old, new in reversed(_replacements(original)):
        if restored.count(new) != 1:
            raise ValueError("C4 L1 derived replacement drift")
        restored = restored.replace(new, old, 1)
    _baseline(restored)
    if transform(restored) != text:
        raise ValueError("C4 L1 inverse is not exact")
    return restored
