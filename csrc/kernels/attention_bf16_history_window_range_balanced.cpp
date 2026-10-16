// SPDX-License-Identifier: Apache-2.0
// GENERATED UNIFIED BF16 KERNEL. Default routing is not numerical acceptance.
// Mother: attention_cv_window_range_balanced.cpp at 6029f2a8c6277ab8136e5a5eeaa2c2c16313ca08; SHA256 1bae967a1b9e0126e627b5bf0c29e6b9799f6202d12a3bc373c2455ea00ca013.
// Generate/check: tools/experiments/generate_bf16_history_kernels.py --check.
// Archive G4/#81-83/#126/#145/#148/#154/#157; full startup D.4 reread.
// D.4 four questions: (1) direct INT2-to-BF16 and all-source BF16 QK/PV;
// (2) failed full-history restore cost6499.8-6655.1ms versus FIA18.5-18.9ms;
// (3) fixed KV256 BF16 operands, exact BF16 window/current K/V; FP32
// softmax/LSE/state/Rv, masks, error order and cross-core flags are retained;
// (4) GM capacity=1.5x mother, extra UB16KiB (+512B on H16); Matmul switches,
// resets and bounded unpack lookup add cost; no speed/acceptance claim.
// One active Matmul resource set avoids simultaneous L1/L0 allocations.
// Re-Init/Reset needs actual SDK and device verification; host has neither.
// The mother's inherited historical comments below are not claims about
// this BF16 candidate. The executable delta and independent gates govern it.
// Isolated actual striped candidate clone. Archive #126/#129/#140/#145/#148/#151/#154/#155.
// D.4: full restore cost6499.8-6655.1ms is not repeated. This changes bounded
// window reads and score-mask writes only, with original FP32 history math,
// source/group boundaries and statuses. No current q4 gain is assumed here.
#include "attention_fast_unpack.h"
#include "attention_striped_unpack.h"
#define oscar_ascend_fast oscar_ascend_striped
#define OSCAR_STRIPED_D256_ONLY 1
#define attention_cv_fast_balanced_launch attention_bf16_history_window_range_balanced_launch
#define oscar_attention_cv_fast_balanced_kernel oscar_attention_bf16_history_window_range_balanced_kernel
// SPDX-License-Identifier: Apache-2.0
// EXPERIMENT ONLY: independent mixed-batch task-owner schedule. Archive
// #126/#129/#140/#143-151 and startup D.4 four questions:
// (1) This changes ownership inside the fused history/window/current FIA
//     phase only; prepare, Q/K/V arithmetic, source masks, status and stores
//     remain the fast CV kernel's byte-for-byte logic.
// (2) D.4's failed full-history restore took 6499.8-6655.1ms/device;
//     phase1 stores took ~209ms and host prepare ~725ms. Never restore a
//     history tensor, move task scheduling to Python, or mask task errors.
// (3) A four-token scheduling tile, independent of the unchanged 21-token
//     GQA6 mathematical query group, distributes 31 q4 history leaders in
//     N16384/S1 from 6 to 20 Cubes. Every token/source/split task is still
//     loaded exactly once; a long leader keeps its full qcount and ordered
//     KV256/FP32 online state. Bounded UB and GM workspace are unchanged.
// (4) N16384/S1 outer work items rise from 2343 to 12288, while inner task
//     loads remain 49152. This is a parallelism hypothesis, not a device
//     latency or full-model speed claim; frozen output/LSE/status, graph and
//     paired target-NPU timing must decide whether the extra scheduling pays.
// EXPERIMENT ONLY: exact INT2 unpack for the fe0 query-major schedule. Archive
// G26-G34/#13-20/#126/#129/#140-151; startup D.4 four questions:
// (1) fused source0 dequant+FIA; (2) failed full-history restore cost
// 6499.8-6655.1ms vs native FIA 18.5-18.9ms; (3) one bounded KV256 work
// unit remains, FP32 QK/PV/softmax, masks, checks, flags and output unchanged;
// only integer-plane UB stride and exact FP16 metadata conversion change;
// (4) unpack work was measured hot but bank relief/net speed is a target-NPU
// hypothesis. Frozen bitwise/math/graph/quality gates precede any speed claim.
// Archive G3-G13/#3/#21/#81-83/#87-92: native A2 interfaces, bounded UB,
// primitive launch ABI, signed conversions and explicit event dependencies.
// Archive G26-G34/#4-20/#53-69: FP32 QK/PV/softmax accumulation, exact INT2
// layout, per-query causal masks, all output/LSE rows written, checked tags.
// Archive #126: last valid Q row must finish MTE3 reads before V zeroes the
// reused UB for padding. MTE3_MTE2 alone cannot order a V-only padding path.
// D.4 four questions: this replaces dequant+FIA and precise-window FIA.
// Prior full-history restore cost 6499.8-6655.1ms vs FIA 18.5-18.9ms at 32K.
// Here Vector unpacks one bounded 256-token work unit in eight 32-token
// subtiles with SIMD Gather/Shift/And/Cast, keeping each packed read single-use;
// Cube computes QK, PV and final Rv^T. Up to 128 GQA/query rows reuse each
// compressed read. No full-history tensor or history inverse rotation.
// A2 uses bounded per-Cube GM tile communication
// (not an on-chip-only claim): capacity (768*D+32768)*4 B/core, total traffic
// remains linear in history. D256 explicit UB is 157632B/AIV, below the
// A2 184KiB usable vector-local limit.
// Target: short step <=0.6-1.1ms and 32K same-order as native FIA; NO target
// precision/latency is claimed before fixed-tolerance NPU/profiler validation.
// #126 adds one local MTE3->V dependency before padding writes; tile storage,
// arithmetic and history traffic stay unchanged (no CPU/global sync or retry).
// #129/D.4: schedule consecutive query tiles within each source/head/split.
// Raw token-task striding aliases GQA6 leaders (stride30) onto 2/20 Cubes.
// This changes ownership only: fixed scratch, identical CV math, no fallback;
// causal task bounds avoid future-only work. NPU latency remains unverified.
// #130/D.4: batch contiguous KV rows into strided DMA instead of per-token
// table/DMA/fence calls. History runs stop at virtual128 boundaries; current
// BF16 uses one DMA+Cast per half-tile. Reuse existing UB, preserve FP32 math,
// metadata/finite checks and GQA reuse. No full-history allocation or restore;
// target remains native FIA's 0.6-1.1ms/18.5-18.9ms order, not a claimed result.
// #134/D.4 (this change): (1) phase=fia. (2) Failure mode: per-element scalar
// SetValue/GetValue masking+finite loops (~512/tile/lane) made the per-tile
// cost latency-bound; measured fia p95 7.166s at 49K KV on the target.
// (3) Structural avoidance: the row-independent LogicalPosition map is hoisted
// per tile; each row's visible set is a union of at most two index intervals;
// only masked slots are scalar-written, and non-finite coverage is one aligned
// whole-row ReduceSum (masked slots are finite by construction since inputs
// were checked at load). Softmax FP32 math, -inf placement, LSE, tiling, UB
// budget and workspace layout are all unchanged; no fallback or history restore.
// (4) Expected: per-tile UB element write/read drops from ~512 slots to the
// masked-slot count (zero on fully visible rows); the remaining interval scan
// is 32 register-level compares per row. p95 must measurably fall on the
// target TIMING_SUMMARY before any further pipelining; parity with native FIA
// is NOT claimed by this change alone.
// #126/#129-135/#140/D.4 short-query follow-up: (1) This is only the fused
// fia softmax phase for each 32-KV tile; prepare, phase1 stores and the INT2
// unpack/Cube QK/PV paths are unchanged. (2) D.4's old full-history dequant
// took 6499.8-6655.1ms versus FIA 18.5-18.9ms at 32K; stores took ~209ms
// and host prepare ~725ms. Do not reintroduce restoration or host work.
// (3) AIV still unpacks its KV half and follows every Cube/Vector flag, but
// softmax visits only real query rows. Padded score rows are bulk-zeroed once
// before publishing the same full P tile; all statuses, LSE/output owners and
// FP32 arithmetic for live rows remain unchanged. (4) GQA6 q_len=1/4 uses
// 6/24 of 64 available rows: scalar softmax row visits fall by 90.6%/62.5%
// per KV tile, not the QK/PV or INT2 work. This is a source-work prediction,
// not a target-NPU speed or precision claim; frozen oracle and graph probes
// must decide whether it improves the measured #140 12.4x K4 wall gap.
// #126/#129-135/#140/D.4 prior 128-KV follow-up: (1) source0 fused dequant+fia
// and source1/2 precise fia share this work unit. (2) D.4's old full-history
// restore cost 6499.8-6655.1ms vs native FIA 18.5-18.9ms at 32K;
// phase1_stores ~209ms and host prepare ~725ms were separate failures.
// (3) Four 32-KV subtiles assemble one bounded 128-KV K/V tile in GM, with
// no [history,D] tensor. Score is reused as P only after Cube QK completes;
// PV is reused for final rotation only after Vector consumes it. All AIVs
// retain the three cross-core handshakes per work unit. FP32 QK/PV/online
// softmax, page/causal/window masks, MTP positions and status remain intact.
// (4) QK/PV MatmulImpl calls and flag rounds fall up to 4x per long sequence
// (16K first-chunk current source: 420761 -> 105805 work units per FULL layer,
// static task formula). D256 GM capacity is 425984 B/core; AIV score UB grows
// by 12KiB. Neither number is a speed claim; exact output/LSE tolerance,
// CANN build, device completion, graph replay and paired K4 remain gates.
// #140/D.4 analytic masks: (1) fused fia row masking; (2) the prior 6.5s
// full-history dequant and this kernel's 128-candidate scalar scan per query
// row are unacceptable long-context work; (3) derive the exact visible
// prefix/suffix intervals from causal/sink/recent boundaries in O(1), without
// changing QK/PV, FP32 softmax or history bytes; (4) for GQA6 a full query
// tile avoids up to 60*128 scalar visibility tests per KV unit. SDK/device
// timing, frozen accuracy and graph replay decide the actual benefit.
// #140/D.4 Cube FP32 tiling: (1) source0/history and source2/current fused
// fia use the same QK/PV MatmulImpl; (2) the old D.4 full dequant was ~6.5s,
// while this kernel's basic16x32x32 split QK/PV into many small MMADs with
// a PIPE_M barrier when the SDK's tile-area threshold is missed. (3) Keep
// SetHF32Mode(false), FP32 buffers and the exact online softmax; use an A2
// L0-safe basic64x64x128 (32KiB A/B each with double buffering, 16KiB C),
// with no history materialization. (4) At M≈60, QK/PV nominal MMAD counts
// per 128-KV work unit fall from 128 each to 4 each; q_len1 tails retain
// smaller actual M and may still barrier. CANN compile and target timing,
// not this arithmetic count, decide whether the #140 9x gap closes.
// #140/D.4 metadata arithmetic: (1) source0 fused dequant+fia; (2) old
// full-history restore cost ~6.5s and per-row Muls/Adds here launch 32
// arithmetic vectors per 16-row K/V subtile. (3) Preserve exact FP16 metadata
// conversion/validation, but Brcb each of 16 FP32 scalars into one 32B block
// and apply one row-strided FP32 Mul then Add per 64D chunk; no history tensor,
// HF32 relaxation or new UB. (4) At D256, arithmetic calls fall 32 -> 8 per
// K/V subtile (plus two Brcb); native A2 hc_pre_base.h uses this exact stride
// pattern. CANN compile, frozen oracle and target device timing must verify.
// #140/D.4 online-softmax follow-up: (1) the AIV score/P phase follows each
// bounded Cube QK work unit. (2) Per-row ReduceMax/Exp/ReduceSum and scalar
// fences serialize up to 32 query rows after Cube tiling improved. (3) CANN
// A2 SoftmaxFlashV2 updates live FP32 rows together, reusing dequant UB only
// after K/V publication; a finite old-max sentinel preserves empty rows.
// Alpha is Brcb-multiplied across accumulator rows in the dead natural UB;
// exact causal masks and FP32 online sum remain. (4) It removes per-row
// reductions/alpha fences, not history bytes or the Cube/Vector flags;
// frozen oracle, real CANN/CPU-debug and target timing still decide speed.
// #143/D.4 larger query reuse and natural V: (1) fused history/window FIA;
// (2) D.4's full-history dequant took 6499.8-6655.1ms while native FIA was
// 18.5-18.9ms, and the current #143 synchronized long-context prefill showed
// about 494ms/CV layer versus about 6.6ms for a no-old-context candidate.
// (3) Preserve every INT2/window/current mask and FP32 online state, but let
// 21 GQA6 queries share one 256-KV unit. Keep each lane's 64-row FP32 running
// accumulator in UB, process its score as two 32-row V2 blocks, and stream PV
// back through dead dequant UB. V stays natural [KV,D] in GM; the SDK/native
// MatmulImpl permits runtime SetTensorB(false) on a transpose-capable B type.
// This removes a per-16-row Gather and a D-block strided 64-byte GM write.
// No [history,D] HBM tensor, precision relaxation or native fallback exists.
// (4) For a long GQA6 query span, query groups shrink from ceil(Q/10) to
// ceil(Q/21); doubling KV width then cuts cross-core rounds and MatmulImpl
// calls by another factor of about two, while packed reads/unpacks shrink only
// with query groups. At D256 the V publish changes 256 scattered 64-byte
// blocks per 16-row subtile to one contiguous 16KiB copy. These are source
// work counts, not a measured 3-5x speedup: CANN/CPU-debug, frozen target NPU
// accuracy, graph replay and paired K4 timing remain mandatory gates.
// #143/D.4 short-query P tail: (1) only the bounded FIA score/P publish;
// (2) writing four padded 32x256 P blocks for a six-row decode would add GM
// traffic after the old 6.5s restore failure. (3) CANN SetSingleShape keeps
// PV's M equal to the real row count, so empty P blocks are never consumed;
// V2 still initializes its local padded rows and the three cross-core flags
// remain unchanged. (4) At GQA6 q_len1, three empty 32-row P writes vanish
// per 256-KV unit and the live block publishes only six rows. Real target
// graph timing and frozen accuracy, not this byte count, decide acceptance.
// #145/D.4 score-buffer ownership: (1) fused FIA softmax/P publication;
// (2) Q6/GQA6 first reuses the same UB for a second score block, exposing
// an MTE3-read/MTE2-write race. (3) Order P copy-out before the next score
// copy-in with MTE3_MTE2; that copy-in's MTE2_V also protects later Vector
// writes. Keep bounded tiles, FP32 math and all cross-core flags unchanged.
// (4) Replace the incorrectly directed local event, without adding a global
// sync or history traffic; this fixes ownership, not a claimed speedup.
// #143/D.4 INT2 plane sequencing: (1) source0 Unpack inside fused fia;
// (2) historical full-history dequant took ~6.5s while native FIA was ~18.7ms,
// and target K4 events now place most prefill time in CV. (3) Eight independent
// ShiftRight writes share one PIPE_V barrier, then eight disjoint in-place And
// writes share one barrier before Gather; exact two-bit positions, FP32 scales
// and single-use packed reads are unchanged. (4) This removes 14 vector
// barriers per K/V unpack, 112 per 128-KV unit; target speed remains unclaimed
// until frozen oracle, CANN CPU-debug and target NPU timing all pass.
// Native precedents: hc_pre_m_k_split_core.h mode-2 Cube/Vector flags;
// hc_pre_cube_compute.h FP32 Mmad; moe_grouped_matmul.h MatmulImpl;
// add_rms_norm_bias_multi_n.h Gather; PR triton_oscar_decode.py INT2 and LSE.
#include "oscar_common.h"
#include "attention_bf16_history_operands.h"
#include "attention_striped_unpack_bf16.h"
#include "../include/oscar_bf16_history_launch.h"
#include "attention_range_mask.h"
#include "attention_fast_unpack.h"
#include "../include/oscar_attention_launch.h"
#define OSCAR_SCHEDULE_FN __aicore__ inline
#include "../include/oscar_cv_schedule.h"
#undef OSCAR_SCHEDULE_FN
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
#include "adv_api/activation/softmaxflashv2.h"
using namespace oscar_ascend_device;
namespace {
constexpr int32_t kQueryRows=oscar_ascend::kAttentionQueryRows;
constexpr int32_t kKvRows=oscar_ascend::kAttentionKvRows;
constexpr int32_t kHalfRows=32, kHalfKv=16;
constexpr int32_t kLaneRows=kQueryRows/2, kRowBlocks=kLaneRows/kHalfRows;
constexpr int32_t kKvSubtiles=kKvRows/(2*kHalfKv);
constexpr uint16_t kVectorReady=8, kCubeReady=9;
constexpr float kNegativeInfinity=-__builtin_inff();
constexpr float kEmptyMax=-__FLT_MAX__;
constexpr SoftmaxConfig kCvSoftmaxConfig={false,0,0,SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
__aicore__ inline int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
__aicore__ inline int64_t Max64(int64_t a,int64_t b){return a>b?a:b;}
using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;
__aicore__ constexpr MatmulConfig CvConfig() {
  auto cfg=GetNormalConfig();
  cfg.basicM=64;cfg.basicN=64;cfg.basicK=128;
  cfg.singleCoreM=128;cfg.singleCoreN=256;cfg.singleCoreK=256;
  cfg.enableSetBias=false;
  return cfg;
}
constexpr auto kCubeConfig=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(CvConfig());
using CubeMatmul=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kCubeConfig>;
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


struct Geometry {
  int64_t tokens,hq,hk,requests,columns,taskCount,blockTokens,blocks,ssmOffset;
  int64_t pageStride,windowStride,tagStride,sink,recent,speculative,splits;
  float scale;
};

template<int32_t D> class AttentionCv {
  static constexpr int32_t kSlotBytes=D/2+8;
  static constexpr int32_t kPackedStride=(kSlotBytes+31)/32*32;
  static constexpr int32_t kElements=kHalfKv*D;
  static constexpr int32_t kWords=kElements/8;
  static constexpr int32_t kPlaneStride=kWords+16;
  static constexpr int32_t kOriginalVectorBytes=
      (128)+
      (kHalfKv*kPackedStride)+
      (oscar_ascend_fast::Geometry<D>::planeBytes)+
      (kElements*2)+
      (kWords*2)+
      (kWords*2)+
      (kWords*4)+
      (kElements*4)+
      (128)+
      (64)+
      (kElements*4)+
      (kHalfRows*kKvRows*4)+
      (kLaneRows*D*4)+
      (D*4)+
      (D*2)+
      ((2*kLaneRows+2*kHalfRows)*4)+
      (512)+
      (128);
  static constexpr int32_t kNaturalPaddingBytes=512;
  static_assert(kOriginalVectorBytes+kNaturalPaddingBytes+oscar_bf16_history::kOperandScratchBytes<=184*1024,
      "BF16 publisher exceeds A2 usable UB; do not relax this bound");
 public:
  __aicore__ void Init(GM_ADDR query,GM_ADDR queryRot,GM_ADDR key,GM_ADDR value,
      GM_ADDR rotation,GM_ADDR raw,GM_ADDR table,GM_ADDR windowKey,
      GM_ADDR windowValue,GM_ADDR tags,GM_ADDR tasks,GM_ADDR partial,
      GM_ADDR lse,GM_ADDR status,GM_ADDR workspace,const Geometry& geometry) {
    g=geometry;
    q.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(query));
    qr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(queryRot));
    ck.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key));
    cv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(value));
    rv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotation));
    bytes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(raw));
    bt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
    wk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(windowKey));
    wv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(windowValue));
    wt.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tags));
    taskGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tasks));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));
    outLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
    outStatus.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    int64_t core=GetBlockIdx();
    if ASCEND_IS_AIV {lane=core%2;core/=2;}
    coreIndex=core;
    const int64_t floatsPerCore=oscar_ascend::attention_workspace_per_core(D)/4;
    // One private region per Cube: unchanged float arena followed by its
    // index-preserving BF16 mirror. Result/state/Rv never alias that mirror.
    auto coreWorkspace=workspace+core*floatsPerCore*6;
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(coreWorkspace));
    operandWork.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
        coreWorkspace+floatsPerCore*4));
    // Q(128D), K(256D), natural V(256D), score/P(128*256),
    // PV/rotation(128D). Query accumulator stays in UB across KV units.
    // QK is complete before Vector overwrites score with P; PV is consumed
    // before the final Rv rotation reuses that same GM tile. No active aliases.
    qOffset=0;kOffset=kQueryRows*D;vOffset=(kQueryRows+kKvRows)*D;
    scoreOffset=(kQueryRows+2*kKvRows)*D;
    pOffset=scoreOffset;pvOffset=pOffset+kQueryRows*kKvRows;rotOffset=pvOffset;
    if ASCEND_IS_AIC {
      // Static tiling is compiler-derived by the real SDK. SetOrg/SingleShape
      // below specify each QK/PV/rotation matrix, never hand-written tiling POD.
      mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);
      SetHF32Mode(false);
    } else {
      pipe.InitBuffer(taskBuf,128);pipe.InitBuffer(packedBuf,kHalfKv*kPackedStride);
      pipe.InitBuffer(planeBuf,oscar_ascend_fast::Geometry<D>::planeBytes);
      pipe.InitBuffer(naturalBuf,kHalfKv*(D+16)*2);
      pipe.InitBuffer(wordBuf,kWords*2);pipe.InitBuffer(maskBuf,kWords*2);
      pipe.InitBuffer(wordIndexBuf,kWords*4);pipe.InitBuffer(laneIndexBuf,kElements*4);
      pipe.InitBuffer(metadataIndexBuf,128);pipe.InitBuffer(metadataHalfBuf,64);
      pipe.InitBuffer(dequantBuf,kElements*4);
      pipe.InitBuffer(scoreBuf,kHalfRows*kKvRows*4);
      pipe.InitBuffer(accBuf,kLaneRows*D*4);
      pipe.InitBuffer(qFloatBuf,D*4);pipe.InitBuffer(qBf16Buf,D*2);
      pipe.InitBuffer(statsBuf,(2*kLaneRows+2*kHalfRows)*4);pipe.InitBuffer(scratchBuf,512);
      pipe.InitBuffer(addressBuf,128);
      pipe.InitBuffer(operandBuf,oscar_bf16_history::kOperandScratchBytes);
      InitIndices();
    }
  }
  __aicore__ void Process() {
    // Scalar Cube metadata reads must invalidate stale cache on changed-input
    // graph replay. The Vector side exclusively DMA-loads mutable metadata.
    if ASCEND_IS_AIC {
      DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>(taskGm);
    }
    // This is an owner tile, not the task's mathematical query group.
    // Prepare still permits qcount<=kQueryRows/(hq/hk); a 21-token leader
    // may cross several four-token owner tiles, whose followers only write
    // their own status. Do not split the leader or its ordered KV traversal.
    constexpr int64_t scheduleTile=4;
    const int64_t perToken=g.hk*3*g.splits;
    const oscar_ascend_schedule::CvTaskSchedule schedule{g.tokens,scheduleTile,perToken};
    // A work item owns one global token tile and one source/head/split.
    // Scan every slot in it: request boundaries/padding holes may put the
    // active leader anywhere, so no task is discarded or counted twice.
    for(int64_t workId=coreIndex;workId<schedule.WorkItems();workId+=GetBlockNum()) {
      const int64_t tokenBegin=schedule.TokenBegin(workId);
      for(int64_t token=tokenBegin;token<Min64(tokenBegin+scheduleTile,g.tokens);++token) {
        const int64_t id=schedule.TaskId(workId,token);
        LoadTask(id);
        if(qcount==0) {if ASCEND_IS_AIV {PublishStatus(id,0);} continue;}
        if(qcount<0 || taskError || !TaskValid()) {
          if ASCEND_IS_AIV {PublishEmpty(id,taskError?taskError:(!TaskValid()?1:0));}
          continue;
        }
        if ASCEND_IS_AIC {CubeTask();}
        else {VectorTask(id);}
      }
    }
    if ASCEND_IS_AIC {if(historyMatmulActive)historyMm.End();else mm.End();}
  }
 private:
  __aicore__ void LoadTask(int64_t id) {
    int64_t data[11];
    if ASCEND_IS_AIC {for(int32_t i=0;i<11;++i)data[i]=taskGm.GetValue(id*16+i);}
    else {
      auto row=taskBuf.Get<int64_t>();DataCopy(row,taskGm[id*16],16);
      Fence<HardEvent::MTE2_S>();
      for(int32_t i=0;i<11;++i)data[i]=row.GetValue(i);
    }
    qbegin=data[0];qcount=data[1];kvhead=data[2];kvbegin=data[3];kvend=data[4];
    request=data[5];split=data[6];kind=data[7];context=data[8];requestBegin=data[9];
    taskError=static_cast<int32_t>(data[10]);
  }
  __aicore__ bool TaskValid() {
    return qbegin>=0 && qbegin<g.tokens && qcount<=kQueryRows/(g.hq/g.hk) &&
        (qcount<0 || qbegin+qcount<=g.tokens) && kvhead>=0 && kvhead<g.hk &&
        kvbegin>=0 && kvend>=kvbegin && request>=0 && request<g.requests &&
        split>=0 && split<3*g.splits && kind>=0 && kind<3 && context>=0 &&
        requestBegin>=0 && requestBegin<=qbegin &&
        (kind!=0 || kvend<=context) &&
        (kind!=2 || (kvbegin>=context && requestBegin+kvend-context<=g.tokens));
  }
  __aicore__ void InitIndices() {
    oscar_ascend_striped_bf16::InitIndices<D,kHalfKv>(wordIndexBuf,laneIndexBuf,
        metadataIndexBuf,maskBuf);
  }
  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,
      bool rotation=false,bool transposeB=true) {
    const bool history=!rotation;
    oscar_bf16_history::SelectMatmul(history,historyMatmulActive,pipe,mm,historyMm);
    if(history) {
      SetHF32Mode(false);
      historyMm.SetOrgShape(m,n,k);
      historyMm.SetSingleShape(m,n,k);
      historyMm.SetTensorA(operandWork[a],false);
      historyMm.SetTensorB(operandWork[b],transposeB);
      historyMm.template IterateAll<true>(work[c]);
      historyMm.End();
      return;
    }
    SetHF32Mode(false); // no implicit HF32/TF32 relaxation of the frozen oracle.
    mm.SetOrgShape(m,n,k);mm.SetSingleShape(m,n,k);
    mm.SetTensorA(work[a],false);
    mm.SetTensorB(rv,true); // The only FP32 matmul is inverse rotation.
    mm.template IterateAll<true>(work[c]);
    // End closes the iteration state; buffers remain initialized for reuse.
    mm.End();
  }
  __aicore__ void CubeTask() {
    const int32_t rows=static_cast<int32_t>(qcount*(g.hq/g.hk));
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows) {
      CrossCoreWaitFlag(kVectorReady);
      Matmul(qOffset,kOffset,scoreOffset,rows,kKvRows,D);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
      Matmul(pOffset,vOffset,pvOffset,rows,D,kKvRows,false,false);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      // Vector consumes PV before the next tile may overwrite shared buffers.
      CrossCoreWaitFlag(kVectorReady);
    }
    if(kind==0 && kvbegin<kvend) {
      CrossCoreWaitFlag(kVectorReady);
      Matmul(qOffset,0,rotOffset,rows,D,D,true);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
    }
  }
  __aicore__ void PublishOperand(int64_t offset,LocalTensor<float> source,int32_t elements) {
    oscar_bf16_history::Publish(operandWork[offset],source,operandBuf,elements);
  }
  __aicore__ void PublishBf16Operand(int64_t offset,LocalTensor<bfloat16_t> source,
      int32_t elements) {
    DataCopy(operandWork[offset],source,elements);
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
  __aicore__ void LoadQueries() {
    auto dst=qFloatBuf.Get<float>();auto src=qBf16Buf.Get<bfloat16_t>();
    const int64_t group=g.hq/g.hk;
    const int64_t rows=qcount*group;
    const int32_t laneBegin=lane*kLaneRows,laneEnd=(lane+1)*kLaneRows;
    const int32_t validEnd=static_cast<int32_t>(Min64(laneEnd,Max64(laneBegin,rows)));
    for(int32_t row=laneBegin;row<validEnd;++row) {
      const int64_t off=((qbegin+row/group)*g.hq+kvhead*group+row%group)*D;
      if(kind==0) {DataCopy(dst,qr[off],D);Fence<HardEvent::MTE2_V>();}
      else {DataCopy(src,q[off],D);Fence<HardEvent::MTE2_V>();
        Cast(dst,src,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();}
      auto check=scratchBuf.Get<float>();
      ReduceSum(check,dst,check[16],D);Fence<HardEvent::V_S>();
      if(!Finite(check.GetValue(0)))error=2;
      Fence<HardEvent::V_MTE3>();
      if(kind==0)PublishOperand(qOffset+row*D,dst,D);
      else PublishBf16Operand(qOffset+row*D,src,D);Fence<HardEvent::MTE3_MTE2>();
    }
    // Pad the inactive query rows in bounded contiguous pieces. This keeps
    // the full initialized Q tile required by the Cube contract without
    // adding one scalar/GM copy per padded row to q_len=1/4 graph replay.
    // The previous live-row MTE3 must finish before Vector reuses the scratch.
    auto zeros=dequantBuf.Get<bfloat16_t>();
    Fence<HardEvent::MTE3_V>();
    for(int32_t row=validEnd;row<laneEnd;row+=kHalfKv) {
      const int32_t count=static_cast<int32_t>(Min64(kHalfKv,laneEnd-row));
      Duplicate(zeros.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),count*D);Fence<HardEvent::V_MTE3>();
      PublishBf16Operand(qOffset+row*D,zeros,count*D);Fence<HardEvent::MTE3_V>();
    }
  }
  __aicore__ int64_t LogicalPosition(int64_t candidate) {
    if(kind!=1) return candidate;
    const int64_t sinkCount=Min64(g.sink,context);
    const int64_t firstPosition=context+qbegin-requestBegin;
    const int64_t tailBegin=Min64(context,Max64(g.sink,firstPosition+1-g.recent));
    return candidate<sinkCount?candidate:tailBegin+candidate-sinkCount;
  }
  __aicore__ bool Physical(int64_t position,int64_t& block,int64_t& inpage) {
    if(position<0 || position/128>=g.columns) {error=1;return false;}
    auto local=addressBuf.Get<int32_t>();
    DataCopyExtParams copy{1,4,0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(local,bt[request*g.columns+position/128],copy,pad);
    Fence<HardEvent::MTE2_S>();
    const int64_t virtualBlock=local.GetValue(0);
    const int64_t ratio=g.blockTokens/128;
    if(virtualBlock<0 || virtualBlock>=g.blocks*ratio) {error=1;return false;}
    block=virtualBlock/ratio;inpage=(virtualBlock%ratio)*128+position%128;
    return true;
  }
  __aicore__ void LoadPacked(int64_t start) {
    liveKvRows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-start-lane*kHalfKv)));
    auto packed=packedBuf.Get<uint8_t>();
    Duplicate(packed.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),
        kHalfKv*kPackedStride/2);Fence<HardEvent::V_MTE2>();
    for(int32_t j=0;j<liveKvRows;) {
      const int64_t candidate=start+lane*kHalfKv+j;
      const int32_t rows=static_cast<int32_t>(Min64(liveKvRows-j,128-candidate%128));
      int64_t block=0,inpage=0;
      if(!Physical(candidate,block,inpage)) {j+=rows;continue;}
      const int64_t off=g.ssmOffset+block*g.pageStride+
          (inpage*g.hk+kvhead)*kSlotBytes;
      // GM rows are interleaved by KV head; UB rows are padded to 32 bytes.
      // Never assume adjacent virtual128 entries point to adjacent pages.
      DataCopyExtParams copy{static_cast<uint16_t>(rows),kSlotBytes,
          static_cast<uint32_t>((g.hk-1)*kSlotBytes),0,0};
      DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
      DataCopyPad(packed[j*kPackedStride],bytes[off],copy,pad);
      j+=rows;
    }
    Fence<HardEvent::MTE2_V>();
  }
  __aicore__ void Unpack(bool value) {
    oscar_ascend_striped_bf16::Unpack<D,kHalfKv>(value,liveKvRows,error,packedBuf,wordBuf,
        planeBuf,naturalBuf,maskBuf,wordIndexBuf,laneIndexBuf,
        metadataIndexBuf,metadataHalfBuf,dequantBuf,scratchBuf,operandBuf);
  }
  __aicore__ void LoadWindowRuns(int64_t start,bool value) {
    auto dst=dequantBuf.Get<bfloat16_t>();
    // These buffers are dead after source0 unpack. Permanent unpack indices
    // (metadataIndexBuf/maskBuf/etc.) remain untouched for the next task.
    auto tags=packedBuf.Get<int64_t>();
    const int64_t first=start+lane*kHalfKv;
    const int32_t live=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-first)));
    const int64_t sinkCount=Min64(g.sink,context);
    const int64_t ring=g.recent+g.speculative;
    for(int32_t j=0;j<live;) {
      const int64_t candidate=first+j;
      const int64_t position=LogicalPosition(candidate);
      int32_t rows=static_cast<int32_t>(Min64(live-j,128-position%128));
      if(candidate<sinkCount)rows=static_cast<int32_t>(Min64(rows,sinkCount-candidate));
      int64_t block=0,inpage=0;
      if(!Physical(position,block,inpage)) {j+=rows;continue;}
      rows=static_cast<int32_t>(Min64(rows,g.blockTokens-inpage));
      int64_t windowRow;
      if(position<g.sink) {
        rows=static_cast<int32_t>(Min64(rows,g.sink-position));
        windowRow=position;
      } else {
        rows=static_cast<int32_t>(Min64(rows,ring-inpage%ring));
        windowRow=g.sink+inpage%ring;
      }
      // #126/#145: protect every old Vector/scalar owner before MTE2
      // overwrites the reused packed (tag) UB.
      Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
      DataCopyExtParams tagCopy{1,static_cast<uint32_t>(rows*8),0,0,0};
      DataCopyPadExtParams<int64_t> tagPad{false,0,0,0};
      DataCopyPad(tags,wt[block*g.tagStride+windowRow],tagCopy,tagPad);
      Fence<HardEvent::MTE2_S>();
      // Invalid tags never cause a K/V read. All invalid and unused output
      // rows retain the exact initial +0, including poisoned ring slots.
      Duplicate(dst[j*D].ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),rows*D);
      Fence<HardEvent::V_MTE2>();
      for(int32_t begin=0;begin<rows;) {
        if(tags.GetValue(begin)!=inpage+begin) {++begin;continue;}
        int32_t end=begin+1;
        while(end<rows && tags.GetValue(end)==inpage+end)++end;
        const int64_t off=block*g.windowStride+((windowRow+begin)*g.hk+kvhead)*D;
        DataCopyExtParams copy{static_cast<uint16_t>(end-begin),D*2,
            static_cast<uint32_t>((g.hk-1)*D*2),0,0};
        DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
        if(value)DataCopyPad(dst[(j+begin)*D],wv[off],copy,pad);
        else DataCopyPad(dst[(j+begin)*D],wk[off],copy,pad);
        begin=end;
      }
      Fence<HardEvent::MTE2_V>();
      // Do NOT publish errors during the tag/DMA prepass. Original order is
      // physical->tag->finite for each row, K side first then V side. A later
      // bad tag must overwrite an earlier finite error, and vice versa.
      for(int32_t row=0;row<rows;++row) {
        if(tags.GetValue(row)!=inpage+row) {error=4;continue;}
        CheckBf16Row(dst[(j+row)*D]);
      }
      Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
      j+=rows;
    }
    Fence<HardEvent::S_V>();
  }
  __aicore__ void LoadPrecise(int64_t start,bool value) {
    auto dst=dequantBuf.Get<bfloat16_t>();
    Duplicate(dst.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),kElements);PipeBarrier<PIPE_V>();
    if(kind==2) {
      const int32_t rows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-start-lane*kHalfKv)));
      if(rows==0)return;
      const int64_t token=requestBegin+start+lane*kHalfKv-context;
      if(token<0 || token+rows>g.tokens) {error=1;return;}
      // Load exact BF16 directly into the bounded operand tile.
      Fence<HardEvent::V_MTE2>();
      DataCopyExtParams copy{static_cast<uint16_t>(rows),D*2,
          static_cast<uint32_t>((g.hk-1)*D*2),0,0};
      DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
      const int64_t off=(token*g.hk+kvhead)*D;
      if(value)DataCopyPad(dst,cv[off],copy,pad);else DataCopyPad(dst,ck[off],copy,pad);
      Fence<HardEvent::MTE2_V>();
      for(int32_t row=0;row<rows;++row) {
        CheckBf16Row(dst[row*D]);
      }
      return;
    }
    if(kind==1) {LoadWindowRuns(start,value);return;}
    error=1;
  }
  __aicore__ void PublishKv(bool value,int32_t subtile) {
    auto src=dequantBuf.Get<bfloat16_t>();
    const int64_t slot=subtile*(2*kHalfKv)+lane*kHalfKv;
    // Both K and V are natural [KV,D] BF16. QK requests B transpose=true;
    // PV requests false, as permitted by the same static B matmul type.
    Fence<HardEvent::V_MTE3>();
    PublishBf16Operand((value?vOffset:kOffset)+slot*D,src,kElements);
    Fence<HardEvent::MTE3_V>();
  }
  __aicore__ void AddVisibleRange(int64_t begin,int64_t end,int64_t start,
      int32_t& lo0,int32_t& hi0,int32_t& lo1,int32_t& hi1) {
    if(end<=begin)return;
    const int32_t lo=static_cast<int32_t>(begin-start);
    const int32_t hi=static_cast<int32_t>(end-start);
    if(hi0==lo0) {lo0=lo;hi0=hi;}
    else if(lo<=hi0) hi0=static_cast<int32_t>(Max64(hi0,hi));
    else {lo1=lo;hi1=hi;}
  }
  __aicore__ void Softmax(int64_t start,int32_t rowBase) {
    auto scores=scoreBuf.Get<float>();
    auto acc=accBuf.Get<float>()[(rowBase-lane*kLaneRows)*D];
    auto stats=statsBuf.Get<float>();auto tmp=scratchBuf.Get<float>();
    const int64_t group=g.hq/g.hk;
    const int32_t activeRows=static_cast<int32_t>(Max64(0,
        Min64(kHalfRows,qcount*group-rowBase)));
    if(activeRows==0) {
      // Matmul SetSingleShape(M=real rows) excludes this full block from PV.
      // Keep the outer flag protocol, but publish no unused GM P rows.
      return;
    }
    DataCopy(scores,work[scoreOffset+rowBase*kKvRows],kHalfRows*kKvRows);
    Fence<HardEvent::MTE2_V>();Muls(scores,scores,g.scale,kHalfRows*kKvRows);
    Fence<HardEvent::V_S>();
    const int32_t softmaxRows=(activeRows+7)/8*8;
    if(activeRows<softmaxRows) {
      // -inf plus the finite empty-max sentinel gives P=0, sum=0 even when
      // this query has no prior visible tile. V2 never sees -inf - (-inf).
      Fence<HardEvent::S_V>();
      Duplicate(scores[activeRows*kKvRows],kNegativeInfinity,
          (softmaxRows-activeRows)*kKvRows);
      Fence<HardEvent::V_S>();
    }
    if(softmaxRows<kHalfRows) {
      Fence<HardEvent::S_V>();
      Duplicate(scores[softmaxRows*kKvRows],0.0F,
          (kHalfRows-softmaxRows)*kKvRows);
      Fence<HardEvent::V_S>();
    }
    // Archive #134/#140/D.4: every source has at most two visible candidate
    // intervals. Derive them once per query row instead of scanning all 128
    // candidate positions. The remaining mask writes and finite check are
    // unchanged, including padded tail columns and empty source rows.
    const int32_t jEnd=static_cast<int32_t>(Min64(kKvRows,kvend-start));
    const int64_t tileEnd=start+jEnd;
    const int64_t sinkCount=Min64(g.sink,context);
    const int64_t firstPosition=context+qbegin-requestBegin;
    const int64_t tailBegin=Min64(context,Max64(g.sink,firstPosition+1-g.recent));
    for(int32_t r=0;r<activeRows;++r) {
      const int64_t row=rowBase+r;
      const int64_t position=context+qbegin+row/group-requestBegin;
      int32_t lo0=0,hi0=0,lo1=0,hi1=0;
      const int64_t cut=Max64(g.sink,position+1-g.recent);
      if(kind==0) {
        AddVisibleRange(Max64(start,g.sink),
            Min64(tileEnd,Min64(context,Min64(cut,position+1))),start,
            lo0,hi0,lo1,hi1);
      } else if(kind==1) {
        // LogicalPosition maps [0,sinkCount) to the exact sink, and the
        // remaining candidate interval to [tailBegin,context). Every query
        // position is >=context, so only the recent cut affects the tail.
        AddVisibleRange(Max64(start,0),Min64(tileEnd,sinkCount),start,
            lo0,hi0,lo1,hi1);
        AddVisibleRange(Max64(start,sinkCount+Max64(0,cut-tailBegin)),
            Min64(tileEnd,sinkCount+context-tailBegin),start,
            lo0,hi0,lo1,hi1);
      } else {
        AddVisibleRange(Max64(start,context),Min64(tileEnd,position+1),start,
            lo0,hi0,lo1,hi1);
      }
      const bool any=hi0>lo0||hi1>lo1;
      if(!any) {Fence<HardEvent::S_V>();Duplicate(scores[r*kKvRows],kNegativeInfinity,kKvRows);
        Fence<HardEvent::V_S>();continue;}
      PipeBarrier<PIPE_V>();
      ReduceSum(tmp,scores[r*kKvRows],tmp[16],kKvRows);Fence<HardEvent::V_S>();
      if(!Finite(tmp.GetValue(0)))error=2;
      if(lo0>0 || hi0<kKvRows) {
        oscar_range_mask::MaskComplement(scores[r*kKvRows],lo0,hi0,lo1,hi1);
        Fence<HardEvent::V_S>();
      }
    }
    // The two 64-row max/sum arrays persist across KV work units. Two shared
    // 32-row input arrays let CANN's FP32 WITHOUT_BRC V2 update one score
    // block at a time without spilling the running state to GM.
    const int32_t block=rowBase-lane*kLaneRows;
    auto outMax=stats[block],outSum=stats[kLaneRows+block];
    auto inMax=stats[2*kLaneRows],inSum=stats[2*kLaneRows+kHalfRows];
    Adds(inMax,outMax,0.0F,softmaxRows);
    Adds(inSum,outSum,0.0F,softmaxRows);
    PipeBarrier<PIPE_V>();
    // All K/V staging into GM finished before CubeReady. The final dequant
    // UB values are dead until the next outer KV tile, so its D64/128/256
    // capacity gives the SDK 4/8/16KiB of temporary space with no new UB.
    auto softWork=dequantBuf.Get<float>();
    const SoftMaxShapeInfo shape{static_cast<uint32_t>(softmaxRows),
        static_cast<uint32_t>(kKvRows),static_cast<uint32_t>(softmaxRows),
        static_cast<uint32_t>(kKvRows)};
    const auto tiling=SoftMaxFlashV2TilingFunc(shape,sizeof(float),sizeof(float),
        softWork.GetSize()*sizeof(float),true,false);
    SoftmaxFlashV2<float,true,true,false,false,kCvSoftmaxConfig>(
        scores,outSum,outMax,scores,tmp,inSum,inMax,
        softWork.ReinterpretCast<uint8_t>(),tiling,shape);
    Fence<HardEvent::V_S>();
    // Reuse the now-dead natural unpack UB for the same row-strided Brcb
    // pattern as the validated scale/zero path. This multiplies all active
    // accumulator rows by their FP32 online alpha without scalar readback.
    auto alphaBlocks=naturalBuf.Get<float>();
    Brcb(alphaBlocks,tmp,static_cast<uint8_t>((activeRows+7)/8),{1,8});
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams alphaParams{1,1,0,static_cast<uint8_t>(D/8),
        static_cast<uint8_t>(D/8),1};
    for(int32_t chunk=0;chunk<D/64;++chunk)
      Mul(acc[chunk*64],acc[chunk*64],alphaBlocks,static_cast<uint64_t>(64),
          static_cast<uint8_t>(activeRows),alphaParams);
    PipeBarrier<PIPE_V>();
    Fence<HardEvent::V_MTE3>();
    // V2 may use padded UB rows to satisfy its 8-row shape, but Cube PV has
    // M=qcount*group and consumes only the real P rows from this block.
    PublishOperand(pOffset+rowBase*kKvRows,scores,activeRows*kKvRows);
    // The next Softmax block refills scoreBuf on MTE2, not Vector. MTE3_V
    // cannot prevent that DMA from overwriting P while MTE3 still reads it.
    // The next copy-in is followed by MTE2_V before any Vector score writes.
    Fence<HardEvent::MTE3_MTE2>();
  }
  __aicore__ void VectorTask(int64_t id) {
    error=0;
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();
    Duplicate(acc,0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
    Duplicate(stats,kEmptyMax,kLaneRows);
    Duplicate(stats[kLaneRows],0.0F,kLaneRows);
    Fence<HardEvent::V_S>();
    LoadQueries();
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows) {
      // Four bounded 32-token subtiles fill one 128-token K/V work unit.
      // Packed history is read once per subtile and reused for K and V;
      // all incomplete tail slots are published as zero before Cube QK.
      for(int32_t subtile=0;subtile<kKvSubtiles;++subtile) {
        const int64_t subStart=start+subtile*(2*kHalfKv);
        if(kind==0) {LoadPacked(subStart);Unpack(false);}
        else LoadPrecise(subStart,false);
        PublishKv(false,subtile);
        if(kind==0)Unpack(true);else LoadPrecise(subStart,true);
        PublishKv(true,subtile);
      }
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      for(int32_t block=0;block<kRowBlocks;++block)
        Softmax(start,lane*kLaneRows+block*kHalfRows);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      // P is fully consumed by Cube before this point. Reuse the now-dead
      // dequant UB for bounded 16-row PV reads; the 64-row FP32 accumulator
      // stays in UB across every 256-KV unit in the same query task.
      auto result=dequantBuf.Get<float>();
      const int32_t activeRows=static_cast<int32_t>(qcount*(g.hq/g.hk));
      for(int32_t row=lane*kLaneRows;row<(lane+1)*kLaneRows && row<activeRows;
          row+=kHalfKv) {
        const int32_t count=static_cast<int32_t>(Min64(kHalfKv,activeRows-row));
        DataCopy(result,work[pvOffset+row*D],count*D);
        Fence<HardEvent::MTE2_V>();
        auto target=acc[(row-lane*kLaneRows)*D];
        Add(target,target,result,count*D);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
      }
      CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
    }
    Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kLaneRows;++r) {
      const float sum=stats.GetValue(kLaneRows+r);
      if(sum>0.0F && Finite(sum))Muls(acc[r*D],acc[r*D],1.0F/sum,D);
      else {if(sum!=0.0F)error=2;Duplicate(acc[r*D],0.0F,D);}
      PipeBarrier<PIPE_V>();
    }
    if(kind==0 && kvbegin<kvend) {
      Fence<HardEvent::V_MTE3>();DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      DataCopy(acc,work[rotOffset+lane*kLaneRows*D],kLaneRows*D);
      Fence<HardEvent::MTE2_V>();CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
    }
    PublishRows(id);
  }
  __aicore__ void PublishRows(int64_t id) {
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();auto scalar=scratchBuf.Get<float>();
    const int64_t group=g.hq/g.hk,rows=qcount*group;
    Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kLaneRows && lane*kLaneRows+r<rows;++r) {
      const int64_t row=lane*kLaneRows+r;
      const int64_t outRow=((qbegin+row/group)*g.hq+kvhead*group+row%group)*3*g.splits+split;
      const float sum=stats.GetValue(kLaneRows+r),maximum=stats.GetValue(r);
      float lse=kNegativeInfinity;
      if(sum>0.0F && Finite(sum)) {
        scalar.SetValue(0,sum);Fence<HardEvent::S_V>();
        Log(scalar[8],scalar,1);Fence<HardEvent::V_S>();lse=maximum+scalar.GetValue(8);
      }
      if(error) {Duplicate(acc[r*D],__builtin_nanf(""),D);lse=__builtin_nanf("");}
      Fence<HardEvent::V_MTE3>();DataCopy(out[outRow*D],acc[r*D],D);
      scalar.SetValue(0,lse);Fence<HardEvent::S_MTE3>();
      DataCopyExtParams copy{1,4,0,0,0};DataCopyPad(outLse[outRow],scalar,copy);
      Fence<HardEvent::MTE3_S>();
    }
    PublishStatus(id,error);
  }
  __aicore__ void PublishEmpty(int64_t id,int32_t code) {
    // Padding has one output owner per token/head/segment, never leaves poison.
    if(qbegin>=0 && qbegin<g.tokens && kvhead>=0 && kvhead<g.hk &&
        split>=0 && split<3*g.splits) {
      qcount=1;error=code;
      auto stats=statsBuf.Get<float>();auto acc=accBuf.Get<float>();
      Duplicate(acc,code?__builtin_nanf(""):0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
      Duplicate(stats,kEmptyMax,kLaneRows);
      Duplicate(stats[kLaneRows],0.0F,kLaneRows);
      Fence<HardEvent::V_S>();
      PublishRows(id);
    }else PublishStatus(id,code?code:1);
  }
  __aicore__ void PublishStatus(int64_t id,int32_t code) {
    auto word=addressBuf.Get<int32_t>();word.SetValue(0,code);
    Fence<HardEvent::S_MTE3>();DataCopyExtParams copy{1,4,0,0,0};
    DataCopyPad(outStatus[id*2+lane],word,copy);Fence<HardEvent::MTE3_S>();
  }
  TPipe pipe;CubeMatmul mm;HistoryMatmul historyMm;
  bool historyMatmulActive=false;
  GlobalTensor<bfloat16_t> operandWork;
  TBuf<TPosition::VECCALC> operandBuf;
  Geometry g;
  GlobalTensor<bfloat16_t> q,ck,cv,wk,wv;
  GlobalTensor<float> qr,rv,out,outLse,work;
  GlobalTensor<uint8_t> bytes;GlobalTensor<int32_t> bt,outStatus;
  GlobalTensor<int64_t> wt,taskGm;
  TBuf<TPosition::VECCALC> taskBuf,packedBuf,planeBuf,naturalBuf,wordBuf,maskBuf,
      wordIndexBuf,laneIndexBuf,metadataIndexBuf,metadataHalfBuf,dequantBuf,
      scoreBuf,accBuf,qFloatBuf,qBf16Buf,statsBuf,scratchBuf,addressBuf;
  int64_t coreIndex,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin;
  int64_t qOffset,kOffset,vOffset,scoreOffset,pOffset,pvOffset,rotOffset;
  int32_t lane=0,error=0,taskError=0,liveKvRows=0;
};
}
#define OSCAR_CV_ARGUMENTS GM_ADDR query, GM_ADDR queryRot, GM_ADDR key, GM_ADDR value, \
    GM_ADDR rotation, GM_ADDR raw, GM_ADDR table, GM_ADDR wk, GM_ADDR wv, GM_ADDR tags, \
    GM_ADDR tasks, GM_ADDR output, GM_ADDR lse, GM_ADDR status, GM_ADDR workspace, \
    int64_t tokens, int64_t hq, int64_t hk, int64_t dim, int64_t requests, \
    int64_t columns, int64_t taskCount, int64_t blockTokens, int64_t blocks, \
    int64_t ssmOffset, int64_t pageStride, int64_t windowStride, int64_t tagStride, \
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits, float scale
#define OSCAR_CV_INIT op.Init(query,queryRot,key,value,rotation,raw,table,wk,wv,tags, \
    tasks,output,lse,status,workspace,g);op.Process()
extern "C" __global__ __aicore__ void oscar_attention_cv_fast_balanced_kernel(OSCAR_CV_ARGUMENTS) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  const Geometry g{tokens,hq,hk,requests,columns,taskCount,blockTokens,blocks,ssmOffset,
      pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale};
#ifdef OSCAR_STRIPED_D256_ONLY
  if(dim!=256) {trap();return;}
  AttentionCv<256> op;OSCAR_CV_INIT;
#else
  if(dim==64) {AttentionCv<64> op;OSCAR_CV_INIT;}
  else if(dim==128) {AttentionCv<128> op;OSCAR_CV_INIT;}
  else {AttentionCv<256> op;OSCAR_CV_INIT;}
#endif
}
#ifndef ASCENDC_CPU_DEBUG
namespace oscar_ascend {
void attention_cv_fast_balanced_launch(void* stream,void* query,void* queryRot,void* key,void* value,
    void* rotation,void* raw,void* table,void* wk,void* wv,void* tags,void* tasks,
    void* output,void* lse,void* status,void* workspace,int64_t tokens,int64_t hq,
    int64_t hk,int64_t dim,int64_t requests,int64_t columns,int64_t taskCount,
    int64_t blockTokens,int64_t blocks,int64_t ssmOffset,int64_t pageStride,
    int64_t windowStride,int64_t tagStride,int64_t sink,int64_t recent,
    int64_t speculative,int64_t splits,float scale,uint32_t cores) {
  oscar_attention_cv_fast_balanced_kernel<<<cores,nullptr,stream>>>(
      static_cast<uint8_t*>(query),static_cast<uint8_t*>(queryRot),
      static_cast<uint8_t*>(key),static_cast<uint8_t*>(value),
      static_cast<uint8_t*>(rotation),static_cast<uint8_t*>(raw),
      static_cast<uint8_t*>(table),static_cast<uint8_t*>(wk),static_cast<uint8_t*>(wv),
      static_cast<uint8_t*>(tags),static_cast<uint8_t*>(tasks),static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(lse),static_cast<uint8_t*>(status),static_cast<uint8_t*>(workspace),
      tokens,hq,hk,dim,requests,columns,taskCount,blockTokens,blocks,ssmOffset,
      pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale);
}
}
#endif

#undef OSCAR_STRIPED_D256_ONLY
