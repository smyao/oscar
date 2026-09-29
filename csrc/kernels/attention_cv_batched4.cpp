// SPDX-License-Identifier: Apache-2.0
// Explicit candidate batched-history route. Archive #126/#129/#145/#148-153
// and startup D.4 four questions: (1) this targets fused INT2 history FIA;
// (2) the failed full-history restore alone took 6499.8-6655.1ms/card while
// native FIA was 18.5-18.9ms, with separate ~725ms host and ~209ms store;
// (3) four same-domain query groups share one fixed KV256 unpack and one
// bounded M512 FP32 QK/PV each, keeping each query's online max/sum/alpha,
// causal mask, status, KV order and old-acc*alpha then +PV arithmetic. No
// history-sized restore, native BF16 fallback, HF32 relaxation, or host loop;
// (4) eight QK/PV Matmul calls become two per shared KV work unit, but FP32 FLOPs,
// score/P GM traffic and accumulator state traffic remain. Neither target
// NPU precision nor wall-time gain is claimed before independent gates.
// #126: MTE3->V protects valid Q followed by padding; #145: MTE3->MTE2
// protects score/P UB reuse, even in the wider M512 score layout.
// Candidate exact INT2 unpack for the C4 KV-major schedule. Archive
// G26-G34/#13-20/#126/#129/#140-151; startup D.4 four questions:
// (1) fused source0 dequant+FIA; (2) failed full-history restore cost
// 6499.8-6655.1ms vs native FIA 18.5-18.9ms; (3) one bounded KV256 work
// unit remains, FP32 QK/PV/softmax, masks, checks, flags and output unchanged;
// only integer-plane UB stride and exact FP16 metadata conversion change;
// (4) unpack work was measured hot but bank relief/net speed is a target-NPU
// hypothesis. Frozen bitwise/math/graph/quality gates precede any speed claim.
// Candidate source0 four-query-group KV-major reuse. This file starts
// from this project's fe0e925 attention_cv.cpp; production fe0 stays byte-for-
// byte unchanged. Archive #126/#129/#140-145 and startup D.4 four questions:
// (1) phase is fused INT2 history dequant + FIA. (2) D.4's failed full-history
// restore cost 6499.8-6655.1ms while native FIA cost 18.5-18.9ms, with host
// prepare ~725ms and stores ~209ms. (3) Four exact-domain query groups reuse
// only one bounded KV256 unpack; independent FP32 online acc/max/sum are
// byte-preserved in bounded per-core GM between tiles. Fe0 QK/PV/softmax,
// pre-mask finite checks, status, rotation and output ownership stay intact;
// no history-sized BF16 tensor or fallback exists. (4) Eligible source0
// unpack work falls up to 4:1, but QK/PV work and extra state traffic remain;
// device precision, graph and same-input wall measurements must prove value.
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
#include "attention_fast_unpack.h"
#include "../include/oscar_attention_launch.h"
#include "../include/oscar_batched4_experimental.h"
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
  cfg.singleCoreM=512;cfg.singleCoreN=256;cfg.singleCoreK=256;
  cfg.enableSetBias=false;
  return cfg;
}
constexpr auto kCubeConfig=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(CvConfig());
using CubeMatmul=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kCubeConfig>;
static_assert(oscar_ascend::attention_batched4_workspace_per_core(64)==
              oscar_ascend_experiment::batched4_workspace_per_core(64));
static_assert(oscar_ascend::attention_batched4_workspace_per_core(128)==
              oscar_ascend_experiment::batched4_workspace_per_core(128));
static_assert(oscar_ascend::attention_batched4_workspace_per_core(256)==
              oscar_ascend_experiment::batched4_workspace_per_core(256));

struct Geometry {
  int64_t tokens,hq,hk,requests,columns,taskCount,blockTokens,blocks,ssmOffset;
  int64_t pageStride,windowStride,tagStride,sink,recent,speculative,splits;
  float scale;
};

struct TaskState {
  int64_t id,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin;
  int32_t taskError;
};

template<int32_t D> class AttentionCv {
  static constexpr int32_t kSlotBytes=D/2+8;
  static constexpr int32_t kPackedStride=(kSlotBytes+31)/32*32;
  static constexpr int32_t kElements=kHalfKv*D;
  static constexpr int32_t kWords=kElements/8;
  static constexpr int32_t kPlaneStride=kWords+16;
 public:
  __aicore__ void Init(GM_ADDR query,GM_ADDR queryRot,GM_ADDR key,GM_ADDR value,
      GM_ADDR rotation,GM_ADDR raw,GM_ADDR table,GM_ADDR windowKey,
      GM_ADDR windowValue,GM_ADDR tags,GM_ADDR tasks,GM_ADDR partial,
      GM_ADDR lse,GM_ADDR status,GM_ADDR workspace,GM_ADDR clusterStatsPtr,
      const Geometry& geometry) {
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
    clusterStats.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(clusterStatsPtr));
    int64_t core=GetBlockIdx();
    if ASCEND_IS_AIV {lane=core%2;core/=2;}
    coreIndex=core;
    const int64_t floatsPerCore=oscar_ascend_experiment::batched4_workspace_per_core(D)/4;
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+core*floatsPerCore);
    // Four Q tiles, bounded K/V, score/P, PV/rotation and four online states.
    // The C1 branch uses Q0 and retains its fe0 accumulator in UB.
    qOffset=0;kOffset=4*kQueryRows*D;vOffset=kOffset+kKvRows*D;
    scoreOffset=vOffset+kKvRows*D;
    pOffset=scoreOffset;pvOffset=pOffset+4*kQueryRows*kKvRows;rotOffset=pvOffset;
    accStateOffset=pvOffset+4*kQueryRows*D;
    statStateOffset=accStateOffset+4*kQueryRows*D;
    alphaOffset=statStateOffset+4*2*kQueryRows;
    for(int32_t i=0;i<8;++i)counters[i]=0;
    if ASCEND_IS_AIC {
      // Static tiling is compiler-derived by the real SDK. SetOrg/SingleShape
      // below specify each QK/PV/rotation matrix, never hand-written tiling POD.
      mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);
      SetHF32Mode(false);
    } else {
      pipe.InitBuffer(taskBuf,128);pipe.InitBuffer(packedBuf,kHalfKv*kPackedStride);
      pipe.InitBuffer(planeBuf,oscar_ascend_fast::Geometry<D>::planeBytes);
      pipe.InitBuffer(naturalBuf,kElements*2);
      pipe.InitBuffer(wordBuf,kWords*2);pipe.InitBuffer(maskBuf,kWords*2);
      pipe.InitBuffer(wordIndexBuf,kWords*4);pipe.InitBuffer(laneIndexBuf,kElements*4);
      pipe.InitBuffer(metadataIndexBuf,128);pipe.InitBuffer(metadataHalfBuf,64);
      pipe.InitBuffer(dequantBuf,kElements*4);
      pipe.InitBuffer(scoreBuf,kHalfRows*kKvRows*4);
      pipe.InitBuffer(accBuf,kLaneRows*D*4);
      pipe.InitBuffer(qFloatBuf,D*4);pipe.InitBuffer(qBf16Buf,D*2);
      pipe.InitBuffer(statsBuf,(2*kLaneRows+2*kHalfRows)*4);pipe.InitBuffer(scratchBuf,512);
      pipe.InitBuffer(addressBuf,128);
      InitIndices();
    }
  }
  __aicore__ void Process() {
    // Scalar Cube metadata reads must invalidate stale cache on changed-input
    // graph replay. The Vector side exclusively DMA-loads mutable metadata.
    if ASCEND_IS_AIC {
      DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>(taskGm);
    }
    const int64_t queryTile=kQueryRows/(g.hq/g.hk);
    const int64_t perToken=g.hk*3*g.splits;
    // Archive #129/#150/#152: the main mixed batch can contain 31 q4 leaders
    // ahead of a long prefill. Ownership tile4 spreads those source0 leaders
    // across 20 Cubes; prepare's mathematical qcount stays queryTile=21 for
    // GQA6, and the second-pass exact C4 bucket remains 4*queryTile=84.
    constexpr int64_t scheduleTile=4;
    const oscar_ascend_schedule::CvTaskSchedule schedule{g.tokens,scheduleTile,perToken};
    // Pass one visits every follower, invalid leader and noneligible leader
    // once with tile4 ownership. Exact C4 members alone defer to pass two.
    // The predicate is read-only and identical on AIC/AIV; no global barrier.
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
        if(IsClusterMember(id,nullptr)) {
          if ASCEND_IS_AIC {++counters[6];}
          continue;
        }
        qOffset=0;
        if ASCEND_IS_AIC {
          if(kind==0) {
            ++counters[2];
            counters[5]+=(kvend-kvbegin+kKvRows-1)/kKvRows;
          }
          CubeTask();
        } else {VectorTask(id);}
      }
    }
    // Pass two maps contiguous 84-token buckets across ALL Cubes, avoiding
    // the original #129 stride/common-divisor alias of anchor%4. Each anchor
    // belongs to exactly one global bucket; its four request-relative members
    // have already been excluded in pass one by the same predicate.
    if(g.tokens>=4*queryTile) {
      const int64_t bucketWidth=4*queryTile;
      const int64_t buckets=(g.tokens+bucketWidth-1)/bucketWidth;
      const int64_t segments=g.hk*g.splits;
      for(int64_t workId=coreIndex;workId<buckets*segments;workId+=GetBlockNum()) {
        const int64_t bucket=workId%buckets;
        const int64_t segment=workId/buckets;
        const int64_t kvHead=segment/g.splits;
        const int64_t splitIndex=segment%g.splits;
        const int64_t taskSegment=(kvHead*3)*g.splits+splitIndex;
        const int64_t begin=bucket*bucketWidth;
        const int64_t end=Min64(begin+bucketWidth,g.tokens);
        for(int64_t token=begin;token<end;++token) {
          const int64_t id=token*perToken+taskSegment;
          LoadTask(id);
          if(qcount!=queryTile || qbegin!=token || kind!=0 ||
              qbegin<requestBegin ||
              (qbegin-requestBegin)%bucketWidth!=0)continue;
          TaskState members[4];
          if(!IsClusterMember(id,members))continue;
          if ASCEND_IS_AIC {
            ++counters[0];counters[1]+=4;++counters[7];
            const int64_t tiles=(members[0].kvend-members[0].kvbegin+kKvRows-1)/kKvRows;
            counters[3]+=tiles;counters[4]+=3*tiles;
            CubeCluster(members);
          } else {VectorCluster(members);}
        }
      }
    }
    if ASCEND_IS_AIC {
      mm.End();
      for(int32_t i=0;i<8;++i)clusterStats.SetValue(coreIndex*8+i,counters[i]);
      // Each AIC owns exactly one 64-byte line; scalar GM SetValue needs an
      // explicit A2 DCache writeback before the host reads the diagnostics.
      DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE,
          DcciDst::CACHELINE_OUT>(clusterStats[coreIndex*8]);
    }
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
  __aicore__ TaskState CurrentTask(int64_t id) const {
    return TaskState{id,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,
        context,requestBegin,taskError};
  }
  __aicore__ void UseTask(const TaskState& t) {
    qbegin=t.qbegin;qcount=t.qcount;kvhead=t.kvhead;kvbegin=t.kvbegin;
    kvend=t.kvend;request=t.request;split=t.split;kind=t.kind;
    context=t.context;requestBegin=t.requestBegin;taskError=t.taskError;
  }
  __aicore__ bool IsClusterMember(int64_t id,TaskState* members) {
    const int64_t queryTile=kQueryRows/(g.hq/g.hk);
    const int64_t width=4*queryTile;
    if(g.tokens<width || kind!=0 || qcount!=queryTile || taskError!=0 ||
        kvbegin>=kvend || kvend!=context || qbegin<requestBegin ||
        (qbegin-requestBegin)%queryTile!=0 || !TaskValid())return false;
    const TaskState original=CurrentTask(id);
    const int64_t anchor=requestBegin+
        ((qbegin-requestBegin)/width)*width;
    if(anchor<0 || anchor+width>g.tokens) return false;
    const int64_t perToken=g.hk*3*g.splits;
    const int64_t segment=(kvhead*3)*g.splits+split;
    bool valid=true;
    TaskState found[4];
    for(int32_t j=0;j<4;++j) {
      const int64_t candidate=anchor+j*queryTile;
      const int64_t candidateId=candidate*perToken+segment;
      LoadTask(candidateId);
      found[j]=CurrentTask(candidateId);
      if(qbegin!=candidate || qcount!=queryTile || kind!=0 || taskError!=0 ||
          request!=original.request || kvhead!=original.kvhead ||
          split!=original.split || context!=original.context ||
          requestBegin!=original.requestBegin ||
          kvbegin!=original.kvbegin || kvend!=original.kvend ||
          kvend!=context || !TaskValid())valid=false;
    }
    UseTask(original);
    if(valid && members!=nullptr)for(int32_t j=0;j<4;++j)members[j]=found[j];
    return valid;
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
    oscar_ascend_fast::InitIndices<D>(wordIndexBuf,laneIndexBuf,
        metadataIndexBuf,maskBuf);
  }
  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,
      bool rotation=false,bool transposeB=true) {
    SetHF32Mode(false); // no implicit HF32/TF32 relaxation of the frozen oracle.
    mm.SetOrgShape(m,n,k);mm.SetSingleShape(m,n,k);
    mm.SetTensorA(work[a],false);
    if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],transposeB);
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
  __aicore__ void CubeCluster(const TaskState* members) {
    const int64_t first=members[0].kvbegin;
    const int64_t last=members[0].kvend;
    // Every member owns 128 padded Q rows in GM, though only qcount*GQA
    // rows are visible. The full M512 QK/PV shape is statically tiled for
    // this experimental kernel; P for the two inactive rows per member is
    // explicitly zeroed by Vector before PV consumes it.
    const int32_t batchRows=4*kQueryRows;
    for(int64_t start=first;start<last;start+=kKvRows) {
      // One AIV KV publish serves one M512 QK and one M512 PV. The score/P
      // matrix is GM staging between physically separate AIC/AIV pipelines.
      CrossCoreWaitFlag(kVectorReady);
      Matmul(0,kOffset,scoreOffset,batchRows,kKvRows,D);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
      Matmul(pOffset,vOffset,pvOffset,batchRows,D,kKvRows,false,false);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
    }
    const int32_t rows=static_cast<int32_t>(members[0].qcount*(g.hq/g.hk));
    for(int32_t member=0;member<4;++member) {
      CrossCoreWaitFlag(kVectorReady);
      qOffset=member*kQueryRows*D;
      Matmul(qOffset,0,rotOffset,rows,D,D,true);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
    }
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
      DataCopy(work[qOffset+row*D],dst,D);Fence<HardEvent::MTE3_MTE2>();
    }
    // Pad the inactive query rows in bounded contiguous pieces. This keeps
    // the full initialized Q tile required by the Cube contract without
    // adding one scalar/GM copy per padded row to q_len=1/4 graph replay.
    // The previous live-row MTE3 must finish before Vector reuses the scratch.
    auto zeros=dequantBuf.Get<float>();
    Fence<HardEvent::MTE3_V>();
    for(int32_t row=validEnd;row<laneEnd;row+=kHalfKv) {
      const int32_t count=static_cast<int32_t>(Min64(kHalfKv,laneEnd-row));
      Duplicate(zeros,0.0F,count*D);Fence<HardEvent::V_MTE3>();
      DataCopy(work[qOffset+row*D],zeros,count*D);Fence<HardEvent::MTE3_V>();
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
    oscar_ascend_fast::Unpack<D>(value,liveKvRows,error,packedBuf,wordBuf,
        planeBuf,naturalBuf,maskBuf,wordIndexBuf,laneIndexBuf,
        metadataIndexBuf,metadataHalfBuf,dequantBuf,scratchBuf);
  }
  __aicore__ void LoadPrecise(int64_t start,bool value) {
    auto dst=dequantBuf.Get<float>();auto src=qBf16Buf.Get<bfloat16_t>();
    Duplicate(dst,0.0F,kElements);PipeBarrier<PIPE_V>();
    if(kind==2) {
      const int32_t rows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-start-lane*kHalfKv)));
      if(rows==0)return;
      const int64_t token=requestBegin+start+lane*kHalfKv-context;
      if(token<0 || token+rows>g.tokens) {error=1;return;}
      // The natural unpack buffer is not used by the precise source. Reuse
      // it for contiguous BF16 rows; no extra UB or full-history buffer.
      auto batch=naturalBuf.Get<bfloat16_t>();
      Duplicate(batch.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),kElements);
      Fence<HardEvent::V_MTE2>();
      DataCopyExtParams copy{static_cast<uint16_t>(rows),D*2,
          static_cast<uint32_t>((g.hk-1)*D*2),0,0};
      DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
      const int64_t off=(token*g.hk+kvhead)*D;
      if(value)DataCopyPad(batch,cv[off],copy,pad);else DataCopyPad(batch,ck[off],copy,pad);
      Fence<HardEvent::MTE2_V>();Cast(dst,batch,RoundMode::CAST_NONE,kElements);
      PipeBarrier<PIPE_V>();
      auto check=scratchBuf.Get<float>();
      for(int32_t row=0;row<rows;++row) {
        ReduceSum(check,dst[row*D],check[16],D);Fence<HardEvent::V_S>();
        if(!Finite(check.GetValue(0)))error=2;
      }
      return;
    }
    for(int32_t j=0;j<kHalfKv;++j) {
      const int64_t candidate=start+lane*kHalfKv+j;
      if(candidate>=kvend)continue;
      const int64_t position=LogicalPosition(candidate);
      {
        int64_t block=0,inpage=0;
        if(!Physical(position,block,inpage))continue;
        const int64_t row=position<g.sink?position:g.sink+inpage%(g.recent+g.speculative);
        auto tag=addressBuf.Get<int64_t>()[4];
        DataCopyExtParams copy{1,8,0,0,0};DataCopyPadExtParams<int64_t> pad{false,0,0,0};
        DataCopyPad(tag,wt[block*g.tagStride+row],copy,pad);
        Fence<HardEvent::MTE2_S>();
        if(tag.GetValue(0)!=inpage) {error=4;continue;}
        const int64_t off=block*g.windowStride+(row*g.hk+kvhead)*D;
        if(value)DataCopy(src,wv[off],D);else DataCopy(src,wk[off],D);
      }
      Fence<HardEvent::MTE2_V>();Cast(dst[j*D],src,RoundMode::CAST_NONE,D);
      PipeBarrier<PIPE_V>();
      auto check=scratchBuf.Get<float>();ReduceSum(check,dst[j*D],check[16],D);
      Fence<HardEvent::V_S>();if(!Finite(check.GetValue(0)))error=2;
      Fence<HardEvent::V_MTE2>();
    }
  }
  __aicore__ void PublishKv(bool value,int32_t subtile) {
    auto src=dequantBuf.Get<float>();
    const int64_t slot=subtile*(2*kHalfKv)+lane*kHalfKv;
    // Both K and V are natural [KV,D] FP32. QK requests B transpose=true;
    // PV requests false, as permitted by the same static B matmul type.
    Fence<HardEvent::V_MTE3>();
    DataCopy(work[(value?vOffset:kOffset)+slot*D],src,kElements);
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
  __aicore__ void Softmax(int64_t start,int32_t rowBase,int32_t member=-1) {
    auto scores=scoreBuf.Get<float>();
    auto acc=accBuf.Get<float>()[(rowBase-lane*kLaneRows)*D];
    auto stats=statsBuf.Get<float>();auto tmp=scratchBuf.Get<float>();
    const int64_t matrixOffset=member<0?0:member*kQueryRows*kKvRows;
    const int64_t group=g.hq/g.hk;
    const int32_t activeRows=static_cast<int32_t>(Max64(0,
        Min64(kHalfRows,qcount*group-rowBase)));
    if(activeRows==0) {
      // Matmul SetSingleShape(M=real rows) excludes this full block from PV.
      // Keep the outer flag protocol, but publish no unused GM P rows.
      return;
    }
    DataCopy(scores,work[scoreOffset+matrixOffset+rowBase*kKvRows],kHalfRows*kKvRows);
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
        Fence<HardEvent::S_V>();
        for(int32_t j=0;j<kKvRows;++j) {
          const bool visible=(j>=lo0&&j<hi0)||(j>=lo1&&j<hi1);
          if(!visible)scores.SetValue(r*kKvRows+j,kNegativeInfinity);
        }
        Fence<HardEvent::S_V>();
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
    if(member<0) {
      // C1 keeps the original V2 -> Brcb -> Mul before PV -> Add ordering.
      auto alphaBlocks=naturalBuf.Get<float>();
      Brcb(alphaBlocks,tmp,static_cast<uint8_t>((activeRows+7)/8),{1,8});
      PipeBarrier<PIPE_V>();
      const BinaryRepeatParams alphaParams{1,1,0,static_cast<uint8_t>(D/8),
          static_cast<uint8_t>(D/8),1};
      for(int32_t chunk=0;chunk<D/64;++chunk)
        Mul(acc[chunk*64],acc[chunk*64],alphaBlocks,static_cast<uint64_t>(64),
            static_cast<uint8_t>(activeRows),alphaParams);
      PipeBarrier<PIPE_V>();
    } else {
      // Save V2's exact FP32 alpha bytes; no arithmetic or host readback.
      // Brcb consumes 8-row groups, so define the two inactive rows too.
      if(activeRows<softmaxRows) {
        // Duplicate requires a 32-byte-aligned destination; the typical
        // GQA6 tail begins at row 30. Scalar writes touch only inactive
        // alpha rows, after V2 has completed, without changing live bits.
        Fence<HardEvent::V_S>();
        for(int32_t row=activeRows;row<softmaxRows;++row)tmp.SetValue(row,0.0F);
        Fence<HardEvent::S_MTE3>();
      }
      Fence<HardEvent::V_MTE3>();
      DataCopy(work[alphaOffset+member*kQueryRows+rowBase],tmp,softmaxRows);
      Fence<HardEvent::MTE3_V>(); // next V2/finite check may reuse tmp.
    }
    Fence<HardEvent::V_MTE3>();
    // V2 may use padded UB rows to satisfy its 8-row shape, but Cube PV has
    // M=qcount*group and consumes only the real P rows from this block.
    DataCopy(work[pOffset+matrixOffset+rowBase*kKvRows],scores,activeRows*kKvRows);
    // The next Softmax block refills scoreBuf on MTE2, not Vector. MTE3_V
    // cannot prevent that DMA from overwriting P while MTE3 still reads it.
    // The next copy-in is followed by MTE2_V before any Vector score writes.
    Fence<HardEvent::MTE3_MTE2>();
  }
  __aicore__ void ApplyDeferredAlpha(int32_t rowBase,int32_t member) {
    const int32_t activeRows=static_cast<int32_t>(Max64(0,
        Min64(kHalfRows,qcount*(g.hq/g.hk)-rowBase)));
    if(activeRows==0)return;
    const int32_t softmaxRows=(activeRows+7)/8*8;
    auto tmp=scratchBuf.Get<float>();auto acc=accBuf.Get<float>()[(rowBase-lane*kLaneRows)*D];
    DataCopy(tmp,work[alphaOffset+member*kQueryRows+rowBase],softmaxRows);
    Fence<HardEvent::MTE2_V>();
    auto alphaBlocks=naturalBuf.Get<float>();
    Brcb(alphaBlocks,tmp,static_cast<uint8_t>((activeRows+7)/8),{1,8});
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams alphaParams{1,1,0,static_cast<uint8_t>(D/8),
        static_cast<uint8_t>(D/8),1};
    for(int32_t chunk=0;chunk<D/64;++chunk)
      Mul(acc[chunk*64],acc[chunk*64],alphaBlocks,static_cast<uint64_t>(64),
          static_cast<uint8_t>(activeRows),alphaParams);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ void ZeroPaddedClusterP(int32_t member) {
    if(lane!=1)return;
    const int32_t liveRows=static_cast<int32_t>(qcount*(g.hq/g.hk));
    auto zeros=dequantBuf.Get<float>();
    constexpr int32_t rowsPerCopy=kElements/kKvRows;
    for(int32_t row=liveRows;row<kQueryRows;row+=rowsPerCopy) {
      const int32_t count=static_cast<int32_t>(Min64(rowsPerCopy,kQueryRows-row));
      Duplicate(zeros,0.0F,count*kKvRows);Fence<HardEvent::V_MTE3>();
      DataCopy(work[pOffset+member*kQueryRows*kKvRows+row*kKvRows],zeros,count*kKvRows);
      Fence<HardEvent::MTE3_V>();
    }
    Fence<HardEvent::MTE3_MTE2>(); // #145: next score load cannot overwrite P UB.
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
  __aicore__ void SpillState(int32_t member) {
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();
    const int64_t accBase=accStateOffset+member*kQueryRows*D+lane*kLaneRows*D;
    const int64_t statBase=statStateOffset+member*2*kQueryRows+lane*kLaneRows;
    Fence<HardEvent::V_MTE3>();
    DataCopy(work[accBase],acc,kLaneRows*D);
    DataCopy(work[statBase],stats,kLaneRows);
    DataCopy(work[statBase+kQueryRows],stats[kLaneRows],kLaneRows);
    // These exact FP32 bits must reach GM before another member reuses UB
    // and before this member restores its own online state on a later tile.
    Fence<HardEvent::MTE3_MTE2>();Fence<HardEvent::MTE3_V>();
  }
  __aicore__ void RestoreState(int32_t member) {
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();
    const int64_t accBase=accStateOffset+member*kQueryRows*D+lane*kLaneRows*D;
    const int64_t statBase=statStateOffset+member*2*kQueryRows+lane*kLaneRows;
    DataCopy(acc,work[accBase],kLaneRows*D);
    DataCopy(stats,work[statBase],kLaneRows);
    DataCopy(stats[kLaneRows],work[statBase+kQueryRows],kLaneRows);
    Fence<HardEvent::MTE2_V>();
  }
  __aicore__ void SpillStats(int32_t member) {
    auto stats=statsBuf.Get<float>();
    const int64_t statBase=statStateOffset+member*2*kQueryRows+lane*kLaneRows;
    Fence<HardEvent::V_MTE3>();
    DataCopy(work[statBase],stats,kLaneRows);
    DataCopy(work[statBase+kQueryRows],stats[kLaneRows],kLaneRows);
    Fence<HardEvent::MTE3_MTE2>();Fence<HardEvent::MTE3_V>();
  }
  __aicore__ void RestoreStats(int32_t member) {
    auto stats=statsBuf.Get<float>();
    const int64_t statBase=statStateOffset+member*2*kQueryRows+lane*kLaneRows;
    DataCopy(stats,work[statBase],kLaneRows);
    DataCopy(stats[kLaneRows],work[statBase+kQueryRows],kLaneRows);
    Fence<HardEvent::MTE2_V>();
  }
  __aicore__ void VectorCluster(const TaskState* members) {
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();
    int32_t memberErrors[4];
    for(int32_t member=0;member<4;++member) {
      UseTask(members[member]);qOffset=member*kQueryRows*D;error=0;
      Duplicate(acc,0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
      Duplicate(stats,kEmptyMax,kLaneRows);
      Duplicate(stats[kLaneRows],0.0F,kLaneRows);
      Fence<HardEvent::V_S>();
      LoadQueries();
      memberErrors[member]=error;
      SpillState(member);
    }
    const int64_t first=members[0].kvbegin,last=members[0].kvend;
    for(int64_t start=first;start<last;start+=kKvRows) {
      UseTask(members[0]);error=0;
      // The four groups have identical page and pre-mask finite-check domain.
      // Keep fe0's K-then-V subtile order and all metadata/status checks.
      for(int32_t subtile=0;subtile<kKvSubtiles;++subtile) {
        const int64_t subStart=start+subtile*(2*kHalfKv);
        LoadPacked(subStart);Unpack(false);PublishKv(false,subtile);
        Unpack(true);PublishKv(true,subtile);
      }
      const int32_t kvError=error;
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady); // All four QK score matrices are published.
      for(int32_t member=0;member<4;++member) {
        UseTask(members[member]);qOffset=member*kQueryRows*D;
        error=memberErrors[member];
        if(kvError)error=kvError;
        RestoreStats(member); // acc stays in GM until after the one batched PV.
        for(int32_t block=0;block<kRowBlocks;++block)
          Softmax(start,lane*kLaneRows+block*kHalfRows,member);
        ZeroPaddedClusterP(member);
        memberErrors[member]=error;
        SpillStats(member);
      }
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady); // P and alpha complete.
      CrossCoreWaitFlag(kCubeReady); // One full-M PV result is complete.
      for(int32_t member=0;member<4;++member) {
        UseTask(members[member]);qOffset=member*kQueryRows*D;
        error=memberErrors[member];
        RestoreState(member);
        for(int32_t block=0;block<kRowBlocks;++block)
          ApplyDeferredAlpha(lane*kLaneRows+block*kHalfRows,member);
        auto result=dequantBuf.Get<float>();
        const int32_t activeRows=static_cast<int32_t>(qcount*(g.hq/g.hk));
        for(int32_t row=lane*kLaneRows;row<(lane+1)*kLaneRows && row<activeRows;
            row+=kHalfKv) {
          const int32_t count=static_cast<int32_t>(Min64(kHalfKv,activeRows-row));
          DataCopy(result,work[pvOffset+member*kQueryRows*D+row*D],count*D);
          Fence<HardEvent::MTE2_V>();
          auto target=acc[(row-lane*kLaneRows)*D];
          Add(target,target,result,count*D);PipeBarrier<PIPE_V>();
          Fence<HardEvent::V_MTE2>();
        }
        SpillState(member);
      }
      CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
    }
    for(int32_t member=0;member<4;++member) {
      UseTask(members[member]);qOffset=member*kQueryRows*D;
      error=memberErrors[member];RestoreState(member);
      Fence<HardEvent::V_S>();
      for(int32_t r=0;r<kLaneRows;++r) {
        const float sum=stats.GetValue(kLaneRows+r);
        if(sum>0.0F && Finite(sum))Muls(acc[r*D],acc[r*D],1.0F/sum,D);
        else {if(sum!=0.0F)error=2;Duplicate(acc[r*D],0.0F,D);}
        PipeBarrier<PIPE_V>();
      }
      Fence<HardEvent::V_MTE3>();
      DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      DataCopy(acc,work[rotOffset+lane*kLaneRows*D],kLaneRows*D);
      Fence<HardEvent::MTE2_V>();CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
      PublishRows(members[member].id);
    }
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
  TPipe pipe;CubeMatmul mm;
  Geometry g;
  GlobalTensor<bfloat16_t> q,ck,cv,wk,wv;
  GlobalTensor<float> qr,rv,out,outLse,work;
  GlobalTensor<uint8_t> bytes;GlobalTensor<int32_t> bt,outStatus;
  GlobalTensor<int64_t> wt,taskGm,clusterStats;
  TBuf<TPosition::VECCALC> taskBuf,packedBuf,planeBuf,naturalBuf,wordBuf,maskBuf,
      wordIndexBuf,laneIndexBuf,metadataIndexBuf,metadataHalfBuf,dequantBuf,
      scoreBuf,accBuf,qFloatBuf,qBf16Buf,statsBuf,scratchBuf,addressBuf;
  int64_t coreIndex,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin;
  int64_t qOffset,kOffset,vOffset,scoreOffset,pOffset,pvOffset,rotOffset;
  int64_t accStateOffset,statStateOffset,alphaOffset,counters[8];
  int32_t lane=0,error=0,taskError=0,liveKvRows=0;
};
}
#define OSCAR_CV_ARGUMENTS GM_ADDR query, GM_ADDR queryRot, GM_ADDR key, GM_ADDR value, \
    GM_ADDR rotation, GM_ADDR raw, GM_ADDR table, GM_ADDR wk, GM_ADDR wv, GM_ADDR tags, \
    GM_ADDR tasks, GM_ADDR output, GM_ADDR lse, GM_ADDR status, GM_ADDR workspace, \
    GM_ADDR clusterStats, \
    int64_t tokens, int64_t hq, int64_t hk, int64_t dim, int64_t requests, \
    int64_t columns, int64_t taskCount, int64_t blockTokens, int64_t blocks, \
    int64_t ssmOffset, int64_t pageStride, int64_t windowStride, int64_t tagStride, \
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits, float scale
#define OSCAR_CV_INIT op.Init(query,queryRot,key,value,rotation,raw,table,wk,wv,tags, \
    tasks,output,lse,status,workspace,clusterStats,g);op.Process()
extern "C" __global__ __aicore__ void oscar_attention_cv_batched4_kernel(OSCAR_CV_ARGUMENTS) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  const Geometry g{tokens,hq,hk,requests,columns,taskCount,blockTokens,blocks,ssmOffset,
      pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale};
  if(dim==64) {AttentionCv<64> op;OSCAR_CV_INIT;}
  else if(dim==128) {AttentionCv<128> op;OSCAR_CV_INIT;}
  else {AttentionCv<256> op;OSCAR_CV_INIT;}
}
#ifndef ASCENDC_CPU_DEBUG
namespace oscar_ascend {
void attention_cv_batched4_launch(void* stream,void* query,void* queryRot,void* key,void* value,
    void* rotation,void* raw,void* table,void* wk,void* wv,void* tags,void* tasks,
    void* output,void* lse,void* status,void* workspace,void* clusterStats,
    int64_t tokens,int64_t hq,
    int64_t hk,int64_t dim,int64_t requests,int64_t columns,int64_t taskCount,
    int64_t blockTokens,int64_t blocks,int64_t ssmOffset,int64_t pageStride,
    int64_t windowStride,int64_t tagStride,int64_t sink,int64_t recent,
    int64_t speculative,int64_t splits,float scale,uint32_t cores) {
  oscar_attention_cv_batched4_kernel<<<cores,nullptr,stream>>>(
      static_cast<uint8_t*>(query),static_cast<uint8_t*>(queryRot),
      static_cast<uint8_t*>(key),static_cast<uint8_t*>(value),
      static_cast<uint8_t*>(rotation),static_cast<uint8_t*>(raw),
      static_cast<uint8_t*>(table),static_cast<uint8_t*>(wk),static_cast<uint8_t*>(wv),
      static_cast<uint8_t*>(tags),static_cast<uint8_t*>(tasks),static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(lse),static_cast<uint8_t*>(status),static_cast<uint8_t*>(workspace),
      static_cast<uint8_t*>(clusterStats),
      tokens,hq,hk,dim,requests,columns,taskCount,blockTokens,blocks,ssmOffset,
      pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale);
}
}
#endif
