"""Generate the isolated, same-math source0 decode pipeline from this repo.

Only the current production SIMD kernel is read. No production source is
overwritten; math methods remain textually identical in the experiment.
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'csrc/kernels/attention_cv_striped_decode_simd.cpp'
DEST = ROOT / 'csrc/experiments/decode_history_pipeline/attention_pipeline.cpp'
BASELINE_SHA256 = 'c67219ce15c98e932acaa8b1c5515a3f31c99dd6e4963823623528ba2b3a3930'

METHODS = r'''
  // source0 only: cache one next packed tile, split K/V dequant across the
  // current QK/PV windows. Each array is bounded by two HalfKv64 subtiles.
  __aicore__ void PipelineUnpack(TBuf<TPosition::VECCALC>& packedInput,bool value) {
    oscar_ascend_striped_simd::Unpack<D,kHalfKv>(value,liveKvRows,error,packedInput,wordBuf,
        planeBuf,naturalBuf,maskBuf,wordIndexBuf,laneIndexBuf,metadataIndexBuf,
        metadataHalfBuf,dequantBuf,scratchBuf);
  }
  __aicore__ void PipelinePublishKv(bool value,int32_t subtile,int32_t buffer) {
    const int64_t slot=subtile*(2*kHalfKv)+lane*kHalfKv;
    Fence<HardEvent::V_MTE3>();
    DataCopy(work[(value?vOffset:kOffset)+buffer*(2*kKvRows*D)+slot*D],
        dequantBuf.Get<float>(),kElements);
    Fence<HardEvent::MTE3_V>();
  }
  __aicore__ void PrefetchKeys(int64_t start,int32_t buffer) {
    const int32_t savedError=error;
    for(int32_t sub=0;sub<kKvSubtiles;++sub) {
      auto& packedInput=sub==0?packedBuf:packedNextBuf;
      error=0;
      LoadPackedInto(start+sub*(2*kHalfKv),packedInput);
      pendingLive[sub]=liveKvRows;
      PipelineUnpack(packedInput,false);
      pendingKeyError[sub]=error;
      PipelinePublishKv(false,sub,buffer);
    }
    error=savedError;
  }
  __aicore__ void PrefetchValues(int32_t buffer) {
    const int32_t savedError=error;
    for(int32_t sub=0;sub<kKvSubtiles;++sub) {
      auto& packedInput=sub==0?packedBuf:packedNextBuf;
      error=0;liveKvRows=pendingLive[sub];
      PipelineUnpack(packedInput,true);
      pendingValueError[sub]=error;
      PipelinePublishKv(true,sub,buffer);
    }
    error=savedError;
  }
  __aicore__ void CommitPrefetchErrors() {
    // Baseline order is Load/K0,V0,Load/K1,V1. Speculative loads may run
    // earlier, but their errors must not overwrite an older score failure
    // until the previous PV accumulation has completed.
    for(int32_t sub=0;sub<kKvSubtiles;++sub) {
      if(pendingKeyError[sub])error=pendingKeyError[sub];
      if(pendingValueError[sub])error=pendingValueError[sub];
    }
  }
  __aicore__ void PipelineConsumePv() {
    auto result=dequantBuf.Get<float>();auto acc=accBuf.Get<float>();
    const int32_t activeRows=static_cast<int32_t>(qcount*(g.hq/g.hk));
    for(int32_t row=lane*kLaneRows;row<(lane+1)*kLaneRows && row<activeRows;
        row+=kHalfKv) {
      const int32_t count=static_cast<int32_t>(Min64(kHalfKv,
          Min64(activeRows,(lane+1)*kLaneRows)-row));
      DataCopy(result,work[pvOffset+row*D],count*D);
      Fence<HardEvent::MTE2_V>();
      auto target=acc[(row-lane*kLaneRows)*D];
      Add(target,target,result,count*D);PipeBarrier<PIPE_V>();
      Fence<HardEvent::V_MTE2>();
    }
  }
  __aicore__ void HistoryCubeTask() {
    const int32_t rows=static_cast<int32_t>(qcount*(g.hq/g.hk));
    int32_t buffer=0;
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows,buffer^=1) {
      CrossCoreWaitFlag(kKvReady);
      Matmul(qOffset,kOffset+buffer*(2*kKvRows*D),scoreOffset,rows,kKvRows,D);
      CrossCoreSetFlag<2,PIPE_FIX>(kScoreReady);
      CrossCoreWaitFlag(kPReady);
      Matmul(pOffset,vOffset+buffer*(2*kKvRows*D),pvOffset,rows,D,kKvRows,false,false);
      CrossCoreSetFlag<2,PIPE_FIX>(kPvReady);
      // Next QK may overwrite score/P after this PV has finished reading it.
      // The next PReady cannot be sent before Vector consumes this PV, so
      // the one PV result buffer cannot be overwritten early.
    }
    CrossCoreWaitFlag(kNormReady);
    Matmul(qOffset,0,rotOffset,rows,D,D,true);
    CrossCoreSetFlag<2,PIPE_FIX>(kRotReady);
    CrossCoreWaitFlag(kTaskDone);
  }
  __aicore__ void HistoryVectorTask(int64_t id) {
    error=0;
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();
    Duplicate(acc,0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
    Duplicate(stats,kEmptyMax,kLaneRows);
    Duplicate(stats[kLaneRows],0.0F,kLaneRows);Fence<HardEvent::V_S>();
    LoadQueries();
    for(int32_t sub=0;sub<kKvSubtiles;++sub) {
      LoadPacked(kvbegin+sub*(2*kHalfKv));
      Unpack(false);PipelinePublishKv(false,sub,0);
      Unpack(true);PipelinePublishKv(true,sub,0);
    }
    CrossCoreSetFlag<2,PIPE_MTE3>(kKvReady);
    int32_t buffer=0;
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows,buffer^=1) {
      const bool hasNext=start+kKvRows<kvend;
      // Packed buffers, metadata scratch and dequant UB are not used by
      // Cube. K is fully published before Softmax reuses dequant UB.
      if(hasNext)PrefetchKeys(start+kKvRows,buffer^1);
      CrossCoreWaitFlag(kScoreReady);
      for(int32_t block=0;block<kRowBlocks;++block)
        Softmax(start,lane*kLaneRows+block*kHalfRows);
      CrossCoreSetFlag<2,PIPE_MTE3>(kPReady);
      if(hasNext) {
        PrefetchValues(buffer^1);
        CrossCoreSetFlag<2,PIPE_MTE3>(kKvReady);
      }
      CrossCoreWaitFlag(kPvReady);
      PipelineConsumePv();
      if(hasNext)CommitPrefetchErrors();
    }
    Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kLaneRows;++r) {
      const float sum=stats.GetValue(kLaneRows+r);
      if(sum>0.0F && Finite(sum))Muls(acc[r*D],acc[r*D],1.0F/sum,D);
      else {if(sum!=0.0F)error=2;Duplicate(acc[r*D],0.0F,D);}
      PipeBarrier<PIPE_V>();
    }
    Fence<HardEvent::V_MTE3>();DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D);
    CrossCoreSetFlag<2,PIPE_MTE3>(kNormReady);
    CrossCoreWaitFlag(kRotReady);
    DataCopy(acc,work[rotOffset+lane*kLaneRows*D],kLaneRows*D);
    Fence<HardEvent::MTE2_V>();
    PublishRows(id);
    CrossCoreSetFlag<2,PIPE_MTE3>(kTaskDone);
  }
'''


def build():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise RuntimeError('Production decode source changed: re-audit substitutions and buffer ownership before regenerating')
    source = SOURCE.read_text()
    body = source[source.index('#include "oscar_common.h"'):]
    body = body.replace('#include "oscar_common.h"', '#include "../../kernels/oscar_common.h"')
    body = body.replace('#include "attention_striped_unpack_simd.h"', '#include "../../kernels/attention_striped_unpack_simd.h"')
    body = body.replace('#include "../include/', '#include "../../include/')
    body = body.replace('constexpr uint16_t kVectorReady=8, kCubeReady=9;',
        'constexpr uint16_t kVectorReady=8, kCubeReady=9;\n'
        'constexpr uint16_t kKvReady=8,kScoreReady=9,kPReady=10,kPvReady=11,\n'
        '                   kNormReady=12,kRotReady=13,kTaskDone=14;')
    body = body.replace('oscar_ascend::attention_workspace_per_core(D)/4',
        '((2*kQueryRows+4*kKvRows)*D+kQueryRows*kKvRows)')
    body = body.replace('scoreOffset=(kQueryRows+2*kKvRows)*D;',
                        'scoreOffset=(kQueryRows+4*kKvRows)*D;')
    body = body.replace('pipe.InitBuffer(taskBuf,128);pipe.InitBuffer(packedBuf,kHalfKv*kPackedStride);',
        'pipe.InitBuffer(taskBuf,128);pipe.InitBuffer(packedBuf,kHalfKv*kPackedStride);\n'
        '      pipe.InitBuffer(packedNextBuf,kHalfKv*kPackedStride);')
    body = body.replace('if ASCEND_IS_AIC {CubeTask();}\n        else {VectorTask(id);}',
        'if(kind==0 && kvbegin<kvend) {\n'
        '          if ASCEND_IS_AIC {HistoryCubeTask();} else {HistoryVectorTask(id);}\n'
        '        } else {\n'
        '          if ASCEND_IS_AIC {CubeTask();} else {VectorTask(id);}\n        }')
    load_start = body.index('  __aicore__ void LoadPacked(int64_t start) {')
    load_end = body.index('  __aicore__ void Unpack(bool value)',load_start)
    load = body[load_start:load_end].replace(
        'LoadPacked(int64_t start)', 'LoadPackedInto(int64_t start,TBuf<TPosition::VECCALC>& destination)').replace(
        'auto packed=packedBuf.Get<uint8_t>();', 'auto packed=destination.Get<uint8_t>();')
    body = body[:load_start] + load + (
        '  __aicore__ void LoadPacked(int64_t start) {LoadPackedInto(start,packedBuf);}\n') + body[load_end:]
    body = body.replace('  __aicore__ void PublishRows(int64_t id) {',
                        METHODS+'\n  __aicore__ void PublishRows(int64_t id) {')
    body = body.replace('taskBuf,packedBuf,planeBuf', 'taskBuf,packedBuf,packedNextBuf,planeBuf')
    body = body.replace('  int32_t lane=0,error=0,taskError=0,liveKvRows=0;',
        '  int32_t lane=0,error=0,taskError=0,liveKvRows=0;\n'
        '  int32_t pendingLive[kKvSubtiles],pendingKeyError[kKvSubtiles],pendingValueError[kKvSubtiles];')
    for old,new in (
        ('oscar_attention_cv_striped_decode_simd_kernel','oscar_attention_history_pipeline_q4_kernel'),
        ('oscar_attention_cv_striped_q1_simd_kernel','oscar_attention_history_pipeline_q1_kernel'),
        ('attention_cv_striped_decode_simd_launch','attention_history_pipeline_q4_launch'),
        ('attention_cv_striped_q1_simd_launch','attention_history_pipeline_q1_launch')):
        body = body.replace(old,new)
    header = (
        '// SPDX-License-Identifier: Apache-2.0\n'
        '// ISOLATED EXPERIMENT. Source: current production SIMD decode kernel, SHA256\n'
        '// '+hashlib.sha256(SOURCE.read_bytes()).hexdigest()+'\n'
        '// Archive #126/#129/#140/#145/#148/#154/#155, startup D.4 four questions:\n'
        '// 1. Fused INT2 history attention for q4/q1; window/current remain original.\n'
        '// 2. D.4 full-history restore cost 6499.8-6655.1ms; no such tensor here.\n'
        '// 3. Two fixed KV256 GM buffers, two packed HalfKv64 UB buffers. Next K\n'
        '//    loads during current QK; next V during current PV. FP32 operations,\n'
        '//    KV order and final RvT stay unchanged; future errors commit in old order.\n'
        '// 4. Overlap is a hypothesis. No CPU/CANN/model/NPU speed claimed yet.\n'
        '// D256 workspace 1146880 bytes/core, additional UB10240 bytes/lane;\n'
        '// no memory grows with historical length. Source0 nonempty only.\n')
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(header+body)
    print(DEST.relative_to(ROOT))


if __name__ == '__main__':
    build()
