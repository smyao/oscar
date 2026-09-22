// SPDX-License-Identifier: Apache-2.0
// D.4: merge/materialize; historical materialize ~0.5 ms, host prepare ~725 ms.
// Structure: one device launch merges bounded split outputs; no history KV
// recovery, no Python per-request loop, no host synchronization or allocation.
// Complexity: O(rows*splits*D), splits<=128, <4 KiB UB/core, independent of
// history except the chosen bounded split count. Target below decode FIA's
// 0.6-1.1 ms reference; NOT compiled or timed on target CANN/NPU here.
// Archive G30-G34/#4-16 require every output/LSE row written, including empty.
// Native: moe_gating_top_k_generalized.h:185-204 reduction/event sequence;
// moe_gating_top_k_without_group.h:238 ReduceSum; swiglu_group_quant_base.h:53-56 infinity.
// No scalar GlobalTensor loads/stores: LSE/status DMA avoids per-core 64B
// DCache lost updates (G30/#12) and stale metadata when graph inputs change.
#include "oscar_common.h"
#include "../include/oscar_launch.h"

using namespace oscar_ascend_device;
constexpr float kNegInf=-__builtin_inff();

class OscarMerge {
 public:
  __aicore__ inline void Init(GM_ADDR partial, GM_ADDR partialLse,
      GM_ADDR output, GM_ADDR lse, GM_ADDR status, int64_t rows,
      int64_t splits, int64_t dim) {
    partial_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));
    partialLse_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partialLse));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    lse_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
    status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    rows_=rows; splits_=splits; dim_=dim;
    pipe_.InitBuffer(weightsBuf_,kMaxSplits*4);
    pipe_.InitBuffer(valuesBuf_,kMaxDim*4);
    pipe_.InitBuffer(accBuf_,kMaxDim*4);
    pipe_.InitBuffer(scalarBuf_,32);
    pipe_.InitBuffer(reduceBuf_,kMaxDim*4);
    pipe_.InitBuffer(publishBuf_,64);
  }
  __aicore__ inline void Process() {
    auto weights=weightsBuf_.Get<float>(); auto values=valuesBuf_.Get<float>();
    auto acc=accBuf_.Get<float>(); auto scalar=scalarBuf_.Get<float>();
    auto reduce=reduceBuf_.Get<float>();
    for (int64_t row=GetBlockIdx(); row<rows_; row+=GetBlockNum()) {
      int32_t rowStatus=0;
      float maximum=kNegInf;
      bool invalid=false;
      Fence<HardEvent::S_MTE2>();
      DataCopyExtParams lseCopy{1,static_cast<uint32_t>(splits_*4),0,0,0};
      DataCopyPadExtParams<float> lsePad{false,0,0,0};
      DataCopyPad(weights,partialLse_[row*splits_],lseCopy,lsePad);
      Fence<HardEvent::MTE2_S>();
      // Bounded scan of LSE scalars, never a loop over history tokens.
      for (int32_t s=0; s<splits_; ++s) {
        const float x=weights.GetValue(s);
        if (x!=kNegInf && !Finite(x)) invalid=true;
        if (x>maximum) maximum=x;
      }
      Duplicate(acc,0.0F,dim_);
      if (invalid || maximum==kNegInf) {
        Fence<HardEvent::V_MTE3>();
        DataCopy(output_[row*dim_],acc,dim_);
        Publish(row,kNegInf,invalid ? 2 : 0);
        Fence<HardEvent::MTE3_V>();
        continue;
      }
      Fence<HardEvent::S_V>();
      Adds(weights,weights,-maximum,splits_); PipeBarrier<PIPE_V>();
      Exp(weights,weights,splits_); Fence<HardEvent::V_S>();
      float sum=0.0F;
      for (int32_t s=0; s<splits_; ++s) sum+=weights.GetValue(s);
      if (!(sum>0.0F) || !Finite(sum)) {
        Fence<HardEvent::V_MTE3>();
        DataCopy(output_[row*dim_],acc,dim_);
        Publish(row,kNegInf,3);
        Fence<HardEvent::MTE3_V>();
        continue;
      }
      for (int32_t s=0; s<splits_; ++s) {
        const float weight=weights.GetValue(s)/sum;
        if (weight==0.0F) continue;  // Empty output may contain poison/NaN.
        DataCopy(values,partial_[(row*splits_+s)*dim_],dim_);
        Fence<HardEvent::MTE2_V>();
        Muls(values,values,weight,dim_); PipeBarrier<PIPE_V>();
        Add(acc,acc,values,dim_); PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
      }
      ReduceSum(scalar,acc,reduce,dim_); Fence<HardEvent::V_S>();
      if (!Finite(scalar.GetValue(0))) rowStatus=2;
      Fence<HardEvent::V_MTE3>();
      DataCopy(output_[row*dim_],acc,dim_);
      // #91: scalar Exp/Log return overload is not supported on A2.
      scalar.SetValue(0,sum); Fence<HardEvent::S_V>();
      Log(scalar,scalar,1); Fence<HardEvent::V_S>();
      Publish(row,maximum+scalar.GetValue(0),rowStatus);
      Fence<HardEvent::MTE3_V>();
    }
  }
 private:
  __aicore__ inline void Publish(int64_t row,float lseValue,int32_t statusValue) {
    auto lseLocal=publishBuf_.Get<float>();
    auto statusLocal=publishBuf_.Get<int32_t>()[8];
    lseLocal.SetValue(0,lseValue); statusLocal.SetValue(0,statusValue);
    Fence<HardEvent::S_MTE3>();
    DataCopyExtParams copy{1,4,0,0,0};
    DataCopyPad(lse_[row],lseLocal,copy);
    DataCopyPad(status_[row],statusLocal,copy);
    Fence<HardEvent::MTE3_S>();
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> weightsBuf_,valuesBuf_,accBuf_,scalarBuf_,reduceBuf_,publishBuf_;
  GlobalTensor<float> partial_,partialLse_,output_,lse_;
  GlobalTensor<int32_t> status_;
  int64_t rows_,splits_,dim_;
};
extern "C" __global__ __aicore__ void oscar_merge_lse_kernel(GM_ADDR partial,
    GM_ADDR partialLse,GM_ADDR output,GM_ADDR lse,GM_ADDR status,
    int64_t rows,int64_t splits,int64_t dim) {
  OscarMerge op;op.Init(partial,partialLse,output,lse,status,rows,splits,dim);
  op.Process();
}
namespace oscar_ascend {
void merge_lse_launch(void* stream,void* partial,void* partialLse,void* output,
    void* lse,void* status,int64_t rows,int64_t splits,int64_t dim,uint32_t cores) {
  oscar_merge_lse_kernel<<<cores,nullptr,stream>>>(static_cast<uint8_t*>(partial),
      static_cast<uint8_t*>(partialLse),static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(lse),static_cast<uint8_t*>(status),rows,splits,dim);
}
}
