// SPDX-License-Identifier: Apache-2.0
// Archive G27, #13-22/#37-49/#70-71/#81-83/#91-92/#111:
// raw window stays unrotated; scales/zero round to FP16 before quantization;
// exact percentile order statistics; each live row and status has one writer.
// D.4 four questions, reread against its COMPLETE original timing JSON:
// 1. rotate/clip/store corresponds to phase0_store/phase1_stores.
// 2. Prior writes took 208.6-216.0 ms / 16K and repeated history recovery took
//    6499.8-6655.1 ms; prepare reached 724.7-726.2 ms on the CPU.
// 3. One fused launch reads ONLY new K/V, keeps transforms/sort/quantization
//    in UB, writes final packed bytes and page-local exact BF16 window rows.
//    Hadamard uses log2(D) vector butterflies; generic rotations reuse each
//    FP32 matrix tile across eight rows, without FP16 operand truncation.
//    Sink rows never duplicate into recent storage. Negative-slot query rows
//    clear their UB input/output and status; all-padding tiles skip transform.
//    No history is read or restored, no host token loop and no tiny-op chain.
// 4. Work is O(new_tokens*heads*D log D) for Hadamard, O(new_tokens*heads*D^2)
//    for calibrated dense rotations; UB <= 64 KiB per core. Targets are below
//    0.6-1.1 ms decode / materially below 215 ms per 16K writes, NOT measured
//    performance claims. Dense-transform throughput requires device profiling.
// API evidence: native moe_gating_top_k_generalized.h (ReduceSum, Sort32,
// MrgSort, Gather); dequant_swiglu_quant.h (Mins/Maxs); bgmv_expand.cpp (DMA).
#include "oscar_common.h"
#include "../include/oscar_rotation_launch.h"

using namespace oscar_ascend_device;
namespace {
constexpr int32_t kRowTile = 8;
constexpr int32_t kColumnTile = 16;

template <typename T>
class RotationEngine {
 public:
  __aicore__ inline void Init(TPipe* pipe, int32_t dim, bool hadamard) {
    pipe_=pipe; dim_=dim; hadamard_=hadamard;
    pipe_->InitBuffer(inputBuf_,kRowTile*kMaxDim*4);
    pipe_->InitBuffer(sourceBuf_,kRowTile*kMaxDim*4);
    pipe_->InitBuffer(outputBuf_,kRowTile*kMaxDim*4);
    pipe_->InitBuffer(matrixBuf_,hadamard ? 32 : kColumnTile*kMaxDim*4);
    pipe_->InitBuffer(workBuf_,kMaxDim*4);
    pipe_->InitBuffer(redBuf_,kMaxDim*4);
    pipe_->InitBuffer(scalarBuf_,64);
    pipe_->InitBuffer(gatherBuf_,hadamard ? 8*kMaxDim*4 : 32);
    pipe_->InitBuffer(signBuf_,hadamard ? 8*kMaxDim*4 : 32);
    if (hadamard_) {
      auto indices=gatherBuf_.template Get<uint32_t>();
      auto signs=signBuf_.template Get<float>();
      int32_t stage=0;
      for (int32_t distance=1;distance<dim_;distance*=2,++stage) {
        for (int32_t i=0;i<dim_;++i) {
          indices.SetValue(stage*dim_+i,static_cast<uint32_t>((i^distance)*4));
          signs.SetValue(stage*dim_+i,(i&distance) ? -1.0F : 1.0F);
        }
      }
      Fence<HardEvent::S_V>();
    }
  }
  __aicore__ inline void Load(const GlobalTensor<T>& source,int64_t offset,
                             int32_t rows) {
    auto original=sourceBuf_.template Get<T>();
    auto input=inputBuf_.template Get<float>();
    DataCopy(original,source[offset],rows*dim_);
    Fence<HardEvent::MTE2_V>();
    if constexpr (IsSameType<T,float>::value) {
      DataCopy(input,original,rows*dim_);
    } else {
      Cast(input,original,RoundMode::CAST_NONE,rows*dim_);
    }
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline LocalTensor<T> Original() {return sourceBuf_.template Get<T>();}
  __aicore__ inline LocalTensor<float> Result() {return outputBuf_.template Get<float>();}
  __aicore__ inline void ZeroInputRow(int32_t row) {
    Duplicate(inputBuf_.template Get<float>()[row*dim_],0.0F,dim_);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Transform(const GlobalTensor<float>& matrix,int32_t rows) {
    auto input=inputBuf_.template Get<float>();
    auto output=outputBuf_.template Get<float>();
    auto work=workBuf_.template Get<float>();
    auto red=redBuf_.template Get<float>();
    auto scalar=scalarBuf_.template Get<float>();
    auto localMatrix=matrixBuf_.template Get<float>();
    if (hadamard_) {
      // Matrix[0,0] preserves the artifact's exact FP32 normalization constant.
      DataCopy(localMatrix,matrix,8); Fence<HardEvent::MTE2_S>();
      const float normalization=localMatrix.GetValue(0);
      Fence<HardEvent::S_V>();
      auto indices=gatherBuf_.template Get<uint32_t>();
      auto signs=signBuf_.template Get<float>();
      for (int32_t row=0;row<rows;++row) {
        auto x=input[row*dim_];
        int32_t stage=0;
        for (int32_t distance=1;distance<dim_;distance*=2,++stage) {
          Gather(work,x,indices[stage*dim_],static_cast<uint32_t>(0),dim_);
          PipeBarrier<PIPE_V>();
          Mul(x,x,signs[stage*dim_],dim_); PipeBarrier<PIPE_V>();
          Add(x,x,work,dim_); PipeBarrier<PIPE_V>();
        }
        Muls(output[row*dim_],x,normalization,dim_); PipeBarrier<PIPE_V>();
      }
      return;
    }
    for (int32_t column=0;column<dim_;column+=kColumnTile) {
      DataCopy(localMatrix,matrix[column*dim_],kColumnTile*dim_);
      Fence<HardEvent::MTE2_V>();
      for (int32_t row=0;row<rows;++row) {
        for (int32_t c=0;c<kColumnTile;++c) {
          Mul(work,input[row*dim_],localMatrix[c*dim_],dim_);
          PipeBarrier<PIPE_V>();
          ReduceSum(scalar,work,red,dim_); Fence<HardEvent::V_S>();
          output.SetValue(row*dim_+column+c,scalar.GetValue(0));
          Fence<HardEvent::S_V>();
        }
      }
      Fence<HardEvent::V_MTE2>();
    }
  }
 private:
  TPipe* pipe_; int32_t dim_; bool hadamard_;
  TBuf<TPosition::VECCALC> inputBuf_,sourceBuf_,outputBuf_,matrixBuf_,workBuf_,
      redBuf_,scalarBuf_,gatherBuf_,signBuf_;
};

// Exact descending Sort32 + two bounded four-way merges, rather than binary
// threshold approximation. Interpolation selects adjacent absolute values.
class ClipQuantize {
 public:
  __aicore__ inline void Init(TPipe* pipe,int32_t dim) {
    dim_=dim;
    pipe->InitBuffer(workBuf_,kMaxDim*4); pipe->InitBuffer(redBuf_,kMaxDim*4);
    pipe->InitBuffer(sortBuf_,2*kMaxDim*4); pipe->InitBuffer(mergeBuf_,2*kMaxDim*4);
    pipe->InitBuffer(indicesBuf_,kMaxDim*4); pipe->InitBuffer(divisorBuf_,kMaxDim*4);
    pipe->InitBuffer(qBuf_,kMaxDim*4); pipe->InitBuffer(scalarBuf_,64);
    pipe->InitBuffer(metaBuf_,32);
    auto indices=indicesBuf_.Get<uint32_t>();
    for (int32_t i=0;i<dim_;++i) indices.SetValue(i,static_cast<uint32_t>(i));
    Fence<HardEvent::S_V>();
  }
  __aicore__ inline int32_t Process(LocalTensor<float> x,float ratio,
                                  LocalTensor<uint8_t> packed,int32_t offset) {
    auto work=workBuf_.Get<float>(); auto red=redBuf_.Get<float>();
    auto scalar=scalarBuf_.Get<float>();
    ReduceSum(scalar,x,red,dim_); Fence<HardEvent::V_S>();
    if (!Finite(scalar.GetValue(0))) return 2;
    if (ratio>0.0F) {
      Abs(work,x,dim_); PipeBarrier<PIPE_V>();
      auto sorted=sortBuf_.Get<float>(); auto merged=mergeBuf_.Get<float>();
      Sort32(sorted,work,indicesBuf_.Get<uint32_t>(),dim_/32); PipeBarrier<PIPE_V>();
      if (dim_==64) {
        Merge(merged,sorted,32,2); sorted=merged;
      } else if (dim_==128) {
        Merge(merged,sorted,32,4); sorted=merged;
      } else {
        Merge(merged,sorted,32,4);
        Merge(merged[256],sorted[256],32,4);
        Merge(sorted,merged,128,2);
      }
      Fence<HardEvent::V_S>();
      const float rank=ratio*static_cast<float>(dim_-1);
      const int32_t lower=static_cast<int32_t>(rank);
      const int32_t upper=lower+1<dim_ ? lower+1 : lower;
      const float fraction=rank-static_cast<float>(lower);
      const float lo=sorted.GetValue(2*(dim_-1-lower));
      const float hi=sorted.GetValue(2*(dim_-1-upper));
      // Match torch's stable lerp: choose the endpoint nearer the rank.
      const float threshold=fraction<0.5F ? lo+fraction*(hi-lo)
          : hi-(hi-lo)*(1.0F-fraction);
      Mins(x,x,threshold,dim_); PipeBarrier<PIPE_V>();
      Maxs(x,x,-threshold,dim_); PipeBarrier<PIPE_V>();
    }
    ReduceMax(scalar,x,red,dim_); Fence<HardEvent::V_S>();
    const float maximum=scalar.GetValue(0);
    Muls(work,x,-1.0F,dim_); PipeBarrier<PIPE_V>();
    ReduceMax(scalar,work,red,dim_); Fence<HardEvent::V_S>();
    const float minimum=-scalar.GetValue(0);
    float scale=(maximum-minimum)/3.0F;
    if (scale<1.0e-8F) scale=1.0e-8F;
    Duplicate(scalar,0.0F,16); Fence<HardEvent::V_S>();
    scalar.SetValue(0,scale); scalar.SetValue(1,minimum);
    Fence<HardEvent::S_V>();
    auto meta=metaBuf_.Get<half>();
    Cast(meta,scalar,RoundMode::CAST_RINT,16); PipeBarrier<PIPE_V>();
    Cast(scalar,meta,RoundMode::CAST_NONE,16); Fence<HardEvent::V_S>();
    scale=scalar.GetValue(0); const float zero=scalar.GetValue(1);
    if (!(scale>0.0F) || !Finite(scale) || !Finite(zero)) return 3;
    auto divisor=divisorBuf_.Get<float>(); auto codes=qBuf_.Get<int32_t>();
    Duplicate(divisor,scale,dim_);
    Adds(work,x,-zero,dim_); PipeBarrier<PIPE_V>();
    Div(work,work,divisor,dim_); PipeBarrier<PIPE_V>();
    Adds(work,work,0.5F,dim_); PipeBarrier<PIPE_V>();
    Maxs(work,work,0.0F,dim_); PipeBarrier<PIPE_V>();
    Mins(work,work,3.0F,dim_); PipeBarrier<PIPE_V>();
    Cast(codes,work,RoundMode::CAST_TRUNC,dim_); Fence<HardEvent::V_S>();
    for (int32_t b=0;b<dim_/4;++b) {
      const int32_t byte=codes.GetValue(4*b)|(codes.GetValue(4*b+1)<<2)
          |(codes.GetValue(4*b+2)<<4)|(codes.GetValue(4*b+3)<<6);
      packed.SetValue(offset+b,static_cast<uint8_t>(byte));
    }
    auto words=meta.ReinterpretCast<uint16_t>();
    const uint16_t sc=words.GetValue(0),zr=words.GetValue(1);
    packed.SetValue(offset+dim_/4,static_cast<uint8_t>(sc&255));
    packed.SetValue(offset+dim_/4+1,static_cast<uint8_t>(sc>>8));
    packed.SetValue(offset+dim_/4+2,static_cast<uint8_t>(zr&255));
    packed.SetValue(offset+dim_/4+3,static_cast<uint8_t>(zr>>8));
    return 0;
  }
 private:
  __aicore__ inline void Merge(LocalTensor<float> dst,LocalTensor<float> src,
                              int32_t length,int32_t lists) {
    MrgSortSrcList<float> source;
    source.src1=src; source.src2=src[2*length];
    source.src3=src[lists==4 ? 4*length : 0];
    source.src4=src[lists==4 ? 6*length : 0];
    MrgSort4Info info;
    info.elementLengths[0]=length; info.elementLengths[1]=length;
    info.elementLengths[2]=lists==4 ? length : 0;
    info.elementLengths[3]=lists==4 ? length : 0;
    info.validBit=lists==4 ? 15 : 3;
    info.ifExhaustedSuspension=false; info.repeatTimes=1;
    MrgSort(dst,source,info); PipeBarrier<PIPE_V>();
  }
  int32_t dim_;
  TBuf<TPosition::VECCALC> workBuf_,redBuf_,sortBuf_,mergeBuf_,indicesBuf_,
      divisorBuf_,qBuf_,scalarBuf_,metaBuf_;
};

template <typename T>
class RotateOnly {
 public:
  __aicore__ inline void Run(GM_ADDR source,GM_ADDR rotation,GM_ADDR output,
      GM_ADDR status,int64_t rows,int32_t dim,bool hadamard,GM_ADDR slots,int64_t heads) {
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source));
    matrix_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotation));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    if (slots!=nullptr) slots_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
    engine_.Init(&pipe_,dim,hadamard);
    pipe_.InitBuffer(statusBuf_,32); pipe_.InitBuffer(redBuf_,kMaxDim*4);
    pipe_.InitBuffer(scalarBuf_,32); pipe_.InitBuffer(slotBuf_,32);
    for (int64_t base=GetBlockIdx()*kRowTile;base<rows;base+=GetBlockNum()*kRowTile) {
      const int32_t count=static_cast<int32_t>(rows-base<kRowTile ? rows-base : kRowTile);
      auto result=engine_.Result(); auto statusLocal=statusBuf_.Get<int32_t>();
      bool active[kRowTile]; bool anyActive=false;
      int64_t previousToken=-1; bool previousActive=true;
      for (int32_t row=0;row<count;++row) {
        if (slots!=nullptr) {
          const int64_t token=(base+row)/heads;
          if (token!=previousToken) {
            auto local=slotBuf_.Get<int64_t>();
            DataCopyExtParams copy{1,8,0,0,0};
            DataCopyPadExtParams<int64_t> pad{false,0,0,0};
            DataCopyPad(local,slots_[token],copy,pad); Fence<HardEvent::MTE2_S>();
            previousActive=local.GetValue(0)>=0; previousToken=token;
            Fence<HardEvent::S_MTE2>();
          }
          active[row]=previousActive;
        } else active[row]=true;
        anyActive=anyActive || active[row];
      }
      if (anyActive) {
        engine_.Load(input_,base*dim,count);
        for (int32_t row=0;row<count;++row) if (!active[row]) engine_.ZeroInputRow(row);
        engine_.Transform(matrix_,count);
      }
      for (int32_t row=0;row<count;++row) {
        if (!active[row]) {
          Duplicate(result[row*dim],0.0F,dim); PipeBarrier<PIPE_V>();
          statusLocal.SetValue(row,0);
          continue;
        }
        ReduceSum(scalarBuf_.Get<float>(),result[row*dim],redBuf_.Get<float>(),dim);
        Fence<HardEvent::V_S>();
        statusLocal.SetValue(row,Finite(scalarBuf_.Get<float>().GetValue(0)) ? 0 : 2);
      }
      Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
      DataCopy(output_[base*dim],result,count*dim);
      DataCopyExtParams statusCopy{1,static_cast<uint32_t>(count*4),0,0,0};
      DataCopyPad(status_[base],statusLocal,statusCopy);
      Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
    }
  }
 private:
  TPipe pipe_; RotationEngine<T> engine_;
  TBuf<TPosition::VECCALC> statusBuf_,redBuf_,scalarBuf_,slotBuf_;
  GlobalTensor<T> input_; GlobalTensor<float> matrix_,output_; GlobalTensor<int32_t> status_;
  GlobalTensor<int64_t> slots_;
};

template <typename T>
class RotateStore {
 public:
  __aicore__ inline void Run(GM_ADDR key,GM_ADDR value,GM_ADDR rk,GM_ADDR rv,
      GM_ADDR slots,GM_ADDR positions,GM_ADDR packed,GM_ADDR rawKey,GM_ADDR rawValue,
      GM_ADDR tags,GM_ADDR status,int64_t tokens,int64_t heads,int32_t dim,
      int64_t blockTokens,int64_t blocks,int64_t offset,int64_t pageStride,
      int64_t kStride,int64_t vStride,int64_t tagStride,int64_t sink,
      int64_t recent,float kClip,float vClip,bool hadamard) {
    k_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key));
    v_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value));
    rk_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rk));
    rv_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rv));
    slots_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
    positions_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions));
    packed_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(packed));
    rawK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rawKey));
    rawV_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rawValue));
    tags_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tags));
    status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    engine_.Init(&pipe_,dim,hadamard); quant_.Init(&pipe_,dim);
    pipe_.InitBuffer(packBuf_,kRowTile*160); pipe_.InitBuffer(statusBuf_,32);
    pipe_.InitBuffer(metaBuf_,32); pipe_.InitBuffer(rawBuf_,kMaxDim*2);
    const int64_t total=tokens*heads; const int32_t headBytes=dim/2+8;
    for (int64_t base=GetBlockIdx()*kRowTile;base<total;base+=GetBlockNum()*kRowTile) {
      const int32_t count=static_cast<int32_t>(total-base<kRowTile ? total-base : kRowTile);
      int32_t errors[kRowTile];
      auto bytes=packBuf_.Get<uint8_t>();
      for (int32_t side=0;side<2;++side) {
        engine_.Load(side==0 ? k_ : v_,base*dim,count);
        engine_.Transform(side==0 ? rk_ : rv_,count);
        for (int32_t row=0;row<count;++row) {
          const int32_t error=quant_.Process(engine_.Result()[row*dim],
              side==0 ? kClip : vClip,bytes,row*160+side*(dim/4+4));
          if (side==0) errors[row]=error;
          else if (!errors[row]) errors[row]=error;
        }
      }
      // No cache row is published until BOTH quantizers succeeded.
      for (int32_t row=0;row<count;++row) {
        const int64_t linear=base+row, token=linear/heads, head=linear%heads;
        const int64_t slot=Read(slots_,token);
        if (slot<0) {errors[row]=0;continue;}
        if (slot>=blocks*blockTokens) {errors[row]=1;continue;}
        const int64_t position=Read(positions_,token);
        if (position<0) {errors[row]=4;continue;}
        if (errors[row]) continue;
        const int64_t page=slot/blockTokens,inPage=slot%blockTokens;
        const int64_t destination=offset+page*pageStride+inPage*heads*headBytes+head*headBytes;
        Fence<HardEvent::S_MTE3>();
        DataCopyExtParams copy{1,static_cast<uint32_t>(headBytes),0,0,0};
        DataCopyPad(packed_[destination],bytes[row*160],copy);
        Fence<HardEvent::MTE3_S>();
        // A later member of the same page/ring owns this raw slot. Native
        // slot mapping supplies one contiguous writable interval per page.
        bool keepRecent=recent>0 && position>=sink;
        if (keepRecent && token+recent<tokens) {
          const int64_t later=Read(slots_,token+recent);
          if (later>=0 && later/blockTokens==page && later==slot+recent) keepRecent=false;
        }
        if (position<sink) {
          WriteRaw(token,head,page,position,kStride,vStride,heads,dim);
          if (head==0) WriteTag(page*tagStride+position,inPage);
        }
        if (keepRecent) {
          const int64_t index=sink+inPage%recent;
          WriteRaw(token,head,page,index,kStride,vStride,heads,dim);
          if (head==0) WriteTag(page*tagStride+index,inPage);
        }
      }
      auto localStatus=statusBuf_.Get<int32_t>();
      for (int32_t row=0;row<count;++row) localStatus.SetValue(row,errors[row]);
      Fence<HardEvent::S_MTE3>();
      DataCopyExtParams copyStatus{1,static_cast<uint32_t>(count*4),0,0,0};
      DataCopyPad(status_[base],localStatus,copyStatus);
      Fence<HardEvent::MTE3_S>();
    }
  }
 private:
  __aicore__ inline int64_t Read(const GlobalTensor<int64_t>& source,int64_t index) {
    auto local=metaBuf_.Get<int64_t>(); DataCopyExtParams copy{1,8,0,0,0};
    DataCopyPadExtParams<int64_t> pad{false,0,0,0};
    DataCopyPad(local,source[index],copy,pad); Fence<HardEvent::MTE2_S>();
    const int64_t value=local.GetValue(0); Fence<HardEvent::S_MTE2>(); return value;
  }
  __aicore__ inline void WriteTag(int64_t index,int64_t value) {
    auto local=metaBuf_.Get<int64_t>(); local.SetValue(0,value);
    Fence<HardEvent::S_MTE3>(); DataCopyExtParams copy{1,8,0,0,0};
    DataCopyPad(tags_[index],local,copy); Fence<HardEvent::MTE3_S>();
  }
  __aicore__ inline void WriteRaw(int64_t token,int64_t head,int64_t page,
      int64_t index,int64_t kStride,int64_t vStride,int64_t heads,int32_t dim) {
    // The source K/V are read again only for the bounded exact window, never
    // for history. No rotation or quantizer rounding can touch these bytes.
    auto original=engine_.Original(); auto converted=rawBuf_.Get<bfloat16_t>();
    for (int32_t side=0;side<2;++side) {
      DataCopy(original,(side==0 ? k_ : v_)[(token*heads+head)*dim],dim);
      Fence<HardEvent::MTE2_V>();
      if constexpr (IsSameType<T,bfloat16_t>::value) {
        DataCopy(converted,original,dim);
      } else if constexpr (IsSameType<T,float>::value) {
        Cast(converted,original,RoundMode::CAST_RINT,dim);
      } else {
        auto temp=engine_.Result();
        Cast(temp,original,RoundMode::CAST_NONE,dim); PipeBarrier<PIPE_V>();
        Cast(converted,temp,RoundMode::CAST_RINT,dim);
      }
      Fence<HardEvent::V_MTE3>();
      DataCopy((side==0 ? rawK_ : rawV_)[page*(side==0 ? kStride : vStride)
          +(index*heads+head)*dim],converted,dim);
      Fence<HardEvent::MTE3_V>(); Fence<HardEvent::V_MTE2>();
    }
  }
  TPipe pipe_; RotationEngine<T> engine_; ClipQuantize quant_;
  TBuf<TPosition::VECCALC> packBuf_,statusBuf_,metaBuf_,rawBuf_;
  GlobalTensor<T> k_,v_; GlobalTensor<float> rk_,rv_;
  GlobalTensor<int64_t> slots_,positions_,tags_;
  GlobalTensor<uint8_t> packed_; GlobalTensor<bfloat16_t> rawK_,rawV_;
  GlobalTensor<int32_t> status_;
};
}

extern "C" __global__ __aicore__ void oscar_rotate_kernel(GM_ADDR source,
    GM_ADDR rotation,GM_ADDR output,GM_ADDR status,int64_t rows,int64_t dim,
    int32_t dtype,bool hadamard,GM_ADDR slots,int64_t heads) {
  if (dtype==0) {RotateOnly<float> op;op.Run(source,rotation,output,status,rows,dim,hadamard,slots,heads);}
  else if (dtype==1) {RotateOnly<half> op;op.Run(source,rotation,output,status,rows,dim,hadamard,slots,heads);}
  else {RotateOnly<bfloat16_t> op;op.Run(source,rotation,output,status,rows,dim,hadamard,slots,heads);}
}

extern "C" __global__ __aicore__ void oscar_rotate_clip_store_kernel(
    GM_ADDR key,GM_ADDR value,GM_ADDR rk,GM_ADDR rv,GM_ADDR slots,GM_ADDR positions,
    GM_ADDR packed,GM_ADDR rawKey,GM_ADDR rawValue,GM_ADDR tags,GM_ADDR status,
    int64_t tokens,int64_t heads,int64_t dim,int32_t dtype,int64_t blockTokens,
    int64_t blocks,int64_t offset,int64_t pageStride,int64_t kStride,int64_t vStride,
    int64_t tagStride,int64_t sink,int64_t recent,float kClip,float vClip,bool hadamard) {
#define OSCAR_RUN_ROTATE_STORE(TYPE) \
  RotateStore<TYPE> op; op.Run(key,value,rk,rv,slots,positions,packed,rawKey,rawValue, \
      tags,status,tokens,heads,dim,blockTokens,blocks,offset,pageStride,kStride,vStride, \
      tagStride,sink,recent,kClip,vClip,hadamard)
  if (dtype==0) {OSCAR_RUN_ROTATE_STORE(float);}
  else if (dtype==1) {OSCAR_RUN_ROTATE_STORE(half);}
  else {OSCAR_RUN_ROTATE_STORE(bfloat16_t);}
#undef OSCAR_RUN_ROTATE_STORE
}

#ifndef ASCENDC_CPU_DEBUG
namespace oscar_ascend {
void rotate_launch(void* stream,void* input,void* rotation,void* output,
    void* status,int64_t rows,int64_t dim,int32_t dtype,bool hadamard,
    void* slots,int64_t heads,uint32_t cores) {
  oscar_rotate_kernel<<<cores,nullptr,stream>>>(static_cast<uint8_t*>(input),
      static_cast<uint8_t*>(rotation),static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(status),rows,dim,dtype,hadamard,static_cast<uint8_t*>(slots),heads);
}
void rotate_clip_store_launch(void* stream,void* key,void* value,void* rk,void* rv,
    void* slots,void* positions,void* packed,void* rawKey,void* rawValue,void* tags,
    void* status,int64_t tokens,int64_t heads,int64_t dim,int32_t dtype,
    int64_t blockTokens,int64_t blocks,int64_t offset,int64_t pageStride,
    int64_t kStride,int64_t vStride,int64_t tagStride,int64_t sink,int64_t recent,
    float kClip,float vClip,bool hadamard,uint32_t cores) {
  oscar_rotate_clip_store_kernel<<<cores,nullptr,stream>>>(static_cast<uint8_t*>(key),
      static_cast<uint8_t*>(value),static_cast<uint8_t*>(rk),static_cast<uint8_t*>(rv),
      static_cast<uint8_t*>(slots),static_cast<uint8_t*>(positions),
      static_cast<uint8_t*>(packed),static_cast<uint8_t*>(rawKey),
      static_cast<uint8_t*>(rawValue),static_cast<uint8_t*>(tags),
      static_cast<uint8_t*>(status),tokens,heads,dim,dtype,blockTokens,blocks,offset,
      pageStride,kStride,vStride,tagStride,sink,recent,kClip,vClip,hadamard);
}
}
#endif
