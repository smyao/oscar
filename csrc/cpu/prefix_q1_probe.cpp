// Archive #26/#37-49/#126/#127/#128/#142/#157/#158: execute the production
// stripe writer, slot-context prepare and default q1 reader at APC geometry.
// D.4: (1) bounded phase1_store/fia seam; (2) never restore full history;
// (3) full-seed mode writes 323 exact rows; incremental mode explicitly seeds
//     old snapshots with its independent oracle then writes four real rows;
// (4) this is correctness evidence, not NPU completion or a timing estimate.
#include "tikicpulib.h"
#include "../include/oscar_attention_launch.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_attention_cv_decode_bundle_q1_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_rotate_clip_store_striped_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int32_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,float,float,bool);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t);

namespace {
struct Gm {
  uint8_t* ptr;size_t size;
  explicit Gm(size_t bytes,uint8_t value=0xA5):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(bytes))),size(bytes) {
    if(!ptr)throw std::runtime_error("GmAlloc failed");std::memset(ptr,value,size);
  }
  ~Gm(){AscendC::GmFree(ptr);}
};
constexpr int64_t kProbeDim=256,kProbeHeads=6,kProbeBlockTokens=2304,kProbeBlocks=5,kProbeConvBytes=30720,kProbeSsmBytes=393216,kProbePageBytes=817152;
constexpr int64_t kProbeSink=64,kProbeRecent=256,kProbeSpec=3,kProbeRing=259,kProbeRows=323,kProbeCores=2;
constexpr int64_t kProbePrefixBytes=kProbeBlocks*kProbeConvBytes,kProbePaddingBytes=kProbePageBytes-kProbeConvBytes-kProbeSsmBytes,kProbeSnapshotBase=kProbeBlocks*(kProbeConvBytes+kProbeSsmBytes);
constexpr int64_t kProbeKeyBytes=kProbeRows*kProbeDim*2,kProbeTagOffset=2*kProbeKeyBytes,kProbeSnapshotBytes=kProbeTagOffset+kProbeRows*8;
uint16_t Bf16(float value) {uint32_t bits;std::memcpy(&bits,&value,4);return bits>>16;}
void ZeroStatus(const Gm& status,const char* stage) {
  for(size_t i=0;i<status.size/4;++i)if(reinterpret_cast<const int32_t*>(status.ptr)[i])
    throw std::runtime_error(std::string(stage)+" status nonzero at "+std::to_string(i));
}
void FillKv(Gm& key,Gm& value,const std::vector<int64_t>& positions) {
  for(size_t row=0;row<positions.size();++row)for(int64_t j=0;j<kProbeDim;++j) {
    reinterpret_cast<uint16_t*>(key.ptr)[row*kProbeDim+j]=Bf16(float(j%4));
    reinterpret_cast<uint16_t*>(value.ptr)[row*kProbeDim+j]=Bf16(float(j%4)+float(positions[row]%8)/8.0F);
  }
}
void Store(Gm& raw,Gm& rotation,Gm& key,Gm& value,Gm& slots,Gm& positions,int64_t tokens) {
  Gm status(tokens*4);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_rotate_clip_store_striped_kernel,4,key.ptr,value.ptr,rotation.ptr,rotation.ptr,
      slots.ptr,positions.ptr,raw.ptr,raw.ptr+kProbeSnapshotBase,raw.ptr+kProbeSnapshotBase+kProbeKeyBytes,
      raw.ptr+kProbeSnapshotBase+kProbeTagOffset,status.ptr,tokens,int64_t{1},kProbeDim,int32_t{2},kProbeBlockTokens,kProbeBlocks,kProbePrefixBytes,kProbeSsmBytes,
      kProbePaddingBytes/2,kProbePaddingBytes/2,kProbePaddingBytes/8,kProbeSink,kProbeRing,0.0F,0.0F,false);
  ZeroStatus(status,"actual stripe writer");
}
}
int main(int argc,char** argv) {
 try {
  const bool incremental=argc==2 && std::string(argv[1])=="--incremental";
  if(argc!=1 && !incremental)throw std::runtime_error("usage: oscar_prefix_q1_cpu [--incremental]");
  Gm raw(kProbeBlocks*kProbePageBytes),rotation(kProbeDim*kProbeDim*4,0);
  for(int64_t i=0;i<kProbeDim;++i)reinterpret_cast<float*>(rotation.ptr)[i*kProbeDim+i]=1.0F;
  // Independent exact INT2 fixture: x[d]=d%4+offset, scale=1. Every word
  // stores dimensions i+32*j, so its 8 two-bit codes are the same i%4.
  const uint16_t halfEighths[8]={0,0x3000,0x3400,0x3600,0x3800,0x3900,0x3A00,0x3B00};
  for(int64_t pos=0;pos<kProbeBlockTokens;++pos) {
    auto* row=raw.ptr+kProbePrefixBytes+3*kProbeSsmBytes+pos*136;
    for(int64_t i=0;i<32;++i) {
      const uint16_t word=static_cast<uint16_t>((i%4)*0x5555);
      std::memcpy(row+2*i,&word,2);std::memcpy(row+64+2*i,&word,2);
    }
    const uint16_t meta[4]={0x3C00,0,0x3C00,halfEighths[pos%8]};
    std::memcpy(row+128,meta,8);
  }
  std::vector<int64_t> seed;
  for(int64_t p=0;p<kProbeSink;++p)seed.push_back(p);
  for(int64_t p=kProbeBlockTokens-kProbeRing;p<kProbeBlockTokens;++p)seed.push_back(p);
  Gm seedKey(kProbeRows*kProbeDim*2),seedValue(kProbeRows*kProbeDim*2),seedSlots(kProbeRows*8),seedPositions(kProbeRows*8);
  FillKv(seedKey,seedValue,seed);
  for(int64_t i=0;i<kProbeRows;++i) {
    reinterpret_cast<int64_t*>(seedSlots.ptr)[i]=3*kProbeBlockTokens+seed[i];
    reinterpret_cast<int64_t*>(seedPositions.ptr)[i]=seed[i];
  }
  if(incremental) {
    // Separate evidence mode after the full 323-row CPU writer timed out.
    // Seed a valid OLD cache independently; this does not certify the full
    // initial writer. Four page-tail rows then change through the real writer.
    for(int64_t i=0;i<kProbeRows;++i) {
      const int64_t p=seed[i],row=p<kProbeSink?p:kProbeSink+p%kProbeRing;
      auto* snapshot=raw.ptr+kProbeSnapshotBase+3*kProbePaddingBytes;
      std::memcpy(snapshot+row*kProbeDim*2,seedKey.ptr+i*kProbeDim*2,kProbeDim*2);
      std::memcpy(snapshot+kProbeKeyBytes+row*kProbeDim*2,seedValue.ptr+i*kProbeDim*2,kProbeDim*2);
      std::memcpy(snapshot+kProbeTagOffset+row*8,&p,8);
    }
    auto* tailRaw=raw.ptr+kProbePrefixBytes+3*kProbeSsmBytes+(kProbeBlockTokens-4)*136;
    const std::vector<uint8_t> expectedPacked(tailRaw,tailRaw+4*136);
    Gm tailKey(4*kProbeDim*2),tailValue(4*kProbeDim*2),tailSlots(32),tailPositions(32);
    std::vector<int64_t> tail;
    for(int64_t i=0;i<4;++i) {
      const int64_t p=kProbeBlockTokens-4+i,row=kProbeSink+p%kProbeRing;
      tail.push_back(p);
      reinterpret_cast<int64_t*>(tailSlots.ptr)[i]=3*kProbeBlockTokens+p;
      reinterpret_cast<int64_t*>(tailPositions.ptr)[i]=p;
      const uint16_t oldZero=static_cast<uint16_t>(0x3C00+(p%8)*0x80);
      std::memcpy(tailRaw+i*136+134,&oldZero,2);
      auto* oldValue=reinterpret_cast<uint16_t*>(raw.ptr+kProbeSnapshotBase+3*kProbePaddingBytes+kProbeKeyBytes)+row*kProbeDim;
      for(int64_t j=0;j<kProbeDim;++j)oldValue[j]=Bf16(float(j%4)+float(p%8)/8.0F+1.0F);
    }
    FillKv(tailKey,tailValue,tail);
    std::cout<<"PREFIX_Q1_CPU seed=independent_oracle real_writer=tail4 starting"<<std::endl;
    Store(raw,rotation,tailKey,tailValue,tailSlots,tailPositions,4);
    if(std::memcmp(tailRaw,expectedPacked.data(),expectedPacked.size()))
      throw std::runtime_error("real tail writer did not replace old INT2 values with oracle bytes");
    for(int64_t i=0;i<4;++i) {
      const int64_t row=kProbeSink+tail[i]%kProbeRing;
      const auto* snapshot=raw.ptr+kProbeSnapshotBase+3*kProbePaddingBytes;
      if(std::memcmp(snapshot+row*kProbeDim*2,tailKey.ptr+i*kProbeDim*2,kProbeDim*2) ||
         std::memcmp(snapshot+kProbeKeyBytes+row*kProbeDim*2,tailValue.ptr+i*kProbeDim*2,kProbeDim*2))
        throw std::runtime_error("real tail writer differs from exact BF16 oracle bytes");
    }
  }else {
    std::cout<<"PREFIX_Q1_CPU seed=production_writer323 starting"<<std::endl;
    Store(raw,rotation,seedKey,seedValue,seedSlots,seedPositions,kProbeRows);
  }
  std::vector<uint8_t> immutablePacked(raw.ptr+kProbePrefixBytes+3*kProbeSsmBytes,raw.ptr+kProbePrefixBytes+3*kProbeSsmBytes+kProbeBlockTokens*136);
  std::vector<uint8_t> immutableSnapshot(raw.ptr+kProbeSnapshotBase+3*kProbePaddingBytes,raw.ptr+kProbeSnapshotBase+3*kProbePaddingBytes+kProbeSnapshotBytes);
  for(int64_t p=0;p<kProbeBlockTokens;++p) {
    if(p>=kProbeSink && p<kProbeBlockTokens-kProbeRing)continue;
    const int64_t row=p<kProbeSink?p:kProbeSink+p%kProbeRing;
    int64_t tag;std::memcpy(&tag,raw.ptr+kProbeSnapshotBase+3*kProbePaddingBytes+kProbeTagOffset+row*8,8);
    if(tag!=p)throw std::runtime_error("real writer omitted prefix exact tag");
  }
  Gm q(kProbeHeads*kProbeDim*2,0),qr(kProbeHeads*kProbeDim*4,0),key(kProbeDim*2),value(kProbeDim*2),slots(8),positions(8);
  Gm starts(8,0),lens(4),table(36*4),tasks(3*16*8);
  reinterpret_cast<int32_t*>(starts.ptr)[1]=1;
  for(int64_t i=0;i<18;++i) {
    reinterpret_cast<int32_t*>(table.ptr)[i]=3*18+i;
    reinterpret_cast<int32_t*>(table.ptr)[18+i]=18+i;
  }
  Gm partial(kProbeHeads*3*kProbeDim*4),lse(kProbeHeads*3*4),status(3*2*4);
  Gm workspace(kProbeCores*oscar_ascend::attention_decode_bundle_workspace_per_core(kProbeDim));
  Gm output(kProbeHeads*kProbeDim*4),outputLse(kProbeHeads*4),mergeStatus(kProbeHeads*4);
  // Cross the full-page boundary, then simulate MTP rollback onto the first
  // private row. The cached full page must stay bitwise immutable throughout.
  for(int64_t context:{kProbeBlockTokens,kProbeBlockTokens+1,kProbeBlockTokens}) {
    FillKv(key,value,{context});
    reinterpret_cast<int64_t*>(slots.ptr)[0]=1*kProbeBlockTokens+(context-kProbeBlockTokens);
    reinterpret_cast<int32_t*>(lens.ptr)[0]=context+1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,tasks.ptr,positions.ptr,
        int64_t{1},int64_t{1},kProbeHeads,int64_t{1},kProbeSink,kProbeRecent,int64_t{1},true,table.ptr,int64_t{36},true);
    if(reinterpret_cast<int64_t*>(positions.ptr)[0]!=context)
      throw std::runtime_error("virtual128 slot context differs across physical2304 boundary");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(oscar_attention_cv_decode_bundle_q1_kernel,kProbeCores,q.ptr,qr.ptr,key.ptr,value.ptr,
        rotation.ptr,raw.ptr,table.ptr,raw.ptr+kProbeSnapshotBase,raw.ptr+kProbeSnapshotBase+kProbeKeyBytes,
        raw.ptr+kProbeSnapshotBase+kProbeTagOffset,tasks.ptr,partial.ptr,lse.ptr,status.ptr,workspace.ptr,
        int64_t{1},kProbeHeads,int64_t{1},kProbeDim,int64_t{1},int64_t{36},int64_t{3},kProbeBlockTokens,kProbeBlocks,kProbePrefixBytes,kProbeSsmBytes,
        kProbePaddingBytes/2,kProbePaddingBytes/8,kProbeSink,kProbeRecent,kProbeSpec,int64_t{1},1.0F/16.0F);
    ZeroStatus(status,"actual bundle q1");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,lse.ptr,output.ptr,outputLse.ptr,mergeStatus.ptr,kProbeHeads,int64_t{3},kProbeDim);
    ZeroStatus(mergeStatus,"actual merge");
    double offset=0;for(int64_t p=0;p<=context;++p)offset+=double(p%8)/8.0;
    offset/=double(context+1);
    for(int64_t h=0;h<kProbeHeads;++h) {
      const float actualLse=reinterpret_cast<float*>(outputLse.ptr)[h];
      const double expectedLse=std::log(double(context+1));
      if(!std::isfinite(actualLse)||std::abs(actualLse-expectedLse)>0.005+0.005*std::abs(expectedLse))
        throw std::runtime_error("independent uniform-attention LSE oracle failed");
      for(int64_t j=0;j<kProbeDim;++j) {
        const float actual=reinterpret_cast<float*>(output.ptr)[h*kProbeDim+j];
        const double expected=double(j%4)+offset;
        if(!std::isfinite(actual)||std::abs(actual-expected)>0.005+0.005*std::abs(expected))
          throw std::runtime_error("independent raw-space output oracle failed");
      }
    }
    Store(raw,rotation,key,value,slots,positions,1);
    if(std::memcmp(immutablePacked.data(),raw.ptr+kProbePrefixBytes+3*kProbeSsmBytes,immutablePacked.size()) ||
       std::memcmp(immutableSnapshot.data(),raw.ptr+kProbeSnapshotBase+3*kProbePaddingBytes,immutableSnapshot.size()))
      throw std::runtime_error("private suffix modified the shared page");
    std::cout<<"PREFIX_Q1_CPU context="<<context<<" status=passed"<<std::endl;
  }
  std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"prefix_q1_writer_reader\","
      "\"physical_block_tokens\":2304,\"virtual_block_tokens\":128,"
      "\"cases\":3,\"independent_raw_space_oracle\":\"passed\","
      "\"initial_seed\":\""<<(incremental?"independent_oracle":"production_writer323")<<"\","
      "\"real_writer_rows_before_queries\":"<<(incremental?4:kProbeRows)<<",\"status\":\"passed\"}"<<std::endl;
  return 0;
 }catch(const std::exception& e){std::cerr<<"PREFIX_Q1_CPU_FAILED: "<<e.what()<<std::endl;return 1;}
}
