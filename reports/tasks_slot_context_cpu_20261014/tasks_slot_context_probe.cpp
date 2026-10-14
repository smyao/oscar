// Archive #34/#36/#157-159: actual frozen prepare, long-table request churn.
// D.4 four questions: (1) prepare; (2) historical host work cost 725ms and
// dequant restored history at 6.5s; (3) CPU-only bounded metadata driver links
// the unmodified production kernel, no KV allocation or production logging;
// (4) O(32*2048) table inputs, no device timing or NPU acceptance claim.
#include "tikicpulib.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
namespace {
struct Gm {
  uint8_t* p;
  explicit Gm(size_t n):p(static_cast<uint8_t*>(AscendC::GmAlloc(n))) {
    if(!p)throw std::runtime_error("GmAlloc");std::memset(p,0x85,n);
  }
  ~Gm(){AscendC::GmFree(p);}
};
void Run(bool wide) {
  constexpr int64_t cap=32,columns=2048,hq=6,hk=1,splits=3,perToken=9;
  Gm starts((cap+1)*4),lens(cap*4),table(cap*columns*4),slots(cap*(wide?8:4));
  Gm tasks(cap*perToken*16*8),positions(cap*8);
  std::mt19937 rng(157159);
  std::vector<int32_t> owners(cap*columns);
  std::iota(owners.begin(),owners.end(),100003);
  std::shuffle(owners.begin(),owners.end(),rng);
  std::vector<int> order(cap);std::iota(order.begin(),order.end(),0);
  int launch=0, valid=0, controls=0;
  for(int64_t b:{32,31,12}) {
    std::shuffle(order.begin(),order.end(),rng);
    for(int draft:{1,2}) {
      auto* qs=reinterpret_cast<int32_t*>(starts.p);
      auto* ls=reinterpret_cast<int32_t*>(lens.p);
      auto* bt=reinterpret_cast<int32_t*>(table.p);
      std::vector<int64_t> expected(cap,-1),slotValues(cap,-1);
      for(int64_t r=0;r<=cap;++r)qs[r]=static_cast<int32_t>(std::min(r,b));
      for(int64_t r=0;r<b;++r) {
        const int id=order[r];
        const int32_t optimistic=32766+(id%9)*128+draft;
        const int32_t rejected=(id+draft)%4;
        ls[r]=optimistic;expected[r]=optimistic-1-rejected;
        std::memcpy(bt+r*columns,owners.data()+id*columns,columns*4);
        slotValues[r]=int64_t(bt[r*columns+expected[r]/128])*128+expected[r]%128;
        // Stale duplicate strictly outside the valid prefix must stay inert.
        bt[r*columns+(optimistic+127)/128]=bt[r*columns+expected[r]/128];
      }
      for(int64_t r=b;r<cap;++r)ls[r]=0;
      auto publish=[&](){for(int64_t r=0;r<cap;++r) {
        if(wide)reinterpret_cast<int64_t*>(slots.p)[r]=slotValues[r];
        else reinterpret_cast<int32_t*>(slots.p)[r]=static_cast<int32_t>(slotValues[r]);
      }};
      publish();
      for(int64_t n:{b,cap}) {
        // All three failure predicates get explicit independent controls at
        // B12 draft2. Real valid inputs run first, unchanged, in both shapes.
        const int variants=(b==12 && draft==2 && n==b)?5:1;
        for(int fault=0;fault<variants;++fault) {
          const auto oldSlot=slotValues[0];const auto oldLength=ls[0];
          const auto oldStart=qs[1];const auto oldEntry=bt[0];
          if(fault==1)qs[1]=2;                 // qlength != 1
          if(fault==2)bt[0]=static_cast<int32_t>(oldSlot/128); // duplicate match
          if(fault==3)slotValues[0]=99999999;  // absent physical page
          if(fault==4)ls[0]=static_cast<int32_t>(expected[0]); // position == lens
          publish();
          std::cout<<"CASE slots="<<(wide?64:32)<<" B="<<b<<" N="<<n<<" draft="<<draft
                   <<" fault="<<fault<<" row0_slot="<<slotValues[0]<<" row0_lens="<<ls[0]
                   <<" row0_qend="<<qs[1]<<std::endl;
          AscendC::SetKernelMode(KernelMode::AIV_MODE);
          ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,32,starts.p,lens.p,slots.p,
              tasks.p,positions.p,b,n,hq,hk,int64_t{4},int64_t{2048},splits,wide,table.p,columns,true);
          for(int64_t token=0;token<n;++token) {
            const int64_t error=(fault && (token==0 || (fault==1 && token==1)))?5:0;
            for(int64_t k=0;k<perToken;++k) {
              const auto* t=reinterpret_cast<int64_t*>(tasks.p)+(token*perToken+k)*16;
              if(t[10]!=error) {
                std::cerr<<"FIRST_ERROR launch="<<launch<<" token="<<token<<" task="<<k
                         <<" expected="<<error<<" actual="<<t[10]<<" request="<<t[5]
                         <<" context="<<t[8]<<std::endl;
                throw std::runtime_error("unexpected prepare metadata error");
              }
            }
            const int64_t want=error?-1:expected[token];
            if(reinterpret_cast<int64_t*>(positions.p)[token]!=want)
              throw std::runtime_error("resolved position mismatch");
          }
          ++launch;if(fault)++controls;else ++valid;
          qs[1]=oldStart;bt[0]=oldEntry;ls[0]=oldLength;slotValues[0]=oldSlot;publish();
        }
      }
    }
  }
  std::cout<<"{\"slot_bits\":"<<(wide?64:32)<<",\"valid_launches\":"<<valid
           <<",\"negative_controls\":"<<controls<<",\"status\":\"passed\"}"<<std::endl;
}
void RunPublishedPosition(bool wide) {
  constexpr int64_t b=3,cols=4,perToken=3;
  Gm starts((b+1)*4),lens(b*4),table(b*cols*4),slots(b*(wide?8:4));
  Gm tasks(b*perToken*16*8),positions(b*8);
  auto* qs=reinterpret_cast<int32_t*>(starts.p);
  auto* ls=reinterpret_cast<int32_t*>(lens.p);
  auto* bt=reinterpret_cast<int32_t*>(table.p);
  const int64_t published[b]={145,127,126};
  for(int r=0;r<=b;++r)qs[r]=r;
  for(int r=0;r<b;++r) {
    ls[r]=128;
    for(int c=0;c<cols;++c)bt[r*cols+c]=100+r*cols+c;
    const int64_t slot=int64_t(bt[r*cols+published[r]/128])*128+published[r]%128;
    if(wide)reinterpret_cast<int64_t*>(slots.p)[r]=slot;
    else reinterpret_cast<int32_t*>(slots.p)[r]=slot;
  }
  for(int mode=0;mode<3;++mode) {
    if(mode)for(int r=0;r<b;++r)ls[r]=published[r]+1;
    if(mode==2)ls[1]=0; // Python validator rejects an inconsistent live slot.
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,32,starts.p,lens.p,slots.p,
        tasks.p,positions.p,b,b,int64_t{6},int64_t{1},int64_t{64},int64_t{256},
        int64_t{1},wide,table.p,cols,mode==0);
    for(int r=0;r<b;++r) {
      const int error=mode==0&&r==0?5:mode==2&&r==1?1:0;
      for(int k=0;k<perToken;++k) {
        const auto* t=reinterpret_cast<int64_t*>(tasks.p)+(r*perToken+k)*16;
        if(t[10]!=error)throw std::runtime_error("published-position error-domain mismatch");
      }
      if(reinterpret_cast<int64_t*>(positions.p)[r]!=(error?-1:published[r]))
        throw std::runtime_error("published-position output mismatch");
    }
  }
  std::cout<<"PUBLISHED_POSITION slots="<<(wide?64:32)
           <<" inverse_error=5 explicit_error=0 invalid_live_error=1 invalid_store_position=-1 passed"<<std::endl;
}
}
int main(){try {Run(false);Run(true);RunPublishedPosition(false);RunPublishedPosition(true);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
