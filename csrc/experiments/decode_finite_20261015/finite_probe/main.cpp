// Archive #134/#140/#145/#148/#154: actual extracted Softmax CPU stress.
// Bitwise output/state/error checks; no Cube/full CV/NPU/performance claim.
#include "tikicpulib.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t
extern "C" void FINITE_BASELINE(ARGS);
extern "C" void FINITE_CANDIDATE(ARGS);
namespace {
struct Gm {
  uint8_t *base,*p;size_t n;
  explicit Gm(size_t z):base(static_cast<uint8_t*>(AscendC::GmAlloc(z+128))),p(base?base+64:nullptr),n(z){if(!base)throw std::runtime_error("GmAlloc");std::memset(base,0xa5,z+128);std::memset(p,0x85,z);}
  ~Gm(){AscendC::GmFree(base);}
  void guard(){for(size_t i=0;i<64;++i)if(base[i]!=0xa5||p[n+i]!=0xa5)throw std::runtime_error("GM guard");}
};
void put(Gm& g,size_t i,float v){std::memcpy(g.p+4*i,&v,4);}
void bit(Gm& g,size_t i,uint32_t v){std::memcpy(g.p+4*i,&v,4);}
struct Case{const char* name;int mode;int64_t source,context,start,end,recent,rowbase,initial;int expected;int64_t postCommit=0;};
const Case cases[]={
 {"normal_full",0,0,320,64,320,1,0,0,0},
 {"second_lane_full_or_empty",0,0,320,64,320,1,16,0,0},
 {"finite_sum_overflow",1,0,320,64,320,1,0,0,2},
 {"finite_cancel",2,0,320,64,320,1,0,0,0},
 {"cancel_intermediate_overflow",3,0,320,64,320,1,0,0,2},
 {"visible_NaN",4,0,320,64,320,1,0,0,2},
 {"visible_positive_Inf",5,0,320,64,320,1,0,0,2},
 {"visible_negative_Inf",6,0,320,64,320,1,0,0,2},
 {"masked_NaN_still_prechecked",7,0,337,320,337,1,0,0,2},
 {"empty_row_NaN_not_checked",8,0,320,64,320,1000,0,0,0},
 {"precise_masked_NaN",7,1,320,0,64,1,0,0,2},
 {"padding_NaN_not_checked",9,0,320,64,320,1,16,0,0},
 {"existing_metadata_error_preserved",0,0,320,64,320,1,0,3,3},
 {"score_NaN_overwrites_incoming_metadata",4,0,320,64,320,1,0,3,2,0},
 {"score_NaN_then_pending_metadata_commit",4,0,320,64,320,1,0,3,3,1},
};
}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: FINITE_PROBE NEW_REPORT_DIR");
 std::filesystem::path out=argv[1];if(std::filesystem::exists(out))throw std::runtime_error("fresh report required");std::filesystem::create_directories(out);
 AscendC::SetKernelMode(KernelMode::AIV_MODE);
 std::ofstream report(out/"report.json");report<<"{\"status\":\"passed\",\"scope\":\"isolated actual Softmax CPU only\",\"cases\":[";
 int index=0;
 for(const auto& c:cases){
  Gm a(32*256*4),b(32*256*4),ia(16*256*4),is(64*4),oa(16*256*4),ob(16*256*4),sa(64*4),sb(64*4),ea(4),eb(4);
  for(size_t i=0;i<32*256;++i)put(a,i,float(int(i%17)-8)/32.0F);
  for(size_t i=0;i<16*256;++i)put(ia,i,float(int(i%7)+1)/8.0F);
  for(size_t i=0;i<64;++i)put(is,i,i<16?0.125F:(i<32?1.25F:0.0F));
  for(int r=0;r<16;++r)for(int k=0;k<256;++k){size_t at=(c.rowbase+r)*256+k;
   if(c.mode==1)put(a,at,1.0e38F);
   if(c.mode==2)put(a,at,k%2?-1.0e20F:1.0e20F);
   if(c.mode==3)put(a,at,k<128?3.0e38F:-3.0e38F);
   if(c.mode==8)bit(a,at,0x7fc12345U);
  }
  if(c.mode>=4&&c.mode<=6)bit(a,(c.rowbase+3)*256+33,c.mode==4?0x7fc12345U:(c.mode==5?0x7f800000U:0xff800000U));
  if(c.mode==7)bit(a,(c.rowbase+3)*256+200,0x7fc12345U);
  if(c.mode==9)bit(a,(c.rowbase+15)*256+33,0x7fc12345U);
  std::memcpy(b.p,a.p,a.n);std::vector<uint8_t> oldAcc(ia.p,ia.p+ia.n),oldStats(is.p,is.p+is.n);
#define RUN(name,w,o,s,e) ICPU_RUN_KF(name,1,w.p,ia.p,is.p,o.p,s.p,e.p,c.source,c.context,int64_t{FINITE_QUERIES},int64_t{6},c.start,c.end,int64_t{64},c.recent,c.rowbase,c.initial,c.postCommit)
  RUN(FINITE_BASELINE,a,oa,sa,ea);RUN(FINITE_CANDIDATE,b,ob,sb,eb);
  if(std::memcmp(a.p,b.p,a.n)||std::memcmp(oa.p,ob.p,oa.n)||std::memcmp(sa.p,sb.p,sa.n)||std::memcmp(ea.p,eb.p,4))throw std::runtime_error(std::string(c.name)+" bitwise mismatch");
  int32_t error;std::memcpy(&error,ea.p,4);if(error!=c.expected)throw std::runtime_error(std::string(c.name)+" original expected error "+std::to_string(error));
  if(std::memcmp(oldAcc.data(),ia.p,ia.n)||std::memcmp(oldStats.data(),is.p,is.n))throw std::runtime_error("input changed");
  for(auto* g:{&a,&b,&ia,&is,&oa,&ob,&sa,&sb,&ea,&eb})g->guard();
  if(index++)report<<',';report<<"{\"case\":\""<<c.name<<"\",\"error\":"<<error<<",\"bitwise\":true,\"guards\":true}";report.flush();
  std::cout<<"FINITE_CASE "<<c.name<<" error="<<error<<" bitwise=passed\n"<<std::flush;
 }
 report<<"],\"full_C320_CPU_CAModel\":\"separate mandatory gate\",\"NPU\":\"not_run\"}\n";if(!report.flush())throw std::runtime_error("report write");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
