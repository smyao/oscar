// Archive #13-22/#126/#145/#154/#155; full D.4. Complete small C4 arm, no timing claim.
// Real prepare + production current suppression, all positions/statuses/guards.
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
#define CV_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float
extern "C" void oscar_c4_b_nz_baseline_kernel(CV_ARGS);
extern "C" void oscar_c4_b_nz_candidate_kernel(CV_ARGS);
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
namespace {
std::vector<uint8_t> Read(const std::filesystem::path& p,size_t n){std::vector<uint8_t>b(n);std::ifstream f(p,std::ios::binary);if(!f.read(reinterpret_cast<char*>(b.data()),n)||f.peek()!=EOF)throw std::runtime_error("byte count "+p.string());return b;}
struct Gm{uint8_t *base,*p;size_t n;explicit Gm(size_t z):base(static_cast<uint8_t*>(AscendC::GmAlloc(z+128))),p(base?base+64:nullptr),n(z){if(!base)throw std::runtime_error("GmAlloc");std::memset(base,0xa5,z+128);std::memset(p,0x85,z);}Gm(const std::filesystem::path& f,size_t z):Gm(z){auto b=Read(f,z);std::memcpy(p,b.data(),z);}~Gm(){AscendC::GmFree(base);}void guard(){for(size_t i=0;i<64;++i)if(base[i]!=0xa5||p[n+i]!=0xa5)throw std::runtime_error("GM guard");}void save(const std::filesystem::path& f){std::ofstream o(f,std::ios::binary);if(!o.write(reinterpret_cast<char*>(p),n)||!o.flush())throw std::runtime_error("save");}};
template<class T>T At(const uint8_t*p,size_t i){T v;std::memcpy(&v,p+i*sizeof(T),sizeof(T));return v;}
}
int main(int argc,char**argv){try{
  if(argc!=4)throw std::runtime_error("usage: CV_CPU FIXTURE NEW_OUTPUT baseline|candidate");
  std::filesystem::path dir=argv[1],out=argv[2];std::string arm=argv[3];
  if(std::filesystem::exists(out)||(arm!="baseline"&&arm!="candidate"))throw std::runtime_error("new output / explicit arm required");
  int64_t a[17];std::ifstream shape(dir/"shape.txt");for(auto& x:a)if(!(shape>>x))throw std::runtime_error("shape");
  if(a[0]!=336||a[1]!=6||a[2]!=1||a[3]!=256||a[4]!=1||a[6]!=1008||a[13]!=64||a[14]!=256||a[15]!=3||a[16]!=1)throw std::runtime_error("undeclared C4 shape");
  int bad=-1;std::ifstream fault(dir/"bad_position.txt");if(!(fault>>bad)||(bad!=-1&&bad!=80&&bad!=96&&bad!=336))throw std::runtime_error("fault domain");
  float atol,rtol;std::ifstream tolerance(dir/"tolerance.txt");if(!(tolerance>>atol>>rtol)||atol!=.005F||rtol!=.005F)throw std::runtime_error("frozen tolerance");
  const auto rvNatural=Read(dir/"rv.bin",256*256*4),rvNz=Read(dir/"rv_nz.bin",256*256*4);
  for(size_t r=0;r<256;++r)for(size_t c=0;c<256;++c)
    if(At<uint32_t>(rvNatural.data(),r*256+c)!=At<uint32_t>(rvNz.data(),(c/8)*256*8+r*8+c%8))
      throw std::runtime_error("Rv full uint32 NZ inverse differs");
  Gm q(dir/"q.bin",336*6*256*2),qr(dir/"qr.bin",336*6*256*4),k(dir/"ck.bin",336*256*2),v(dir/"cv.bin",336*256*2),rv(dir/(arm=="candidate"?"rv_nz.bin":"rv.bin"),256*256*4),raw(dir/"raw.bin",a[9]+a[8]*a[10]);
  Gm table(dir/"table.bin",a[5]*4),wk(dir/"wk.bin",a[8]*a[11]*2),wv(dir/"wv.bin",a[8]*a[11]*2),tags(dir/"tags.bin",a[8]*a[12]*8);
  Gm starts(dir/"starts.bin",8),lens(dir/"lens.bin",4),slots(dir/"slots.bin",336*8),tasks(1008*128),positions(336*8);
  Gm partial(336*6*3*256*4),lse(336*6*3*4),status(1008*2*4),workspace(20*2494464),stats(20*64);
  std::vector<Gm*> inputs={&q,&qr,&k,&v,&rv,&raw,&table,&wk,&wv,&tags,&starts,&lens,&slots};std::vector<std::vector<uint8_t>> before;
  for(auto* x:inputs)before.emplace_back(x->p,x->p+x->n);
  const int context=At<int32_t>(lens.p,0)-336;
  if(context!=337||bad>=context)throw std::runtime_error("declared context/fault required");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,20,starts.p,lens.p,slots.p,tasks.p,positions.p,int64_t{1},int64_t{336},int64_t{6},int64_t{1},int64_t{64},int64_t{256},int64_t{1},true,table.p,a[5],false);
  for(int i=0;i<336;++i)if(At<int64_t>(positions.p,i)!=context+i)throw std::runtime_error("position oracle");
  for(int id=0;id<1008;++id){auto row=reinterpret_cast<int64_t*>(tasks.p)+id*16;if(row[7]==2&&row[10]==0&&row[4]>=row[3])row[4]=row[3];}
  auto taskBefore=std::vector<uint8_t>(tasks.p,tasks.p+tasks.n);
  // Archive #94/#95/#129: retain completed metadata independently of the
  // later full-C4 arm. This is not attention/numeric completion evidence.
  tasks.guard();positions.guard();
  for(size_t i=0;i<inputs.size();++i){inputs[i]->guard();if(std::memcmp(before[i].data(),inputs[i]->p,inputs[i]->n))throw std::runtime_error("prepare changed input");}
  std::filesystem::create_directories(out/"prepared");tasks.save(out/"prepared/tasks.bin");positions.save(out/"prepared/positions.bin");
  {std::ofstream metadata(out/"prepared/metadata.json");metadata<<"{\"status\":\"metadata_only\",\"backend\":\"real_ascendc_cpu_prepare\",\"tokens\":336,\"context\":"<<context<<",\"task_rows\":1008,\"source2_suppressed\":true,\"positions_checked\":true,\"guards\":true,\"input_immutable\":true,\"attention_execution\":\"not_established\"}\n";if(!metadata.flush())throw std::runtime_error("metadata checkpoint save");}
  std::cout<<"C4_B_NZ_METADATA status=metadata_only tasks=1008 positions=passed\n"<<std::flush;
  AscendC::SetKernelMode(KernelMode::MIX_MODE);
#define RUN(name) ICPU_RUN_KF(name,20,q.p,qr.p,k.p,v.p,rv.p,raw.p,table.p,wk.p,wv.p,tags.p,tasks.p,partial.p,lse.p,status.p,workspace.p,stats.p,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],.0625F)
  if(arm=="baseline"){RUN(oscar_c4_b_nz_baseline_kernel);}else{RUN(oscar_c4_b_nz_candidate_kernel);}
  std::filesystem::create_directories(out);partial.save(out/"partial.bin");lse.save(out/"lse.bin");status.save(out/"status.bin");stats.save(out/"cluster_stats.bin");tasks.save(out/"tasks.bin");
  rv.save(out/"rv_input.bin");
  {std::ofstream f(out/"rv_natural.bin",std::ios::binary),g(out/"rv_input.guarded.bin",std::ios::binary);
   f.write(reinterpret_cast<const char*>(rvNatural.data()),rvNatural.size());
   g.write(reinterpret_cast<const char*>(rv.base),rv.n+128);
   if(!f.flush()||!g.flush())throw std::runtime_error("Rv ownership evidence save");}
  for(size_t i=0;i<inputs.size();++i){inputs[i]->guard();if(std::memcmp(before[i].data(),inputs[i]->p,inputs[i]->n))throw std::runtime_error("input modified");}
  for(auto* x:{&tasks,&positions,&partial,&lse,&status,&workspace,&stats})x->guard();
  if(std::memcmp(taskBefore.data(),tasks.p,tasks.n))throw std::runtime_error("tasks modified");
  int faults=0;for(int id=0;id<1008;++id){auto row=reinterpret_cast<int64_t*>(tasks.p)+id*16;for(int lane=0;lane<2;++lane){
    int expected=(bad>=0&&row[1]>0&&row[7]==0&&row[3]<=bad&&bad<row[4]&&((bad-row[3])%32)/16==lane)?3:0;
    if(At<int32_t>(status.p,id*2+lane)!=expected)throw std::runtime_error("status/lane oracle at "+std::to_string(id)+":"+std::to_string(lane));faults+=expected!=0;}}
  int64_t clusters=0,leaders=0,skips=0;for(int core=0;core<20;++core){clusters+=At<int64_t>(stats.p,core*8);leaders+=At<int64_t>(stats.p,core*8+1);skips+=At<int64_t>(stats.p,core*8+6);}
  if(clusters!=1||leaders!=4||skips!=4||(bad>=0&&faults==0))throw std::runtime_error("actual C4/fault not exercised");
  if(bad<0){auto expected=Read(dir/"expected_output.bin",336*6*256*4),logs=Read(dir/"expected_lse.bin",336*6*4);
    for(int row=0;row<336*6;++row){double maximum=-INFINITY,weights[3],sum=0;for(int s=0;s<3;++s)maximum=std::fmax(maximum,double(At<float>(lse.p,row*3+s)));
      for(int s=0;s<3;++s){weights[s]=std::exp(double(At<float>(lse.p,row*3+s))-maximum);sum+=weights[s];}
      double wantL=At<float>(logs.data(),row),gotL=maximum+std::log(sum);if(!std::isfinite(gotL)||std::fabs(gotL-wantL)>atol+rtol*std::fabs(wantL))throw std::runtime_error("LSE oracle");
      for(int d=0;d<256;++d){double got=0;for(int s=0;s<3;++s)if(weights[s]>0)got+=weights[s]*At<float>(partial.p,(row*3+s)*256+d);got/=sum;double want=At<float>(expected.data(),row*256+d);
        if(!std::isfinite(got)||std::fabs(got-want)>atol+rtol*std::fabs(want))throw std::runtime_error("output oracle");}}
  }
  std::ofstream report(out/"report.json");report<<"{\"status\":\"passed\",\"arm\":\""<<arm<<"\",\"context\":"<<context<<",\"bad_position\":"<<bad<<",\"clusters\":1,\"fault_statuses\":"<<faults<<",\"guards\":true,\"input_immutable\":true,\"all_status_lanes\":true,\"oracle\":\""<<(bad<0?"numeric":"invalid_metadata_status")<<"\",\"rv_inverse\":true,\"rv_layout\":\""<<(arm=="candidate"?"NZ_C0_8":"ND")<<"\",\"NPU\":\"not_run\"}\n";
  std::cout<<"C4_B_NZ_CPU arm="<<arm<<" bad="<<bad<<" clusters=1 status=passed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
