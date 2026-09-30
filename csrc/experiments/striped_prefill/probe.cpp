// Official CPU-debug: same task/Q/K/V, old packed pages versus striped pages.
// Archive #126/#129/#145/#154 and startup D.4. A long case may run reference
// and candidate as separate bounded processes; actual partial/LSE/status bytes
// and input SHA are carried between them. CPU evidence is not NPU acceptance.
#include "probe_common.h"
#include "../../include/oscar_attention_launch.h"
#include <filesystem>
#include <sstream>

#define CV_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float
#define CV_CLUSTER_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float
extern "C" void oscar_attention_cv_fast_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_fast_balanced_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_striped_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_striped_balanced_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_fast_cluster4_kernel(CV_CLUSTER_ARGS);
extern "C" void oscar_attention_cv_fast_cluster16_kernel(CV_CLUSTER_ARGS);
extern "C" void oscar_attention_cv_striped_cluster4_kernel(CV_CLUSTER_ARGS);
extern "C" void oscar_attention_cv_striped_cluster16_kernel(CV_CLUSTER_ARGS);
#undef CV_ARGS
#undef CV_CLUSTER_ARGS
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t);

namespace {
void CheckStatus(const Gm& status) {
  for(size_t offset=0;offset<status.size;offset+=4) {
    int32_t code;std::memcpy(&code,status.ptr+offset,4);
    if(code)throw std::runtime_error("CV status="+std::to_string(code)+
                                     " byte="+std::to_string(offset));
  }
}
void CheckOracle(const std::string& dir,const Gm& partial,const Gm& partLse,
    int64_t tokens,int64_t hq,int64_t dim,int64_t segments) {
  Gm merged(tokens*hq*dim*4),mergedLse(tokens*hq*4),mergeStatus(tokens*hq*4);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,partLse.ptr,merged.ptr,
      mergedLse.ptr,mergeStatus.ptr,tokens*hq,segments,dim);
  const auto expected=Read(dir+"/expected_output.bin",merged.size);
  const auto expectedLse=Read(dir+"/expected_lse.bin",mergedLse.size);
  const size_t sampleBytes=std::filesystem::file_size(dir+"/oracle_samples.bin");
  if(sampleBytes==0 || sampleBytes%8)throw std::runtime_error("invalid oracle sample file");
  const auto sampleData=Read(dir+"/oracle_samples.bin",sampleBytes);
  for(size_t index=0;index<sampleBytes;index+=8) {
    int64_t token;std::memcpy(&token,sampleData.data()+index,8);
    if(token<0 || token>=tokens)throw std::runtime_error("oracle token out of range");
    for(int64_t head=0;head<hq;++head) {
      const int64_t row=token*hq+head;
      int32_t code;std::memcpy(&code,mergeStatus.ptr+row*4,4);
      if(code)throw std::runtime_error("sampled merge status nonzero");
      float actual,reference;
      std::memcpy(&actual,mergedLse.ptr+row*4,4);
      std::memcpy(&reference,expectedLse.data()+row*4,4);
      if(!std::isfinite(actual) ||
          std::abs(actual-reference)>0.005F+0.005F*std::abs(reference))
        throw std::runtime_error("sampled LSE exceeds frozen oracle");
      for(int64_t col=0;col<dim;++col) {
        const int64_t off=(row*dim+col)*4;
        std::memcpy(&actual,merged.ptr+off,4);
        std::memcpy(&reference,expected.data()+off,4);
        if(!std::isfinite(actual) ||
            std::abs(actual-reference)>0.005F+0.005F*std::abs(reference))
          throw std::runtime_error("sampled output exceeds frozen oracle");
      }
    }
  }
}
std::string InputHash(const std::string& dir) {
  std::ifstream file(dir+"/input_sha256.txt");std::string result;
  if(!(file>>result) || result.size()!=64)throw std::runtime_error("invalid input SHA");
  return result;
}
}

int main(int argc,char** argv) {
  try {
    if(argc!=5)throw std::runtime_error(
        "usage: oscar_striped_prefill_cpu base|balanced|cluster4|cluster16 reference|candidate|both GOLDEN_DIR REF_DIR");
    const std::string mode=argv[1],phase=argv[2],dir=argv[3],refDir=argv[4];
    const bool cluster4=mode=="cluster4",cluster16=mode=="cluster16";
    if(!(mode=="base" || mode=="balanced" || cluster4 || cluster16) ||
        !(phase=="reference" || phase=="candidate" || phase=="both"))
      throw std::runtime_error("invalid mode/phase");
    std::ifstream shape(dir+"/shape.txt");
    int64_t n,hq,hk,d,context,sink,recent,spec,b,blocks,prefix,stride,cores;
    if(!(shape>>n>>hq>>hk>>d>>context>>sink>>recent>>spec>>b>>blocks>>prefix>>stride>>cores))
      throw std::runtime_error("invalid golden shape");
    int64_t requests=1,splits=1,columns=8;
    shape>>requests;shape>>splits;shape>>columns;
    if(d!=256 || hk!=1 || hq!=6 || cores<1 || splits<1 || columns<8 || columns>32)
      throw std::runtime_error("invalid D256 striped CPU shape");
    const int64_t segments=3*splits,tasksCount=n*hk*segments;
    const int64_t windowRows=sink+recent+spec;
    Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*4);
    Gm k(dir+"/ck.bin",n*hk*d*2),v(dir+"/cv.bin",n*hk*d*2),rv(dir+"/rv.bin",d*d*4);
    Gm raw(dir+"/raw.bin",prefix+blocks*stride),striped(raw.size);
    Gm table(dir+"/table.bin",requests*columns*4);
    Gm wk(dir+"/wk.bin",blocks*windowRows*hk*d*2),wv(dir+"/wv.bin",blocks*windowRows*hk*d*2);
    Gm tags(dir+"/tags.bin",blocks*windowRows*8),starts(dir+"/starts.bin",(requests+1)*4);
    Gm lens(dir+"/lens.bin",requests*4),slots(dir+"/slots.bin",n*8);
    Gm tasks(tasksCount*16*8),positions(n*8);
    const int64_t workspacePerCore=cluster16?
        oscar_ascend::attention_cluster16_workspace_per_core(d):
        (cluster4?oscar_ascend::attention_cluster4_workspace_per_core(d):
         oscar_ascend::attention_workspace_per_core(d));
    Gm partial(n*hq*segments*d*4),partLse(n*hq*segments*4);
    Gm status(tasksCount*2*4),workspace(cores*workspacePerCore);
    Gm stats(cores*8*8);
    StripeRaw(raw,striped,prefix,blocks,stride,b,hk);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,
        tasks.ptr,positions.ptr,requests,n,hq,hk,sink,recent,splits,true,
        static_cast<uint8_t*>(nullptr),columns,false);
    const auto expectedPositions=Read(dir+"/expected_positions.bin",positions.size);
    if(std::memcmp(positions.ptr,expectedPositions.data(),positions.size))
      throw std::runtime_error("task positions differ from golden");
    const float scale=1.0F/std::sqrt(float(d));
#define RUN_BASE(kernel,rawPtr) \
    ICPU_RUN_KF(kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,rawPtr,table.ptr, \
        wk.ptr,wv.ptr,tags.ptr,tasks.ptr,partial.ptr,partLse.ptr,status.ptr,workspace.ptr, \
        n,hq,hk,d,requests,columns,tasksCount,b,blocks,prefix,stride, \
        windowRows*hk*d,windowRows,sink,recent,spec,splits,scale)
#define RUN_CLUSTER(kernel,rawPtr) \
    ICPU_RUN_KF(kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,rawPtr,table.ptr, \
        wk.ptr,wv.ptr,tags.ptr,tasks.ptr,partial.ptr,partLse.ptr,status.ptr,workspace.ptr, \
        stats.ptr,n,hq,hk,d,requests,columns,tasksCount,b,blocks,prefix,stride, \
        windowRows*hk*d,windowRows,sink,recent,spec,splits,scale)
    auto launch=[&](bool candidate) {
      AscendC::SetKernelMode(KernelMode::MIX_MODE);
      if(mode=="base") {
        if(candidate)RUN_BASE(oscar_attention_cv_striped_kernel,striped.ptr);
        else RUN_BASE(oscar_attention_cv_fast_kernel,raw.ptr);
      } else if(mode=="balanced") {
        if(candidate)RUN_BASE(oscar_attention_cv_striped_balanced_kernel,striped.ptr);
        else RUN_BASE(oscar_attention_cv_fast_balanced_kernel,raw.ptr);
      } else if(cluster4) {
        if(candidate)RUN_CLUSTER(oscar_attention_cv_striped_cluster4_kernel,striped.ptr);
        else RUN_CLUSTER(oscar_attention_cv_fast_cluster4_kernel,raw.ptr);
      } else {
        if(candidate)RUN_CLUSTER(oscar_attention_cv_striped_cluster16_kernel,striped.ptr);
        else RUN_CLUSTER(oscar_attention_cv_fast_cluster16_kernel,raw.ptr);
      }
      CheckStatus(status);
      if(cluster4 || cluster16) {
        int64_t clusters=0,shared=0;
        for(int64_t core=0;core<cores;++core) {
          int64_t x,y;std::memcpy(&x,stats.ptr+(core*8)*8,8);
          std::memcpy(&y,stats.ptr+(core*8+3)*8,8);
          if(x<0 || y<0)throw std::runtime_error("negative cluster counters");
          clusters+=x;shared+=y;
        }
        if(clusters==0 || shared==0)throw std::runtime_error("fixture did not activate shared cluster");
      }
      CheckOracle(dir,partial,partLse,n,hq,d,segments);
    };
    const bool doReference=phase=="reference" || phase=="both";
    const bool doCandidate=phase=="candidate" || phase=="both";
    if(doReference) {
      launch(false);
      if(phase=="reference") {
        std::filesystem::create_directories(refDir);
        Write(refDir+"/partial.bin",partial.ptr,partial.size);
        Write(refDir+"/lse.bin",partLse.ptr,partLse.size);
        Write(refDir+"/status.bin",status.ptr,status.size);
        if(cluster4 || cluster16)Write(refDir+"/stats.bin",stats.ptr,stats.size);
        std::ofstream hash(refDir+"/input_sha256.txt");
        if(!(hash<<InputHash(dir)<<"\n") || !hash.flush())
          throw std::runtime_error("failed to write reference input hash");
      }
    }
    if(doCandidate) {
      if(phase=="candidate") {
        std::ifstream hash(refDir+"/input_sha256.txt");std::string previous;
        if(!(hash>>previous) || previous!=InputHash(dir))
          throw std::runtime_error("candidate inputs differ from completed reference");
      }
      const auto refPartial=phase=="both"?std::vector<uint8_t>(partial.ptr,partial.ptr+partial.size):
          Read(refDir+"/partial.bin",partial.size);
      const auto refLse=phase=="both"?std::vector<uint8_t>(partLse.ptr,partLse.ptr+partLse.size):
          Read(refDir+"/lse.bin",partLse.size);
      const auto refStatus=phase=="both"?std::vector<uint8_t>(status.ptr,status.ptr+status.size):
          Read(refDir+"/status.bin",status.size);
      const auto refStats=(cluster4 || cluster16)?
          (phase=="both"?std::vector<uint8_t>(stats.ptr,stats.ptr+stats.size):
           Read(refDir+"/stats.bin",stats.size)):std::vector<uint8_t>{};
      std::memset(partial.ptr,0x85,partial.size);
      std::memset(partLse.ptr,0x85,partLse.size);
      std::memset(status.ptr,0x85,status.size);
      std::memset(workspace.ptr,0x85,workspace.size);
      std::memset(stats.ptr,0x85,stats.size);
      launch(true);
      if(std::memcmp(partial.ptr,refPartial.data(),partial.size) ||
          std::memcmp(partLse.ptr,refLse.data(),partLse.size) ||
          std::memcmp(status.ptr,refStatus.data(),status.size) ||
          ((cluster4 || cluster16) && std::memcmp(stats.ptr,refStats.data(),stats.size)))
        throw std::runtime_error("striped partial/LSE/status/stats differs bytewise");
    }
#undef RUN_BASE
#undef RUN_CLUSTER
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"mode\":\""<<mode
             <<"\",\"phase\":\""<<phase<<"\",\"status\":\"passed\","
             <<"\"oracle_scope\":\"sampled_independent\",\"bitwise\":"
             <<(doCandidate?"true":"null")<<"}"<<std::endl;
    return 0;
  }catch(const std::exception& e) {
    std::cerr<<"STRIPED_PREFILL_CPU_FAILED: "<<e.what()<<std::endl;return 1;
  }
}
