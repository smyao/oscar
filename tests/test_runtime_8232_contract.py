"""Archive #145/#154/#157/#165/#166/#167: bounded 16384+8232 source audit.

Execute the actual task generator, C16 eligibility/scheduler and packed-Q
reader on the host. DMA/arithmetic/event APIs are synchronous stubs, so these
tests prove constructed address/owner contracts, not CANN, NPU or Trap cause.
The scheduler dump can belong to an older future than the failing sync.
"""
from enum import Enum
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.current_attention import use_native_current
from oscar_ascend.integration.metadata import from_common
from oscar_ascend.ops.cv_dispatch import select_cv_op
from oscar_ascend.runtime import WorkspaceGeometry
from tools.serving_variants import variant_config
from test_bf16_query_producer import STUB, producer_source
from test_bf16_query_reader import model as query_model
from test_bf16_reader_protocol import _without_comments

ROOT = Path(__file__).resolve().parents[1]
C16 = ROOT / "csrc/kernels/attention_bf16_history_window_range_cluster16.cpp"


def _compile(tmp_path, source):
    compiler = os.environ.get("OSCAR_TEST_CLANGXX") or shutil.which("clang++")
    if not compiler:
        pytest.skip("host compiler unavailable; no device result implied")
    path, binary = tmp_path / "contract.cpp", tmp_path / "contract"
    path.write_text(source)
    result = subprocess.run([compiler, "-std=c++17", "-O2", "-ffp-contract=off",
                             str(path), "-o", str(binary)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return binary


def _method(source, name):
    match = re.search(r"__aicore__\s+(?:void|bool|TaskState)\s+" + name + r"\(", source)
    assert match, name
    begin = source.index("{", match.start())
    depth, end = 1, begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def test_current_metadata_routes_warm_8232_to_c16_without_dropping_history():
    config = variant_config(json.loads((ROOT / "configs/target.json").read_text()), "candidate")
    state = Enum("AscendAttentionState", ["ChunkedPrefill"])
    starts = torch.tensor([0, 8232], dtype=torch.int32)
    common = SimpleNamespace(causal=True, query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([24616], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([24616], dtype=torch.int32),
        slot_mapping=torch.arange(16384, 24616, dtype=torch.int64),
        block_table_tensor=torch.arange(193, dtype=torch.int32).view(1, -1),
        num_reqs=1, num_actual_tokens=8232, num_input_tokens=8232,
        max_query_len=8232, max_seq_len=24616, attn_state=state.ChunkedPrefill)
    metadata = from_common(common, current_only=True)
    assert metadata.current_only_plan is None
    assert use_native_current(metadata) and metadata.current_cumulative == (8232,)
    geometry = WorkspaceGeometry(config["max_num_batched_tokens"], 6, 1, 256,
        cube_cores=20, history_cluster_size=16, c4_member_pipeline=True, bf16_history=True)
    assert geometry.splits_for_tokens(8232) == 1
    flags = {name: config["experimental_" + name] for name in (
        "fast_unpack", "mixed_cv", "striped_cache", "decode_bundle",
        "c4_member_pipeline", "mdl_decode", "bf16_history")}
    assert select_cv_op(16, 6, 1, 8232, metadata.max_query_len, **flags) == \
        "attention_bf16_history_window_range_cluster16_out"
    assert 8232*6*256*4 <= geometry.buffer_bytes["query_rot"]
    assert 8232*6*3*256*4 <= geometry.buffer_bytes["partial"]
    assert 8232*3*16*8 <= geometry.buffer_bytes["tasks"]
    assert geometry.cv_bytes == 20*5_259_264


def _scheduler_program():
    tasks = (ROOT / "csrc/kernels/attention_tasks.cpp").read_text()
    tasks = tasks[tasks.index("constexpr int32_t kMetadataRequests"):
                  tasks.index('\n}\nextern "C"')]
    kernel = _without_comments(C16.read_text())
    fields = kernel[kernel.index("struct TaskState {"):kernel.index("template<int32_t D>")]
    schedule = (ROOT / "csrc/include/oscar_cv_schedule.h").read_text()
    schedule = schedule[schedule.index("namespace oscar_ascend_schedule"):
                        schedule.index("#ifdef OSCAR_SCHEDULE_LOCAL_MACRO")]
    stub = STUB.replace("#include <vector>", "#include <vector>\n#include <unordered_map>\n#include <cassert>")
    stub = stub.replace("template<class T>struct Tensor", "std::unordered_map<void*,size_t> gmBytes;\ntemplate<class T>struct Tensor")
    stub = stub.replace("p=v;size=1000000;", "p=v;size=gmBytes.at(v)/sizeof(T);")
    return stub + r'''
namespace oscar_ascend {constexpr int64_t kAttentionQueryRows=128;}
#define OSCAR_SCHEDULE_FN inline
int actor=0;
#define ASCEND_IS_AIC (actor==0)
#define ASCEND_IS_AIV (actor!=0)
enum class CacheLine{ENTIRE_DATA_CACHE,SINGLE_CACHE_LINE};
enum class DcciDst{CACHELINE_OUT};
template<class T,CacheLine C,DcciDst D=DcciDst::CACHELINE_OUT>
void DataCacheCleanAndInvalid(Tensor<T>){}
int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
constexpr int64_t kQueryRows=128,kKvRows=256;
''' + schedule + tasks + fields + r'''
struct Model {
 struct{int64_t tokens=0,hq=6,hk=1,requests=1,splits=1;}g;
 int64_t coreIndex=0,qbegin=0,qcount=0,kvhead=0,kvbegin=0,kvend=0,request=0;
 int64_t split=0,kind=0,context=0,requestBegin=0,qOffset=0,counters[8]={};
 int32_t taskError=0;
 bool historyMatmulActive=true;
 struct Dummy{void End(){}}mm,historyMm;
 TPipe pipe;TBuf<TPosition::VECCALC>taskBuf;
 GlobalTensor<int64_t>taskGm,clusterStats;
 std::vector<int>*owners=nullptr,*rows=nullptr;
 int clusters=0,soloHistory=0;
 Model(){pipe.InitBuffer(taskBuf,128);}
 void Record(int64_t id,bool output){
  if(id<0 || id>=int64_t(owners->size()))throw std::runtime_error("task owner bound");
  ++owners->at(id);
  if(output)for(int64_t i=0;i<qcount;++i){
   int64_t index=(qbegin+i)*3+kind;
   if(index<0 || index>=int64_t(rows->size()))throw std::runtime_error("partial owner bound");
   ++rows->at(index);}}
 void PublishStatus(int64_t id,int code){if(code)throw std::runtime_error("unexpected status");Record(id,false);}
 void PublishEmpty(int64_t id,int code){if(code)throw std::runtime_error("unexpected malformed task");Record(id,false);}
 void VectorTask(int64_t id){if(kind==0)++soloHistory;Record(id,true);}
 void VectorCluster(const TaskState* members){++clusters;for(int i=0;i<16;++i){UseTask(members[i]);Record(members[i].id,true);}}
 void CubeTask(){VectorTask(qbegin*3+kind);}
 void CubeCluster(const TaskState* members){VectorCluster(members);}
''' + "\n".join(_method(kernel, n) for n in (
        "Process", "LoadTask", "CurrentTask", "UseTask", "IsClusterMember", "TaskValid")) + r'''
};
int main(int argc,char**argv){
 assert(argc==2);const int n=std::atoi(argv[1]),actual=8232;
 assert(n>=actual);blocks=20;
 std::vector<int32_t>starts{0,actual},lens{16384+actual};
 std::vector<int64_t>slots(n,-1),positions(n,-99),descriptors(n*3*16,-99),stats(20*8,-99);
 for(int t=0;t<actual;++t)slots[t]=16384+t;
 auto bind=[](auto&v){gmBytes[v.data()]=v.size()*sizeof(v[0]);};
 bind(starts);bind(lens);bind(slots);bind(positions);bind(descriptors);bind(stats);
 for(block=0;block<blocks;++block){AttentionTasks<int64_t>p;p.Run(
  (GM_ADDR)starts.data(),(GM_ADDR)lens.data(),(GM_ADDR)slots.data(),
  (GM_ADDR)descriptors.data(),(GM_ADDR)positions.data(),1,n,6,1,64,256,1,nullptr,0,false);}
 for(int t=0;t<n;++t){assert(positions[t]==(t<actual?16384+t:-1));
  for(int kind=0;kind<3;++kind){auto row=descriptors.data()+(t*3+kind)*16;
   assert(row[0]==t && row[7]==kind && row[10]==0);
   assert(row[1]==(t>=actual?-1:(t%21?0:21)));}}
 // Run the real scheduler/predicate separately for AIC and each AIV actor.
 // Arithmetic and cross-core completion are deliberately not simulated.
 for(actor=0;actor<3;++actor){std::vector<int>owners(n*3),outputs(n*3);int clusters=0,solo=0;
  for(int core=0;core<20;++core){Model m;m.g.tokens=n;m.coreIndex=core;m.owners=&owners;m.rows=&outputs;
   m.taskGm.SetGlobalBuffer(descriptors.data());m.clusterStats.SetGlobalBuffer(stats.data());m.Process();
   clusters+=m.clusters;solo+=m.soloHistory;}
  assert(clusters==23 && solo==24);
  for(int t=0;t<n;++t)for(int kind=0;kind<3;++kind){int i=t*3+kind;
   assert(outputs[i]==(t<actual?1:0));
   // The Cube has no work/status publication for follower or padding rows.
   int expected=(actor==0 && (t>=actual || t%21))?0:1;
   assert(owners[i]==expected);}
 }
 std::puts("actual prepare/C16 scheduler: 23 clusters, 24 history solos, exact owners");
}
'''


@pytest.mark.parametrize("padded", [8232, 8256])
def test_actual_prepare_and_c16_schedule_cover_8232_once(tmp_path, padded):
    binary = _compile(tmp_path, _scheduler_program())
    result = subprocess.run([str(binary), str(padded)], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_c16_packed_query_reader_handles_final_full_group_and_bad_stride(tmp_path):
    # Exercise the actual producer and LoadQueries at the end of the long
    # buffer, with a full 126 live rows spanning both lanes/two score blocks.
    stub = STUB.replace("MTE3_S};", "MTE3_S,MTE3_MTE2};")
    stub = stub.replace("CAST_RINT};", "CAST_RINT,CAST_NONE};")
    stub = stub.replace("T GetValue(size_t i)const",
        "template<class U>Tensor<U> ReinterpretCast(){return{reinterpret_cast<U*>(p),size*sizeof(T)/sizeof(U)};}\n T GetValue(size_t i)const")
    # Producer tensors are bound to exact allocations rather than the small
    # fixture's default upper bound; the offset model checks every access.
    stub = stub.replace("#include <vector>", "#include <vector>\n#include <unordered_map>")
    stub = stub.replace("template<class T>struct Tensor", "std::unordered_map<void*,size_t> gmBytes;\ntemplate<class T>struct Tensor")
    stub = stub.replace("p=v;size=1000000;", "p=v;size=gmBytes.at(v)/sizeof(T);")
    stub += r'''
int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
int64_t Max64(int64_t a,int64_t b){return a>b?a:b;}
namespace oscar_ascend{constexpr int kAttentionQueryRows=128,kAttentionKvRows=256;}
void Cast(Tensor<float>d,Tensor<bfloat16_t>s,RoundMode,size_t n){for(size_t i=0;i<n;++i){
 uint32_t b=uint32_t(s.GetValue(i).bits)<<16;float v;std::memcpy(&v,&b,4);d.SetValue(i,v);}}
'''
    model = query_model(C16.name).replace("hq=12,hk=2", "hq=6,hk=1")
    model = model.replace("qbegin=3,kvhead=1", "qbegin=8211,kvhead=0")
    model = model.replace("tokens=32,heads=12", "tokens=8232,heads=6")
    model = model.replace("for(int count:{1,4})", "for(int count:{21})")
    model = model.replace("*heads+6+r%6", "*heads+r%6")
    model = model.replace("r==3*heads+6", "r==8211*heads")
    model = model.replace("r==4*heads+7", "r==8231*heads+5")
    marker = "for(int repeat=0;repeat<2;++repeat){"
    assert marker in model
    model = model.replace(marker, r'''
 auto bind=[](auto&v){gmBytes[v.data()]=v.size()*sizeof(v[0]);};
 bind(input);bind(packed);bind(matrix);bind(status);
''' + marker)
    cpp = stub + producer_source() + model + "int main(){return check();}"
    binary = _compile(tmp_path, cpp)
    result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    # Negative control: the old D-stride must not pass the long-row contract.
    assert "DataCopy(dst,qr[off],D)" in cpp
    binary = _compile(tmp_path, cpp.replace("DataCopy(dst,qr[off],D)", "DataCopy(dst,qr[(off/D)*(D+16)],D)"))
    result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=20)
    assert result.returncode != 0
