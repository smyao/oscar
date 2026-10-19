"""Archive #13-22/#126/#140/#148/#151/#154/#155/#163/#167; D.4 reread.

D.4: count the bounded fused reconstruction/attention path, never propose
full-history expansion; old 6.5-second restore is a warning, not calibration.
Execute current C++ Cube control flow and shared Unpack API calls on host.
No scalar-stub elapsed time is interpreted as device latency. No production
kernel, numerical oracle, acceptance threshold or serving route is changed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / 'csrc/kernels'
FILES = {
 'mdl_q4':'attention_bf16_history_mdl.cpp',
 'mdl_q1':'attention_bf16_history_mdl.cpp',
 'c4':'attention_bf16_history_member_pipeline_cluster4.cpp',
 'c16':'attention_bf16_history_window_range_cluster16.cpp',
 'range':'attention_bf16_history_window_range.cpp',
 'balanced':'attention_bf16_history_window_range_balanced.cpp',
}


def clean(s):
    return re.sub(r'//[^\n]*|/\*.*?\*/', '', s, flags=re.S)


def function(s, name):
    s=clean(s)
    m=re.search(r'__aicore__\s+(?:void|int32_t|int64_t)\s+'+name+r'\(',s)
    if not m: raise ValueError(name)
    first=s.index('{',m.start());depth=1;end=first+1
    while depth:
        depth+=(s[end]=='{')-(s[end]=='}');end+=1
    return s[m.start():end].replace('__aicore__','')


STUB=r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#define __aicore__
std::map<std::string,long long> ops;
int actor=0;
enum Pipe{PIPE_FIX,PIPE_MTE3,PIPE_MTE2,PIPE_V};
void CrossCoreWaitFlag(int){++ops["cube_wait_api"];}
template<int,Pipe>void CrossCoreSetFlag(int){++ops["cube_set_api"];}
int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
struct TaskState{int64_t id,kvbegin,kvend,qcount;};
struct half{};struct bfloat16_t{};
namespace AscendC{
enum class TPosition{VECCALC};
enum class RoundMode{CAST_NONE,CAST_RINT,CAST_ROUND};
enum class HardEvent{V_S,S_V};
enum class CMPMODE{GT,LT,GE};
template<class T>struct LocalTensor{
 LocalTensor operator[](int32_t)const{return{};}
 template<class U>LocalTensor<U>ReinterpretCast()const{return{};}
 T GetValue(int32_t)const{return T{};}void SetValue(int32_t,T){}
};
template<TPosition>struct TBuf{template<class T>LocalTensor<T>Get(){return{};}};
struct UnaryRepeatParams{uint16_t dst,src;uint8_t dstRep,srcRep;};
struct BinaryRepeatParams{uint8_t dst,src0,src1,dstRep,src0Rep,src1Rep;};
template<Pipe>void PipeBarrier(){++ops["PipeBarrier"];}
#define VEC(name) template<class...A>void name(A...){++ops[#name];}
VEC(Duplicate) VEC(ShiftRight) VEC(And) VEC(Gather) VEC(Cast)
VEC(Mul) VEC(Add) VEC(Muls) VEC(Adds)
template<class T,class U,class...A>void Compares(A...){++ops["Compares"];}
#undef VEC
}
namespace oscar_ascend_device{template<AscendC::HardEvent>void Fence(){++ops["local_Fence"];}}
'''


def cpp_program():
    source=STUB
    common=clean((ROOT/'csrc/include/oscar_attention_launch.h').read_text())
    source+='namespace oscar_ascend {\n'+'\n'.join(re.findall(
        r'constexpr int64_t kAttention(?:Query|Kv)Rows\s*=[^;]+;',common))+'\n}\n'
    unpack=(KERNELS/'attention_striped_unpack_bf16.h').read_text()
    source+=unpack.replace('#include "oscar_common.h"','').replace('#pragma once','')
    schedule=(ROOT/'csrc/include/oscar_cv_schedule.h').read_text()
    source+=schedule.replace('#include <cstdint>','').replace('#pragma once','')
    for i,(family,filename) in enumerate(FILES.items()):
        text=clean((KERNELS/filename).read_text())
        constants=text[text.index('constexpr int32_t kQueryRows='):text.index('constexpr SoftmaxConfig')]
        bodies=function(text,'CubeTask')
        if family.startswith('mdl'):bodies+=function(text,'HistoryCubeTask')+function(text,'TailColumns')
        if family in ('c4','c16'):bodies+=function(text,'CubeCluster')
        if family=='c4':bodies+=function(text,'SelectClusterSlots')
        source+=f'\nnamespace f{i}{{\n'+constants+r'''
struct Model{
 static constexpr int D=256;
 struct{int hq=6,hk=1;}g;
 int64_t qcount=1,kind=0,kvbegin=0,kvend=0,qOffset=0,kOffset=0,vOffset=0;
 int64_t scoreOffset=0,pOffset=0,pvOffset=0,rotOffset=0,normOffset=0;
 int64_t accStateOffset=0,extraKvOffset=0,baseKOffset=0,extraScoreOffset=0;
 int64_t baseScoreOffset=0,extraPOffset=0,basePOffset=0;
 void Matmul(int64_t,int64_t,int64_t,int32_t m,int32_t n,int32_t k,
             bool rotation=false,bool transposeB=true){
  const char*type=rotation?"rot":(transposeB?"qk":"pv");
  ++ops[std::string(type)+"_calls"];
  ops[std::string(type)+"_flops"]+=2LL*m*n*k;
  int bytes=rotation?4:2;
  ops[std::string(type)+"_a_min_read"]+=1LL*m*k*bytes;
  ops[std::string(type)+"_b_min_read"]+=1LL*n*k*bytes;
  ops[std::string(type)+"_c_min_write"]+=1LL*m*n*4;
 }
 void MatmulAtomic(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k){
  Matmul(a,b,c,m,n,k,false,false);ops["pv_atomic_read"]+=1LL*m*n*4;}
'''+bodies+'\n};}\n'
    source+=r'''
template<int H>void unpack_count(){
 using namespace AscendC;TBuf<TPosition::VECCALC>b;int error=0;
 oscar_ascend_striped_bf16::Unpack<256,H>(false,H,error,b,b,b,b,b,b,b,b,b,b,b,b);
}
int main(int argc,char**argv){if(argc<3)return 1;
 std::string mode=argv[1];int x=std::atoi(argv[2]);
 if(mode=="unpack"){if(x==16)unpack_count<16>();else if(x==32)unpack_count<32>();
 else if(x==64)unpack_count<64>();else return 2;}
 else if(mode=="schedule"){
  int q=std::atoi(argv[3]),padded=std::atoi(argv[4]),splits=std::atoi(argv[5]);
  int cores=20;oscar_ascend_schedule::CvTaskSchedule s{padded,q,3*splits};
  for(int c=0;c<cores;++c){long long live=0,scan=0;
   for(int64_t w=c;w<s.WorkItems();w+=cores)
    for(int64_t t=s.TokenBegin(w);t<Min64(s.TokenBegin(w)+q,padded);++t){
     ++scan;if(t<x*q && t%q==0 && s.Segment(w)<splits)++live;}
   std::printf("core %d %lld %lld\n",c,live,scan);}
  return 0;
 }
 else if(mode=="cube"){
  int kind=std::atoi(argv[3]),length=std::atoi(argv[4]),q=std::atoi(argv[5]),cluster=std::atoi(argv[6]);
'''
    for i,(family,_) in enumerate(FILES.items()):
        source+=f'if(x=={i}){{f{i}::Model m;m.kind=kind;m.kvend=length;m.qcount=q;'
        if family in ('c4','c16'):
            c=4 if family=='c4' else 16
            source+=f'TaskState members[{c}];for(int j=0;j<{c};++j)members[j]={{j,0,length,q}};if(cluster)m.CubeCluster(members);else '
        if family.startswith('mdl'):source+='if(kind==0 && length)m.HistoryCubeTask();else m.CubeTask();'
        else:source+='m.CubeTask();'
        source+='}'
    source+='}else return 3;for(auto&p:ops)std::printf("%s %lld\\n",p.first.c_str(),p.second);return 0;}\n'
    return source


class Profiler:
    def __init__(self,directory):
        compiler=os.environ.get('OSCAR_TEST_CLANGXX') or shutil.which('clang++')
        if not compiler:raise RuntimeError('No host compiler; profile not executed')
        self.binary=Path(directory)/'count';src=Path(directory)/'count.cpp';src.write_text(cpp_program())
        run=subprocess.run([compiler,'-std=c++17','-O1',str(src),'-o',str(self.binary)],capture_output=True,text=True,timeout=45)
        if run.returncode:raise RuntimeError(run.stderr)
        self.cache={};self.cpp_sha256=hashlib.sha256(src.read_bytes()).hexdigest()
    def run(self,*args):
        key=tuple(args)
        if key not in self.cache:
            p=subprocess.run([str(self.binary),*map(str,args)],capture_output=True,text=True,timeout=10)
            if p.returncode:raise RuntimeError(p.stderr or p.stdout)
            self.cache[key]=p.stdout
        return self.cache[key]
    def cube(self,family,kind,length,q,cluster=False):
        text=self.run('cube',list(FILES).index(family),kind,length,q,int(cluster))
        return Counter({k:int(v) for k,v in (line.split() for line in text.splitlines())})
    def unpack(self,h):
        text=self.run('unpack',h)
        return {k:int(v) for k,v in (line.split() for line in text.splitlines())}


def ceildiv(a,b):return (a+b-1)//b


def source_constants():
    out={}
    for family,name in FILES.items():
        s=clean((KERNELS/name).read_text())
        def val(name):
            if name=='kQueryRows' and 'kQueryRows=oscar_ascend::kAttentionQueryRows' in s:
                common=(ROOT/'csrc/include/oscar_attention_launch.h').read_text()
                return int(re.search(r'kAttentionQueryRows\s*=\s*(\d+)',common).group(1))
            return int(re.search(r'\b'+name+r'\s*=\s*(\d+)',s).group(1))
        out[family]={'m':val('kQueryRows'),'history_h':val('kHistoryHalfKv') if family=='c4' else val('kHalfKv'),
                     'half_rows':val('kHalfRows'),'kv':256}
    return out


def one_task(p,family,kind,first,last,q,cluster=1):
    c=source_constants()[family];length=last-first;tiles=ceildiv(length,256);m=q*6
    metrics=p.cube(family,kind,length,q,cluster>1)
    # GM API payloads below are source-level bytes, NOT measured HBM bytes.
    # SDK Matmul may reread operands; caching may service these without HBM.
    metrics['packed_payload_read']=length*136 if kind==0 else 0
    metrics['packed_int2_read']=length*128 if kind==0 else 0
    metrics['packed_fp16_metadata_read']=length*8 if kind==0 else 0
    metrics['exact_bf16_kv_read']=length*1024 if kind else 0
    metrics['exact_tag_read']=length*16 if kind==1 else 0
    metrics['bf16_kv_publish']=tiles*256*1024
    if family.startswith('mdl') and kind:
        metrics['bf16_kv_publish']=sum(ceildiv(min(256,length-start),64)*64*1024 for start in range(0,length,256))
        metrics['score_padding_zero_write']=sum(
            c['m']*256*4 for start in range(0,length,256)
            if ceildiv(min(256,length-start),16)*16<256)
    metrics['q_payload_read']=cluster*m*(544 if kind==0 else 512)
    metrics['q_operand_publish']=cluster*c['m']*256*2
    metrics['score_vector_read']=cluster*tiles*ceildiv(m,c['half_rows'])*c['half_rows']*256*4
    metrics['p_bf16_publish']=cluster*tiles*m*256*2
    metrics['pv_vector_read']=0 if family=='c4' and cluster>1 else cluster*tiles*m*256*4
    metrics['cluster_state_read_write']=2*(tiles+1)*cluster*c['m']*(256+2)*4 if cluster>1 else 0
    metrics['rotation_norm_publish']=cluster*c['m']*256*4 if kind==0 and length else 0
    metrics['rotation_result_read']=metrics['rotation_norm_publish']
    metrics['partial_output_write']=cluster*m*256*4
    metrics['partial_lse_write']=cluster*m*4
    metrics['softmax_live_row_visits']=cluster*tiles*m
    metrics['softmax_api_calls']=cluster*tiles*ceildiv(m,c['half_rows'])
    metrics['softmax_shape_elements']=cluster*tiles*ceildiv(m,8)*8*256
    metrics['p_cast_elements']=cluster*tiles*m*256
    metrics['fp32_alpha_mul_elements']=cluster*tiles*m*256
    metrics['fp32_pv_acc_add_elements']=0 if family=='c4' and cluster>1 else cluster*tiles*m*256
    metrics['history_tile_decodes']=tiles if kind==0 else 0
    metrics['unpack_helper_calls']=tiles*(256//c['history_h'])*2 if kind==0 else 0
    metrics['lut_fp32_mul_add_flops']=tiles*256*16 if kind==0 else 0
    metrics['lut_bf16_cast_elements']=tiles*256*8 if kind==0 else 0
    metrics['bf16_lookup_output_elements']=tiles*256*512 if kind==0 else 0
    metrics['packed_dma_calls']=0
    metrics['page_table_read']=0
    if kind==0:
        # Exact logical DataCopyPad runs across each H-row AIV producer and
        # virtual128 page; physical pages may be scattered but no rows coalesce
        # across that boundary in current LoadPacked[Into].
        h=c['history_h']
        for tile in range(first,last,256):
            for start in range(tile,tile+256,h):
                end=min(start+h,last)
                if start<end:
                    calls=1+(end-1)//128-start//128
                    metrics['packed_dma_calls']+=calls
                    metrics['page_table_read']+=calls*4
    return metrics


def data_bytes(counts):
    keys=['packed_payload_read','page_table_read','exact_bf16_kv_read','exact_tag_read',
          'bf16_kv_publish','q_payload_read','q_operand_publish','score_vector_read',
          'p_bf16_publish','pv_vector_read','cluster_state_read_write',
          'rotation_norm_publish','rotation_result_read','partial_output_write','partial_lse_write',
          'score_padding_zero_write']
    keys += [f'{op}_{field}' for op in ('qk','pv','rot') for field in ('a_min_read','b_min_read','c_min_write')]
    keys+=['pv_atomic_read']
    return sum(counts[k] for k in keys)


def scenario(p,batch,q,context,padded,splits):
    from oscar_ascend.runtime import WorkspaceGeometry
    family='mdl_q1' if q==1 else 'mdl_q4'
    geo=WorkspaceGeometry(16384,6,1,256,cube_cores=20,history_cluster_size=16,c4_member_pipeline=True,bf16_history=True)
    actual_s=geo.splits_for_tokens(padded)
    if splits is None:splits=actual_s
    hist_last=min(context,max(64,context+q-256));window_last=min(64,context)+context-min(context,max(64,context+1-256))
    ranges=((min(64,context),hist_last),(0,window_last),(context,context+q))
    total=Counter();per_source=[]
    for kind,(first,last) in enumerate(ranges):
        c=Counter();width=ceildiv(last-first,splits)
        for split in range(splits):
            a=min(last,first+split*width);z=min(last,a+width)
            c.update(one_task(p,family,kind,a,z,q))
        per_source.append(dict(c));total.update(c)
    total=Counter({k:v*batch for k,v in total.items()})
    schedule=[list(map(int,line.split()[1:])) for line in p.run('schedule',batch,q,padded,splits).splitlines()]
    live=[x[1] for x in schedule]
    # Source0 dominates long history. Unequal split tails add a small cost not
    # represented by this task-count balance factor; full GM/FLOP totals keep it.
    return {'batch':batch,'q':q,'context':context,'padded_tokens':padded,'splits':splits,
     'actual_runtime_splits_for_this_shape':actual_s,'family':family,
     'per_request_source_counts':per_source,'counts':dict(total),
     'source_gm_payload_and_min_cube_bytes':data_bytes(total),
     'ideal_native_bf16_kv_read_bytes':batch*(context+q)*1024,
     'history_tasks_per_core':live,'history_task_balance_efficiency':sum(live)/(20*max(live)) if max(live) else 0,
     'active_history_cube_cores':sum(x>0 for x in live),
     'descriptor_scan_rows':sum(x[2] for x in schedule),
     'meaning':'Source-level counts per TP card per FULL layer; no NPU time or whole-model speed claim.'}


def resource_breakdown(counts):
    """Keep bytes for persistent payload and temporary GM separate, never HBM."""
    persistent=sum(counts.get(k,0) for k in ('packed_payload_read','page_table_read',
        'exact_bf16_kv_read','exact_tag_read','q_payload_read','rot_b_min_read'))
    total=data_bytes(Counter(counts))
    return {'persistent_source_bytes':persistent,'temporary_gm_min_bytes':total-persistent,
        'total_gm_payload_and_min_cube_bytes':total,
        'bf16_qk_pv_flops':counts.get('qk_flops',0)+counts.get('pv_flops',0),
        'fp32_inverse_rotation_flops':counts.get('rot_flops',0),
        'qk_calls':counts.get('qk_calls',0),'pv_calls':counts.get('pv_calls',0),
        'inverse_rotation_calls':counts.get('rot_calls',0),
        'cube_crosscore_wait_api_calls':counts.get('cube_wait_api',0),
        'cube_crosscore_set_api_calls':counts.get('cube_set_api',0),
        'unpack_helper_calls':counts.get('unpack_helper_calls',0)}


def full_layer_step(p,batch,context,captures):
    """17 q4 + 2 q1 CV calls; first source2 uses native current FIA."""
    captured=min(x for x in captures if x>=4*batch)
    target=scenario(p,batch,4,context,captured,None)
    first=scenario(p,batch,4,context,4*batch,None)
    # Native draft enforce_eager=true preserves eager first-pass N=4B;
    # later rounds inherit that N, not target capture-rounding.
    later=scenario(p,batch,1,context,4*batch,None)
    first_counts=Counter(first['counts'])
    first_counts.subtract({k:v*batch for k,v in first['per_request_source_counts'][2].items()})
    first['counts']=dict(first_counts)
    first['per_request_source_counts'][2]={}
    first['source_gm_payload_and_min_cube_bytes']=data_bytes(first_counts)
    first['meaning']='First MTP q4 CV source0+1 only; source2 uses native current FIA whose internal cost is not modelled.'
    # Current descriptor layouts may retain holes; live leaders are modelled
    # contiguous. Token scan count is exact for N; live-core balance is only
    # for this explicitly stated placement, never a proof of target placement.
    total=Counter({k:v*16 for k,v in target['counts'].items()})
    total.update(first['counts']);total.update({k:v*2 for k,v in later['counts'].items()})
    return {'batch':batch,'context':context,'target_q4':target,'first_q4':first,
        'later_q1_retained_eager_first_n':later,'counts':dict(total),'resources':resource_breakdown(total),
        'ideal_native_bf16_kv_read_bytes':batch*(17*(context+4)+2*(context+1))*1024,
        'descriptor_scan_rows':16*target['descriptor_scan_rows']+first['descriptor_scan_rows']+2*later['descriptor_scan_rows'],
        'omitted_native_current_FIA':{'calls':1,'batch':batch,'q_per_request':4,
            'source':2,'internal_bytes_flops_latency':'not modelled'},
        'scope':'Per card per model step: 16 target q4 three-source CV, 1 first MTP q4 history/window CV plus omitted native current FIA, 2 later MTP q1 three-source CV. Quant/store, Q rotation producer, prepare, status/guard, merge and final copy are excluded; no whole-model latency claim.'}


def thresholds(p):
    rows=[]
    for q in (1,4):
        counts=one_task(p,'mdl_q1' if q==1 else 'mdl_q4',0,0,256,q)
        # Remove once-per-task Q publication, inverse rotation and partial
        # writes to isolate the steady-state full history tile.
        for key in list(counts):
            if key.startswith('rot_') or key in ('q_payload_read','q_operand_publish','rotation_norm_publish',
                'rotation_result_read','partial_output_write','partial_lse_write'):counts[key]=0
        total=data_bytes(counts);persistent=counts['packed_payload_read']+counts['page_table_read']
        scratch=total-persistent;native=256*1024;saving=native-persistent
        for bw in (0.5,1.0,1.5):
            bytes_s=bw*1e12
            rows.append({'q':q,'assumed_effective_hbm_TBps':bw,'native_ideal_bytes':native,
             'packed_and_page_bytes':persistent,'workspace_gm_min_bytes':scratch,
             'workspace_equivalent_miss_fraction_limit':saving/scratch,
             'aggregate_extra_decode_budget_ns':saving/bytes_s*1e9,
             'per_cube_extra_decode_budget_us_at_20_active':saving/bytes_s*20*1e6,
             'unpack_min_aggregate_Gvalues_per_s_if_serial_with_packed_read':256*512/(saving/bytes_s)/1e9,
             'bf16_qk_pv_effective_TFps_to_fit_native_memory_floor':
                (counts['qk_flops']+counts['pv_flops'])/(native/bytes_s)/1e12,
             'bf16_qk_pv_effective_TFps_to_be_packed_memory_bound':
                (counts['qk_flops']+counts['pv_flops'])/(persistent/bytes_s)/1e12,
             'interpretation':'Hypothetical effective bandwidth, not device specification. Extra decode budget ignores other work; any positive workspace HBM traffic or sync overhead reduces it. min operands exclude SDK reloads.'})
    return rows


def tile_resources(p,helpers):
    out=[]
    for q in (1,4):
        counts=one_task(p,'mdl_q1' if q==1 else 'mdl_q4',0,0,256,q)
        for key in list(counts):
            if key.startswith('rot_') or key in ('q_payload_read','q_operand_publish',
                'rotation_norm_publish','rotation_result_read','partial_output_write','partial_lse_write'):
                counts[key]=0
        # The one-task cross-core fence counters include two waits and one
        # signal for inverse rotation; exclude those from the steady tile.
        counts['cube_wait_api']-=2;counts['cube_set_api']-=1
        out.append({'q':q,'history_tokens':256,'counts':dict(counts),
            'resources':resource_breakdown(counts),
            'unpack_vector_API_invocations':{k:v*8 for k,v in helpers['64'].items()},
            'scope':'Steady full history KV256 tile, one KV head, GQA6, excluding per-task query initialization, inverse rotation and partial publication. Matmul requests actual source; Vector API counts actual shared Unpack helper, not lowered ISA.'})
    return out


def old_anchors():
    paths=['reports/q1_mdl_qualification_20261013/completed/comparison.json',
     'reports/decode_mdl_load_20261012/completed/reports/mdl-comparison.json',
     'reports/decode_integrated_batch32_retry_after_oom_20261007/q4_r32_n128_20to30k_s3-comparison.json']
    out=[]
    for path in paths:
        obj=json.loads((ROOT/path).read_text())
        if not str(obj['status']).startswith('completed_'):raise ValueError('uncompleted anchor')
        out.append({'path':path,'sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
         'status':obj['status'],'case':obj.get('case','q4_r2_n8_c1341_s1'),
         'task_ticks':obj['task_ticks'],'ratio':obj.get('ratio',obj.get('baseline_over_candidate')),
         'scope':'Old FP32 operand operators only; not current BF16 clock calibration. No NPU/graph/whole-model extrapolation.',
         'NPU':obj.get('NPU',obj.get('npu','not_run'))})
    return out


def build_report():
    config=json.loads((ROOT/'configs/target.json').read_text());captures=config['compilation_config']['cudagraph_capture_sizes']
    with tempfile.TemporaryDirectory(prefix='oscar-batch-count-') as td:
        p=Profiler(td);constants=source_constants()
        helpers={str(h):p.unpack(h) for h in (16,32,64)}
        cases=[]
        batches=(1,4,6,8,11,16,19,20,31,32,46,64,128)
        for length in (32768,65536,131072):
            for q in (1,4):
                for b in batches:
                    n=min(x for x in captures if x>=b*q)
                    cases.append(scenario(p,b,q,length,n,None))
        steps=[full_layer_step(p,b,length,captures)
            for length,bs in ((32768,(20,46)),(65536,(11,31)),(131072,(6,19))) for b in bs]
        clusters=[]
        for family,size in (('c4',4),('c16',16)):
            c=one_task(p,family,0,0,4096,21,size)
            solo=one_task(p,family,0,0,4096,21,1)
            clusters.append({'family':family,'members':size,'query_tokens':size*21,'history_tokens':4096,
                'eligibility':'Ideal mature equal-domain cluster within ONE request; actual eligibility must be checked separately.',
                'cluster':dict(c),'independent_same_family_groups':{k:v*size for k,v in solo.items()},
                'cluster_source_gm_min_bytes':data_bytes(c),'independent_source_gm_min_bytes':data_bytes(solo)*size})
        tests=[]
        # Executed control-flow identities, not a mirror of performance timing.
        for family in FILES:
            q=1 if family=='mdl_q1' else 4
            for length in (1,255,256,257,65536):
                x=p.cube(family,0,length,q)
                assert x['qk_calls']==x['pv_calls']==ceildiv(length,256)
                assert x['rot_calls']==1
                assert x['qk_flops']==x['pv_flops']==2*q*6*256*256*ceildiv(length,256)
                tests.append([family,length])
        for h,count in helpers.items():
            assert count['ShiftRight']==count['And']==16
            assert count['Mul']==count['Add']==4
            assert count['Gather']==1+int(h)//8
        sources=set(FILES.values())|{'attention_striped_unpack_bf16.h'}
        paths=[str((KERNELS/f).relative_to(ROOT)) for f in sources]+[
          'csrc/kernels/attention_tasks.cpp','csrc/include/oscar_cv_schedule.h','oscar_ascend/runtime.py',
          'oscar_ascend/ops/cv_dispatch.py','tools/experiments/batch_kernel_cost_20261019.py','configs/target.json']
        return {'schema_version':2,'status':'host_source_count_profile_completed','returncode':0,
         'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'source_sha256':{x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in sorted(paths)},
         'generated_counter_cpp_sha256':p.cpp_sha256,'constants':constants,'unpack_api_counts_per_one_K_or_V_H_rows':helpers,
         'unpack_api_count_scope':'Actual whole helper executed with API counters; counts API calls, not lowered ISA/repeat cycles. All data ops are stubs.',
         'cube_scope':'Actual six Cube control-flow bodies; Matmul/API implementation stubbed. Requested logical FLOPs and minimum per-call operand bytes only.',
         'schedule_scope':'Actual CvTaskSchedule methods plus current MDL OwnerTile=q ownership loop. Healthy contiguous uniform requests, 20 Cubes, GQA6/Hkv1/D256.',
         'self_checks':{'actual_cube_shape_cases':len(tests),'helper_H_variants':3,'passed':True},
         'decode_scenarios':cases,'capacity_matched_full_layer_steps':steps,
         'steady_history_tile_resources':tile_resources(p,helpers),
         'ideal_mature_cluster_profiles':clusters,'thresholds':thresholds(p),'old_completed_FP32_CAModel_anchors':old_anchors(),
         'scaling_conclusions':[
          'Requests have independent KV. Increasing B does not amortize historical INT2 unpack or QK/PV/P/merge work per request.',
          'C4/C16 share one KV decode/publication only across eligible query groups inside the same request, not across B requests.',
          'Runtime S depends on padded tokens using M128-derived groups, even though MDL arithmetic uses M32. S changes task tail overhead and inverse-rotation multiplicity.',
          'Small B can already occupy 20 Cubes through history splits; B growth can improve scheduling granularity but does not guarantee proportional attention speedup.',
          'Persistent INT2 payload is 136B vs BF16 KV1024B per historical token/head. Extra BF16 workspace traffic can exceed the native KV payload if treated as cold HBM. L2 residency and pipeline overlap are decisive unknowns.',
          'Logical matrix FLOPs are unchanged by BF16 dtype alone; FP32 inverse rotation remains additional. Lower BF16 execution cost requires current device evidence.',
          '19-CV-call step uses identical starting context for all calls; actual next MTP positions can add a few tokens. Later q1 retains eager first-pass N=4B for S and descriptor scans, not target capture-rounding; live leaders are modelled contiguous and padding placement may change per-core balance. First MTP source2 is native current FIA and is explicitly omitted from internal cost.',
          'Quant/store, Q rotation producer, prepare, status/guard, merge/final-copy, kernel launch, projection/weights, GDN/MLP/collectives and MTP accepted-output efficiency are outside this per-FULL-layer CV count; whole-model B amortization must be modelled separately.'
         ],
         'precision_gate':'Existing BF16 screen remains 7/11 failed against frozen FP32 oracle. This cost profile does not confer acceptance.',
         'not_run':['CANN compile','official CPU debug','CAModel current BF16','NPU','graph','model quality','end-to-end performance'],
         'production_qualified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=build_report();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'scenarios':len(report['decode_scenarios']),
                     'self_checks':report['self_checks'],'output':str(args.output)}))

if __name__=='__main__':main()
