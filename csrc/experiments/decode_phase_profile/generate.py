"""Archive #140/#144/#148/#154/#155 and startup D.4: exact-source diagnostic clones.

No production edits. Reversible counted edits add clocks/counters only, no completion
fence/PIPE_ALL, and assert original source plus math/control flow can be restored.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
INPUTS={
 'simd':('csrc/kernels/attention_cv_striped_decode_simd.cpp','c67219ce15c98e932acaa8b1c5515a3f31c99dd6e4963823623528ba2b3a3930'),
 'pipeline':('csrc/experiments/decode_history_pipeline/attention_pipeline.cpp','f9cb6862783dc2289944405766d3cbc56101e24bf538b6da913fe082e0ccb372'),
}
def sha(value):return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()
def masked(text):
 return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27',lambda m:''.join('\n' if c=='\n' else ' ' for c in m[0]),text,flags=re.S)
def function(text,name):
 m=list(re.finditer(r'__aicore__\s+(?:inline\s+)?(?:void|bool|int(?:32|64)_t)\s+'+re.escape(name)+r'\s*\([^;{}]*\)\s*\{',masked(text)))
 assert len(m)==1,(name,len(m))
 start=m[0].end()-1;depth=0
 for end,c in enumerate(masked(text)[start:],start):
  if c=='{':depth+=1
  elif c=='}':
   depth-=1
   if depth==0:return start,end+1
 raise AssertionError('unbalanced function '+name)
ENUM='''// Profile ABI schema2: same [core,3,4,24] layout as the old profile kernel.
// Fields20/21 are explicitly metadata/writeback, NOT legacy bits/meta labels.
enum ProfileField : int32_t {
  kTasks=0,kKvUnits=1,kKvRowsObserved=2,kTaskSpan=3,kQueryStage=4,
  kPackedDma=5,kUnpack=6,kPreciseLoad=7,kKvPublish=8,
  kAivQkWait=9,kSoftmaxP=10,kAivPvWait=11,kPvAcc=12,
  kAicKvWait=13,kAicQk=14,kAicPWait=15,kAicPv=16,kAicAccWait=17,
  kAicRotation=18,kAivFinalize=19,kTaskMetadata=20,kFinalWriteback=21,
  kRotationWait=22,kComputeSpan=23,
};
'''
HELPERS='''  __aicore__ inline void TraceMarker(int32_t field,bool end) {
#ifdef OSCAR_CAMODEL_MARKERS
    // SYS_CNT is zero in local CAModel. The simulator's instruction trace
    // supplies actual virtual completion cycles for this explicit MOV.
    // No memory write, completion fence or clock value is fabricated.
    const uint64_t source=(field==20 || field==23)?3:
        ((kind>=0 && kind<3)?static_cast<uint64_t>(kind):3);
    const uint64_t tag=0x5a77c01d00000000ULL | (source<<8) |
        (static_cast<uint64_t>(field)<<1) | static_cast<uint64_t>(end);
    uint64_t output;
    __asm__ __volatile__("MOV %0, %1\\n" : "=l"(output) : "l"(tag) : "memory");
#endif
  }
  __aicore__ inline void ProfileAdd(int32_t field,uint64_t delta) {
    if(kind>=0 && kind<3)profileCounters[kind][field]+=static_cast<int64_t>(delta);
    profileCounters[3][field]+=static_cast<int64_t>(delta);
  }
  class ProfileScope {
   public:
    __aicore__ inline ProfileScope(AttentionCv& owner,int32_t field)
        :owner_(owner),field_(field),start_(GetSystemCycle()) {owner_.TraceMarker(field_,false);}
    __aicore__ inline ~ProfileScope() {owner_.TraceMarker(field_,true);owner_.ProfileAdd(field_,GetSystemCycle()-start_);}
   private:
    AttentionCv& owner_;int32_t field_;uint64_t start_;
  };
  __aicore__ inline void ProfileWait(int32_t field,uint16_t flag) {
    const uint64_t start=GetSystemCycle();
    TraceMarker(field,false);
    CrossCoreWaitFlag(flag);
    TraceMarker(field,true);
    ProfileAdd(field,GetSystemCycle()-start);
  }
  __aicore__ void ProfileFlush() {
    constexpr int64_t ownerElements=oscar_ascend::kAttentionProfileSources*
        oscar_ascend::kAttentionProfileFields;
    static_assert(oscar_ascend::kAttentionProfileFields==24,"profile ABI drift");
    static_assert(ownerElements*sizeof(int64_t)%64==0,"actor cache-line ownership");
    const int64_t base=(coreIndex*oscar_ascend::kAttentionProfileEngines+engineIndex)*ownerElements;
    for(int32_t source=0;source<4;++source)
      for(int32_t field=0;field<24;++field)
        profileGm.SetValue(base+source*24+field,profileCounters[source][field]);
    // Only disjoint diagnostic cache lines are published after measurement.
    for(int32_t line=0;line<ownerElements*sizeof(int64_t)/64;++line) {
      __asm__ __volatile__("");
      DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE,
          DcciDst::CACHELINE_OUT>(profileGm[base+line*8]);
    }
  }
'''
FIELDS=['tasks','kv_units','kv_rows_observed','task_span','query_stage','packed_dma','unpack','precise_load','kv_publish','aiv_qk_wait','softmax_p','aiv_pv_wait','pv_acc','aic_kv_wait','aic_qk','aic_p_wait','aic_pv','aic_acc_wait','aic_rotation','aiv_finalize','task_metadata','final_writeback','rotation_wait','compute_span']

def generate(kind,path,pin):
 original=(ROOT/path).read_text();assert sha(original)==pin,(path,'source hash changed; re-audit before repinning',sha(original))
 text=original;edits=[]
 def replace(old,new,count=1,label=''):
  nonlocal text
  assert text.count(old)==count,(kind,label or old[:60],text.count(old),count)
  text=text.replace(old,new);edits.append((old,new,count,label))
 def edit_method(name,change):
  a,b=function(text,name);body=text[a:b];updated=change(body)
  assert body!=updated,name
  replace(body,updated,1,'method '+name)
 def scope(name,field,task=False):
  line='\n    ProfileScope profileMethod(*this,'+field+');'
  if task:line+='\n    ProfileAdd(kTasks,1);'
  edit_method(name,lambda body:body[:1]+line+body[1:])
 def waits(name,flag,fields):
  token='CrossCoreWaitFlag('+flag+');'
  def change(body):
   assert body.count(token)==len(fields),(name,flag,body.count(token),fields)
   for field in fields:body=body.replace(token,'ProfileWait('+field+','+flag+');',1)
   return body
  edit_method(name,change)
 replace('template<int32_t D,int32_t OwnerTile> class AttentionCv {',ENUM+'\ntemplate<int32_t D,int32_t OwnerTile> class AttentionCv {',label='enum')
 replace('GM_ADDR lse,GM_ADDR status,GM_ADDR workspace,const Geometry& geometry)', 'GM_ADDR lse,GM_ADDR status,GM_ADDR workspace,GM_ADDR profile,const Geometry& geometry)',label='Init profile argument')
 replace('    g=geometry;','    TraceMarker(kComputeSpan,false);\n    profileStart=GetSystemCycle();\n    g=geometry;',label='actor start')
 replace('    outStatus.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));','    outStatus.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));\n    profileGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(profile));',label='profile pointer')
 replace('    coreIndex=core;','''    coreIndex=core;
    engineIndex=0;
    if ASCEND_IS_AIV {engineIndex=lane+1;}
    for(int32_t source=0;source<4;++source)
      for(int32_t field=0;field<24;++field)profileCounters[source][field]=0;''',label='counter init')
 replace('  __aicore__ void Process() {',HELPERS+'  __aicore__ void Process() {',label='helpers')
 replace('    if ASCEND_IS_AIC {mm.End();}\n  }','''    if ASCEND_IS_AIC {mm.End();}
    profileCounters[3][kComputeSpan]=static_cast<int64_t>(GetSystemCycle()-profileStart);
    TraceMarker(kComputeSpan,true);
    ProfileFlush();
  }''',label='actor end+flush')
 replace('  Geometry g;','''  Geometry g;
  GlobalTensor<int64_t> profileGm;
  int64_t profileCounters[4][24];uint64_t profileStart;int32_t engineIndex=0;''',label='counter storage')
 for name,field in [('LoadTask','kTaskMetadata'),('LoadQueries','kQueryStage'),('Unpack','kUnpack'),('LoadPrecise','kPreciseLoad'),('PublishKv','kKvPublish'),('Softmax','kSoftmaxP'),('PublishRows','kFinalWriteback')]:scope(name,field)
 scope('LoadPacked' if kind=='simd' else 'LoadPackedInto','kPackedDma')
 scope('Matmul','rotation?kAicRotation:(transposeB?kAicQk:kAicPv)')
 for name in ['CubeTask','VectorTask']+(['HistoryCubeTask','HistoryVectorTask'] if kind=='pipeline' else []):scope(name,'kTaskSpan',True)
 if kind=='pipeline':
  for name,field in [('PipelineUnpack','kUnpack'),('PipelinePublishKv','kKvPublish'),('PipelineConsumePv','kPvAcc')]:scope(name,field)
 waits('CubeTask','kVectorReady',['kAicKvWait','kAicPWait','kAicAccWait','kRotationWait','kRotationWait'])
 waits('VectorTask','kCubeReady',['kAivQkWait','kAivPvWait','kRotationWait'])
 if kind=='pipeline':
  for flag,field in [('kKvReady','kAicKvWait'),('kPReady','kAicPWait'),('kNormReady','kRotationWait'),('kTaskDone','kRotationWait')]:waits('HistoryCubeTask',flag,[field])
  for flag,field in [('kScoreReady','kAivQkWait'),('kPvReady','kAivPvWait'),('kRotReady','kRotationWait')]:waits('HistoryVectorTask',flag,[field])
 for header,count in [('for(int64_t start=kvbegin;start<kvend;start+=kKvRows) {',2)]+([('for(int64_t start=kvbegin;start<kvend;start+=kKvRows,buffer^=1) {',2)] if kind=='pipeline' else []):
  replace(header,header+'\n      ProfileAdd(kKvUnits,1);\n      ProfileAdd(kKvRowsObserved,Min64(kKvRows,kvend-start));',count,'KV counts')
 def acc_scope(body):
  start=body.index('      // P is fully consumed by Cube before this point.')
  end=body.index('      CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);',start)
  return body[:start]+'      { ProfileScope profileAcc(*this,kPvAcc);\n'+body[start:end]+'      }\n'+body[end:]
 edit_method('VectorTask',acc_scope)
 for name in ['VectorTask']+(['HistoryVectorTask'] if kind=='pipeline' else []):
  def finalize_scope(body):
   token='    Fence<HardEvent::V_S>();\n    for(int32_t r=0;r<kLaneRows;++r) {'
   assert body.count(token)==1
   return body.replace(token,'    ProfileScope profileFinalize(*this,kAivFinalize);\n'+token)
  edit_method(name,finalize_scope)
 replace('GM_ADDR status, GM_ADDR workspace, \\\n    int64_t tokens','GM_ADDR status, GM_ADDR workspace, GM_ADDR profile, \\\n    int64_t tokens',label='kernel ABI')
 replace('tasks,output,lse,status,workspace,g);','tasks,output,lse,status,workspace,profile,g);',label='Init call')
 replace('void* status,void* workspace,int64_t tokens','void* status,void* workspace,void* profile,int64_t tokens',label='launcher ABI')
 replace('static_cast<uint8_t*>(workspace), \\\n      tokens,hq','static_cast<uint8_t*>(workspace),static_cast<uint8_t*>(profile), \\\n      tokens,hq',label='launcher argument')
 names=([('oscar_attention_cv_striped_decode_simd_kernel','oscar_decode_simd_profile_q4_kernel'),('oscar_attention_cv_striped_q1_simd_kernel','oscar_decode_simd_profile_q1_kernel'),('attention_cv_striped_decode_simd_launch','decode_simd_profile_q4_launch'),('attention_cv_striped_q1_simd_launch','decode_simd_profile_q1_launch')] if kind=='simd' else [('oscar_attention_history_pipeline_q4_kernel','oscar_history_pipeline_profile_q4_kernel'),('oscar_attention_history_pipeline_q1_kernel','oscar_history_pipeline_profile_q1_kernel'),('attention_history_pipeline_q4_launch','history_pipeline_profile_q4_launch'),('attention_history_pipeline_q1_launch','history_pipeline_profile_q1_launch')])
 for old,new in names:replace(old,new,2 if old.startswith('oscar_') else 1,'symbol '+old)
 if kind=='simd':
  replace('#include "oscar_common.h"','#include "../../kernels/oscar_common.h"',label='include')
  replace('#include "attention_striped_unpack_simd.h"','#include "../../kernels/attention_striped_unpack_simd.h"',label='include')
  replace('"../include/','"../../include/',2,'includes')
 header='''// GENERATED DIAGNOSTIC ONLY. Archive #140/#144/#148/#154/#155; startup D.4.
// Source math/control flow is hash-pinned and reversibly restored by generate.py.
// Scalar clock scopes add NO completion fence or PIPE_ALL. Queue/SDK spans can
// include existing waits and can end before async engines finish. Original
// cross-core waits remain; their separate counters measure that actor's waits.
// Same-actor task/finalize/writeback scopes nest: never add nested totals or
// actor maxima as wall. Profile GM owns disjoint cache lines; flush is excluded
// from field23. No production selection, NPU/graph/quality/performance claim.
'''
 # Reversing every counted edit must recover every original byte, including
 # all math calls, all original fences, flags, error paths and KV ordering.
 restored=text
 for old,new,count,label in reversed(edits):
  assert restored.count(new)==count,(kind,'inverse',label,restored.count(new),count)
  restored=restored.replace(new,old)
 assert restored==original,'profiling edit altered unaccounted math/control flow'
 for token in ('Fence<HardEvent::','PipeBarrier<','CrossCoreSetFlag<','SetHF32Mode('):
  assert masked(text).count(token)==masked(original).count(token),(kind,token,'new synchronization/math control')
 methods=['Matmul','LoadQueries','LoadPrecise','Unpack','PublishKv','Softmax','PublishRows','PublishEmpty','PublishStatus']
 method_hashes={name:sha(original[slice(*function(original,name))]) for name in methods}
 return header+text,{'source':path,'source_sha256':pin,'all_original_bytes_restored':True,'new_completion_fences':0,'math_method_body_sha256':method_hashes,'edits':[{'label':label,'count':count,'before_sha256':sha(old),'after_sha256':sha(new)} for old,new,count,label in edits]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
 manifest={'schema_version':2,'diagnostic_only':True,'abi':{'pointer_count':16,'profile_pointer_position_0based':15,'int64_attributes':17,'float_attributes':1,'dtype':'int64','shape':['cube_cores',3,4,24],'bytes_per_actor':768,'bytes_per_cube':2304,'engines':['AIC','AIV0','AIV1'],'sources':['history','window','current','total'],'fields':dict(enumerate(FIELDS)),'field_20_21_not_legacy_unpack_subitems':True,'all_slots_written':True},'variants':{}}
 for kind,(path,pin) in INPUTS.items():
  text,entry=generate(kind,path,pin);dest=OUT/(kind+'_profile.cpp')
  entry.update(output=str(dest.relative_to(ROOT)),output_sha256=sha(text));manifest['variants'][kind]=entry
  if args.check:assert dest.read_text()==text,('stale generated file',dest)
  else:dest.write_text(text)
 encoded=json.dumps(manifest,indent=2)+'\n'
 if args.check:assert (OUT/'manifest.json').read_text()==encoded,'stale manifest'
 else:(OUT/'manifest.json').write_text(encoded)
 print(json.dumps({'status':'checked' if args.check else 'generated','sources_pinned':2,'math_and_control_flow_restored':True,'new_completion_fences':0,'abi':manifest['abi']}))
if __name__=='__main__':main()
