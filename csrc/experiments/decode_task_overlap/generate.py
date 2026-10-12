"""Current-bundle two-task overlap proof; no production/VM changes.

Archive #126/#140/#145/#148/#154-156 and full startup D.4. Same owner,
all sources/tasks, two persistent UB states, two bounded GM contexts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_phase_profile.generate import function
from csrc.experiments.c4_member_pipeline.generate import validate_declarations
SOURCE='csrc/kernels/attention_cv_decode_bundle.cpp'
SOURCE_SHA='d8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401'

STRUCTS='''struct OverlapTask {
  int64_t id,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin;
  int32_t taskError;
};
struct OverlapContext {
  OverlapTask task;
  int32_t error,live[2],keyError[2],valueError[2];
};
'''

PROCESS='''{
    if ASCEND_IS_AIC {
      DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>(taskGm);
    }
    const oscar_ascend_schedule::CvTaskSchedule schedule{g.tokens,OwnerTile,g.hk*3*g.splits};
    SelectWorkspace(0);
    // First pass preserves every original owner/status for padding, invalid,
    // precise and short/unpairable history tasks. No pending state is live.
    for(int64_t workId=coreIndex;workId<schedule.WorkItems();workId+=GetBlockNum()) {
      const int64_t begin=schedule.TokenBegin(workId);
      for(int64_t token=begin;token<Min64(begin+OwnerTile,g.tokens);++token) {
        const int64_t id=schedule.TaskId(workId,token);LoadTask(id);
        if(qcount==0) {if ASCEND_IS_AIV {PublishStatus(id,0);} continue;}
        if(qcount<0 || taskError || !TaskValid()) {
          if ASCEND_IS_AIV {PublishEmpty(id,taskError?taskError:(!TaskValid()?1:0));}
          continue;
        }
        if(PairEligible())continue;
        RunSerialTask(id);
      }
    }
    bool pending=false;OverlapTask first{};
    // Second pass only defers eligible long history. Every task remains on
    // its original core; no numerical state starts until a pair is chosen.
    for(int64_t workId=coreIndex;workId<schedule.WorkItems();workId+=GetBlockNum()) {
      const int64_t begin=schedule.TokenBegin(workId);
      for(int64_t token=begin;token<Min64(begin+OwnerTile,g.tokens);++token) {
        const int64_t id=schedule.TaskId(workId,token);LoadTask(id);
        if(!PairEligible())continue;
        const OverlapTask current=CaptureTask(id);
        if(!pending) {first=current;pending=true;continue;}
        const int64_t aTiles=(first.kvend-first.kvbegin+kKvRows-1)/kKvRows;
        const int64_t bTiles=(current.kvend-current.kvbegin+kKvRows-1)/kKvRows;
        if(first.request!=current.request && aTiles==bTiles) {
          RunPair(first,current);pending=false;
        } else {
          ApplyTask(first);RunSerialTask(first.id);first=current;
        }
      }
    }
    if(pending) {ApplyTask(first);RunSerialTask(first.id);}
    if ASCEND_IS_AIC {mm.End();}
  }'''

HELPERS='''  __aicore__ LocalTensor<float> ActiveAcc() {
    return activeContext==0?accBuf.Get<float>():secondAccBuf.Get<float>();
  }
  __aicore__ LocalTensor<float> ActiveStats() {
    return activeContext==0?statsBuf.Get<float>():secondStatsBuf.Get<float>();
  }
  __aicore__ void SelectWorkspace(int32_t index) {
    activeContext=index;work.SetGlobalBuffer(workBase+index*contextFloats);
  }
  __aicore__ uint16_t PairFlag(int32_t index,uint16_t flag) {
    // Two disjoint mode2 banks. Original8..14 and second0..6; no +16 lane alias.
    return static_cast<uint16_t>(flag-(index?8:0));
  }
  __aicore__ OverlapTask CaptureTask(int64_t id) {
    return {id,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin,taskError};
  }
  __aicore__ void ApplyTask(const OverlapTask& t) {
    qbegin=t.qbegin;qcount=t.qcount;kvhead=t.kvhead;kvbegin=t.kvbegin;kvend=t.kvend;
    request=t.request;split=t.split;kind=t.kind;context=t.context;requestBegin=t.requestBegin;taskError=t.taskError;
  }
  __aicore__ bool PairEligible() {
    return g.requests>=2 && qcount>0 && !taskError && TaskValid() && kind==0 && kvend-kvbegin>kKvRows;
  }
  __aicore__ void Activate(int32_t index) {
    SelectWorkspace(index);ApplyTask(contexts[index].task);error=contexts[index].error;
    for(int32_t s=0;s<kKvSubtiles;++s) {
      pendingLive[s]=contexts[index].live[s];pendingKeyError[s]=contexts[index].keyError[s];
      pendingValueError[s]=contexts[index].valueError[s];
    }
  }
  __aicore__ void SaveContext(int32_t index) {
    contexts[index].error=error;
    for(int32_t s=0;s<kKvSubtiles;++s) {
      contexts[index].live[s]=pendingLive[s];contexts[index].keyError[s]=pendingKeyError[s];
      contexts[index].valueError[s]=pendingValueError[s];
    }
  }
  __aicore__ void RunSerialTask(int64_t id) {
    SelectWorkspace(0);
    if(kind==0 && kvbegin<kvend) {
      if ASCEND_IS_AIC {HistoryCubeTask();} else {HistoryVectorTask(id);}
    } else {
      if ASCEND_IS_AIC {CubeTask();} else {VectorTask(id);}
    }
  }
  __aicore__ void RunPair(const OverlapTask& a,const OverlapTask& b) {
    contexts[0].task=a;contexts[1].task=b;
    for(int32_t c=0;c<2;++c) {
      contexts[c].error=0;
      for(int32_t s=0;s<kKvSubtiles;++s) {
        contexts[c].live[s]=0;contexts[c].keyError[s]=0;contexts[c].valueError[s]=0;
      }
    }
    if ASCEND_IS_AIC {HistoryPairCube();} else {HistoryPairVector();}
    SelectWorkspace(0);
  }
  __aicore__ void HistoryPairCube() {
    const int64_t rounds=(contexts[0].task.kvend-contexts[0].task.kvbegin+kKvRows-1)/kKvRows;
    for(int64_t round=0;round<rounds;++round) {
      const int32_t buffer=static_cast<int32_t>(round%2);
      for(int32_t c=0;c<2;++c) {
        Activate(c);CrossCoreWaitFlag(PairFlag(c,kKvReady));
        Matmul(qOffset,kOffset+buffer*(2*kKvRows*D),scoreOffset,
            static_cast<int32_t>(qcount*(g.hq/g.hk)),kKvRows,D);
        CrossCoreSetFlag<2,PIPE_FIX>(PairFlag(c,kScoreReady));
      }
      for(int32_t c=0;c<2;++c) {
        Activate(c);CrossCoreWaitFlag(PairFlag(c,kPReady));
        Matmul(pOffset,vOffset+buffer*(2*kKvRows*D),pvOffset,
            static_cast<int32_t>(qcount*(g.hq/g.hk)),D,kKvRows,false,false);
        CrossCoreSetFlag<2,PIPE_FIX>(PairFlag(c,kPvReady));
      }
    }
    for(int32_t c=0;c<2;++c) {
      Activate(c);CrossCoreWaitFlag(PairFlag(c,kNormReady));
      Matmul(qOffset,0,rotOffset,static_cast<int32_t>(qcount*(g.hq/g.hk)),D,D,true);
      CrossCoreSetFlag<2,PIPE_FIX>(PairFlag(c,kRotReady));
      CrossCoreWaitFlag(PairFlag(c,kTaskDone));
    }
  }
  __aicore__ void HistoryPairVector() {
    for(int32_t c=0;c<2;++c) {
      Activate(c);auto acc=ActiveAcc();auto stats=ActiveStats();
      Duplicate(acc,0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
      Duplicate(stats,kEmptyMax,kLaneRows);
      Duplicate(stats[kLaneRows],0.0F,kLaneRows);Fence<HardEvent::V_S>();
      LoadQueries();
      for(int32_t sub=0;sub<kKvSubtiles;++sub) {
        LoadPacked(kvbegin+sub*(2*kHalfKv));
        Unpack(false);PipelinePublishKv(false,sub,0);
        Unpack(true);PipelinePublishKv(true,sub,0);
      }
      SaveContext(c);CrossCoreSetFlag<2,PIPE_MTE3>(PairFlag(c,kKvReady));
    }
    const int64_t rounds=(contexts[0].task.kvend-contexts[0].task.kvbegin+kKvRows-1)/kKvRows;
    for(int64_t round=0;round<rounds;++round) {
      const bool hasNext=round+1<rounds;const int32_t next=static_cast<int32_t>((round+1)%2);
      // Shared packed UB belongs to A until its next V is expanded. SA/SB
      // use only score/natural/dequant; they never overwrite packed buffers.
      Activate(0);
      if(hasNext)PrefetchKeys(kvbegin+(round+1)*kKvRows,next);
      CrossCoreWaitFlag(PairFlag(0,kScoreReady));
      for(int32_t block=0;block<kRowBlocks;++block)
        Softmax(kvbegin+round*kKvRows,lane*kLaneRows+block*kHalfRows);
      SaveContext(0);CrossCoreSetFlag<2,PIPE_MTE3>(PairFlag(0,kPReady));
      Activate(1);CrossCoreWaitFlag(PairFlag(1,kScoreReady));
      for(int32_t block=0;block<kRowBlocks;++block)
        Softmax(kvbegin+round*kKvRows,lane*kLaneRows+block*kHalfRows);
      SaveContext(1);CrossCoreSetFlag<2,PIPE_MTE3>(PairFlag(1,kPReady));
      if(hasNext) {
        Activate(0);PrefetchValues(next);SaveContext(0);
        CrossCoreSetFlag<2,PIPE_MTE3>(PairFlag(0,kKvReady));
        // Only now may B overwrite shared packed UB. CA below uses dequant
        // scratch, but B's packed bytes and saved live/errors remain intact.
        Activate(1);PrefetchKeys(kvbegin+(round+1)*kKvRows,next);SaveContext(1);
      }
      Activate(0);CrossCoreWaitFlag(PairFlag(0,kPvReady));PipelineConsumePv();
      if(hasNext)CommitPrefetchErrors();SaveContext(0);
      Activate(1);
      if(hasNext) {
        PrefetchValues(next);SaveContext(1);
        CrossCoreSetFlag<2,PIPE_MTE3>(PairFlag(1,kKvReady));
      }
      CrossCoreWaitFlag(PairFlag(1,kPvReady));PipelineConsumePv();
      if(hasNext)CommitPrefetchErrors();SaveContext(1);
    }
    FinishPairContext(0);FinishPairContext(1);
  }
'''

HEADER='''// INDEPENDENT TWO-TASK DECODE OVERLAP; no production registration.
// Archive #126/#140/#145/#148/#154-156 and full startup D.4 four questions:
// 1. Fused INT2 decode dequant/QK/softmax/PV; 2. no D.4 full-history6.5s
// restore, no BF16 relaxation; 3. two original-owner tasks retain per-task
// FP32/KV256 order, two persistent UB acc/stats and fixed two GM contexts.
// Packed scratch is shared only after its current owner's V expansion.
// 4. Full same-library oracle/paired-byte/completion/timing gates required;
// equal-tile pairs only, no odd-triad or full K32/2-3x performance claim.
'''


def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()


def generate(candidate):
    original=(ROOT/SOURCE).read_text()
    if sha(original)!=SOURCE_SHA:raise ValueError('current decode-bundle mother drift')
    text=original;edits=[]
    def replace(old,new,count=1,label=''):
        nonlocal text
        if text.count(old)!=count:raise ValueError((label,text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count,label))
    if candidate:
        replace('template<int32_t D,int32_t OwnerTile> class AttentionCv {',STRUCTS+'\ntemplate<int32_t D,int32_t OwnerTile> class AttentionCv {',label='bounded scalar contexts')
        replace('work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+core*floatsPerCore);',
                'contextFloats=floatsPerCore;workBase=reinterpret_cast<__gm__ float*>(workspace)+core*(2*floatsPerCore);\n    work.SetGlobalBuffer(workBase);',label='two fixed GM contexts')
        replace('pipe.InitBuffer(addressBuf,128);',
                'pipe.InitBuffer(addressBuf,128);\n      pipe.InitBuffer(secondAccBuf,kLaneRows*D*4);\n      pipe.InitBuffer(secondStatsBuf,(2*kLaneRows+2*kHalfRows)*4);',label='independent original-size UB state')
        for old,new in [('accBuf.Get<float>()','ActiveAcc()'),('statsBuf.Get<float>()','ActiveStats()')]:
            replace(old,new,text.count(old),label='select exact-size UB view')
        replace('int32_t lane=0,error=0,taskError=0,liveKvRows=0;',
                'int32_t lane=0,error=0,taskError=0,liveKvRows=0;\n  int32_t activeContext=0;int64_t contextFloats=0;__gm__ float* workBase=nullptr;\n  TBuf<TPosition::VECCALC> secondAccBuf,secondStatsBuf;\n  OverlapContext contexts[2];',label='fixed context state')
        a,b=function(text,'HistoryVectorTask');body=text[a:b]
        tail=body[body.index('    Fence<HardEvent::V_S>();\n    for(int32_t r=0;r<kLaneRows;++r)'):]
        for flag in ('kNormReady','kRotReady','kTaskDone'):tail=tail.replace(flag,'PairFlag(index,'+flag+')')
        finish='  __aicore__ void FinishPairContext(int32_t index) {\n    Activate(index);\n    const int64_t id=contexts[index].task.id;\n    auto acc=ActiveAcc();auto stats=ActiveStats();\n'+tail
        a,b=function(text,'Process');replace(text[a:b],PROCESS,label='all-task two-pass owner traversal')
        replace('  __aicore__ void HistoryCubeTask() {',HELPERS+finish+'\n  __aicore__ void HistoryCubeTask() {',label='two-task protocol')
    arm='candidate' if candidate else 'baseline'
    for old,new in [
        ('oscar_attention_cv_decode_bundle_q1_kernel','oscar_task_overlap_'+arm+'_q1_kernel'),
        ('oscar_attention_cv_decode_bundle_kernel','oscar_task_overlap_'+arm+'_q4_kernel'),
        ('attention_cv_decode_bundle_q1_launch','task_overlap_'+arm+'_q1_launch'),
        ('attention_cv_decode_bundle_launch','task_overlap_'+arm+'_q4_launch')]:
        replace(old,new,2 if old.startswith('oscar_') else 1,label='isolated symbol')
    for header in ('oscar_common.h','attention_striped_unpack_padded.h','attention_range_mask.h'):
        replace('#include "'+header+'"','#include "../../kernels/'+header+'"',label='relative include')
    replace('#include "../include/','#include "../../include/',2,label='relative include')
    validate_declarations(text)
    restored=text
    for old,new,count,label in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse drift '+label)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('mother byte restoration failed')
    unchanged=('Matmul','TaskValid','LoadTask','LoadQueries','LoadPackedInto','LoadPacked','Unpack',
               'LoadWindowRuns','LoadPrecise','Softmax','PipelineUnpack','PipelinePublishKv',
               'PrefetchKeys','PrefetchValues','CommitPrefetchErrors','PipelineConsumePv',
               'HistoryCubeTask','HistoryVectorTask','CubeTask','VectorTask','PublishRows','PublishStatus','PublishEmpty')
    for name in unchanged:
        body=text[slice(*function(text,name))].replace('ActiveAcc()','accBuf.Get<float>()').replace('ActiveStats()','statsBuf.Get<float>()')
        if body!=original[slice(*function(original,name))]:raise ValueError('math method changed '+name)
    result=HEADER+text
    return result,{'arm':arm,'source':SOURCE,'source_sha256':SOURCE_SHA,'sha256':sha(result),
        'symbols':{'q4':'oscar_task_overlap_'+arm+'_q4_kernel','q1':'oscar_task_overlap_'+arm+'_q1_kernel'},
        'mother_bytes_restored':True,'math_methods_retained':unchanged,
        'workspace_bytes_per_core':2293760 if candidate else 1146880,
        'UB_bytes_per_AIV':175296 if candidate else 158656,
        'pair_domain':'valid source0 different-request same-original-owner >=2KV and equal tile count' if candidate else 'original bundle',
        'unpaired':'original complete OSCAR task math; all source/status paths retained',
        'flags':'mode2 8..14 and 0..6; final all tokens drained',
        'CPU':'not_run','CANN':'not_run','CAModel':'not_run','NPU':'not_run','graph':'not_run'}


def probe():
    path=ROOT/'csrc/experiments/window_batch_decode/probe.cpp';original=path.read_text()
    edits=[('oscar_attention_cv_striped_decode_simd_kernel','oscar_task_overlap_baseline_q4_kernel'),
           ('oscar_attention_cv_striped_q1_simd_kernel','oscar_task_overlap_baseline_q1_kernel'),
           ('oscar_ascend::attention_workspace_per_core(d)','oscar_ascend::attention_decode_bundle_workspace_per_core(d)'),
           ('current_production_simd','current_production_decode_bundle')]
    text=original
    for old,new in edits:
        if old not in text:raise ValueError('shared probe changed')
        text=text.replace(old,new)
    restored=text
    for old,new in reversed(edits):restored=restored.replace(new,old)
    if restored!=original:raise ValueError('shared probe restoration failed')
    return text,{'source':str(path.relative_to(ROOT)),'source_sha256':sha(original),'sha256':sha(text),'restored':True}


def build_binding():
    return {'schema':1,'source_sha256':SOURCE_SHA,'sources':{
        'attention_'+a+'.cpp':generate(a=='candidate')[1]['sha256'] for a in ('baseline','candidate')},
        'generator_sha256':sha(Path(__file__).read_bytes()),'library_sha':'recorded by actual build/run receipt'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--build-binding',type=Path);args=p.parse_args();manifest={}
    files={}
    for candidate in (False,True):
        body,item=generate(candidate);files['attention_'+item['arm']+'.cpp']=body;manifest[item['arm']]=item
    body,item=probe();files['probe.cpp']=body;manifest['probe']=item
    files['manifest.json']=json.dumps(manifest,indent=2)+'\n'
    for name,text in files.items():
        path=HERE/name
        if args.check:
            if path.read_text()!=text:raise ValueError('stale generated '+name)
        else:path.write_text(text)
    if args.build_binding:
        if not args.check:raise ValueError('binding requires check')
        args.build_binding.write_text(json.dumps(build_binding(),indent=2)+'\n')
    print(json.dumps({'status':'checked' if args.check else 'generated','production_changed':False,'execution':'not_run'}))


if __name__=='__main__':main()
