"""Archive #13-22/#126/#140/#145/#148/#151/#154-156; startup D.4.

Isolated H32 C4 member pipeline. Every edit reverses to the previously paired
H32 source; production source/math/dispatch are untouched. Atomic immediate
readback is a separate prerequisite, never inferred from source generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_unpack32 import generate_c4
from csrc.experiments.decode_phase_profile.generate import function,masked


def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()


CUBE='''{
    const int64_t first=members[0].kvbegin,last=members[0].kvend;
    const int32_t rows=static_cast<int32_t>(members[0].qcount*(g.hq/g.hk));
    int32_t buffer=0;
    for(int64_t start=first;start<last;start+=kKvRows,buffer^=1) {
      CrossCoreWaitFlag(kClusterKvReady);
      // QK0,QK1,PV0,QK2,PV1,QK3,PV2,PV3. Reusing score slot i%2
      // follows the completed PV that consumed its prior P, never a V guess.
      for(int32_t member=0;member<2;++member) {
        SelectClusterSlots(buffer,member);
        Matmul(member*kQueryRows*D,kOffset,scoreOffset,rows,kKvRows,D);
        CrossCoreSetFlag<2,PIPE_FIX>(kClusterScore0+member);
      }
      for(int32_t member=0;member<4;++member) {
        const int32_t slot=member%2;
        CrossCoreWaitFlag(kClusterP0+slot);
        SelectClusterSlots(buffer,slot);
        MatmulAtomic(pOffset,vOffset,accStateOffset+member*kQueryRows*D,
            rows,D,kKvRows);
        if(member+2<4) {
          Matmul((member+2)*kQueryRows*D,kOffset,scoreOffset,rows,kKvRows,D);
          CrossCoreSetFlag<2,PIPE_FIX>(kClusterScore0+slot);
        }
      }
      // This orders every member's complete-K atomic before next-round
      // RestoreState. The separate atomic-v2 immediate-read gate is mandatory.
      CrossCoreSetFlag<2,PIPE_FIX>(kClusterRoundDone);
    }
    SelectClusterSlots(0,0);
    for(int32_t member=0;member<4;++member) {
      CrossCoreWaitFlag(kVectorReady);
      qOffset=member*kQueryRows*D;
      Matmul(qOffset,0,rotOffset,rows,D,D,true);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
    }
  }'''

LOOP='''    const int64_t first=members[0].kvbegin,last=members[0].kvend;
    UseTask(members[0]);error=0;SelectClusterSlots(0,0);
    LoadClusterKv(first);
    int32_t kvError=error,buffer=0;
    CrossCoreSetFlag<2,PIPE_MTE3>(kClusterKvReady);
    for(int64_t start=first;start<last;start+=kKvRows,buffer^=1) {
      for(int32_t member=0;member<4;++member) {
        UseTask(members[member]);qOffset=member*kQueryRows*D;
        error=memberErrors[member];
        if(kvError)error=kvError;
        SelectClusterSlots(buffer,member%2);
        CrossCoreWaitFlag(kClusterScore0+member%2);
        RestoreState(member);
        for(int32_t block=0;block<kRowBlocks;++block)
          Softmax(start,lane*kLaneRows+block*kHalfRows);
        memberErrors[member]=error;
        // FP32 alpha*acc and max/sum reach GM before PReady; Cube atomic
        // updates ONLY this member's acc. Other members restore disjoint state.
        SpillState(member);
        CrossCoreSetFlag<2,PIPE_MTE3>(kClusterP0+member%2);
      }
      if(start+kKvRows<last) {
        // P3 and all alpha states are published. Natural/dequant/packed UB
        // are dead; next KV writes only the buffer not read by current PV3.
        // Future metadata errors remain separate from all current members.
        UseTask(members[0]);error=0;SelectClusterSlots(buffer^1,0);
        LoadClusterKv(start+kKvRows);kvError=error;
        CrossCoreSetFlag<2,PIPE_MTE3>(kClusterKvReady);
      }
      CrossCoreWaitFlag(kClusterRoundDone);
    }
    // No final next-KV token; all two score/P tokens and RoundDone drained.
    // The unchanged final norm/Rv protocol may now reuse flags8/9.
    SelectClusterSlots(0,0);
'''

HELPERS='''  __aicore__ void SelectClusterSlots(int32_t buffer,int32_t slot) {
    kOffset=buffer?extraKvOffset:baseKOffset;
    vOffset=kOffset+kKvRows*D;
    scoreOffset=slot?extraScoreOffset:baseScoreOffset;pOffset=scoreOffset;
  }
  __aicore__ void LoadClusterKv(int64_t start) {
    // Preserve the tested H32 two-old-16-row lane domains and K/V order.
    for(int32_t subtile=0;subtile<kHistorySubtiles;++subtile) {
      const int64_t subStart=start+subtile*(2*kHistoryHalfKv);
      LoadPacked(subStart);Unpack(false);PublishHistoryKv(false,subtile);
      Unpack(true);PublishHistoryKv(true,subtile);
    }
  }
  __aicore__ void MatmulAtomic(int64_t a,int64_t b,int64_t c,
      int32_t m,int32_t n,int32_t k) {
    SetHF32Mode(false);
    mm.SetOrgShape(m,n,k);mm.SetSingleShape(m,n,k);
    mm.SetTensorA(work[a],false);mm.SetTensorB(work[b],false);
    // Same K128 accumulation order; atomic=1 applies only to final FP32 C.
    mm.template IterateAll<true>(work[c],static_cast<uint8_t>(1));
    mm.End();
  }
'''


def generate(candidate):
    original,origin=generate_c4.generate(True)
    if original!=(ROOT/'csrc/experiments/c4_unpack32/c4_candidate.cpp').read_text():
        raise ValueError('qualified H32 source drift')
    text=original;edits=[]
    def replace(old,new,count=1,label=''):
        nonlocal text
        if text.count(old)!=count:raise ValueError((label,text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count,label))
    def method(name,new):
        a,b=function(text,name);replace(text[a:b],new,label=name)
    if candidate:
        replace('const int64_t floatsPerCore=oscar_ascend::attention_cluster4_workspace_per_core(D)/4;',
            'const int64_t baseFloats=oscar_ascend::attention_cluster4_workspace_per_core(D)/4;\n    const int64_t floatsPerCore=baseFloats+2*kKvRows*D+kQueryRows*kKvRows;')
        replace('statStateOffset=accStateOffset+4*kQueryRows*D;',
            'statStateOffset=accStateOffset+4*kQueryRows*D;\n    baseKOffset=kOffset;baseScoreOffset=scoreOffset;\n    extraKvOffset=baseFloats;extraScoreOffset=baseFloats+2*kKvRows*D;')
        replace('int64_t accStateOffset,statStateOffset,counters[8];',
            'int64_t accStateOffset,statStateOffset,counters[8];\n  int64_t baseKOffset,baseScoreOffset,extraKvOffset,extraScoreOffset;')
        replace('constexpr uint16_t kVectorReady=8, kCubeReady=9;',
            'constexpr uint16_t kVectorReady=8, kCubeReady=9;\nconstexpr uint16_t kClusterKvReady=8,kClusterScore0=9,kClusterP0=11,kClusterRoundDone=13;')
        method('CubeCluster',CUBE)
        a,b=function(text,'VectorCluster');body=text[a:b]
        begin=body.index('    const int64_t first=')
        end=body.index('    for(int32_t member=0;member<4;++member) {\n      UseTask(members[member]);qOffset=member*kQueryRows*D;\n      error=memberErrors[member];RestoreState(member);',begin)
        method('VectorCluster',body[:begin]+LOOP+body[end:])
        replace('  __aicore__ void CubeTask() {',HELPERS+'  __aicore__ void CubeTask() {',label='bounded pipeline helpers')
    arm='candidate' if candidate else 'baseline'
    replace('c4_unpack32_candidate_launch','c4_member_pipeline_'+arm+'_launch')
    replace('oscar_c4_unpack32_candidate_kernel','oscar_c4_member_pipeline_'+arm+'_kernel')
    replace('#include "unpack32.h"','#include "../c4_unpack32/unpack32.h"')
    # function() returns braces/body, not the preceding declaration. Byte
    # reversibility alone cannot reject accidentally inserting a second one.
    validate_declarations(text)
    restored=text
    for old,new,count,label in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse drift '+label)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('H32 source restoration failed')
    unchanged=('Process','IsClusterMember','Matmul','LoadQueries','LoadPacked','Unpack',
        'PublishHistoryKv','LoadWindowRuns','LoadPrecise','Softmax','SpillState','RestoreState',
        'CubeTask','VectorTask','PublishRows','PublishStatus','TaskValid')
    for name in unchanged:
        if text[slice(*function(text,name))]!=original[slice(*function(original,name))]:
            raise ValueError('unrelated source changed '+name)
    if text[text.index('using Matrix='):text.index('struct Geometry {')]!=original[original.index('using Matrix='):original.index('struct Geometry {')]:
        raise ValueError('Cube type/tiling/config changed')
    header='''// ISOLATED C4 MEMBER PIPELINE; not registered or enabled in production.
// Archive #13-22/#126/#140/#145/#148/#151/#154-156; startup D.4 four questions:
// (1) fused INT2 history dequant/QK/softmax/PV. (2) D.4's full-history
// restore6499.8-6655.1ms is forbidden. (3) only two fixed KV256 buffers,
// two score/P slots, four original FP32 states; no history-sized tensor.
// Same Matmul/K128/FP32 and per-member KV order; tested H32 old-lane ownership.
// Atomic immediate-read v2 is a prerequisite, not established by this file.
// (4) >=1.5 long-history is a conditional hypothesis; no NPU/graph/E2E claim.
'''
    result=header+text
    return result,{'arm':arm,'symbol':'oscar_c4_member_pipeline_'+arm+'_kernel',
        'production_source_sha256':origin['source_sha256'],'H32_source_sha256':sha(original),
        'source_sha256':origin['source_sha256'],'sha256':sha(result),'original_H32_bytes_restored':True,
        'unchanged_methods':unchanged,'AIV_explicit_UB_bytes':157760,
        'GM_bytes_per_core':1839104+(655360 if candidate else 0),
        'source2':'production_prefill_suppressed_all_tasks_retained',
        'atomic_immediate_readback_gate':'required_separate_evidence',
        'CPU':'not_run','CANN':'not_run','NPU':'not_run','graph':'not_run'}


def validate_declarations(text):
    code=masked(text)
    if re.search(r'__aicore__\b[^{};]*\b__aicore__\b',code):
        raise ValueError('repeated __aicore__ declaration prefix')
    names=re.findall(r'__aicore__\s+(?:inline\s+)?(?:void|bool|int(?:32|64)_t)\s+(\w+)\s*\(',code)
    for name in names:function(text,name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();manifest={}
    for candidate in (False,True):
        text,entry=generate(candidate);path=HERE/('c4_'+entry['arm']+'.cpp');manifest[entry['arm']]=entry
        if a.check:
            if path.read_text()!=text:raise ValueError('generated source changed '+str(path))
        else:path.write_text(text)
    encoded=json.dumps(manifest,indent=2)+'\n';path=HERE/'manifest.json'
    if a.check:
        if path.read_text()!=encoded:raise ValueError('manifest changed')
    else:path.write_text(encoded)
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False,'execution':'not_run'}))


if __name__=='__main__':main()
