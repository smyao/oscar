"""Archive #126/#129/#145/#148/#154-157 and full D.4; C16 single-B1 reuse.

Independent candidate over the C16 pipeline. Sixteen score/P slots, four
acknowledged flag slots, one B1, unchanged per-member floating-point order.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_phase_profile.generate import function
from csrc.experiments.c4_l1_reuse_20261013.generate import HELPERS,MATMUL,ATOMIC,official_bindings
from csrc.experiments.c16_member_pipeline import generate as pipeline_generator

SOURCE=ROOT/'csrc/kernels/attention_cv_member_pipeline_cluster16.cpp'
HEADER='''// GENERATED C16 SINGLE-B1 RESIDENCY. Archive #126/#129/#145/#148/#154-157.
// D.4 four questions; complete original log reread:
// 1. Fused history dequant/FIA repeated FP32 B GM->L1 loading.
// 2. Full-history dequant6499.8-6655.1ms, host725ms, store209-215ms forbidden.
// 3. Fixed256KiB B1: one K load for16QK, one V load for16PV. Sixteen
//    fixed score/P slots retain every member's original FP32/K128/atomic
//    sequence. Four score flags9..12 and P flags0..3 are acknowledged before
//    reuse: before QK(i+4), wait P(i); drain last4 P before loading V.
//    MTE1->MTE2 protects B1 overwrite; MTE2->MTE1 publishes B1. Original
//    error/softmax/Spill/Restore/INT2 publisher untouched. No history tensor.
// 4. B1 logical loading falls8MiB->512KiB per matureKV, not an HBM/speed
//    claim. GM7487488B/core versus pipeline5652480/stable4997120;20cores
//    add47.5MiB/rank versusstable. Full same-library oracle/bitwise/task/
//    completion and>=1.15 over pipeline are mandatory, not model acceptance.
'''

CUBE='''{
    const int64_t first=members[0].kvbegin,last=members[0].kvend;
    const int32_t rows=static_cast<int32_t>(members[0].qcount*(g.hq/g.hk));
    int32_t buffer=0;
    for(int64_t start=first;start<last;start+=kKvRows,buffer^=1) {
      CrossCoreWaitFlag(kClusterKvReady);
      SelectClusterSlots(buffer,0);LoadResidentB(kOffset);
      for(int32_t member=0;member<16;++member) {
        // P(i-4) acknowledges that AIV already consumed Score(i-4).
        // Only then may the same 4-bit Score/P flag pair be reused.
        if(member>=4)CrossCoreWaitFlag(kClusterP0+member%4);
        SelectClusterSlots(buffer,member);
        MatmulResidentB(member*kQueryRows*D,scoreOffset,rows,kKvRows,D);
        CrossCoreSetFlag<2,PIPE_FIX>(kClusterScore0+member%4);
      }
      for(int32_t slot=0;slot<4;++slot)CrossCoreWaitFlag(kClusterP0+slot);
      // All sixteen P/alpha states are now published. K's last MTE1 read
      // is protected by the explicit B1 ownership fence inside LoadResidentB.
      SelectClusterSlots(buffer,0);LoadResidentB(vOffset);
      for(int32_t member=0;member<16;++member) {
        SelectClusterSlots(buffer,member);
        MatmulResidentB(pOffset,accStateOffset+member*kQueryRows*D,
            rows,D,kKvRows,false,static_cast<uint8_t>(1));
      }
      CrossCoreSetFlag<2,PIPE_FIX>(kClusterRoundDone);
    }
    SelectClusterSlots(0,0);
    for(int32_t member=0;member<16;++member) {
      CrossCoreWaitFlag(kVectorReady);
      qOffset=member*kQueryRows*D;
      Matmul(qOffset,0,rotOffset,rows,D,D,true);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
    }
  }'''


def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()


def generate_arm(arm):
    if arm not in ('stable','baseline','candidate'):raise ValueError('explicit three-arm identity required')
    if arm!='stable':return generate(arm=='candidate')
    source,entry=pipeline_generator.generate(False)
    source=source.replace('c16_member_pipeline_baseline_launch','c16_l1_reuse_stable_launch')
    source=source.replace('oscar_c16_member_pipeline_baseline_kernel','oscar_c16_l1_reuse_stable_kernel')
    return source,{**entry,'arm':'stable','symbol':'oscar_c16_l1_reuse_stable_kernel',
        'generated_sha256':sha(source),'workspace_per_core':4997120,
        'denominator':'frozen stable C16, not pipeline'}


def generate(candidate):
    original=SOURCE.read_text()
    pin=json.loads((HERE/'pins.json').read_text())['pipeline_c16']
    if sha(original)!=pin:raise ValueError('pipeline C16 mother drift')
    text=original;edits=[]
    def edit(old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError(('source edit',old[:80],text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count))
    def body(name,new):edit(text[slice(*function(text,name))],new)
    if candidate:
        edit('using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;',
             'using TransposedMatrix=MatmulType<TPosition::B1,CubeFormat::NZ,float,true>;')
        anchor='constexpr auto kCubeConfig=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(CvConfig());'
        edit(anchor,anchor+'''
static_assert(kCubeConfig.baseM==64 && kCubeConfig.baseN==64 && kCubeConfig.baseK==128,
    "original FP32 M/N/K tiling changed");
static_assert(kCubeConfig.shareL1Size>0 && kCubeConfig.shareL1Size+256*1024<=512*1024,
    "single external B1 plus SDK A/reserve exceeds physical L1");
''')
        edit('TPipe pipe;CubeMatmul mm;', 'TPipe pipe;CubeMatmul mm;TBuf<TPosition::B1> residentB;')
        edit('mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);',
             'pipe.InitBuffer(residentB,256*1024);\n      mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);')
        edit('baseFloats+2*kKvRows*D+kQueryRows*kKvRows;',
             'baseFloats+2*kKvRows*D+15*kQueryRows*kKvRows;')
        edit('kClusterKvReady=8,kClusterScore0=9,kClusterP0=11,kClusterRoundDone=13;',
             'kClusterKvReady=8,kClusterScore0=9,kClusterP0=0,kClusterRoundDone=13;')
        edit('  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,',
             HELPERS+'  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,')
        body('Matmul',MATMUL);body('MatmulAtomic',ATOMIC)
        edit('scoreOffset=slot?extraScoreOffset:baseScoreOffset;pOffset=scoreOffset;',
             'scoreOffset=slot?extraScoreOffset+(slot-1)*kQueryRows*kKvRows:baseScoreOffset;pOffset=scoreOffset;')
        edit('''      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
      Matmul(pOffset,vOffset,pvOffset,rows,D,kKvRows,false,false);''',
             '''      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      LoadResidentB(vOffset);
      CrossCoreWaitFlag(kVectorReady);
      MatmulResidentB(pOffset,pvOffset,rows,D,kKvRows,false);''')
        body('CubeCluster',CUBE)
        old=text[slice(*function(text,'VectorCluster'))];new=old
        for before,after in [('SelectClusterSlots(buffer,member%2);','SelectClusterSlots(buffer,member);'),
            ('kClusterScore0+member%2','kClusterScore0+member%4'),
            ('kClusterP0+member%2','kClusterP0+member%4')]:
            if new.count(before)!=1:raise ValueError('vector protocol edit drift')
            new=new.replace(before,after)
        body('VectorCluster',new)
    arm='candidate' if candidate else 'baseline'
    edit('attention_cv_member_pipeline_cluster16_launch','c16_l1_reuse_'+arm+'_launch')
    edit('oscar_attention_cv_member_pipeline_cluster16_kernel','oscar_c16_l1_reuse_'+arm+'_kernel')
    for header in ('attention_fast_unpack.h','attention_striped_unpack.h','attention_striped_unpack32.h',
                   'attention_range_mask.h','oscar_common.h'):
        edit('#include "'+header+'"','#include "../../kernels/'+header+'"',2 if header=='attention_fast_unpack.h' else 1)
    edit('"../include/','"../../include/',text.count('"../include/'))
    restored=text
    for old,new,count in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse drift '+new[:80])
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('mother not restored')
    unchanged={}
    for name in ('Process','IsClusterMember','LoadQueries','LoadPacked','Unpack','PublishHistoryKv',
        'LoadWindowRuns','LoadPrecise','PublishKv','Softmax','VectorTask','SpillState','RestoreState',
        'LoadClusterKv','PublishRows','PublishEmpty','PublishStatus','TaskValid'):
        before=original[slice(*function(original,name))];after=text[slice(*function(text,name))]
        if before!=after:raise ValueError('unchanged producer/math/domain drift '+name)
        unchanged[name]=sha(before)
    text=(HEADER if candidate else '// C16 pipeline denominator, archive #126/#145/#154 and D.4.\n')+text
    return text,{'arm':arm,'symbol':'oscar_c16_l1_reuse_'+arm+'_kernel','source_sha256':pin,'H32_source_sha256':pin,
        'generated_sha256':sha(text),'mother_bytes_restored':True,'unchanged_methods':unchanged,
        'workspace_per_core':7487488 if candidate else 5652480,'external_B1':262144 if candidate else 0,
        'explicit_UB':157760,'validation':'not_executed'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    result={'official_sdk':official_bindings(),'arms':{}}
    for arm in ('stable','baseline','candidate'):
        text,entry=generate_arm(arm);path=HERE/('c16_'+entry['arm']+'.cpp')
        if a.check:
            if path.read_text()!=text:raise ValueError('generated source drift')
        else:path.write_text(text)
        result['arms'][entry['arm']]=entry
    text=json.dumps(result,indent=2)+'\n'
    if a.check:
        if (HERE/'manifest.json').read_text()!=text:raise ValueError('manifest drift')
    else:(HERE/'manifest.json').write_text(text)
    print(json.dumps({'status':'checked' if a.check else 'generated','execution':'not_run'}))
