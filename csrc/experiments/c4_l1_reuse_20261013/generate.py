"""Archive #126/#129/#140/#145/#148/#154-157 and startup D.4.

Current qualified H32/member-pipeline mother only. Single external B1 with
four-member K/V residency and ordinary-source V prefetch; no production edit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from csrc.experiments.decode_phase_profile.generate import function

SOURCE = 'csrc/kernels/attention_cv_member_pipeline_cluster4.cpp'
PIN = '0795a07b1db0ba3d12e7ae762929b752d3bfb6d7475a925cf9e25e62ddeeb4bc'
HEADER = '''// GENERATED ISOLATED C4 SINGLE-B1 REUSE. Archive #126/#129/#140/#145/#148/#154-157.
// D.4 four questions, original logs reread:
// 1. Fused INT2 history dequant/FIA, specifically repeated FP32 B GM->L1 load.
// 2. Old full-history restore6499.8-6655.1ms/store209-215ms are forbidden.
// 3. One fixed256KiB B1; K for four QK then V for four PV; four score/P slots.
//    AIV INT2 publisher/FP32/softmax/error order and each member KV order stay.
//    MTE1->MTE2 prevents overwrite of live B1; MTE2->MTE1 publishes new B1.
//    All C1/precise tasks remain; their V load moves into existing P wait.
//    No whole-history restore, no extra Matmul instance, no GDN/FE0 changes.
// 4. Same-library full oracle/bitwise/60actors and>=1.15 complete path needed.
//    Traffic8->2 is NOT a speed claim; Vector softmax/extra score traffic charged.
// ABI workspace2756608B/core versus mother2494464; UB157760 unchanged.
'''

HELPERS = '''  __aicore__ void LoadResidentB(int64_t b,bool rotation=false) {
    // AIC-only direct MatmulImpl: no KFC/SyncAll cross-core flag bank.
    // End resets SDK iteration state, but cannot own this external B1.
    Fence<HardEvent::MTE1_MTE2>();
    auto local=residentB.Get<float>();
    Nd2NzParams copy;
    copy.ndNum=1;copy.nValue=D;copy.dValue=D;
    copy.srcNdMatrixStride=0;copy.srcDValue=D;
    copy.dstNzC0Stride=D;copy.dstNzNStride=1;copy.dstNzMatrixStride=0;
    if(rotation)DataCopy(local,rv,copy);else DataCopy(local,work[b],copy);
    Fence<HardEvent::MTE2_MTE1>();
  }
  __aicore__ void MatmulResidentB(int64_t a,int64_t c,int32_t m,int32_t n,int32_t k,
      bool transposeB=true,uint8_t atomic=0) {
    SetHF32Mode(false);
    // Normal scheduler external-L1 InitSplitBParams uses SingleCoreN/K.
    // This D256 C4 family always has N=K=256, including bounded precise tiles.
    mm.SetOrgShape(m,n,k);mm.SetSingleShape(m,n,k);
    mm.SetTensorA(work[a],false);
    mm.SetTensorB(residentB.Get<float>(),transposeB);
    mm.template IterateAll<true>(work[c],atomic);
    mm.End();
  }
'''

MATMUL = '''{
    LoadResidentB(b,rotation);
    MatmulResidentB(a,c,m,n,k,rotation?true:transposeB);
  }'''
ATOMIC = '''{
    LoadResidentB(b);
    MatmulResidentB(a,c,m,n,k,false,static_cast<uint8_t>(1));
  }'''
CLUSTER = '''{
    const int64_t first=members[0].kvbegin,last=members[0].kvend;
    const int32_t rows=static_cast<int32_t>(members[0].qcount*(g.hq/g.hk));
    int32_t buffer=0;
    for(int64_t start=first;start<last;start+=kKvRows,buffer^=1) {
      CrossCoreWaitFlag(kClusterKvReady);
      SelectClusterSlots(buffer,0);LoadResidentB(kOffset);
      for(int32_t member=0;member<4;++member) {
        SelectClusterSlots(buffer,member);
        MatmulResidentB(member*kQueryRows*D,scoreOffset,rows,kKvRows,D);
        CrossCoreSetFlag<2,PIPE_FIX>(kClusterScore0+member);
      }
      // All K's MTE1 reads precede this sole overwrite. V was already
      // published by the same KVReady as K; each P/state has its own flag.
      SelectClusterSlots(buffer,0);LoadResidentB(vOffset);
      for(int32_t member=0;member<4;++member) {
        CrossCoreWaitFlag(kClusterP0+member);
        SelectClusterSlots(buffer,member);
        MatmulResidentB(pOffset,accStateOffset+member*kQueryRows*D,
            rows,D,kKvRows,false,static_cast<uint8_t>(1));
      }
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


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def official_bindings():
    result = {}
    for folder in ('cann_sdk_l1_interface_20261013', 'cann_sdk_crosscore_flags_20261011'):
        root = ROOT / 'artifacts' / folder
        source = json.loads((root / 'provenance.json').read_text())
        for row in source['files']:
            relative = row.get('relative', row.get('path'))
            path = root / relative
            if sha(path.read_bytes()) != row['sha256']: raise ValueError('official SDK snapshot drift ' + str(path))
            result[str(path.relative_to(ROOT))] = row['sha256']
    return result


def generate(candidate):
    original = (ROOT / SOURCE).read_text()
    if sha(original) != PIN: raise ValueError('current qualified member-pipeline mother changed')
    text = original; edits = []
    def edit(old, new, count=1):
        nonlocal text
        if text.count(old) != count: raise ValueError(('source anchor', old[:120], text.count(old), count))
        text = text.replace(old, new); edits.append((old, new, count))
    def body(name, replacement):
        left, right = function(text, name); edit(text[left:right], replacement)
    if candidate:
        edit('using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;',
             'using TransposedMatrix=MatmulType<TPosition::B1,CubeFormat::NZ,float,true>;')
        # Keep GetNormalConfig/base64/64/128; this is not an MDL composition.
        anchor = 'constexpr auto kCubeConfig=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(CvConfig());'
        edit(anchor, anchor + '''
static_assert(kCubeConfig.baseM==64 && kCubeConfig.baseN==64 && kCubeConfig.baseK==128,
    "original FP32 M/N/K tiling changed");
static_assert(kCubeConfig.shareL1Size>0 && kCubeConfig.shareL1Size+256*1024<=512*1024,
    "Normal SDK A/reserve plus the only external B1 exceeds physical L1");
''')
        edit('  TPipe pipe;CubeMatmul mm;', '  TPipe pipe;CubeMatmul mm;TBuf<TPosition::B1> residentB;')
        edit('mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);',
             'pipe.InitBuffer(residentB,256*1024);\n      mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);')
        edit('baseFloats+2*kKvRows*D+kQueryRows*kKvRows;',
             'baseFloats+2*kKvRows*D+3*kQueryRows*kKvRows;')
        edit('kClusterKvReady=8,kClusterScore0=9,kClusterP0=11,kClusterRoundDone=13;',
             'kClusterKvReady=8,kClusterScore0=9,kClusterP0=0,kClusterRoundDone=13;')
        edit('  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,',
             HELPERS + '  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,')
        body('Matmul', MATMUL); body('MatmulAtomic', ATOMIC)
        edit('scoreOffset=slot?extraScoreOffset:baseScoreOffset;pOffset=scoreOffset;',
             'scoreOffset=slot?extraScoreOffset+(slot-1)*kQueryRows*kKvRows:baseScoreOffset;pOffset=scoreOffset;')
        # Preserve original C1/precise handshake, but preload V while AIV
        # executes softmax. All existing integer checks/status remain on AIV.
        edit('''      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
      Matmul(pOffset,vOffset,pvOffset,rows,D,kKvRows,false,false);''',
             '''      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      LoadResidentB(vOffset);
      CrossCoreWaitFlag(kVectorReady);
      MatmulResidentB(pOffset,pvOffset,rows,D,kKvRows,false);''')
        body('CubeCluster', CLUSTER)
        left,right=function(text,'VectorCluster');vector=text[left:right]
        for before,after in (
                ('SelectClusterSlots(buffer,member%2);','SelectClusterSlots(buffer,member);'),
                ('CrossCoreWaitFlag(kClusterScore0+member%2);','CrossCoreWaitFlag(kClusterScore0+member);'),
                ('CrossCoreSetFlag<2,PIPE_MTE3>(kClusterP0+member%2);','CrossCoreSetFlag<2,PIPE_MTE3>(kClusterP0+member);')):
            if vector.count(before)!=1:raise ValueError('Vector flag/slot mapping drift')
            vector=vector.replace(before,after)
        body('VectorCluster',vector)
    arm = 'candidate' if candidate else 'baseline'
    edit('attention_cv_member_pipeline_cluster4_launch', f'c4_l1_reuse_{arm}_launch')
    edit('oscar_attention_cv_member_pipeline_cluster4_kernel', f'oscar_c4_l1_reuse_{arm}_kernel')
    for header in ('oscar_common.h','attention_fast_unpack.h','attention_striped_unpack.h',
                   'attention_striped_unpack32.h','attention_range_mask.h'):
        edit(f'#include "{header}"',f'#include "../../kernels/{header}"',2 if header=='attention_fast_unpack.h' else 1)
    edit('"../include/', '"../../include/', 3)
    restored = text
    for old, new, count in reversed(edits):
        if restored.count(new) != count: raise ValueError('inverse alias ' + new[:100])
        restored = restored.replace(new, old)
    if restored != original: raise ValueError('full mother bytes not restored')
    methods = ('Process','IsClusterMember','TaskValid','LoadQueries','LoadPacked','Unpack','PublishHistoryKv',
        'LoadWindowRuns','LoadPrecise','PublishKv','Softmax','VectorTask','SpillState','RestoreState',
        'LoadClusterKv','PublishRows','PublishEmpty','PublishStatus')
    unchanged = {}
    for method in methods:
        before = original[slice(*function(original, method))]
        after = text[slice(*function(text, method))]
        if before != after: raise ValueError('untouched math/domain/producer changed: ' + method)
        unchanged[method] = sha(before)
    final = (HEADER if candidate else '// Unchanged qualified member-pipeline research denominator.\n') + text
    return final, {'arm': arm, 'mother_source_sha256': PIN, 'source_sha256': PIN, 'sha256':sha(final),
        'symbol': f'oscar_c4_l1_reuse_{arm}_kernel', 'mother_bytes_restored': True,
        'unchanged_methods': unchanged, 'workspace_per_core': 2756608 if candidate else 2494464,
        'explicit_UB_per_AIV': 157760, 'external_B1_bytes': 262144 if candidate else 0,
        'Normal_config_preserved': True, 'validation': 'not_compiled_or_executed'}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--check',action='store_true')
    a = p.parse_args(); manifest = {'official_sdk': official_bindings(), 'arms': {}}
    for candidate in (False, True):
        text, record = generate(candidate); name = 'c4_' + record['arm'] + '.cpp'
        if a.check:
            if (HERE / name).read_text() != text: raise ValueError('generated drift ' + name)
        else: (HERE / name).write_text(text)
        manifest['arms'][record['arm']] = record
    content = json.dumps(manifest,indent=2)+'\n'
    if a.check:
        if (HERE/'manifest.json').read_text()!=content:raise ValueError('manifest drift')
    else:(HERE/'manifest.json').write_text(content)
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False,
                      'mother':PIN,'CANN':'not_run','CPU':'not_run','CAModel':'not_run'}))


if __name__=='__main__':main()
