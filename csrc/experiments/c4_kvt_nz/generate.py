"""Complete Kt/Vt-NZ publisher/consumer trial; old ND pipeline is baseline.

Archive #126/#145/#148/#154 and full startup D.4. Only lossless layout and
matching B transpose change; no old experiment/production/native edit.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_b_nz_vt import generate as mother
from csrc.experiments.decode_phase_profile.generate import function
from csrc.experiments.c4_member_pipeline.generate import validate_declarations
sha=mother.sha
SOURCE='csrc/experiments/c4_b_nz_vt/c4_candidate.cpp'
SOURCE_SHA='6ea0ad44ad458813710a57f5c6ff2517b8a2b8f01d0b29ad5bfe5c0ea1c8c24e'

PUBLISH='''{
    auto output=scoreBuf.Get<float>()[4096];
    Fence<HardEvent::MTE3_V>();
    // Kt and Vt share [KV/8,D,8]. Only exact0..16380 INDEX bytes undergo
    // these two Casts; both sides' FP32 KV payload passes through Gather.
    auto expanded=naturalBuf.Get<float>(4096);
    auto integers=scoreBuf.Get<int32_t>(4096);
    Cast(expanded,vtIndexBuf.Get<int16_t>(4096),RoundMode::CAST_NONE,4096);
    PipeBarrier<PIPE_V>();
    Cast(integers,expanded,RoundMode::CAST_TRUNC,4096);PipeBarrier<PIPE_V>();
    Gather(output,dequantBuf.Get<float>()[sourceRow*D],integers.ReinterpretCast<uint32_t>(),
        static_cast<uint32_t>(0),4096);
    Fence<HardEvent::V_MTE3>();
    // Each original16-row lane segment is two whole C0 blocks:16KiB.
    DataCopy(work[(value?vOffset:kOffset)+logicalRow*D],output,4096);
    Fence<HardEvent::MTE3_V>();
  }'''

HEADER='''// INDEPENDENT COMPLETE Kt/Vt-NZ RESEARCH; no production registration.
// Archive #126/#145/#148/#154, full startup D.4 four questions:
// 1. Bounded INT2 dequant publication plus QK/PV Cube B input movement.
// 2. D.4 full-history6.5s restore and ND-transpose tiny scattered writes
//    cannot return; CPU does not establish NPU asynchronous correctness.
// 3. Kt/Vt share [KV/8,D,8] and the same8KiB exact index. Original lane16,
//    all tasks/errors, FP32/K128 math, ND A/P and crosscore flags survive.
//    QK reads physical Kt with B=false; PV physical Vt with B=true. Rv is
//    the unchanged original-matrix NZ bit-pack, B=true, never transposed twice.
// 4. MTE3 bursts528->32 is only a work count: K gains two index Casts per
//    publish and uses SDK non-transposed Load3D. Charge all costs to full C4.
// Candidate UB165952B/AIV; fixed GM2494464B/core; no history-sized buffer.
'''


def generate(candidate):
    if not candidate:
        original,origin=mother.generate(False)
        text=original.replace('c4_b_nz_vt_baseline_launch','c4_kvt_nz_baseline_launch').replace(
            'oscar_c4_b_nz_vt_baseline_kernel','oscar_c4_kvt_nz_baseline_kernel')
        if text.replace('c4_kvt_nz_baseline_launch','c4_b_nz_vt_baseline_launch').replace(
                'oscar_c4_kvt_nz_baseline_kernel','oscar_c4_b_nz_vt_baseline_kernel')!=original:
            raise ValueError('ND baseline restoration failed')
        return text,{**origin,'symbol':'oscar_c4_kvt_nz_baseline_kernel','sha256':sha(text)}
    original,origin=mother.generate(True)
    if sha(original)!=SOURCE_SHA or (ROOT/SOURCE).read_text()!=original:
        raise ValueError('completed Vt mother source drift')
    text=original;edits=[]
    def replace(old,new,count=1,label=''):
        nonlocal text
        if text.count(old)!=count:raise ValueError((label,text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count,label))
    a,b=function(text,'BuildNz16Indices');start=text.rfind('  __aicore__ void BuildNz16Indices(',0,a)
    if start<0:raise ValueError('K index method declaration missing')
    replace(text[start:b],'  // Kt and Vt share BuildVt16Indices; no separate K constructor.',label='remove redundant16KiB index method')
    replace('      pipe.InitBuffer(nzIndexBuf,16384);','      // No separate16KiB K index allocation.',label='remove only new K index buffer')
    replace('      BuildNz16Indices();','      // Shared compact index initialized below once.',label='remove K constructor call')
    replace('TBuf<TPosition::VECCALC> nzIndexBuf,vtIndexBuf;','TBuf<TPosition::VECCALC> vtIndexBuf;',label='retain8KiB shared index')
    a,b=function(text,'PublishNz16');replace(text[a:b],PUBLISH,label='two contiguous publishers')
    replace('if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],true);',
            'if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],!transposeB);',
            label='QK false / ordinary PV true / original Rv true')
    replace('c4_b_nz_vt_candidate_launch','c4_kvt_nz_candidate_launch',label='launch')
    replace('oscar_c4_b_nz_vt_candidate_kernel','oscar_c4_kvt_nz_candidate_kernel',label='kernel')
    validate_declarations(text)
    restored=text
    for old,new,count,label in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse drift '+label)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('Vt source restoration failed')
    unchanged=('BuildVt16Indices','PublishHistoryKv','PublishKv','InitIndices','MatmulAtomic',
               'Process','TaskValid','IsClusterMember','CubeTask','CubeCluster','VectorTask','VectorCluster',
               'LoadQueries','LoadPacked','Unpack','LoadPrecise','LoadWindowRuns','Softmax','SpillState',
               'RestoreState','PublishRows','PublishStatus','LoadClusterKv','SelectClusterSlots')
    for name in unchanged:
        if text[slice(*function(text,name))]!=original[slice(*function(original,name))]:
            raise ValueError('unrelated method changed '+name)
    for token in ('CrossCoreSetFlag<','CrossCoreWaitFlag(','SetHF32Mode('):
        if text.count(token)!=original.count(token):raise ValueError('protocol changed '+token)
    result=HEADER+text
    return result,{**origin,'symbol':'oscar_c4_kvt_nz_candidate_kernel','sha256':sha(result),
        'VT_source_sha256':SOURCE_SHA,'mother_source':SOURCE,'VT_bytes_restored':True,
        'unchanged_methods':unchanged,'K_format':'Kt-NZ','V_format':'Vt-NZ',
        'Rv_format':'original NZ unchanged','AIV_explicit_UB_bytes':165952,
        'K_V_bursts_per_full_tile':32,'K_V_publisher_casts_per_full_tile':64,
        'K_index_bytes':0,'shared_KV_index_bytes':8192,
        'CPU':'not_run','CANN':'not_run','CAModel':'not_run','NPU':'not_run','graph':'not_run'}


def build_binding():
    return {'schema':1,'scope':'configured inputs; actual library hash recorded separately',
        'VT_source_sha256':SOURCE_SHA,'ND_pipeline_source_sha256':mother.mother.SOURCE_SHA,
        'sources':{f'c4_{a}.cpp':generate(a=='candidate')[1]['sha256'] for a in ('baseline','candidate')},
        'dependencies':{p:sha((ROOT/p).read_bytes()) for p in mother.mother.DEPENDENCIES},
        'generator_sha256':sha(Path(__file__).read_bytes())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--build-binding',type=Path);a=p.parse_args();manifest={}
    for candidate in (False,True):
        text,entry=generate(candidate);path=HERE/('c4_'+entry['arm']+'.cpp');manifest[entry['arm']]=entry
        if a.check:
            if path.read_text()!=text:raise ValueError('stale source '+str(path))
        else:path.write_text(text)
    text=json.dumps(manifest,indent=2)+'\n';path=HERE/'manifest.json'
    if a.check:
        if path.read_text()!=text:raise ValueError('stale manifest')
    else:path.write_text(text)
    if a.build_binding:
        if not a.check:raise ValueError('binding requires check')
        a.build_binding.write_text(json.dumps(build_binding(),indent=2)+'\n')
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False,'execution':'not_run'}))


if __name__=='__main__':main()
