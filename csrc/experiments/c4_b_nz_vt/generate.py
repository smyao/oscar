"""Reversible Vt-NZ candidate from the completed B-only C4 source.

Archive #126/#145/#148/#154 and full startup D.4. No production modifications.
K/Rv stay NZ; V changes lossless layout and its matching PV transpose flag.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_b_nz import generate as mother
from csrc.experiments.decode_phase_profile.generate import function
sha=mother.sha
SOURCE='csrc/experiments/c4_b_nz/c4_candidate.cpp'
SOURCE_SHA='e33cf9485a3a754857c9cb747732ff33d58e43397b27779650dfc1ac1734a236'

PUBLISH='''{
    auto output=scoreBuf.Get<float>()[4096];
    Fence<HardEvent::MTE3_V>();
    if(value) {
      // All compressed byte indices are0..16380: signed16 -> float32 ->
      // signed32 is exact. This casts INDEX metadata only, never a KV value.
      // Natural and score's first half are dead after Unpack/Softmax/spill.
      auto expanded=naturalBuf.Get<float>(4096);
      auto integers=scoreBuf.Get<int32_t>(4096);
      Cast(expanded,vtIndexBuf.Get<int16_t>(4096),RoundMode::CAST_NONE,4096);
      PipeBarrier<PIPE_V>();
      Cast(integers,expanded,RoundMode::CAST_TRUNC,4096);PipeBarrier<PIPE_V>();
      Gather(output,dequantBuf.Get<float>()[sourceRow*D],integers.ReinterpretCast<uint32_t>(),
          static_cast<uint32_t>(0),4096);
      Fence<HardEvent::V_MTE3>();
      // A relative16-KV row segment is two complete Vt-NZ C0 blocks:
      // [KV/8,D,8], hence one contiguous16KiB copy at logicalRow*D.
      DataCopy(work[vOffset+logicalRow*D],output,4096);
    } else {
      // Original K-NZ [D/8,KV,8] publisher, with identical Gather/2D copy.
      Gather(output,dequantBuf.Get<float>()[sourceRow*D],nzIndexBuf.Get<uint32_t>(4096),
          static_cast<uint32_t>(0),4096);
      Fence<HardEvent::V_MTE3>();
      const DataCopyParams copy{32,16,0,240};
      DataCopy(work[kOffset+logicalRow*8],output,copy);
    }
    Fence<HardEvent::MTE3_V>();
  }'''

HEADER='''// ISOLATED Vt-NZ COMPLETE C4 RESEARCH; never a production registration.
// Archive #126/#145/#148/#154, startup D.4 four questions:
// 1. Bounded fused K/V publication and PV Cube B transfer.
// 2. D.4's full-cache6.5s restore and old ND transpose scattered writes
//    must not return; CPU alone cannot prove asynchronous NPU ownership.
// 3. Only V changes to [KV/8,D,8]; K/Rv remain NZ, A/P remain ND. Keep
//    FP32/K128, all tasks/status, old lane16, tail values and crosscore flags.
//    Fixed KV256 only; V indices8KiB add UB to182336B, no buffer shrink.
// 4. V publisher512->16 bursts is not a speedup: Cube B bursts may grow,
//    and both index Casts/Gather/fences are charged to complete kernel time.
// Index narrowing uses the qualified A2 s32->s16 CAST_NONE path from
// striped_store_pack.h;0..16380 fits signed16. KV bits never undergo Cast.
// Baseline Rv is natural ND; candidate Rv is the same matrix in NZ.
// Never transpose Rv twice.
'''


def generate(candidate):
    if not candidate:
        original,origin=mother.mother.generate(True)
        if sha(original)!=mother.SOURCE_SHA or (ROOT/mother.SOURCE).read_text()!=original:
            raise ValueError('qualified ND pipeline changed')
        text=original.replace('c4_member_pipeline_candidate_launch','c4_b_nz_vt_baseline_launch').replace(
            'oscar_c4_member_pipeline_candidate_kernel','oscar_c4_b_nz_vt_baseline_kernel')
        restored=text.replace('c4_b_nz_vt_baseline_launch','c4_member_pipeline_candidate_launch').replace(
            'oscar_c4_b_nz_vt_baseline_kernel','oscar_c4_member_pipeline_candidate_kernel')
        if restored!=original:raise ValueError('ND baseline restoration failed')
        header='// Qualified ND H32+pipeline baseline; natural Rv. Same-library comparison to Vt-NZ.\n'
        result=header+text
        return result,{**origin,'arm':'baseline','symbol':'oscar_c4_b_nz_vt_baseline_kernel',
            'sha256':sha(result),'pipeline_source_sha256':mother.SOURCE_SHA,
            'mother_source':mother.SOURCE,'pipeline_bytes_restored':True,
            'V_format':'ND','K_format':'ND','Rv_format':'natural ND',
            'A_format':'ND','P_format':'ND','rv_input_format':'ND','rv_pack_bytes':0,
            'AIV_explicit_UB_bytes':157760,'GM_bytes_per_core':2494464,
            'CPU':'not_run','CANN':'not_run','CAModel':'not_run','NPU':'not_run','graph':'not_run'}
    original,origin=mother.generate(True)
    if sha(original)!=SOURCE_SHA or (ROOT/SOURCE).read_text()!=original:
        raise ValueError('completed B-only mother source changed')
    text=original;edits=[]
    def replace(old,new,count=1,label=''):
        nonlocal text
        if text.count(old)!=count:raise ValueError((label,text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count,label))
    def body(name,new):
        a,b=function(text,name);replace(text[a:b],new,label=name)
    if candidate:
        replace('pipe.InitBuffer(nzIndexBuf,16384);','pipe.InitBuffer(nzIndexBuf,16384);\n      pipe.InitBuffer(vtIndexBuf,8192);',label='8KiB permanent V index')
        replace('      BuildNz16Indices();','      BuildNz16Indices();\n      BuildVt16Indices();',label='one V index initialization')
        replace('TBuf<TPosition::VECCALC> nzIndexBuf;','TBuf<TPosition::VECCALC> nzIndexBuf,vtIndexBuf;',label='additional fixed TBuf')
        a,b=function(text,'BuildNz16Indices');build=text[a:b]
        changes=[('nzIndexBuf.Get<int32_t>(4096)','naturalBuf.Get<int32_t>(4096)'),
                 ('static_cast<uint16_t>(2),8192);PipeBarrier<PIPE_V>();\n    Duplicate(temp,static_cast<uint16_t>(120)',
                  'static_cast<uint16_t>(10),8192);PipeBarrier<PIPE_V>();\n    Duplicate(temp,static_cast<uint16_t>(2048)'),
                 ('static_cast<uint16_t>(7),8192);PipeBarrier<PIPE_V>();\n    Or(parts,parts,temp',
                  'static_cast<uint16_t>(2),8192);PipeBarrier<PIPE_V>();\n    Or(parts,parts,temp'),
                 ('static_cast<uint16_t>(3968)','static_cast<uint16_t>(2040)'),
                 ('ShiftRight(temp,temp,static_cast<uint16_t>(2)','ShiftRight(temp,temp,static_cast<uint16_t>(1)')]
        for old,new in changes:
            if build.count(old)!=1:raise ValueError('index construction source drift')
            build=build.replace(old,new)
        # Replace explanatory comments; executable statements above remain explicit.
        start=build.index('    // One initialization');end=build.index('    auto index=',start)
        build=build[:start]+'    // Init-only scratch: natural16KiB and the two score16KiB halves.\n'+build[end:]
        start=build.index('    // t indexes');end=build.index('    Duplicate(temp',start)
        build=build[:start]+'    // Vt source byte=((t&7)<<10)|((t&2048)<<2)|((t&2040)>>1).\n'+build[end:]
        build=build[:-1]+'  // Compress exact byte offsets, not floating KV payload.\n    Cast(vtIndexBuf.Get<int16_t>(4096),index,RoundMode::CAST_NONE,4096);\n    PipeBarrier<PIPE_V>();\n  }'
        replace('  __aicore__ void PublishNz16(',
                '  __aicore__ void BuildVt16Indices() '+build+'\n  __aicore__ void PublishNz16(',label='bounded V-index constructor')
        body('PublishNz16',PUBLISH)
        replace('if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],transposeB);',
                'if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],true);',label='K true / Vt true / original Rv true')
        replace('mm.SetTensorA(work[a],false);mm.SetTensorB(work[b],false);',
                'mm.SetTensorA(work[a],false);mm.SetTensorB(work[b],true);',label='atomic PV reads Vt')
    arm='candidate' if candidate else 'baseline'
    replace('c4_b_nz_candidate_launch','c4_b_nz_vt_'+arm+'_launch',label='launch')
    replace('oscar_c4_b_nz_candidate_kernel','oscar_c4_b_nz_vt_'+arm+'_kernel',label='kernel')
    mother.mother.validate_declarations(text)
    restored=text
    for old,new,count,label in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse drift '+label)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('mother restoration failed')
    unchanged=('BuildNz16Indices','PublishHistoryKv','PublishKv','InitIndices','Process',
               'TaskValid','IsClusterMember','CubeTask','CubeCluster','VectorTask','VectorCluster',
               'LoadQueries','LoadPacked','Unpack','LoadPrecise','LoadWindowRuns','Softmax',
               'SpillState','RestoreState','PublishRows','PublishStatus','LoadClusterKv','SelectClusterSlots')
    for name in unchanged:
        if text[slice(*function(text,name))]!=original[slice(*function(original,name))]:
            raise ValueError('unrelated method changed '+name)
    for token in ('CrossCoreSetFlag<','CrossCoreWaitFlag(','SetHF32Mode('):
        if text.count(token)!=original.count(token):raise ValueError('protocol changed '+token)
    result=HEADER+text
    return result,{**origin,'arm':arm,'symbol':'oscar_c4_b_nz_vt_'+arm+'_kernel',
        'sha256':sha(result),'B_NZ_source_sha256':SOURCE_SHA,'mother_source':SOURCE,
        'B_NZ_bytes_restored':True,'unchanged_methods':unchanged,
        'V_format':'Vt-NZ' if candidate else 'NZ','K_format':'NZ','Rv_format':'original NZ unchanged',
        'AIV_explicit_UB_bytes':182336 if candidate else 174144,
        'V_index_bytes':8192 if candidate else 0,'GM_bytes_per_core':2494464,
        'K_V_bursts_per_full_tile':528 if candidate else 1024,
        'V_publisher_casts_per_16rows':2 if candidate else 0,
        'CPU':'not_run','CANN':'not_run','CAModel':'not_run','NPU':'not_run','graph':'not_run'}


def build_binding():
    return {'schema':1,'scope':'configured build inputs; actual library hash recorded separately',
        'B_NZ_source_sha256':SOURCE_SHA,
        'ND_pipeline_source_sha256':mother.SOURCE_SHA,
        'sources':{f'c4_{a}.cpp':generate(a=='candidate')[1]['sha256'] for a in ('baseline','candidate')},
        'dependencies':{p:sha((ROOT/p).read_bytes()) for p in mother.DEPENDENCIES},
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
        if not a.check:raise ValueError('build binding requires check')
        a.build_binding.write_text(json.dumps(build_binding(),indent=2)+'\n')
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False,'execution':'not_run'}))


if __name__=='__main__':main()
