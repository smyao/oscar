"""Isolated complete C4 B-NZ publisher/consumer from the measured H32 pipeline.

Archive G25-G34/#13-22/#126/#145/#148/#154 and full startup D.4. Each edit
reverses to one pinned current-project mother. The same real K/V values, lane
error domains, full task set, FP32 math and two-slot protocol remain intact.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.c4_member_pipeline import generate as mother
from csrc.experiments.decode_phase_profile.generate import function

sha = mother.sha
SOURCE = 'csrc/experiments/c4_member_pipeline/c4_candidate.cpp'
SOURCE_SHA = 'da59a923a44cdb7ac0916596cd89f802a24fc1ffef4426952402d90b5d2e7b31'
DEPENDENCIES = (
    'csrc/kernels/oscar_common.h', 'csrc/kernels/attention_range_mask.h',
    'csrc/kernels/attention_fast_unpack.h', 'csrc/kernels/attention_striped_unpack.h',
    'csrc/include/oscar_attention_launch.h', 'csrc/include/oscar_cv_schedule.h',
    'csrc/experiments/c4_unpack32/unpack32.h',
)

HELPERS = '''  __aicore__ void BuildNz16Indices() {
    static_assert(D==256,"this independent NZ publisher is D256 only");
    // One initialization per AIV before any task uses scoreBuf. Permanent
    // indices cost16KiB; construction temporaries reuse both score halves.
    // Never shrink natural/dequant/Softmax views to make room.
    auto index=nzIndexBuf.Get<int32_t>(4096);
    auto temp=scoreBuf.Get<uint16_t>(8192);
    auto parts=scoreBuf.Get<uint16_t>()[8192];
    ArithProgression<int32_t>(index,0,1,4096);
    PipeBarrier<PIPE_V>();
    auto words=index.ReinterpretCast<uint16_t>();
    // t indexes [32 C0,16 rows,8 lanes]. Its natural source byte index is
    // ((t&7)<<2) | ((t&120)<<7) | ((t&3968)>>2), all below16384.
    // Both halves of each int32 are processed; the high half stays zero.
    Duplicate(temp,static_cast<uint16_t>(7),8192);PipeBarrier<PIPE_V>();
    And(parts,words,temp,8192);PipeBarrier<PIPE_V>();
    ShiftLeft(parts,parts,static_cast<uint16_t>(2),8192);PipeBarrier<PIPE_V>();
    Duplicate(temp,static_cast<uint16_t>(120),8192);PipeBarrier<PIPE_V>();
    And(temp,words,temp,8192);PipeBarrier<PIPE_V>();
    ShiftLeft(temp,temp,static_cast<uint16_t>(7),8192);PipeBarrier<PIPE_V>();
    Or(parts,parts,temp,8192);PipeBarrier<PIPE_V>();
    Duplicate(temp,static_cast<uint16_t>(3968),8192);PipeBarrier<PIPE_V>();
    And(temp,words,temp,8192);PipeBarrier<PIPE_V>();
    ShiftRight(temp,temp,static_cast<uint16_t>(2),8192);PipeBarrier<PIPE_V>();
    Or(words,parts,temp,8192);PipeBarrier<PIPE_V>();
  }
  __aicore__ void PublishNz16(bool value,int32_t logicalRow,int32_t sourceRow) {
    // The second score half is no longer index-construction scratch.
    // Each C0 block writes16*8 FP32 values, exactly512B. NZ physical rows256.
    auto output=scoreBuf.Get<float>()[4096];
    auto indices=nzIndexBuf.Get<uint32_t>(4096);
    Fence<HardEvent::MTE3_V>();
    Gather(output,dequantBuf.Get<float>()[sourceRow*D],indices,
        static_cast<uint32_t>(0),4096);
    Fence<HardEvent::V_MTE3>();
    const DataCopyParams copy{32,16,0,240};
    DataCopy(work[(value?vOffset:kOffset)+logicalRow*8],output,copy);
    Fence<HardEvent::MTE3_V>();
  }
'''

HISTORY = '''{
    const int32_t slot=subtile*(2*kHistoryHalfKv)+lane*kHalfKv;
    // Retain old16-row lane domains: UB rows0..15 and16..31 belong to
    // logical slot..slot+15 and slot+32..slot+47. Every padding value is
    // published too, exactly as in the qualified full natural publisher.
    PublishNz16(value,slot,0);
    PublishNz16(value,slot+2*kHalfKv,kHalfKv);
  }'''

PRECISE = '''{
    const int32_t slot=subtile*(2*kHalfKv)+lane*kHalfKv;
    // Source1/window and source2/current use the same B format, including
    // their original +0 tail and invalid-tag/address output values.
    PublishNz16(value,slot,0);
  }'''

HEADER = '''// ISOLATED COMPLETE C4 B-NZ RESEARCH; never registered in production.
// Archive G25-G34/#13-22/#126/#145/#148/#154; full startup D.4 four questions:
// 1. Fused history dequant publication and Cube QK/PV B movement.
// 2. D.4 full-history restore6499.8-6655.1ms and #145/#154 UB reuse races
//    are forbidden; CPU alone does not certify real asynchronous hardware.
// 3. Only the current two KV256 buffers change lossless layout. FP32/K128,
//    complete tasks, old lane16 metadata errors, ND A/score/P and all
//    CrossCore flags stay unchanged. Real Gather/index/DMA/fences are timed.
// 4. No speed claim: extra publisher bursts can erase the NZ consumer gain.
//    Require complete same-library C4 net timing, oracle and paired bits.
// Permanent NZ indices add16KiB UB (174144B/AIV total); initialization cost
// is inside the complete kernel. No original buffer/view is reduced.
// Candidate Rv is a distinct initialization-time exact NZ pack,256KiB/D256;
// host packing is declared separately, not presented as per-step kernel work.
// SDK C0=float8: CopyNZ2NZ offset=(col/8)*align16(rows)*8+row*8+col%8.
'''


def generate(candidate):
    original, origin = mother.generate(True)
    if sha(original) != SOURCE_SHA or original != (ROOT / SOURCE).read_text():
        raise ValueError('measured pipeline mother changed; re-audit before generating')
    text = original
    edits = []

    def replace(old, new, count=1, label=''):
        nonlocal text
        if text.count(old) != count:
            raise ValueError((label, text.count(old), count))
        text = text.replace(old, new)
        edits.append((old, new, count, label))

    def method(name, body):
        a, b = function(text, name)
        replace(text[a:b], body, label=name)

    if candidate:
        replace('using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;',
                'using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::NZ,float,true>;',
                label='B format only; A/C retain ND')
        replace('pipe.InitBuffer(addressBuf,128);\n      InitIndices();',
                'pipe.InitBuffer(addressBuf,128);\n      pipe.InitBuffer(nzIndexBuf,16384);\n      InitIndices();\n      BuildNz16Indices();',
                label='permanent bounded index storage and one initialization')
        replace('scoreBuf,accBuf,qFloatBuf,qBf16Buf,statsBuf,scratchBuf,addressBuf;',
                'scoreBuf,accBuf,qFloatBuf,qBf16Buf,statsBuf,scratchBuf,addressBuf;\n  TBuf<TPosition::VECCALC> nzIndexBuf;',
                label='one independent index TBuf')
        method('PublishHistoryKv', HISTORY)
        method('PublishKv', PRECISE)
        replace('  __aicore__ void PublishHistoryKv(',
                HELPERS+'  __aicore__ void PublishHistoryKv(', label='bounded NZ helpers')
    arm = 'candidate' if candidate else 'baseline'
    replace('c4_member_pipeline_candidate_launch', 'c4_b_nz_'+arm+'_launch', label='launch')
    replace('oscar_c4_member_pipeline_candidate_kernel', 'oscar_c4_b_nz_'+arm+'_kernel', label='kernel')
    mother.validate_declarations(text)
    restored = text
    for old, new, count, label in reversed(edits):
        if restored.count(new) != count:
            raise ValueError('inverse edit drift: '+label)
        restored = restored.replace(new, old)
    if restored != original:
        raise ValueError('full mother byte restoration failed')
    unchanged = ('InitIndices','Process','TaskValid','IsClusterMember',
                 'Matmul','MatmulAtomic','CubeTask','CubeCluster','VectorTask',
                 'VectorCluster','LoadQueries','LoadPacked','Unpack','LoadPrecise',
                 'LoadWindowRuns','Softmax','SpillState','RestoreState',
                 'PublishRows','PublishStatus','LoadClusterKv','SelectClusterSlots')
    for name in unchanged:
        if text[slice(*function(text,name))] != original[slice(*function(original,name))]:
            raise ValueError('unrelated method changed: '+name)
    for token in ('CrossCoreSetFlag<','CrossCoreWaitFlag(','SetHF32Mode('):
        if text.count(token) != original.count(token):
            raise ValueError('protocol/math marker count changed: '+token)
    result = HEADER + text
    return result, {
        **origin, 'arm':arm, 'symbol':'oscar_c4_b_nz_'+arm+'_kernel',
        'sha256':sha(result), 'pipeline_source_sha256':SOURCE_SHA,
        'mother_source':SOURCE, 'pipeline_bytes_restored':True,
        'unchanged_methods':unchanged, 'A_format':'ND', 'P_format':'ND',
        'B_format':'NZ' if candidate else 'ND',
        'rv_input_format':'NZ bit-pack of unchanged logical matrix' if candidate else 'ND',
        'rv_pack_bytes':256*256*4 if candidate else 0,
        'rv_pack_scope':'initialization outside per-step kernel; separate memory/cost',
        'KV_publisher':'real in-kernel conversion; no host K/V prepack',
        'AIV_explicit_UB_bytes':174144 if candidate else 157760, 'GM_bytes_per_core':2494464,
        'indices_elements':4096 if candidate else 0,
        'indices_initializations_per_AIV':1 if candidate else 0,
        'indices_init_cost':'inside complete kernel; no original buffer/view reduced',
        'K_V_bursts_per_full_tile':1024 if candidate else 32,
        'oracle':'same natural logical inputs; all tasks retained',
        'CPU':'not_run','CANN':'not_run','CAModel':'not_run','NPU':'not_run','graph':'not_run',
    }


def build_binding():
    return {'schema':1, 'scope':'configured build inputs; library SHA recorded separately',
            'pipeline_source_sha256':SOURCE_SHA,
            'sources': {f'c4_{arm}.cpp':generate(arm=='candidate')[1]['sha256']
                        for arm in ('baseline','candidate')},
            'dependencies': {p:sha((ROOT/p).read_bytes()) for p in DEPENDENCIES},
            'generator_sha256':sha(Path(__file__).read_bytes())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--build-binding',type=Path)
    args=parser.parse_args()
    manifest={}
    for candidate in (False,True):
        content,entry=generate(candidate)
        path=HERE/('c4_'+entry['arm']+'.cpp');manifest[entry['arm']]=entry
        if args.check:
            if path.read_text()!=content:raise ValueError('stale generated source: '+str(path))
        else:path.write_text(content)
    encoded=json.dumps(manifest,indent=2)+'\n'
    if args.check:
        if (HERE/'manifest.json').read_text()!=encoded:raise ValueError('stale manifest')
    else:(HERE/'manifest.json').write_text(encoded)
    if args.build_binding:
        if not args.check:raise ValueError('build binding requires --check')
        args.build_binding.write_text(json.dumps(build_binding(),indent=2)+'\n')
    print(json.dumps({'status':'checked' if args.check else 'generated',
                      'production_changed':False,'execution':'not_run'}))


if __name__=='__main__':main()
