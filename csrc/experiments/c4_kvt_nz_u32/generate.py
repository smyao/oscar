"""Kt/Vt layout v2: permanent u32 indices, no per-publisher index Cast.

Archive #126/#145/#148/#154 and full startup D.4. Keep v1 evidence intact;
baseline remains qualified ND H32/member pipeline, not the losing v1 arm.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_kvt_nz import generate as mother
from csrc.experiments.decode_phase_profile.generate import function
from csrc.experiments.c4_member_pipeline.generate import validate_declarations
sha=mother.sha
SOURCE='csrc/experiments/c4_kvt_nz/c4_candidate.cpp'
SOURCE_SHA='c5e0791c3213d81f1c655c4a09fe52dba1718c0ea01f7d9836ac722b89fab45d'

PUBLISH='''{
    auto output=scoreBuf.Get<float>()[4096];
    Fence<HardEvent::MTE3_V>();
    // Permanent0..16380 byte offsets, no repeated index conversions and
    // no KV arithmetic. Original half16 row ownership and tail bits survive.
    Gather(output,dequantBuf.Get<float>()[sourceRow*D],vtIndexBuf.Get<uint32_t>(4096),
        static_cast<uint32_t>(0),4096);
    Fence<HardEvent::V_MTE3>();
    DataCopy(work[(value?vOffset:kOffset)+logicalRow*D],output,4096);
    Fence<HardEvent::MTE3_V>();
  }'''

HEADER='''// ISOLATED Kt/Vt-NZ U32-INDEX V2; never a production registration.
// Archive #126/#145/#148/#154, full startup D.4 four questions:
// 1. Fused current-KV256 dequant publisher. 2. No full-history6.5s restore
//    or unguarded UB reuse. 3. Preserve exact payload/layout/FP32/K128,
//    all tasks/errors/flags and Kt/Vt consumer; only permanent index width
//    changes8->16KiB, UB165952->174144B. No original buffer/view is reduced.
// 4. Remove64*4096 index conversions/full KV pair plus initialization
//    narrowing; all remaining Gather/DMA/fences stay in complete timing.
//    Compare to ND H32+pipeline; prior losing layout is not the denominator.
//    No speed/NPU/graph claim until complete same-library gates pass.
'''


def generate(candidate):
    original,origin=mother.generate(candidate)
    if not candidate:
        text=original.replace('c4_kvt_nz_baseline_launch','c4_kvt_nz_u32_baseline_launch').replace(
            'oscar_c4_kvt_nz_baseline_kernel','oscar_c4_kvt_nz_u32_baseline_kernel')
        if text.replace('c4_kvt_nz_u32_baseline_launch','c4_kvt_nz_baseline_launch').replace(
                'oscar_c4_kvt_nz_u32_baseline_kernel','oscar_c4_kvt_nz_baseline_kernel')!=original:
            raise ValueError('ND baseline restoration failed')
        return text,{**origin,'symbol':'oscar_c4_kvt_nz_u32_baseline_kernel','sha256':sha(text)}
    if sha(original)!=SOURCE_SHA or (ROOT/SOURCE).read_text()!=original:
        raise ValueError('completed Kt/Vt v1 mother changed')
    text=original;edits=[]
    def replace(old,new,count=1,label=''):
        nonlocal text
        if text.count(old)!=count:raise ValueError((label,text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count,label))
    replace('pipe.InitBuffer(vtIndexBuf,8192);','pipe.InitBuffer(vtIndexBuf,16384);',label='u32 permanent index only')
    a,b=function(text,'BuildVt16Indices');old=text[a:b];new=old
    new=new.replace('// Init-only scratch: natural16KiB and the two score16KiB halves.',
                    '// Init directly in permanent u32 table; only the two score halves are scratch.')
    new=new.replace('auto index=naturalBuf.Get<int32_t>(4096);','auto index=vtIndexBuf.Get<int32_t>(4096);')
    narrow='    // Compress exact byte offsets, not floating KV payload.\n    Cast(vtIndexBuf.Get<int16_t>(4096),index,RoundMode::CAST_NONE,4096);\n    PipeBarrier<PIPE_V>();\n'
    if new.count(narrow)!=1:raise ValueError('initialization narrowing changed')
    new=new.replace(narrow,'    // Full u32 byte offsets remain live for every K/V publication.\n')
    replace(old,new,label='same index formula, no narrowing')
    a,b=function(text,'PublishNz16');replace(text[a:b],PUBLISH,label='remove per-publisher index conversions')
    replace('c4_kvt_nz_candidate_launch','c4_kvt_nz_u32_candidate_launch',label='launch')
    replace('oscar_c4_kvt_nz_candidate_kernel','oscar_c4_kvt_nz_u32_candidate_kernel',label='kernel')
    validate_declarations(text)
    restored=text
    for old,new,count,label in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse drift '+label)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('v1 full source restoration failed')
    unchanged=('Matmul','MatmulAtomic','PublishHistoryKv','PublishKv','InitIndices','Process',
        'TaskValid','IsClusterMember','CubeTask','CubeCluster','VectorTask','VectorCluster',
        'LoadQueries','LoadPacked','Unpack','LoadPrecise','LoadWindowRuns','Softmax',
        'SpillState','RestoreState','PublishRows','PublishStatus','LoadClusterKv','SelectClusterSlots')
    for name in unchanged:
        if text[slice(*function(text,name))]!=original[slice(*function(original,name))]:raise ValueError('unrelated method changed '+name)
    for token in ('CrossCoreSetFlag<','CrossCoreWaitFlag(','SetHF32Mode('):
        if text.count(token)!=original.count(token):raise ValueError('protocol changed '+token)
    result=HEADER+text
    return result,{**origin,'symbol':'oscar_c4_kvt_nz_u32_candidate_kernel','sha256':sha(result),
        'KVT_v1_source_sha256':SOURCE_SHA,'mother_source':SOURCE,'v1_bytes_restored':True,
        'unchanged_methods':unchanged,'AIV_explicit_UB_bytes':174144,
        'shared_KV_index_bytes':16384,'shared_index_dtype':'uint32 byte offsets',
        'K_V_publisher_casts_per_full_tile':0,'removed_index_conversions_per_full_tile':64*4096,
        'CPU':'not_run','CANN':'not_run','CAModel':'not_run','NPU':'not_run','graph':'not_run'}


def build_binding():
    parent=mother.build_binding()
    return {**parent,'KVT_v1_source_sha256':SOURCE_SHA,
        'sources':{f'c4_{a}.cpp':generate(a=='candidate')[1]['sha256'] for a in ('baseline','candidate')},
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
