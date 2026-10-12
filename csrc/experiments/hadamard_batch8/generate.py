"""Archive #13-22/#70-73/#126/#145/#148/#154-156; startup D.4.

Compact stage-major Hadamard only. Each row retains its original FP32
Gather/Mul/Add order; stages share fences, never data or reductions. Read
current sources and frozen writer evidence, never regenerate the baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.striped_store_simd_pack import generate as frozen
from csrc.experiments.decode_phase_profile.generate import masked

CANONICAL='csrc/kernels/rotate_clip_store.cpp'
CURRENT='csrc/experiments/striped_store_simd_pack/writer_pack_init.cpp'
OLD='csrc/experiments/striped_store_simd_pack/writer_baseline.cpp'
HEADER='''// ISOLATED COMPACT BATCH8 HADAMARD. Archive #13-22/#70-73/#126/#145/#148/#154-156.
// D.4: rotate/phase1_stores; never the failed full-history restore6499.8-6655.1ms.
// Fixed8-row work only. Row-local FP32 Gather/Mul/Add and scale order retained;
// work[row*D] keeps every pre-Mul gather alive until Add. No dtype relaxation.
// Full8 rows: vector calls200->137, PIPE_V barriers200->25; not a timing claim.
// Fixed UB writer61440B/RotateOnly51392B. NPU, graph and service unverified.
'''

OLD_LOOP='''      for (int32_t row=0;row<rows;++row) {
        auto x=input[row*dim_];
        int32_t stage=0;
        for (int32_t distance=1;distance<dim_;distance*=2,++stage) {
          Gather(work,x,indices[stage*dim_],static_cast<uint32_t>(0),dim_);
          PipeBarrier<PIPE_V>();
          Mul(x,x,signs[stage*dim_],dim_); PipeBarrier<PIPE_V>();
          Add(x,x,work,dim_); PipeBarrier<PIPE_V>();
        }
        Muls(output[row*dim_],x,normalization,dim_); PipeBarrier<PIPE_V>();
      }'''
NEW_LOOP='''      int32_t stage=0;
      for (int32_t distance=1;distance<dim_;distance*=2,++stage) {
        // Every row's old input is gathered before any row is multiplied.
        // Gather still uses the original D indices and one row-local source.
        for (int32_t row=0;row<rows;++row)
          Gather(work[row*dim_],input[row*dim_],indices[stage*dim_],
              static_cast<uint32_t>(0),dim_);
        PipeBarrier<PIPE_V>();
        for (int32_t row=0;row<rows;++row)
          Mul(input[row*dim_],input[row*dim_],signs[stage*dim_],dim_);
        PipeBarrier<PIPE_V>();
        Add(input,input,work,rows*dim_);PipeBarrier<PIPE_V>();
      }
      Muls(output,input,normalization,rows*dim_);PipeBarrier<PIPE_V>();'''


def sha(data):return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def kernel(kind):
    frozen.check();frozen.verify_production()
    source=CURRENT if kind=='writer_batch' else CANONICAL
    original=(ROOT/source).read_text();text=original;edits=[]
    def edit(old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError((kind,old[:70],text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count))
    if kind!='writer_batch':
        edit('#include "oscar_common.h"','#include "../../kernels/oscar_common.h"\n#include "../hadamard_indices/vector_indices.h"')
        edit('#include "../include/oscar_rotation_launch.h"','#include "../../include/oscar_rotation_launch.h"')
        begin=text.index('      int32_t stage=0;');end=text.index('      Fence<HardEvent::S_V>();',begin)+len('      Fence<HardEvent::S_V>();')
        edit(text[begin:end],'''      oscar_hadamard_indices_trial::Build(indices,signs,
          inputBuf_.template Get<int32_t>(),workBuf_.template Get<int32_t>(),
          redBuf_.template Get<int32_t>(),dim_);''')
    else:
        edit('#include "pack.h"','#include "../striped_store_simd_pack/pack.h"')
    if kind in ('writer_batch','rotate_batch'):
        edit('pipe_->InitBuffer(workBuf_,kMaxDim*4);',
             'pipe_->InitBuffer(workBuf_,hadamard ? kRowTile*kMaxDim*4 : kMaxDim*4);')
        edit(OLD_LOOP,NEW_LOOP)
    if kind=='writer_batch':
        edit('rotate_clip_store_striped_simd_pack_init','rotate_clip_store_striped_batch8',3)
        edit('rotate_striped_simd_pack_init_aux','rotate_striped_batch8_aux',3)
    else:
        suffix='batch8_init' if kind=='rotate_init' else 'batch8'
        edit('oscar_rotate_clip_store_kernel','oscar_store_'+suffix+'_aux_kernel',2)
        edit('rotate_clip_store_launch','rotate_clip_store_'+suffix+'_aux_launch')
        edit('oscar_rotate_kernel','oscar_rotate_'+suffix+'_kernel',2)
        edit('rotate_launch','rotate_'+suffix+'_launch')
    if re.search(r'__aicore__\b[^{};]*\b__aicore__\b',masked(text)):
        raise ValueError('duplicate function qualifier')
    restored=text
    for old,new,count in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse edit ambiguity '+kind)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('source restoration failed')
    return HEADER+text,{'source':source,'source_sha256':sha(original),'original_bytes_restored':True,
        'math_change':'row-independent stage scheduling only' if kind!='rotate_init' else 'none; Init metadata only'}


def generated():return {name+'.cpp':kernel(name)[0] for name in ('writer_batch','rotate_init','rotate_batch')}


def source_binding():
    files=[OLD,CURRENT,CANONICAL]+['csrc/experiments/hadamard_batch8/'+n for n in generated()]
    return {'scope':'complete writer old/current/compact and canonical rotate old/Init/compact',
        'frozen_writer_binding':frozen.source_binding(),
        'sources':{p:sha((ROOT/p).read_bytes()) for p in files},
        'dependencies':{p:sha((ROOT/p).read_bytes()) for p in (
            'csrc/experiments/striped_store_simd_pack/pack.h',
            'csrc/experiments/hadamard_indices/vector_indices.h')}}


def check():
    for name,value in generated().items():
        if (HERE/name).read_text()!=value:raise ValueError('generated source changed '+name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--manifest',type=Path);a=p.parse_args()
    if not a.check:
        for name,value in generated().items():(HERE/name).write_text(value)
    check()
    if a.manifest:a.manifest.write_text(json.dumps(source_binding(),indent=2)+'\n')
    print('compact Hadamard source restoration passed; production unchanged; no execution')
