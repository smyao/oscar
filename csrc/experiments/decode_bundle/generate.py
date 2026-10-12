"""Compose independently verified decode mechanisms for a full-kernel trial.

History retains KV256 FP32 pipeline (including its padded last tile); precise
window/current alone use shortened zero-padded Cube widths. Thus clearing a
tail score plane cannot race with the history pipeline's previous PV reader.
"""
from pathlib import Path
import sys
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from csrc.experiments.prefill_window_range.generate import full_method
OUT=Path(__file__).resolve().parent


def main():
    pipeline=ROOT/'csrc/experiments/decode_history_pipeline/attention_pipeline.cpp'
    text=pipeline.read_text()
    assert hashlib.sha256(text.encode()).hexdigest()=='f9cb6862783dc2289944405766d3cbc56101e24bf538b6da913fe082e0ccb372'
    window=(ROOT/'csrc/experiments/window_batch_decode/window_batch_decode.cpp').read_text()
    tail=(ROOT/'csrc/experiments/decode_tail_range/attention_tail_range.cpp').read_text()
    def edit(old,new,count=1):
        nonlocal text
        assert text.count(old)==count,(old[:70],text.count(old),count)
        text=text.replace(old,new)
    edit('#include "../../kernels/attention_striped_unpack_simd.h"',
         '#include "../decode_natural_padding/padded_unpack.h"\n#include "../range_mask/range_mask.h"')
    edit('oscar_ascend_striped_simd::','oscar_ascend_padded_natural::',3)
    edit('pipe.InitBuffer(naturalBuf,kElements*2);','pipe.InitBuffer(naturalBuf,kHalfKv*(D+16)*2);')
    edit(full_method(text,'LoadPrecise'),
         full_method(window,'LoadWindowRuns')+'\n'+full_method(window,'LoadPrecise'))
    edit(full_method(text,'Matmul'),full_method(tail,'Matmul'))
    helpers=full_method(tail,'TailColumns')+'\n'+full_method(tail,'ClearPaddedScores')+'\n'
    helpers=helpers.replace('    return static_cast<int32_t>((Min64(kKvRows,kvend-start)+15)/16*16);',
        '    if(kind==0)return kKvRows; // history pipeline shared score/P plane remains unchanged\n'
        '    return static_cast<int32_t>((Min64(kKvRows,kvend-start)+15)/16*16);')
    edit(full_method(text,'CubeTask'),helpers+full_method(tail,'CubeTask'))
    edit(full_method(text,'VectorTask'),full_method(tail,'VectorTask'))
    edit('''        Fence<HardEvent::S_V>();
        for(int32_t j=0;j<kKvRows;++j) {
          const bool visible=(j>=lo0&&j<hi0)||(j>=lo1&&j<hi1);
          if(!visible)scores.SetValue(r*kKvRows+j,kNegativeInfinity);
        }
        Fence<HardEvent::S_V>();''',
        '''        oscar_range_mask::MaskComplement(scores[r*kKvRows],lo0,hi0,lo1,hi1);
        Fence<HardEvent::V_S>();''')
    for old,new in (
        ('oscar_attention_history_pipeline_q4_kernel','oscar_decode_bundle_q4_kernel'),
        ('oscar_attention_history_pipeline_q1_kernel','oscar_decode_bundle_q1_kernel'),
        ('attention_history_pipeline_q4_launch','decode_bundle_q4_launch'),
        ('attention_history_pipeline_q1_launch','decode_bundle_q1_launch')):
        edit(old,new,2 if old.startswith('oscar_') else 1)
    header='''// ISOLATED COMBINED DECODE TRIAL. Archive #126/#129/#140/#145/#148/#151/#154/#155.
// D.4 four questions: complete source0/1/2 q4/q1 FIA; no full-history restore
// repeating6499.8-6655.1ms. Fixed history KV256 pipeline + padded natural
// INT16 scratch + batched exact-window reads + exact range masks. Only serial
// precise sources omit padded-zero Cube columns. No new score clearing in
// pipelined history, preserving its score/P and PV ownership proof.
// UB158656B/AIV, GM1146880B/core, independent of history. No production/NPU
// performance claim: combined numerical and complete cycle gates still apply.
'''
    (OUT/'attention_bundle.cpp').write_text(header+text)
    cmake=(ROOT/'csrc/experiments/decode_history_pipeline/CMakeLists.txt').read_text()
    cmake=cmake.replace('oscar_decode_history_pipeline','oscar_decode_bundle').replace('OSCAR_DECODE_PIPELINE','OSCAR_DECODE_BUNDLE')
    cmake=cmake.replace('attention_pipeline.cpp','attention_bundle.cpp').replace('history_pipeline','decode_bundle')
    cmake=cmake.replace('oscar_attention_decode_bundle_q4_kernel','oscar_decode_bundle_q4_kernel').replace('oscar_attention_decode_bundle_q1_kernel','oscar_decode_bundle_q1_kernel')
    (OUT/'CMakeLists.txt').write_text(cmake)
    refs=[pipeline,ROOT/'csrc/experiments/window_batch_decode/window_batch_decode.cpp',
          ROOT/'csrc/experiments/decode_tail_range/attention_tail_range.cpp',
          ROOT/'csrc/experiments/decode_natural_padding/padded_unpack.h',
          ROOT/'csrc/experiments/range_mask/range_mask.h']
    (OUT/'source_identity.json').write_text(json.dumps({
        'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in refs},
        'output_sha256':hashlib.sha256((header+text).encode()).hexdigest(),
        'history_score_clearing':'unchanged_no_tail_shortening_in_pipeline',
        'workspace_per_core':1146880,'UB_per_AIV':158656,'production_enabled':False},indent=2)+'\n')
    print('DECODE_BUNDLE_GENERATED pipeline/window/range/natural-padding; tail only precise sources')


if __name__=='__main__':main()
