"""Apply the already bit-exhausted range mask to CURRENT full SIMD decode.

Archive #126/#129/#140/#145/#148/#151/#154/#155 and complete startup D.4.
New 2026-10-07 instruction evidence: q4 source0 Softmax calls 7178/5537/85416
cycles for two full tiles and one partial tile. This is not an old profile
fraction. Mask only the complement intervals, preserving old pre-mask finite
check, V2, FP32 arithmetic and KV order. Full A/B must establish net gain.
"""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def main():
    text=(ROOT/'csrc/kernels/attention_cv_striped_decode_simd.cpp').read_text()
    assert hashlib.sha256(text.encode()).hexdigest()=='c67219ce15c98e932acaa8b1c5515a3f31c99dd6e4963823623528ba2b3a3930'
    def edit(old,new,count=1):
        nonlocal text
        assert text.count(old)==count,(old[:65],text.count(old),count)
        text=text.replace(old,new)
    edit('#include "oscar_common.h"','#include "../../kernels/oscar_common.h"\n#include "../range_mask/range_mask.h"')
    edit('#include "attention_striped_unpack_simd.h"','#include "../../kernels/attention_striped_unpack_simd.h"')
    edit('"../include/','"../../include/',2)
    edit('''        Fence<HardEvent::S_V>();
        for(int32_t j=0;j<kKvRows;++j) {
          const bool visible=(j>=lo0&&j<hi0)||(j>=lo1&&j<hi1);
          if(!visible)scores.SetValue(r*kKvRows+j,kNegativeInfinity);
        }
        Fence<HardEvent::S_V>();''',
        '''        // The original ReduceSum V->S above completed score reads.
        // Scalar edges precede disjoint aligned Duplicate interiors.
        oscar_range_mask::MaskComplement(scores[r*kKvRows],lo0,hi0,lo1,hi1);
        Fence<HardEvent::V_S>();''')
    for old,new in (
        ('oscar_attention_cv_striped_decode_simd_kernel','oscar_range_mask_q4_kernel'),
        ('oscar_attention_cv_striped_q1_simd_kernel','oscar_range_mask_q1_kernel'),
        ('attention_cv_striped_decode_simd_launch','range_mask_q4_launch'),
        ('attention_cv_striped_q1_simd_launch','range_mask_q1_launch')):
        edit(old,new,2 if old.startswith('oscar_') else 1)
    header='''// ISOLATED FULL DECODE EXPERIMENT. Archive #126/#129/#140/#145/#148/#151/#154/#155.
// D.4 four questions: 1. pre-softmax causal/tail mask in full q4/q1 attention.
// 2. No repetition of the failed 6499.8-6655.1ms full-history restore.
// 3. Existing source intervals become <=3 masked ranges. Aligned FP32
// Duplicate replaces per-column scalar writes; <=14 edge words/range remain.
// Original pre-mask ReduceSum, all float math, task/status, KV256 and memory
// sizes stay unchanged. Explicit V->S retires the added vector mask writes.
// 4. Current marker trace motivates this; it does not prove whole-run speed.
'''
    (OUT/'attention_range_mask.cpp').write_text(header+text)
    cmake=(ROOT/'csrc/experiments/decode_static_matmul/CMakeLists.txt').read_text()
    cmake=cmake.replace('oscar_decode_static','oscar_range_mask_full').replace('OSCAR_DECODE_STATIC','OSCAR_RANGE_MASK_FULL')
    cmake=cmake.replace('attention_static.cpp','attention_range_mask.cpp').replace('static_decode','range_mask_full')
    cmake=cmake.replace('oscar_range_mask_full_q4_kernel','oscar_range_mask_q4_kernel').replace('oscar_range_mask_full_q1_kernel','oscar_range_mask_q1_kernel')
    (OUT/'CMakeLists.txt').write_text(cmake)
    print('FULL_RANGE_MASK_GENERATED current SIMD source, unchanged attention math')

if __name__=='__main__':main()
