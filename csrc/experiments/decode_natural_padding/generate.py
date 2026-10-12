"""Archive #126/#145/#148/#151/#154/#155, D.4: test actual SIMD row strides.

The current ShiftRight/And access eight natural INT16 rows with 512-byte
spacing. Give only this temporary a 544-byte row and cast each 64-column
slice back into the original contiguous FP32 buffer. Bank relief remains a
hypothesis until matched CANN CAModel; dtype/arithmetic/cache format stay.
"""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
PIN='c67219ce15c98e932acaa8b1c5515a3f31c99dd6e4963823623528ba2b3a3930'

def main():
    kernel=(ROOT/'csrc/kernels/attention_cv_striped_decode_simd.cpp').read_text()
    assert hashlib.sha256(kernel.encode()).hexdigest()==PIN
    helper=(ROOT/'csrc/kernels/attention_striped_unpack_simd.h').read_text()
    assert hashlib.sha256(helper.encode()).hexdigest()=='8baeac1d37395bd445f87b1f913b85e6d61bd1c681f5df0cfa21d04e0c6396ad'
    def change(text,old,new,count=1):
        assert text.count(old)==count,(old[:65],text.count(old),count)
        return text.replace(old,new)
    helper=change(helper,'#include "oscar_common.h"','#include "../../kernels/oscar_common.h"')
    helper=change(helper,'oscar_ascend_striped_simd','oscar_ascend_padded_natural',2)
    helper=change(helper,'  constexpr int32_t elements=Geometry<D,HalfKv>::elements;',
        '  constexpr int32_t elements=Geometry<D,HalfKv>::elements;\n'
        '  constexpr int32_t naturalStride=D+16;')
    helper=change(helper,'static_cast<uint16_t>(D/16),','static_cast<uint16_t>(naturalStride/16),')
    helper=change(helper,'static_cast<uint8_t>(8*D/16),','static_cast<uint8_t>(8*naturalStride/16),',3)
    helper=change(helper,'static_cast<uint8_t>(D/16),','static_cast<uint8_t>(naturalStride/16),',2)
    helper=change(helper,'  Cast(values,natural,RoundMode::CAST_NONE,elements);',
        '  const UnaryRepeatParams castStride{1,1,static_cast<uint8_t>(D/8),\n'
        '      static_cast<uint8_t>(naturalStride/16)};\n'
        '  for(int32_t chunk=0;chunk<D/64;++chunk)\n'
        '    Cast(values[chunk*64],natural[chunk*64],RoundMode::CAST_NONE,\n'
        '        static_cast<uint64_t>(64),static_cast<uint8_t>(HalfKv),castStride);')
    kernel=change(kernel,'#include "oscar_common.h"','#include "../../kernels/oscar_common.h"')
    kernel=change(kernel,'#include "attention_striped_unpack_simd.h"','#include "padded_unpack.h"')
    kernel=change(kernel,'"../include/','"../../include/',2)
    kernel=change(kernel,'oscar_ascend_striped_simd::','oscar_ascend_padded_natural::',2)
    kernel=change(kernel,'pipe.InitBuffer(naturalBuf,kElements*2);',
        'pipe.InitBuffer(naturalBuf,kHalfKv*(D+16)*2);')
    for old,new in (
        ('oscar_attention_cv_striped_decode_simd_kernel','oscar_natural_pad_q4_kernel'),
        ('oscar_attention_cv_striped_q1_simd_kernel','oscar_natural_pad_q1_kernel'),
        ('attention_cv_striped_decode_simd_launch','natural_pad_q4_launch'),
        ('attention_cv_striped_q1_simd_launch','natural_pad_q1_launch')):
        kernel=change(kernel,old,new,2 if old.startswith('oscar_') else 1)
    header='''// ISOLATED EXPERIMENT. Archive #126/#145/#148/#151/#154/#155; startup D.4.
// 1. Existing SIMD INT2 source0 dequant inside full q4/q1 fused attention.
// 2. D.4 restore cost 6499.8-6655.1ms/card; no history-sized tensor here.
// 3. Natural INT16 rows change512->544B, removing power-of-two row stride
//    from the eight-row bit operations. Four exact Cast slices restore the
//    original FP32 layout before unchanged metadata guards and Mul/Add.
//    Extra UB2048B (D256 total148416B), unchanged GM/workspace/task/flags.
// 4. Bank-conflict relief is a hypothesis, not an 8-minute or NPU claim.
'''
    (OUT/'padded_unpack.h').write_text(header+helper)
    (OUT/'attention_natural_pad.cpp').write_text(header+kernel)
    template=(ROOT/'csrc/experiments/decode_static_matmul/CMakeLists.txt').read_text()
    cmake=template.replace('oscar_decode_static','oscar_natural_pad').replace('OSCAR_DECODE_STATIC','OSCAR_NATURAL_PAD')
    cmake=cmake.replace('attention_static.cpp','attention_natural_pad.cpp').replace('static_decode','natural_pad')
    cmake=cmake.replace('oscar_decode_static_q4_kernel','oscar_natural_pad_q4_kernel').replace('oscar_decode_static_q1_kernel','oscar_natural_pad_q1_kernel')
    (OUT/'CMakeLists.txt').write_text(cmake)
    print('NATURAL_PADDING_GENERATED same FP32 shape, plus2048B UB; no production change')

if __name__=='__main__':main()
