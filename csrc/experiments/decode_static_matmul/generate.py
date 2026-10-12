"""Archive #126/#129/#140/#145/#148/#154/#155, D.4: fixed decode geometry.

Current MMAD really has M6/M24; this is NOT removing imaginary M64 FLOPs.
Pinned SDK MLoopNormBase::SetSingleShapeFromCFG removes runtime shape division
when singleCoreM is fixed and enableSetTail=false. GetMatmulApiTiling sets the
same original M/N/K when enableSetOrgShape=false. Try compiler specialization
only; unchanged FP32 K128 accumulation, no HF32. Full matched timing decides.
"""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'csrc/kernels/attention_cv_striped_decode_simd.cpp'
PIN='c67219ce15c98e932acaa8b1c5515a3f31c99dd6e4963823623528ba2b3a3930'

def main():
    text=SOURCE.read_text()
    assert hashlib.sha256(text.encode()).hexdigest()==PIN
    def edit(old,new,count=1):
        nonlocal text
        assert text.count(old)==count,(old[:65],text.count(old),count)
        text=text.replace(old,new)
    edit('#include "oscar_common.h"','#include "../../kernels/oscar_common.h"')
    edit('#include "attention_striped_unpack_simd.h"','#include "../../kernels/attention_striped_unpack_simd.h"')
    edit('"../include/','"../../include/',2)
    edit('MatmulConfig CvConfig()','MatmulConfig CvConfig(int32_t rows)')
    edit('cfg.singleCoreM=128;cfg.singleCoreN=256;cfg.singleCoreK=256;',
         'cfg.singleCoreM=rows;cfg.singleCoreN=256;cfg.singleCoreK=256;\n'
         '  cfg.enableSetTail=false;cfg.enableSetOrgShape=false;')
    old='constexpr auto kCubeConfig=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(CvConfig());\nusing CubeMatmul=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kCubeConfig>;'
    edit(old,'')
    edit('template<int32_t D,int32_t OwnerTile> class AttentionCv {',
         'template<int32_t D,int32_t OwnerTile> class AttentionCv {\n'
         '  static constexpr int32_t kMatrixRows=OwnerTile*6;\n'
         '  static constexpr auto kCubeConfig=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(CvConfig(kMatrixRows));\n'
         '  using CubeMatmul=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kCubeConfig>;')
    edit('    mm.SetOrgShape(m,n,k);mm.SetSingleShape(m,n,k);',
         '    // N=K=D=256, M is compile-time 6 (q1) or 24 (q4).\n'
         '    // SetSingleShapeFromCFG retains a valid static M tail; no M64 execution claim.')
    edit('if(dim!=256) return;','if(dim!=256 || hk<=0 || hq!=6*hk) return;')
    edit('      // Matmul SetSingleShape(M=real rows) excludes this full block from PV.\n'
         '      // Keep the outer flag protocol, but publish no unused GM P rows.\n'
         '      return;',
         '      // q4 with fewer than four queries still executes static M24.\n'
         '      // Publish initialized P for those unconsumed output rows.\n'
         '      const int32_t fixedRows=static_cast<int32_t>(Max64(0,Min64(kHalfRows,kMatrixRows-rowBase)));\n'
         '      if(fixedRows>0) {\n'
         '        Duplicate(scores,0.0F,fixedRows*kKvRows);Fence<HardEvent::V_MTE3>();\n'
         '        DataCopy(work[pOffset+rowBase*kKvRows],scores,fixedRows*kKvRows);\n'
         '        Fence<HardEvent::MTE3_MTE2>();\n'
         '      }\n'
         '      return;')
    edit('DataCopy(work[pOffset+rowBase*kKvRows],scores,activeRows*kKvRows);',
         'DataCopy(work[pOffset+rowBase*kKvRows],scores,\n'
         '        static_cast<int32_t>(Min64(kHalfRows,kMatrixRows-rowBase))*kKvRows);')
    for before,after in (
        ('oscar_attention_cv_striped_decode_simd_kernel','oscar_decode_static_q4_kernel'),
        ('oscar_attention_cv_striped_q1_simd_kernel','oscar_decode_static_q1_kernel'),
        ('attention_cv_striped_decode_simd_launch','decode_static_q4_launch'),
        ('attention_cv_striped_q1_simd_launch','decode_static_q1_launch')):
        edit(before,after,2 if before.startswith('oscar_') else 1)
    header='''// ISOLATED STATIC-GEOMETRY EXPERIMENT, NOT SERVING.
// Archive #126/#129/#140/#145/#148/#154/#155 and startup D.4 four questions:
// 1. Full q4/q1 INT2 decode, unchanged three-source task/FP32 attention math.
// 2. D.4 failed full-history restore was 6499.8-6655.1ms; no such tensor here.
// 3. Same KV256/FP32/HF32=false. N/K=256, static M6 or M24 lets SDK remove
//    runtime shape logic. q4 tails initialize every added P row; outputs and
//    status still visit only original queries. No extra UB/workspace.
// 4. Extra padded q4 work may outweigh savings; no speed claim until A/B.
// Actual current MMAD is M6/M24, never claimed to waste M64 compute.
'''
    path=Path(__file__).with_name('attention_static.cpp')
    path.write_text(header+text)
    print(path.relative_to(ROOT))

if __name__=='__main__':main()
