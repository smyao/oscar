"""Archive #126/#140/#145/#154, D.4: atomic -> normal mode witness.

Preserve v1/v2 exactly. Revision3 appends two GM outputs and two ordinary
transposed Matmuls around v2's unchanged PV/add/atomic/immediate readback.
No numerical threshold change, SDK edit, production registration or run.
"""
import argparse
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
V2_KERNEL_SHA='bcd752fbf0448fa61c18aab9e1bd620e5de5fe202fbd81a26f6834eea590ce93'
V2_DRIVER_SHA='67efd092e98f8cec1471d10e2dfec7bb45c89ce200cfece7277c37d4ea80b1a0'


def replace(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError('v2 source drift at '+old[:90])
    return text.replace(old,new)


def generated():
    kernel=(HERE/'kernel_readback.cpp').read_text()
    driver=(HERE/'probe_readback.cpp').read_text()
    if hashlib.sha256(kernel.encode()).hexdigest()!=V2_KERNEL_SHA:raise ValueError('v2 kernel changed')
    if hashlib.sha256(driver.encode()).hexdigest()!=V2_DRIVER_SHA:raise ValueError('v2 driver changed')
    kernel=replace(kernel,'GM_ADDR status,GM_ADDR atomic_readback)',
                   'GM_ADDR status,GM_ADDR atomic_readback,GM_ADDR normal_before,GM_ADDR normal_after)',2)
    kernel=replace(kernel,'    status_.SetGlobalBuffer',
'''    before_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normal_before));
    after_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normal_after));
    status_.SetGlobalBuffer''')
    kernel=replace(kernel,'      Matmul(pv_,0);',
'''      // Revision3: ordinary transposed output before any atomic call.
      NormalTransposed(before_);
      Matmul(pv_,0);''')
    kernel=replace(kernel,'      CrossCoreWaitFlag(kVectorsDone);',
'''      CrossCoreWaitFlag(kVectorsDone);
      // Both AIV immediate readbacks are complete. The same mm_ must now
      // overwrite an output prefilled with 1.0 using DEFAULT atomic=0.
      NormalTransposed(after_);''')
    kernel=replace(kernel,'  TPipe pipe_;CubeMatmul mm_;',
'''  __aicore__ void NormalTransposed(const GlobalTensor<float>& output) {
    SetHF32Mode(false);
    mm_.SetOrgShape(kRows,kCols,kK);mm_.SetSingleShape(kRows,kCols,kK);
    mm_.SetTensorA(p_,false);mm_.SetTensorB(v_,true);
    mm_.template IterateAll<true>(output); // SDK default enAtomic=0
    mm_.End();
  }
  TPipe pipe_;CubeMatmul mm_;''')
    kernel=replace(kernel,'atomic_,readback_;','atomic_,readback_,before_,after_;')
    kernel=replace(kernel,'status,atomic_readback);probe.Run();',
                   'status,atomic_readback,normal_before,normal_after);probe.Run();')
    kernel=kernel.replace('oscar_c4_atomic_state_readback_kernel','oscar_c4_atomic_state_recovery_kernel')
    kernel='''// Independent revision3. Archive #126/#140/#145/#154 and complete D.4.
// D.4: FIA state update; no historical restore; same one Matmul/HF32-off,
// plus ordinary P@V.T before/after atomic; this is a legality witness only.
// v1/v2 files and evidence remain unchanged. New outputs have separate GM.
'''+kernel

    driver=replace(driver,'uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);',
                   'uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);')
    driver=replace(driver,'readback(MP*N*4);','readback(MP*N*4),before(MP*N*4),after(MP*N*4);')
    driver=replace(driver,'    AscendC::SetKernelMode',
'''    before.Load(fixture/"before_initial.bin");after.Load(fixture/"after_initial.bin");
    for(size_t i=0;i<MP*N;++i)
      if(Word(before,i)!=0x85858585U||Word(after,i)!=0x3f800000U)
        throw std::runtime_error("normal recovery initial bytes differ");
    AscendC::SetKernelMode''')
    driver=replace(driver,'status.ptr,readback.ptr);','status.ptr,readback.ptr,before.ptr,after.ptr);')
    driver=replace(driver,'readback.Save(output,"atomic_readback");',
                   'readback.Save(output,"atomic_readback");before.Save(output,"normal_before");after.Save(output,"normal_after");')
    driver=replace(driver,'&&readback.Guards();','&&readback.Guards()&&before.Guards()&&after.Guards();')
    driver=replace(driver,'    const bool pass=',
'''    size_t normalLiveDiff=0,normalBeforeTail=0,normalAfterTail=0,normalNonfinite=0;
    size_t normalFirst=MP*N;
    for(size_t i=0;i<MP*N;++i) {
      const auto x=Word(before,i),y=Word(after,i);
      if(i<M*N) {
        normalLiveDiff+=x!=y;normalNonfinite+=Class(x)!=0||Class(y)!=0;
        if(x!=y&&normalFirst==MP*N)normalFirst=i;
      } else {normalBeforeTail+=x!=0x85858585U;normalAfterTail+=y!=0x3f800000U;}
    }
    const std::string normal="{\\\"live_bit_differences\\\":"+std::to_string(normalLiveDiff)+
      ",\\\"before_padding_differences\\\":"+std::to_string(normalBeforeTail)+
      ",\\\"after_padding_differences\\\":"+std::to_string(normalAfterTail)+
      ",\\\"nonfinite_live_elements\\\":"+std::to_string(normalNonfinite)+
      ",\\\"first_difference\\\":"+(normalFirst==MP*N?"null":std::to_string(normalFirst))+"}";
    const bool pass=''')
    driver=replace(driver,'&&!rbAtomicTail&&!rbReferenceTail;',
                   '&&!rbAtomicTail&&!rbReferenceTail&&!normalLiveDiff&&!normalBeforeTail&&!normalAfterTail&&!normalNonfinite;')
    driver=replace(driver,r'''",\"revision\":2,\"atomic_readback\":"+rb+''',
                   r'''",\"revision\":3,\"normal_recovery\":"+normal+",\"atomic_readback\":"+rb+''')
    driver=driver.replace('oscar_c4_atomic_state_readback_kernel','oscar_c4_atomic_state_recovery_kernel')
    driver=driver.replace('oscar_c4_atomic_state_readback_cpu','oscar_c4_atomic_state_recovery_cpu')
    driver='// Independent revision3: immediate visibility and atomic -> normal recovery.\n'+driver
    return {'kernel_recovery.cpp':kernel,'probe_recovery.cpp':driver}


def check():
    result=generated()
    for name,text in result.items():
        if not (HERE/name).is_file() or (HERE/name).read_text()!=text:raise ValueError('revision3 drift: '+name)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:check();print('atomic recovery revision3 identity passed; no execution')
    else:
        for name,text in generated().items():(HERE/name).write_text(text)
        print('generated revision3; v1/v2 preserved; no execution')
