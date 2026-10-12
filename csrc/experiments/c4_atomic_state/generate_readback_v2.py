"""Archive #126/#145/#154, D.4: explicit visibility-only revision of the probe.

Preserve v1 sources/evidence. Generate separate 8-pointer files; mathematics,
shapes, flag IDs and complete-K Matmul are unchanged. No compiler/kernel run.
"""
import argparse
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
V1_KERNEL_SHA='7145f951e9e7b073c2918a3968986394999effd4d7d897526504c612ec1a4f51'


def replace(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError('v1 source drift at '+old[:80])
    return text.replace(old,new)


def generated():
    kernel=(HERE/'kernel.cpp').read_text()
    if hashlib.sha256(kernel.encode()).hexdigest()!=V1_KERNEL_SHA:raise ValueError('v1 kernel source drift')
    kernel=replace(kernel,'GM_ADDR reference,GM_ADDR atomic,GM_ADDR status)',
                   'GM_ADDR reference,GM_ADDR atomic,GM_ADDR status,GM_ADDR atomic_readback)',2)
    kernel=replace(kernel,'    status_.SetGlobalBuffer',
                   '    readback_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(atomic_readback));\n    status_.SetGlobalBuffer')
    kernel=replace(kernel,'      CrossCoreWaitFlag(kAtomicDone);',
'''      CrossCoreWaitFlag(kAtomicDone);
      // Revision2: verify immediate cross-core MTE2 visibility after PIPE_FIX.
      // acc's reference writeback completed before ReferenceDone above.
      DataCopy(acc,atomic_[first*kCols],kLaneRows*kCols);
      Fence<HardEvent::MTE2_MTE3>();
      DataCopy(readback_[first*kCols],acc,kLaneRows*kCols);''')
    kernel=replace(kernel,'p_,v_,alpha_,pv_,reference_,atomic_;','p_,v_,alpha_,pv_,reference_,atomic_,readback_;')
    kernel=replace(kernel,'probe.Init(p,v,alpha,pv,reference,atomic,status);',
                   'probe.Init(p,v,alpha,pv,reference,atomic,status,atomic_readback);')
    kernel=replace(kernel,'oscar_c4_atomic_state_kernel(','oscar_c4_atomic_state_readback_kernel(')
    kernel='// Independent revision2: append readback only; v1 kernel.cpp remains unchanged.\n'+kernel

    driver=(HERE/'probe.cpp').read_text()
    driver=replace(driver,'uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);',
                   'uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);')
    driver=replace(driver,'status(3*64);','status(3*64),readback(MP*N*4);')
    driver=replace(driver,'atomic.ptr,status.ptr);','atomic.ptr,status.ptr,readback.ptr);')
    driver=replace(driver,'status.Save(output,"status");','status.Save(output,"status");readback.Save(output,"atomic_readback");')
    driver=replace(driver,'&&status.Guards();','&&status.Guards()&&readback.Guards();')
    driver=replace(driver,'    std::string first="[";',
                   '    size_t rbAtomic=0,rbReference=0,rbAtomicTail=0,rbReferenceTail=0;\n    std::string first="[";')
    driver=replace(driver,'      const auto left=Word(reference,i),right=Word(atomic,i);',
'''      const auto left=Word(reference,i),right=Word(atomic,i),read=Word(readback,i);
      if(i<M*N) {rbAtomic+=read!=right;rbReference+=read!=left;}
      else {rbAtomicTail+=read!=right;rbReferenceTail+=read!=left;}''')
    driver=replace(driver,'      if(left==right)continue;','      if(left==right&&read==right)continue;')
    driver=replace(driver,'      if(i<M*N)++liveDiff;else ++tailDiff;',
                   '      if(left!=right) {if(i<M*N)++liveDiff;else ++tailDiff;}')
    driver=replace(driver,r'''+std::to_string(right)+"}";''',
                   r'''+std::to_string(right)+",\"readback_bits\":"+std::to_string(read)+"}";''')
    driver=replace(driver,'&&anti&&!liveDiff&&!tailDiff;',
                   '&&anti&&!liveDiff&&!tailDiff&&!rbAtomic&&!rbReference&&!rbAtomicTail&&!rbReferenceTail;')
    driver=replace(driver,'    const std::string report=',
'''    const std::string rb="{\\\"live_vs_atomic\\\":"+std::to_string(rbAtomic)+
      ",\\\"live_vs_reference\\\":"+std::to_string(rbReference)+
      ",\\\"padding_vs_atomic\\\":"+std::to_string(rbAtomicTail)+
      ",\\\"padding_vs_reference\\\":"+std::to_string(rbReferenceTail)+"}";
    const std::string report=''')
    driver=replace(driver,r'''",\"first_differences\":"+first+''',
                   r'''",\"revision\":2,\"atomic_readback\":"+rb+",\"first_differences\":"+first+''')
    driver=driver.replace('oscar_c4_atomic_state_kernel','oscar_c4_atomic_state_readback_kernel')
    driver=driver.replace('oscar_c4_atomic_state_cpu','oscar_c4_atomic_state_readback_cpu')
    driver='// Independent revision2: cross-core atomic state visibility; preserve v1 evidence.\n'+driver

    replay=(HERE/'replay.py').read_text()
    replay=replace(replay,'import fixture as cases','import fixture_readback as cases')
    replay=replace(replay,"    pv=struct.unpack('<32768I',outputs['pv']);live=126*256",
                   "    readback=struct.unpack('<32768I',outputs['atomic_readback']);rb=[0,0,0,0]\n    pv=struct.unpack('<32768I',outputs['pv']);live=126*256")
    replay=replace(replay,'        lc,rc=classification(left),classification(right)',
'''        read=readback[index];offset=0 if index<live else 2
        rb[offset]+=read!=right;rb[offset+1]+=read!=left
        lc,rc=classification(left),classification(right)''')
    replay=replace(replay,'        if left==right:continue','        if left==right and read==right:continue')
    replay=replace(replay,'        if index<live:live_diff+=1\n        else:padding_diff+=1',
                   '        if left!=right:\n            if index<live:live_diff+=1\n            else:padding_diff+=1')
    replay=replace(replay,"'reference_class':lc,'atomic_class':rc}",
                   "'reference_class':lc,'atomic_class':rc,'readback_bits':hex(read)}")
    replay=replace(replay,'and anti and actors else','and anti and actors and not any(rb) else')
    replay=replace(replay,"        'reference_classes':reference_classes,",
                   "        'atomic_readback':dict(zip(('live_vs_atomic','live_vs_reference','padding_vs_atomic','padding_vs_reference'),rb)),\n        'reference_classes':reference_classes,")
    replay=replace(replay,"report={'status':'initializing'","report={'revision':2,'status':'initializing'")
    replay=replace(replay,"kernel='oscar_c4_atomic_state_kernel'","kernel='oscar_c4_atomic_state_readback_kernel'")
    replay=replace(replay,'[C.c_void_p]*7;launch.restype','[C.c_void_p]*8;launch.restype')
    replay=replace(replay,"'atomic':128*256*4,'status':3*64}","'atomic':128*256*4,'status':3*64,'atomic_readback':128*256*4}")
    replay=replace(replay,"('p','v','alpha','pv','reference','atomic','status')",
                   "('p','v','alpha','pv','reference','atomic','status','atomic_readback')")
    return {'kernel_readback.cpp':kernel,'probe_readback.cpp':driver,'replay_readback.py':replay}


def check():
    result=generated()
    for name,text in result.items():
        if not (HERE/name).is_file() or (HERE/name).read_text()!=text:raise ValueError('readback revision drift: '+name)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:check();print('atomic readback revision2 identity passed; no execution')
    else:
        for name,text in generated().items():(HERE/name).write_text(text)
        print('generated independent atomic readback revision2; v1 preserved')
