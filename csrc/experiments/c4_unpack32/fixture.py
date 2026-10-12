"""Archive #13-22/#126/#145/#154/#155; D.4: independent exact INT2 helper data.

No device execution. Tests all 65536 packed-word patterns, every live tail,
boundary metadata and invalid scale/zero. Oracle is PR q*FP32(scale)+zero,
with two distinct FP32 roundings; never a fused or BF16 reconstruction.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

HERE=Path(__file__).resolve().parent
SCALES=(0x0001,0x0002,0x03ff,0x0400,0x3555,0x3800,0x3c00,0x7bff)
ZEROS=(0,0x8000,0x0001,0x8001,0x3c00,0xbc00,0x7bff,0xfbff)
FAULTS=(0x7c00,0xfc00,0x7e01,0xfe01,0,0x8000,0x8001,0xbc00)
def half(bits):return struct.unpack('<e',struct.pack('<H',bits))[0]
def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
def sha(value):return hashlib.sha256(value).hexdigest()


def cases():
    result=[(f'all_words_{i:02d}',32,None) for i in range(64)]
    result += [(f'live_{n:02d}',n,None) for n in range(33)]
    result += [(f'dead_nan_{n:02d}',n,(n,0,0x7e01)) for n in range(32)]
    result += [(f'live_negative_zero_{n:02d}',n,(n-1,2,0x8000)) for n in range(1,33)]
    result += [(f'meta_{channel}_{i}',32,(0 if i%2==0 else 31,channel,bits))
               for channel in range(4) for i,bits in enumerate(FAULTS)]
    return result


def make(index,live,fault):
    packed=bytearray(b'\xa5'*(32*160));expected=[];status=0
    for row in range(32):
        for word in range(32):
            value=(index*1024+row*32+word)&65535
            struct.pack_into('<H',packed,row*160+word*2,value)
            struct.pack_into('<H',packed,row*160+64+word*2,65535-value)
        metadata=[SCALES[row%8],ZEROS[(row+index)%8],SCALES[(row+3)%8],ZEROS[(row+index+5)%8]]
        if fault and row==fault[0]:metadata[fault[1]]=fault[2]
        struct.pack_into('<4H',packed,row*160+128,*metadata)
        if row<live and any(not (math.isfinite(half(metadata[s])) and half(metadata[s])>0 and
                                 math.isfinite(half(metadata[s+1]))) for s in (0,2)):status=3
    for side in range(2):
        for row in range(32):
            scale,zero=struct.unpack_from('<2e',packed,row*160+128+4*side)
            for d in range(256):
                word=struct.unpack_from('<H',packed,row*160+64*side+(d%32)*2)[0]
                level=(word>>(2*(d//32)))&3
                expected.append(f32(f32(float(level)*scale)+zero))
    return bytes(packed),struct.pack('<i',live),struct.pack('<16384f',*expected),struct.pack('<i',status)


def export(path,selection=None,repeat=1):
    if path.exists():raise ValueError('new fixture directory required')
    chosen=[(i,*row) for i,row in enumerate(cases()) if selection is None or row[0]==selection]
    if not chosen:raise ValueError('unknown case')
    if type(repeat) is not int or repeat<1 or len(chosen)*repeat>256:raise ValueError('at most 256 declared cases')
    chosen*=repeat
    buffers=[bytearray() for _ in range(4)];rows=[]
    for i,name,live,fault in chosen:
        parts=make(i,live,fault)
        for dest,part in zip(buffers,parts):dest.extend(part)
        rows.append({'name':name,'live_rows':live,'fault':fault,'expected_status':struct.unpack('<i',parts[3])[0]})
    path.mkdir(parents=True)
    names=('packed','live','expected','expected_status');hashes={}
    for name,data in zip(names,buffers):(path/(name+'.bin')).write_bytes(data);hashes[name]=sha(data)
    (path/'shape.txt').write_text(str(len(rows))+'\n')
    manifest={'scope':'32 logical rows D256 K+V helper, not full attention','cases':rows,'case_count':len(rows),
        'sha256':hashes,'oracle':'separate FP32 Mul then Add; exact finite/signed-zero bits; nonfinite classes plus exact A/B bits',
        'all_word_patterns_covered':selection is None,'device_execution':'not_run'}
    (path/'fixture.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def check_guard():
    # Exhaustive host algebra check of signed-int16 compare masks against
    # independent IEEE binary16 validity, including every NaN/subnormal/zero.
    for bits in range(65536):
        signed=bits if bits<32768 else bits-65536;value=half(bits)
        assert (signed>0 and signed<31744)==(math.isfinite(value) and value>0)
        assert (signed<-1024 or (signed>=0 and signed<31744))==math.isfinite(value)
    for live in range(33):
        mask=(1<<live)-1
        for row in range(32):assert bool(mask&(1<<row))==(row<live)
    return {'all_fp16_patterns':65536,'tails':33,'status':'passed_host_only'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--case');p.add_argument('--repeat',type=int,default=1);p.add_argument('--check-guard',action='store_true');a=p.parse_args()
    if a.check_guard:print(json.dumps(check_guard()))
    else:
        if a.output is None:p.error('--output required')
        result=export(a.output,a.case,a.repeat);print(json.dumps({'cases':result['case_count'],'output':str(a.output),'device_execution':'not_run'}))
