"""Archive #126/#140/#145/#154, D.4: model-free FP32 PV/atomic falsification.

Host-only input generation. No CANN launch, simulated speed or precision claim.
Raw IEEE words preserve NaN payloads; acceptance never changes on observed data.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
ROWS,PADDED,COLS,K=126,128,256,256
SOURCE='csrc/kernels/attention_cv_window_range_cluster4.cpp'
PINS={SOURCE:'14b28f9181220432a15bcc3f23679e539473b12e01d858bbb00ff3345522ba65',
      'csrc/kernels/oscar_common.h':'20574a83f919df5bd6450cd581d1dba5a5a6f169768e7e237ba62f51b7499964'}
CASES=('anti_k128','random','ties','subnormal_zero','nonfinite')


def sha(data):return hashlib.sha256(data).hexdigest()


def config_block(text):
    begin=text.index('using Matrix=')
    end=text.index('using CubeMatmul=',begin)
    return text[begin:text.index(';',end)+1]


def check_source():
    for name,digest in PINS.items():
        if sha((ROOT/name).read_bytes())!=digest:raise ValueError('re-audit changed source: '+name)
    if config_block((ROOT/SOURCE).read_text())!=config_block((HERE/'kernel.cpp').read_text()):
        raise ValueError('atomic prototype changed the current C4 static Matmul config')
    return {'pinned_source_sha256':PINS,'kernel_sha256':sha((HERE/'kernel.cpp').read_bytes())}


def bits(value):return struct.unpack('<I',struct.pack('<f',value))[0]


def raw(values):
    data=array('I',values)
    if sys.byteorder!='little':data.byteswap()
    return data.tobytes()


def inputs(name):
    p=[0]*(PADDED*K);v=[0]*(K*COLS);alpha=[0]*(PADDED*COLS)
    if name=='anti_k128':
        p=[bits(1.)]*len(p)
        v=[bits(32768. if row<128 else -32768.) for row in range(K) for _ in range(COLS)]
        alpha=[bits(.25)]*len(alpha)
    elif name=='random':
        rng=random.Random(194126256)
        p=[bits(rng.randint(-64,64)/64) for _ in p]
        v=[bits(rng.randint(-64,64)/64) for _ in v]
        alpha=[bits(rng.randint(-64,64)/64) for _ in alpha]
    else:
        for row in range(PADDED):p[row*K]=bits(1.)
        if name=='ties':
            values=[bits(x) for x in (2**-24,3*2**-24,-2**-24,-3*2**-24,2**-25,-2**-25)]
            acc=[bits(1.),bits(-1.),bits(1.+2**-23),bits(-1.-2**-23)]
        elif name=='subnormal_zero':
            values=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000]
            acc=[0x80000000,0,0x007fffff,0x807fffff,0x00800000,0x80800000,1,0x80000001]
        elif name=='nonfinite':
            values=[0x7fc00001,0x7f800001,0xffc00017,0x7f800000,0xff800000,bits(1.),bits(-1.),0]
            acc=[0x7fc00123,0x7f800007,0xffc00001,0x7f800000,0xff800000,0,0x80000000,bits(1.)]
        else:raise ValueError('unknown case')
        for col in range(COLS):v[col]=values[col%len(values)]
        alpha=[acc[(row+col)%len(acc)] for row in range(PADDED) for col in range(COLS)]
    # Rows126/127 are deliberately signed-zero padding witnesses, not live M.
    for i in range(ROWS*COLS,len(alpha)):alpha[i]=0x80000000 if i%2 else 0
    return {'p':raw(p),'v':raw(v),'alpha':raw(alpha)}


def export(output):
    source=check_source();output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for name in CASES:
        directory=output/name;directory.mkdir()
        tensors=inputs(name)
        for key,data in tensors.items():(directory/(key+'.bin')).write_bytes(data)
        (directory/'shape.txt').write_text(f'{ROWS} {PADDED} {COLS} {K}\n')
        row={'case':name,'shape':[ROWS,PADDED,COLS,K],'cube_cores':1,'aiv_actors':2,
             'input_sha256':{k:sha(v) for k,v in tensors.items()},
             'input_bytes':{k:len(v) for k,v in tensors.items()},
             'acceptance':'all paired output bits, unchanged inputs, guards, actor completion',
             'nonfinite_classification_separate':name=='nonfinite',
             'expected_anti_k128_live_reference_bits':'0x3e800000' if name=='anti_k128' else None,
             'npu':'not_run','graph':'not_run','whole_c4_error_contract':'not_run'}
        (directory/'fixture.json').write_text(json.dumps(row,indent=2)+'\n');cases.append(row)
    report={'schema':1,'source':source,'cases':cases,'cpu_debug':'not_run','cann_compile':'not_run',
            'camodel':'not_run','npu':'not_run','performance':'not_run',
            'scope':'isolated FP32 final-add legality; not a C4 optimization or production acceptance'}
    (output/'cases.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-source',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.check_source:print(json.dumps(check_source(),indent=2))
    elif args.output:print(json.dumps({'status':'inputs_exported_not_executed','cases':len(export(args.output)['cases'])}))
    else:parser.error('choose --check-source or --output NEW_DIRECTORY')
