"""Archive #13-22/#126/#145/#148/#154, D.4: fixed full C4 B-NZ experiment.

The existing independent source0+1 oracle and every real task are retained.
Only constant Rv is host-reordered (256KiB); K/V must be published by the real
candidate kernel. Neither packing cost nor CPU execution proves a device gain.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_member_pipeline.fixture import export as mother_export


def sha(data):return hashlib.sha256(data).hexdigest()


def reorder_rv(data,*,inverse=False):
    if len(data)!=256*256*4:raise ValueError('Rv must be physical FP32[256,256]')
    src=struct.unpack('<65536I',data);dst=[0]*65536
    for row in range(256):
        for col in range(256):
            nd=row*256+col;nz=(col//8)*256*8+row*8+col%8
            if inverse:dst[nd]=src[nz]
            else:dst[nz]=src[nd]
    result=array('I',dst)
    if result.itemsize!=4:raise ValueError('uint32 width required')
    if sys.byteorder!='little':result.byteswap()
    return result.tobytes()


def verify_rv(natural,packed):
    if reorder_rv(natural)!=packed or reorder_rv(packed,inverse=True)!=natural:
        raise ValueError('Rv NZ differs from logical matrix raw bits')
    return {'status':'passed','elements':65536,'dtype':'uint32 bit permutation only',
            'natural_sha256':sha(natural),'packed_sha256':sha(packed),'packed_bytes':len(packed)}


def export(path,bad=-1):
    meta=mother_export(path,context=337,bad=bad)
    meta['declared_deadlines']={'CPU_each_arm_s':120,'CAModel_each_arm_s':600}
    natural=(path/'rv.bin').read_bytes();packed=reorder_rv(natural)
    witness=verify_rv(natural,packed)
    # Distinct uint32 labels establish all positions, even if actual matrix
    # values repeat. They are not numerical inputs to either attention arm.
    labels=struct.pack('<65536I',*range(65536))
    verify_rv(labels,reorder_rv(labels))
    (path/'rv_nz.bin').write_bytes(packed)
    meta['rv_nz_sha256']=sha(packed);meta['rv_relayout']=witness
    meta['rv_layout_by_arm']={'baseline':'ND','candidate':'NZ_C0_8'}
    meta['host_prepacked_KV']=False
    meta['publisher_scope']='both K/V produced from original packed INT2/window inputs inside the kernel'
    meta['producer_cost']='real K/V publisher included; constant Rv prepack 256KiB outside launch timing'
    meta['mother_fixture_generator_sha256']=sha((ROOT/'csrc/experiments/c4_member_pipeline/fixture.py').read_bytes())
    meta['fixture_generator_sha256']=sha(Path(__file__).read_bytes())
    (path/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bad-position',type=int,choices=(-1,80,96,336),default=-1);args=parser.parse_args()
    value=export(args.output,args.bad_position)
    print(json.dumps({'context':337,'tokens':336,'bad_position':value['bad_position'],
                      'rv_bits':'host_inverse_passed','K_V_host_prepack':False,'execution':'not_run'}))
