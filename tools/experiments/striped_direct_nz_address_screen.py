"""Archive #13-22/#126/#145/#148/#154-156; D.4 address-only host proof.

Check a proposed compressed-word Gather plus direct NZ INT16 decode mapping.
Host-only proof; the accompanying AscendC research helper requires a separate
official CPU-debug qualification. No timing or VM execution in this script.
The real padded-natural producer currently has NO code-data Gather; this
proposal adds a small pre-expansion Gather and must pay its full cost.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[2]
D,H=256,64


def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
def half(value):return struct.unpack('<e',struct.pack('<e',value))[0]
def bits(value):return struct.pack('<f',value)


def run():
    # The existing raw packed row is160B, K words0..31, V words32..63.
    indices=[]
    for group in range(8):
        for offset in range(272):
            if offset<256:
                row=group*8+offset%8
                word=offset//8
                indices.append(row*160+word*2)
            else:
                indices.append(0)  # never used by Shift/And or output Cast
    assert len(indices)==2176 and max(indices)+64+2<=64*160
    seen=set()
    for group in range(8):
        for bit in range(8):
            for offset in range(256):
                row,dim=group*8+offset%8,bit*32+offset//8
                dense_nz=group*2048+bit*256+offset
                assert dense_nz==(row//8)*D*8+dim*8+row%8
                seen.add((row,dim))
    assert len(seen)==H*D
    # Exact raw uint16 Gather semantics are modeled without treating payloads
    # as half. This includes all signaling/quiet NaN encodings. The real
    # vector Gather must independently pass helper_probe.cpp on CPU-debug.
    uint16_seen=[set(),set()]
    for batch in range(32):
        packed=[0]*(H*80)
        for row in range(H):
            for word in range(32):
                x=batch*2048+row*32+word
                packed[row*80+word]=x
                packed[row*80+32+word]=65535-x
        for side in range(2):
            gathered=[packed[index//2+side*32] for index in indices]
            for group in range(8):
                for element in range(256):
                    x=gathered[group*272+element]
                    row,word=group*8+element%8,element//8
                    assert x==packed[row*80+side*32+word]
                    uint16_seen[side].add(x)
                    for bit in range(8):
                        signed=x if x<32768 else x-65536
                        assert ((signed>>(2*bit))&3)==((x>>(2*bit))&3)
    assert all(len(x)==65536 for x in uint16_seen)
    all_checks=[]
    for live in (0,1,7,8,9,63,64):
        packed=[0]*(H*80)
        scales=[half((1+r%7)*2**-9) if r<live else 0.0 for r in range(H)]
        zeros=[half((r%11-5)*.125) if r<live else 0.0 for r in range(H)]
        for row in range(live):
            for word in range(64):packed[row*80+word]=(row*313+word*997+32768)&0xffff
        for value in (False,True):
            word_offset=32 if value else 0
            gathered=[packed[index//2+word_offset] for index in indices]
            natural=[None]*(8*2064)
            for bit in range(8):
                for chunk in range(2):
                    for repeat in range(8):
                        for element in range(128):
                            offset=chunk*128+element
                            src=gathered[repeat*272+offset]
                            signed=src if src<32768 else src-65536
                            natural[repeat*2064+bit*256+offset]=(signed>>(2*bit))&3
            for group in range(8):
                for offset in range(2048):
                    row=group*8+offset%8
                    dim=offset//8
                    expected=(packed[row*80+word_offset+dim%32]>>(2*(dim//32)))&3
                    actual=natural[group*2064+offset]
                    assert actual==expected
                    # Same per-element FP32 Mul followed by Add, only layout differs.
                    assert bits(f32(f32(actual*scales[row])+zeros[row]))==bits(f32(f32(expected*scales[row])+zeros[row]))
                assert all(x is None for x in natural[group*2064+2048:(group+1)*2064])
            for slot in (0,64,128,192):
                for row,dim in seen:
                    local=(row//8)*D*8+dim*8+row%8
                    assert slot*D+local==((slot+row)//8)*D*8+dim*8+(slot+row)%8
            all_checks.append({'live_rows':live,'value':value,'elements':H*D,'layout_bijective':True,
                               'same_FP32_element_Mul_Add_bits':True})
    return {'status':'host_address_mapping_passed_not_operator_qualification','checks':all_checks,
        'uint16_bit_patterns_enumerated_per_side':65536,'all_eight_Shift_And_planes_checked':True,
        'raw_half_nan_payloads_included':True,'actual_A2_Gather_semantics':'requires_CPU_debug_probe',
        'packed_gather_uint32_indices':2176,'packed_gather_payload_bytes':4352,
        'natural_int16_group_stride_elements':2064,'natural_int16_used_bytes':33024,
        'existing_natural_allocation_bytes':34816,'shift_and_each_calls':16,
        'shift_repeat_stride_32B_units':{'source':17,'destination':129},
        'FP32_Cast_calls':8,'existing_FP32_Cast_calls':4,
        'FP32_Mul_calls':8,'FP32_Add_calls':8,'existing_each':4,'removed_Brcb_calls':2,
        'source0_publish':'same contiguous64KiB per halfKV; no cross-lane exchange',
        'new_UB_if_index_after_existing32B_guard':{'wordIndexBuf':8736,'planeBuf':4352,
            'increment':13024,'new_total':171680,'old_total':158656},
        'not_proved':['A2 Gather/Shift/Cast timing','metadata error and nonfinite implementation',
            'source1/2 finite checks and format conversion','new Cube consumer','NPU/graph','whole speed'],
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'csrc/kernels/attention_striped_unpack_padded.h','csrc/kernels/attention_cv_decode_bundle.cpp',
            'tools/experiments/striped_direct_nz_address_screen.py')}}


if __name__=='__main__':
    result=run();target=ROOT/'reports/striped_direct_nz_address_screen_20261012.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(result['checks']),'output':str(target)}))
