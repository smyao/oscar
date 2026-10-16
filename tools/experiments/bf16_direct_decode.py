"""Archive #13-22/#126/#148/#154/#155: CPU checks of direct BF16 lookup.

D.4: only four FP32 levels per vector are evaluated; production must not
expand a full FP32 K/V tile. This module is test-only, never a serving oracle.
The independent reference uses NumPy float64 plus explicit FP32/RNE bits.
No CANN/CPU-debug/NPU result is implied by these arithmetic checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools.phase import atomic_json

ROOT=Path(__file__).resolve().parents[2]
CONTRACT='all_sources_bf16_qkvp_fp32_online_merge_direct_int2_v2'


def four_levels(scale,zero,*,live=None):
    """Host model: exactly four FP32 multiply/add values, then BF16 RNE."""
    if scale.dtype!=torch.float16 or zero.dtype!=torch.float16 or scale.shape!=zero.shape:
        raise ValueError('metadata must have matching FP16 scale/zero shapes')
    if live is None:live=torch.ones_like(scale,dtype=torch.bool)
    if live.dtype!=torch.bool or live.shape!=scale.shape:raise ValueError('invalid live metadata mask')
    invalid=(~torch.isfinite(scale))|(scale<=0)|(~torch.isfinite(zero))
    if bool((invalid&live).any()):raise ValueError('error3: invalid live FP16 metadata')
    # Padded metadata is outside the logical domain and may contain poison.
    s=torch.where(live,scale,torch.ones_like(scale)).float().unsqueeze(-1)
    z=torch.where(live,zero,torch.zeros_like(zero)).float().unsqueeze(-1)
    product=s*torch.arange(4,dtype=torch.float32)
    levels=(product+z).bfloat16()
    return torch.where(live.unsqueeze(-1),levels,torch.zeros_like(levels))


def decode_words(words,scale,zero,*,live=None):
    """Return natural dimensions: striped word w/bit b maps to w+b*words."""
    if words.dtype not in (torch.int16,torch.int32,torch.int64) or words.ndim!=2:
        raise ValueError('packed words require a two-dimensional integer tensor')
    if scale.shape!=(len(words),) or zero.shape!=scale.shape:raise ValueError('row metadata mismatch')
    if bool(((words.to(torch.int64)&~0xffff)!=0).any()) and words.dtype!=torch.int16:
        raise ValueError('packed words outside uint16 domain')
    table=four_levels(scale,zero,live=live)
    code=(words.to(torch.int64).unsqueeze(1) >> (torch.arange(8)*2).view(1,8,1))&3
    return table.gather(1,code.reshape(len(words),-1))


def decode_canonical_kv(raw,dim=256):
    """Diagnostic canonical-byte adapter; no full FP32 restored array."""
    if raw.dtype!=torch.uint8 or raw.shape[-1]!=dim//2+8 or dim%8:
        raise ValueError('invalid canonical KV byte layout')
    outputs=[];width=dim//4+4;shape=raw.shape[:-1]
    for start in (0,width):
        value=raw[...,start:start+width].reshape(-1,width)
        scale_bits=value[:,dim//4].int()|(value[:,dim//4+1].int()<<8)
        zero_bits=value[:,dim//4+2].int()|(value[:,dim//4+3].int()<<8)
        table=four_levels(scale_bits.to(torch.int16).view(torch.float16),zero_bits.to(torch.int16).view(torch.float16))
        codes=((value[:,:dim//4].long().unsqueeze(-1)>>(torch.arange(4)*2))&3).reshape(-1,dim)
        outputs.append(table.gather(1,codes).reshape(*shape,dim))
    return tuple(outputs)


def independent_level_bits(scale_bits,zero_bits):
    """Separate FP64 arithmetic/explicit integer nearest-even BF16 oracle.

    FP16 significands have <=11 bits; multiplying by 0..3 is exact in FP64
    and FP32. The largest half exponent separation spans fewer than 53 bits,
    so FP64 addition is exact here before the explicit FP32 rounding step.
    """
    s=np.asarray(scale_bits,dtype=np.uint16).view(np.float16).astype(np.float64)
    z=np.asarray(zero_bits,dtype=np.uint16).view(np.float16).astype(np.float64)
    product=(s[...,None]*np.arange(4,dtype=np.float64)).astype(np.float32)
    value=(product.astype(np.float64)+z[...,None]).astype(np.float32)
    bits=value.view(np.uint32)
    high=bits>>16;low=bits&65535
    rounded=high+((low>32768)|((low==32768)&((high&1)!=0)))
    return rounded.astype(np.uint16)


def metadata_sweep():
    valid_scales=np.arange(1,0x7c00,dtype=np.uint16)
    finite_zeros=np.concatenate((np.arange(0,0x7c00,dtype=np.uint16),np.arange(0x8000,0xfc00,dtype=np.uint16)))
    # Exhaust each metadata coordinate over its entire valid half domain;
    # cross with signs, signed zeros, extremes and exponent boundaries.
    zero_anchors=np.array([0,0x8000,1,0x8001,0x3c00,0xbc00,0x7bff,0xfbff],dtype=np.uint16)
    scale_anchors=np.array([1,0x3ff,0x400,0x3c00,0x3e00,0x7bff],dtype=np.uint16)
    return (np.concatenate((np.repeat(valid_scales,len(zero_anchors)),np.tile(scale_anchors,len(finite_zeros)))),
            np.concatenate((np.tile(zero_anchors,len(valid_scales)),np.repeat(finite_zeros,len(scale_anchors)))))


def validate():
    torch.set_num_threads(1)
    # Every possible packed 16-bit word, including every bit position and code.
    words=torch.arange(65536,dtype=torch.int32).reshape(-1,1)
    scale=torch.ones(65536,dtype=torch.float16);zero=torch.zeros_like(scale)
    decoded=decode_words(words,scale,zero)
    from oscar_ascend.ops.reference import unpack_int2
    packed=torch.stack(((words[:,0]&255),(words[:,0]>>8)),dim=1).to(torch.uint8)
    expected=unpack_int2(packed,8).bfloat16()
    if not torch.equal(decoded.view(torch.int16),expected.view(torch.int16)):
        raise RuntimeError('packed-word byte order differs from the independent codec')
    scales,zeros=metadata_sweep();checked=0;ties=0
    for first in range(0,len(scales),8192):
        sb,zb=scales[first:first+8192],zeros[first:first+8192]
        table=four_levels(torch.from_numpy(sb.view(np.float16).copy()),torch.from_numpy(zb.view(np.float16).copy()))
        actual=table.view(torch.int16).numpy().view(np.uint16);gold=independent_level_bits(sb,zb)
        if not np.array_equal(actual,gold):raise RuntimeError('metadata RNE table differs from independent arithmetic')
        checked+=actual.size
        exact=(sb.view(np.float16).astype(np.float64)[:,None]*np.arange(4)+zb.view(np.float16).astype(np.float64)[:,None]).astype(np.float32).view(np.uint32)
        ties+=int(np.count_nonzero((exact&65535)==32768))
    # Code zero is +0 + -0 -> +0, never a direct signed-zero bit copy.
    negative_zero=four_levels(torch.tensor([1.],dtype=torch.float16),torch.tensor([-0.],dtype=torch.float16))
    if int(negative_zero.view(torch.int16)[0,0])!=0:raise RuntimeError('negative-zero add semantics changed')
    return {'contract':CONTRACT,'status':'cpu_direct_decode_passed','returncode':0,
        'packed_words':65536,'packed_code_positions_checked':65536*8,
        'valid_scale_patterns':0x7bff,'finite_zero_patterns':2*0x7c00,
        'metadata_pairs':len(scales),'levels_checked':checked,'exact_halfway_values_checked':ties,
        'metadata_cartesian_product_exhaustive':False,
        'oracle':'independent FP64 exact half-domain arithmetic, explicit FP32 then integer BF16 RNE',
        'negative_zero_code0_bits':0,'device_completion':'not_run','production_qualified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('refusing to replace an earlier decode report')
    try:report=validate()
    except Exception as error:
        report={'contract':CONTRACT,'status':'cpu_direct_decode_failed','returncode':2,'first_error':str(error),
                'device_completion':'not_run','production_qualified':False}
    report['source_sha256']={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'oscar_ascend/ops/reference.py':hashlib.sha256((ROOT/'oscar_ascend/ops/reference.py').read_bytes()).hexdigest()}
    atomic_json(args.output,report);print(json.dumps(report),flush=True);return report['returncode']


if __name__=='__main__':raise SystemExit(main())
