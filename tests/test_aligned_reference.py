import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import aligned_reference as ref
from tools import generate_aligned_codec as generate

ROOT=Path(__file__).resolve().parents[1]

def test_contract_frozen_and_legacy_separate():
    assert ref.load_acceptance()["slot_bytes_D256"]==144
    assert ref.CONTRACT!="oscar_bf16_attention_v2"
    for name,pin in generate.PINS.items():
        assert hashlib.sha256((ROOT/"csrc/kernels"/name).read_bytes()).hexdigest()==pin

def test_clip_uses_official_integer_index_not_quantile():
    row=torch.arange(256).bfloat16()[None]
    assert ref.clip_rows(row,.96).max().item()==245
    assert ref.clip_rows(row,.92).max().item()==235
    assert ref.clip_rows(row,0).max().item()==255

def test_quantizer_half_ties_go_up_and_parameters_stay_fp32():
    row=torch.tensor([0,.5,1.5,2.5,3]+[0]*59,dtype=torch.bfloat16)[None]
    codes,scale,zero=ref.quantize_rows(row)
    assert codes[0,:5].tolist()==[0,1,2,3,3]
    assert scale.dtype==zero.dtype==torch.float32
    assert scale.item()==1 and zero.item()==0

def test_constant_scale_clamps_range_before_division():
    _,scale,zero=ref.quantize_rows(torch.zeros((1,64),dtype=torch.bfloat16))
    assert scale.item()==torch.tensor(1e-8,dtype=torch.float32).div(3).item()
    assert zero.item()==0

@pytest.mark.parametrize("dim",[64,128,256])
@pytest.mark.parametrize("mode",[0,1])
def test_code_byte_permutation_and_metadata_roundtrip(dim,mode):
    gen=torch.Generator().manual_seed(20261018+dim)
    k=torch.randn((7,2,dim),generator=gen).bfloat16()
    v=(torch.randn((7,2,dim),generator=gen)*3).bfloat16()
    packed=ref.pack_kv(k,v)
    assert packed.shape==(7,2,dim//2+16)
    metadata=packed[...,dim//2:].contiguous().view(torch.float32)
    decoded=ref.unpack_kv(packed,dim,mode)
    for side,original,clip in [(0,k,.96),(1,v,.92)]:
        codes,scale,zero=ref.quantize_rows(original,clip)
        assert torch.equal(metadata[...,side*2],scale)
        assert torch.equal(metadata[...,side*2+1],zero)
        assert torch.equal(decoded[side],ref.dequantize_rows(codes,scale,zero,mode))
        # Independently reconstruct physical word i from its eight planes.
        for i in (0,dim//8-1):
            expected=sum(codes[...,i+plane*(dim//8)].long()<<(2*plane) for plane in range(8))
            offset=side*(dim//4)+2*i
            actual=packed[...,offset].long()+(packed[...,offset+1].long()<<8)
            assert torch.equal(actual,expected)

def test_prefill_and_decode_rounding_are_intentionally_distinct():
    codes=torch.tensor([[0,1,2,3]],dtype=torch.uint8)
    scale=torch.tensor([.12345]);zero=torch.tensor([2.3456])
    prefill=ref.dequantize_rows(codes,scale,zero,0)
    decode=ref.dequantize_rows(codes,scale,zero,1)
    assert not torch.equal(prefill,decode)
    # Rounding stored metadata would preserve decode but alter prefill.
    bfscale=scale.bfloat16().float();bfzero=zero.bfloat16().float()
    assert torch.equal(decode,ref.dequantize_rows(codes,bfscale,bfzero,1))
    assert not torch.equal(prefill,ref.dequantize_rows(codes,bfscale,bfzero,0))

@pytest.mark.parametrize("mode",[0,1])
def test_corrupt_metadata_fails_closed(mode):
    codes=torch.zeros((1,64),dtype=torch.uint8)
    for scale,zero in [(0.,0.),(-1.,0.),(float("nan"),0.),(1.,float("inf"))]:
        with pytest.raises(ValueError):
            ref.dequantize_rows(codes,torch.tensor([scale]),torch.tensor([zero]),mode)

def test_generated_sources_match_and_cast_fix_is_retained():
    assert (ROOT/"csrc/kernels/attention_cv_aligned_decode.cpp").read_text()==generate.reader()
    assert (ROOT/"csrc/kernels/store_aligned_int2.cpp").read_text()==generate.writer()
    helper=(ROOT/"csrc/kernels/attention_aligned_unpack.h").read_text()
    assert helper==generate.unpack_helper()
    assert "oscar_bf16::CastRne(rounded,values,elements)" in helper
    assert "value?8:0,2*HalfKv" in helper
    assert "g.dequantMode" in generate.reader()

@pytest.mark.parametrize("mode",[0,1])
def test_actual_task_oracle_all_three_sources_uses_v3_raw(mode):
    rng=torch.Generator().manual_seed(18)
    k=torch.randn((128,1,256),generator=rng).bfloat16()
    v=torch.randn((128,1,256),generator=rng).bfloat16()
    q=torch.randn((1,1,256),generator=rng).bfloat16()
    ck=torch.randn((1,1,256),generator=rng).bfloat16()
    cv=torch.randn((1,1,256),generator=rng).bfloat16()
    wk=torch.zeros((1,68,1,256),dtype=torch.bfloat16);wv=torch.zeros_like(wk)
    wk[0,:64]=k[:64];wv[0,:64]=v[:64]
    tags=torch.full((1,68),-1,dtype=torch.int64);tags[0,:64]=torch.arange(64)
    fixture={"cpu":{"q":q,"qr":q.clone(),"ck":ck,"cv":cv,
                    "rv":torch.eye(256,dtype=torch.bfloat16),
                    "raw":ref.pack_kv(k,v).flatten(),"table":torch.tensor([[0]],dtype=torch.int32),
                    "wk":wk,"wv":wv,"tags":tags,"slots":torch.tensor([70])},
             "spec":SimpleNamespace(kv_heads=1,splits=1,recent_tokens=1),
             "stride":128*144,"blocks":1,"scale":256**-.5}
    tasks=torch.zeros((3,16),dtype=torch.int64)
    tasks[0,:11]=torch.tensor([0,1,0,64,70,0,0,0,70,0,0])
    tasks[1,:11]=torch.tensor([0,1,0,0,64,0,1,1,70,0,0])
    tasks[2,:11]=torch.tensor([0,1,0,70,71,0,2,2,70,0,0])
    result=ref.fixture_tasks_reference(fixture,tasks,mode)
    decoded=ref.unpack_kv(ref.pack_kv(k,v)[64:70],256,mode)
    expected=ref.attention_tiles(q.float(),*decoded,value_rotation=fixture["cpu"]["rv"])
    assert torch.equal(result["partial"][:,:,0],expected.output)
    assert torch.equal(result["lse"][:,:,0],expected.lse)
    assert result["live_output_rows"]==3 and result["contract"]==ref.CONTRACT
    assert torch.isfinite(result["output"]).all()
