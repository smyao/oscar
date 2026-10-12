"""Archive #37-49/#126/#145/#154/#155, D.4: host contracts, not NPU proof."""
from pathlib import Path
import importlib.util

import pytest
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("request_window_store_fixture",HERE/"fixture.py")
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
torch.set_num_threads(min(4,torch.get_num_threads()))


def test_rotation_quantization_helpers_remain_verbatim_baseline():
    baseline=(HERE.parents[2]/"csrc/kernels/rotate_clip_store_striped.cpp").read_text()
    generated=(HERE/"packed_store.cpp").read_text()
    begin="using namespace oscar_ascend_device;"
    expected=baseline[baseline.index(begin):baseline.index("template <typename T>\nclass RotateOnly")]
    actual=generated[generated.index(begin):generated.index("template <typename T>\nclass RequestPackedStore")]
    assert actual==expected


@pytest.mark.parametrize("segment,heads,dim",[(128,1,64),(129,2,64),(2816,1,256)])
def test_dual_region_addresses_equal_baseline_remapped_slots(segment,heads,dim):
    blocks,offset=7,13;stride=segment*heads*(dim//2+8)+17
    spans=[]
    for page in (0,2,6):
        for within in (0,segment-1,segment,2*segment-1):
            slot=page*2*segment+within
            for h in range(heads):
                a=fixture.packed_address(slot,h,segment=segment,blocks=blocks,offset=offset,stride=stride,heads=heads,dim=dim)
                region,inner=divmod(within,segment)
                oldslot=(region*blocks+page)*segment+inner
                oldpage,oldinner=divmod(oldslot,segment)
                b=offset+oldpage*stride+(oldinner*heads+h)*(dim//2+8)
                assert a==b
                spans.append((a,a+dim//2+8))
    spans.sort();assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))


def test_rollback_zero_to_three_keeps_complete_recent_window_and_sink():
    tensors,_,meta=fixture.build_case("rollback",lengths=(700,),mode="rollback",in_page=0)
    s=meta["shape_fields"];tags=tensors["expected_tags"].view(torch.int64)
    for rejected in range(4):
        accepted_end=700-rejected
        needed=list(range(64))+list(range(accepted_end-256,accepted_end))
        for pos in needed:
            ring=pos if pos<64 else 64+pos%259
            assert int(tags[ring])==(7<<18)|pos
            dst=ring*s["heads"]*s["dim"]*2
            expected=tensors["key"][pos].to(torch.bfloat16).view(torch.uint8).flatten()
            assert torch.equal(tensors["expected_raw_key"][dst:dst+len(expected)],expected)


def test_reordering_uses_lease_not_batch_row_and_raw_is_unrotated():
    tensors,_,meta=fixture.build_case("reorder",lengths=(4,7,3),heads=2,hadamard=True)
    s=meta["shape_fields"]
    for req,chosen in meta["winners"].items():
        lease,epoch=tensors["mapping"][req].tolist()
        for ring,token in chosen.items():
            dst=(lease*s["key_stride"]+ring*s["heads"]*s["dim"])*2
            raw=tensors["key"][token].to(torch.bfloat16).view(torch.uint8).flatten()
            assert torch.equal(tensors["expected_raw_key"][dst:dst+len(raw)],raw)
            tag=tensors["expected_tags"].view(torch.int64)[lease*s["tag_stride"]+ring]
            assert int(tag)==(epoch<<18)|int(tensors["positions"][token])


def test_epoch_reuse_retains_old_bytes_but_they_cannot_match_new_owner():
    tensors,_,meta=fixture.build_case("epoch",lengths=(7,),mode="epoch_reuse",in_page=0)
    tags=tensors["expected_tags"].view(torch.int64)
    assert all(int(tags[p])==(7<<18)|p for p in range(7))
    for ring in range(7,323):
        assert int(tags[ring])>>18==6
        assert int(tags[ring])!=(7<<18)|ring
    assert torch.equal(tensors["expected_raw_key"][7*64*2:],tensors["initial_raw_key"][7*64*2:])


@pytest.mark.parametrize("mode,lengths,expected",[("bad_epoch",(7,),[2]),("duplicate_lease",(3,4),[3,3]),("bad_position",(7,),[4]),("bad_qstarts",(7,),[1]),("positive_qstarts_origin",(7,),[6])])
def test_invalid_metadata_publishes_no_sidecar_and_reports_error(mode,lengths,expected):
    tensors,_,meta=fixture.build_case(mode,lengths=lengths,mode=mode)
    assert meta["window_status"]==expected
    for name in ("raw_key","raw_value","tags"):
        assert torch.equal(tensors["expected_"+name],tensors["initial_"+name])
    assert tensors["expected_window_status"].numel()==meta["shape_fields"]["capacity"]


def test_faulty_head_does_not_publish_token_and_padding_clears_status():
    tensors,_,meta=fixture.build_case("faults",lengths=(17,),heads=2,mode="faults")
    selected={token for chosen in meta["winners"].values() for token in chosen.values()}
    assert not selected.intersection(range(6))
    assert tensors["expected_status"][0].tolist()==[2,0]
    assert tensors["expected_status"][1].tolist()==[0,2]
    assert tensors["expected_status"][2].tolist()==[3,3]
    assert tensors["expected_status"][3].tolist()==[1,1]
    assert tensors["expected_status"][4].tolist()==[4,4]
    assert tensors["expected_status"][5].tolist()==[0,0]


def test_q1_32_live_rows_and_96_poison_padding_with_hkv2():
    tensors,_,meta=fixture.build_case("q1",lengths=(1,)*32,heads=2,token_capacity=128,window_capacity=128)
    assert tensors["qstarts"].tolist()==list(range(33))
    assert torch.all(tensors["slots"][32:]==-1)
    assert torch.all(tensors["expected_status"][32:]==0)
    assert tensors["expected_window_status"].shape==(128,)
    assert torch.all(tensors["expected_window_status"]==0)
    assert meta["scan_visits"]==32
    assert tensors["mapping"].shape==(32,2)


def test_active_tail_is_explicit_batch_failure_without_req0_publication():
    tensors,_,meta=fixture.build_case("tail",mode="active_tail",token_capacity=12)
    assert meta["window_status"]==[7]
    for name in ("raw_key","raw_value","tags"):
        assert torch.equal(tensors["expected_"+name],tensors["initial_"+name])


def test_capture_all_padding_does_not_need_published_map():
    tensors,_,meta=fixture.build_case("capture",mode="capture_all_padding")
    assert tensors["mapping"].tolist()==[[-1,0]]
    assert meta["window_status"]==[0]
    for name in ("raw_key","raw_value","tags"):
        assert torch.equal(tensors["expected_"+name],tensors["initial_"+name])


def test_native_q1_capture_nonempty_padding_requests_have_no_lease():
    tensors,_,meta=fixture.build_case("capture128",lengths=(1,)*128,heads=2,padding_requests_after=32,window_capacity=128)
    assert tensors["qstarts"].tolist()==list(range(129))
    assert torch.all(tensors["mapping"][32:,0]==-1)
    assert torch.all(tensors["expected_window_status"]==0)
    assert all(not chosen for req,chosen in meta["winners"].items() if req>=32)


def test_negative_holes_scan_is_linear_and_keeps_last_legal_congruent_position():
    n=20000;slots=list(range(n));positions=list(range(n));status=[[0,0] for _ in range(n)]
    for i in range(1,n,5):slots[i]=-1
    for i in range(3,n,17):status[i][1]=2
    chosen,errors,visits=fixture.logical_winners([0,n],[[0,7]],slots,positions,status,1)
    assert errors==[0] and visits==n
    for ring,token in chosen[0].items():
        expected=max(i for i in range(n) if slots[i]>=0 and status[i]==[0,0] and (i if i<64 else 64+i%259)==ring)
        assert token==expected
