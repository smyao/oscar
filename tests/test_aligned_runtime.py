"""Archive #111/#126/#145/#164: versioned dtype/byte/dispatch boundaries.

Native matmul and real forward routing are exercised on host; no NPU claim.
"""
from enum import Enum
from types import SimpleNamespace
import json
from pathlib import Path

import pytest
import torch

from oscar_ascend.integration.aligned_math import dequant_mode,rotate_query,store_aligned
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.layout import SlotLayout,HybridPageLayout
from oscar_ascend.lifecycle import SnapshotLayout
from oscar_ascend.runtime import WorkspaceGeometry


@pytest.mark.parametrize('n,later,name',[
    (8,False,'attention_cv_aligned_decode_out'),(8,True,'attention_cv_aligned_decode_q1_out'),
    (256,False,'attention_cv_aligned_cluster4_out'),(8192,False,'attention_cv_aligned_cluster16_out'),
    (10,False,'attention_cv_aligned_cluster4_out')])
def test_shared_candidate_actual_forward_selects_versioned_reader_and_bf16_rotations(n,later,name):
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        n,1,1,n,16,later,True,True,True,_member_pipeline=True,_c16_l1=True,
        _bf16_expected=name,_aligned=True)


def test_144_byte_pages_and_exact_windows_fit_native_pool_without_gdn_changes():
    slot=SlotLayout(256,metadata_bytes=4)
    assert slot.slot_bytes==144 and slot.token_bytes==144
    page=HybridPageLayout(slot,30720,393216,262144,native_page_size_bytes=817152,native_mamba_cache_mode='none')
    assert page.block_size==2688 and page.virtual_blocks_per_page==21
    snapshot=SnapshotLayout(page,64,256,3)
    for p in range(4):
        begin,end=page.packed_interval(4,p)
        assert end<=4*30720+(p+1)*393216
        assert snapshot.interval(4,p)[1]<=4*817152
    old=WorkspaceGeometry(128,6,1,256)
    new=WorkspaceGeometry(128,6,1,256,aligned_math=True)
    assert new.total_bytes-old.total_bytes==128*2*256*2


def test_native_classification_controls_dequant_not_query_count_or_draft_flag():
    states=Enum('State',['DecodeOnly','SpecDecoding','ChunkedPrefill','PrefillNoCache','PrefillCacheHit'])
    for state in states:
        meta=SimpleNamespace(attn_state=state,capture_origin=False,is_draft=True,max_query_len=4)
        assert dequant_mode(meta)==int(state.name in {'DecodeOnly','SpecDecoding'})
    assert dequant_mode(SimpleNamespace(attn_state=states.ChunkedPrefill,capture_origin=True))==1
    with pytest.raises(OscarReadinessError):dequant_mode(SimpleNamespace(attn_state=None))


def test_native_bf16_rotation_is_rounded_before_abi_and_store_retains_original_windows():
    d,n,h,hk=256,3,6,1
    rng=torch.Generator().manual_seed(42)
    q=torch.randn(n,h,d,generator=rng).bfloat16()
    k=torch.randn(n,hk,d,generator=rng).bfloat16();v=torch.randn(n,hk,d,generator=rng).bfloat16()
    r=torch.eye(d).roll(1,0).bfloat16()
    ws=SimpleNamespace(query_rot_bf16=torch.empty_like(q),key_rot_bf16=torch.empty_like(k),
        value_rot_bf16=torch.empty_like(v),store_status=torch.empty(n,hk,dtype=torch.int32),geometry=SimpleNamespace(cube_cores=1))
    state=SimpleNamespace(workspace=ws,rotation_k_bf16=r,rotation_v_bf16=r.T.contiguous(),
        raw=object(),window_key=object(),window_value=object(),window_tags=object(),num_blocks=1,
        spec=SimpleNamespace(block_size=2688,conv_bytes=30720,ssm_bytes=393216),
        snapshots=SimpleNamespace(sink_tokens=64,ring_tokens=259))
    qr=torch.empty(n,h,d,dtype=torch.bfloat16);status=torch.empty(n,h,dtype=torch.int32);slots=torch.arange(n)
    class Ops:
        def copy_validate_bf16_query_out(self,rounded,actual_slots,output,status,cores):
            assert rounded.data_ptr()==ws.query_rot_bf16.data_ptr()
            assert torch.equal(rounded,torch.matmul(q,r))
            output.copy_(rounded);status.zero_()
        def store_aligned_int2_out(self,rk,rv,original_k,original_v,*args):
            assert original_k is k and original_v is v
            assert torch.equal(rk,torch.matmul(k,r)) and torch.equal(rv,torch.matmul(v,r.T))
            assert args[-2:]==(.96,.92)
    rotate_query(Ops(),state,q,slots,qr,status)
    assert qr.dtype==torch.bfloat16 and torch.equal(qr,torch.matmul(q,r))
    store_aligned(Ops(),state,k,v,slots,torch.arange(n),{})
