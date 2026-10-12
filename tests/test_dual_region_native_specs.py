"""Archive #26/#34/#36/#37-49/#127-129: execute native cache-spec contracts.

The request-window format must preserve GDN identity and physical allocation,
while metadata128 copies retain the new physical token capacity.
"""
from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch

from test_runtime_native import native_specs, native_groups, _native_coordinator  # noqa: F401


def source_specs(interface):
    full = interface.FullAttentionSpec(
        block_size=768, num_kv_heads=1, head_size=256,
        dtype=torch.bfloat16, page_size_padded=817152)
    gdn = interface.MambaSpec(
        block_size=262144, shapes=((6, 2560), (12, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16), page_size_padded=817152,
        num_speculative_blocks=3, mamba_cache_mode="none")
    return {"full": full, "gdn": gdn}


def test_dual_spec_preserves_native_storage_and_metadata_copy(native_specs):
    interface, ours, registry = native_specs
    source = source_specs(interface)
    normal = ours.transform_native_specs(source)
    dual = ours.transform_native_specs(source, request_windows=True)
    assert dual["gdn"] is source["gdn"] is normal["gdn"]
    spec = dual["full"]
    assert spec.block_size == 5632 == 2 * normal["full"].block_size
    assert spec.page_size_bytes == normal["full"].page_size_bytes == 817152
    assert spec.real_page_size_bytes == 765952
    assert registry.KVCacheSpecRegistry.get_manager_class(spec).__name__ == "FullAttentionManager"
    view = spec.copy_with_new_block_size(128)
    assert view.request_window_storage and view.metadata_block_view
    assert view.storage_block_size == 5632 and view.page_size_bytes == 817152
    assert view.layout == spec.layout
    assert view.copy_with_new_block_size(5632) == spec
    with pytest.raises(AssertionError, match="incompatible geometry"):
        ours.OscarFullAttentionSpec.merge([normal["full"], spec])
    with pytest.raises(ValueError, match="physical layout"):
        replace(spec, block_size=2816)


def test_dual_raw_owner_is_not_fake_contiguous_packed_view(native_specs):
    interface, ours, _ = native_specs
    spec = ours.transform_native_specs(source_specs(interface), request_windows=True)["full"]
    raw = torch.empty(3 * 817152, dtype=torch.uint8)
    owner = ours.packed_view(raw, 3, spec.layout)
    assert owner is raw and owner.ndim == 1
    offsets = [spec.layout.packed_offset(3, 1, p) for p in (2815, 2816, 5631)]
    assert offsets[1] != offsets[0] + 136
    assert offsets[2] + 136 <= raw.numel()
    with pytest.raises(ValueError, match="page budget"):
        ours.packed_view(raw[:-1], 3, spec.layout)


def test_dual_format_rejects_unsupported_native_layouts(native_specs):
    interface, ours, _ = native_specs
    source = source_specs(interface)
    source["gdn"] = replace(source["gdn"], block_size=768, mamba_cache_mode="align")
    with pytest.raises(ValueError, match="mode none"):
        ours.transform_native_specs(source, request_windows=True)
    for value in (1, "yes", None):
        with pytest.raises(ValueError, match="boolean"):
            ours.transform_native_specs(source_specs(interface), request_windows=value)


def test_native_shared_pool_dual_capacity_and_gdn_lifetime(native_specs,monkeypatch):
    interface,ours,_=native_specs
    original=ours.transform_native_specs
    monkeypatch.setattr(ours,'transform_native_specs',lambda source:original(source,request_windows=True))
    native=native_groups.__wrapped__(native_specs,monkeypatch,SimpleNamespace(param=False))
    _,_,_,utils,config,groups=native
    assert [g.kv_cache_spec.block_size for g in groups]==[5632,262144,262144,262144]
    pool=_native_coordinator(native,monkeypatch)
    assert pool.scheduler_block_size==2883584  # native LCM unchanged
    assert len(pool.kv_cache_config.kv_cache_tensors)==17
    assert utils._pool_bytes_per_block(groups)==17*817152
    managers=pool.single_type_managers
    assert all(manager.block_pool is pool.block_pool for manager in managers)
    assert all(manager.kv_cache_spec is group.kv_cache_spec for manager,group in zip(managers,groups))
    before=pool.block_pool.get_num_free_blocks()
    for tokens in (65100,65612,262140):
        for manager in managers:
            need=manager.get_num_blocks_to_allocate('request',tokens+3,[],0,tokens)
            allocated=manager.allocate_new_blocks('request',tokens+3,tokens)
            assert len(allocated)==need
        ids=[[block.block_id for block in manager.req_to_blocks['request']] for manager in managers]
        assert list(map(len,ids))==[math.ceil((tokens+3)/5632),4,4,4]
        flat=[idx for group in ids for idx in group]
        assert 0 not in flat and len(set(flat))==len(flat)
    for manager in managers:manager.free('request')
    assert pool.block_pool.get_num_free_blocks()==before
