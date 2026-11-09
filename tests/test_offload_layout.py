"""Archive #34/#36/#127/#128/#157/#163/#171: physical offload byte contracts.

CPU destruction/remapping verifies complete state transport, not NPU execution,
graph completion, asynchronous ownership, or performance acceptance.
"""
import ctypes
from types import SimpleNamespace

import pytest
import torch

from test_runtime_native import native_specs  # noqa: F401
from oscar_ascend.lifecycle import SnapshotLayout
from oscar_ascend.offload.layout import (
    CacheComponent, LayerCacheExport, export_kv_caches, export_layer_cache,
    registration_regions, unique_page_spans,
)


def make_exports(native_specs, mode="align", *, offset=4096, blocks=7):
    interface, ours, _ = native_specs
    full = interface.FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256,
                                       dtype=torch.bfloat16, page_size_padded=817152)
    gdn = interface.MambaSpec(
        block_size=768 if mode == "align" else 262144,
        shapes=((6, 2560), (12, 128, 128)), dtypes=(torch.bfloat16, torch.bfloat16),
        page_size_padded=817152, num_speculative_blocks=3, mamba_cache_mode=mode)
    specs = ours.transform_native_specs({"full": full, "gdn": gdn}, metadata_bytes=4)
    spec = specs["full"]
    owner = torch.full((offset + blocks * spec.page_size_bytes + 64,), 233, dtype=torch.uint8)
    raw = owner[offset:-64]
    raw.copy_((torch.arange(raw.numel(), dtype=torch.int32) % 251).to(torch.uint8))
    packed = ours.packed_view(raw, blocks, spec.layout)
    snapshot = SnapshotLayout(spec.layout, 64, 256, 3)
    wk, wv, tags = snapshot.views(raw, blocks)
    tags.copy_(torch.arange(snapshot.rows).expand(blocks, -1))
    state = SimpleNamespace(raw=raw, packed=packed, spec=spec, snapshots=snapshot,
                            window_key=wk, window_value=wv, window_tags=tags,
                            window_map=None, num_blocks=blocks,
                            cache_format="striped_rotated_fp32_v3")
    half = raw.view(torch.bfloat16)
    conv = half.as_strided((blocks, *gdn.shapes[0]), (15360, 2560, 1), offset // 2)
    ssm = half.as_strided((blocks, *gdn.shapes[1]), (196608, 16384, 128, 1),
                          (offset + blocks * spec.conv_bytes) // 2)
    native = {"full": packed, "gdn": (conv, ssm)}
    exports = export_kv_caches(native, {"full": state}, specs)
    return exports, native, state, specs, owner


@pytest.mark.parametrize("mode", ["none", "align"])
def test_complete_components_survive_destructive_remap_and_partial_tail(native_specs, mode):
    exports, native, state, _, owner = make_exports(native_specs, mode)
    full, gdn = exports["full"], exports["gdn"]
    original_view = native["full"]
    pointer, stride = original_view.data_ptr(), original_view.stride()
    sources = {"full": [1, 4], "gdn": [0, 3]}
    destinations = {"full": [2, 5], "gdn": [4, 0]}
    counts = {"full": [full.storage_block_size, 17], "gdn": [gdn.storage_block_size, 1]}
    saved = {name: [ctypes.string_at(span.address, span.length)
                    for span in export.page_spans(sources[name], valid_tokens=counts[name])]
             for name, export in exports.items()}
    expected_tags = state.window_tags[sources["full"]].clone()
    state.raw.fill_(0)
    for name, export in exports.items():
        spans = export.page_spans(destinations[name], valid_tokens=counts[name])
        assert sum(span.length for span in spans) == 2 * export.page_bytes
        for span, data in zip(spans, saved[name]):
            ctypes.memmove(span.address, data, len(data))
        assert [ctypes.string_at(span.address, span.length) for span in spans] == saved[name]
    assert torch.equal(state.window_tags[destinations["full"]], expected_tags)
    assert native["full"] is original_view
    assert (original_view.data_ptr(), original_view.stride()) == (pointer, stride)
    assert full.connector_cache[0] is original_view
    assert len(full.connector_cache) == 4 and len(gdn.connector_cache) == 2
    assert torch.all(owner[:4096] == 233) and torch.all(owner[-64:] == 233)
    # SSM all speculative columns and conv all offsets remain in the payload.
    assert gdn.components[1].page_bytes == 12 * 128 * 128 * 2
    assert gdn.components[0].page_bytes == 6 * 2560 * 2


def test_shared_storage_registration_and_aliases_are_deduplicated(native_specs):
    exports, _, _, _, owner = make_exports(native_specs)
    regions = registration_regions(exports, alignment=1)
    assert len(regions) == 1
    spans = [component.span(page) for export in exports.values() for component in export.components
             for page in (0, export.num_pages - 1)]
    assert regions[0].address == owner.data_ptr() + 4096
    assert regions[0].address + regions[0].length == max(span.address + span.length for span in spans)
    aliases = dict(exports, full_alias=exports["full"])
    spans = unique_page_spans(aliases, {"full": [1], "full_alias": [1], "gdn": [3]})
    assert len(spans) == 6
    assert spans[0].payload_offset == 0
    assert all(right.payload_offset == left.payload_offset + left.length
               for left, right in zip(spans, spans[1:]))
    with pytest.raises(ValueError, match="overlap"):
        unique_page_spans(exports, {"full": [1], "gdn": [1]})


def test_physical_spec_and_native_view_are_required(native_specs):
    _, native, state, specs, _ = make_exports(native_specs)
    with pytest.raises(ValueError, match="metadata128"):
        export_layer_cache("full", native["full"], state=state,
                           spec=specs["full"].copy_with_new_block_size(128))
    with pytest.raises(ValueError, match="native attention view"):
        export_layer_cache("full", native["full"].clone(), state=state)
    with pytest.raises(ValueError, match="request-window"):
        export_layer_cache("full", native["full"], state=state,
                           spec=SimpleNamespace(request_window_storage=True))
    wrong = SimpleNamespace(**vars(state))
    wrong.window_tags = torch.empty_like(state.window_tags)
    with pytest.raises(ValueError, match="byte layout"):
        export_layer_cache("full", native["full"], state=wrong)


def test_page_validation_does_not_scale_partial_components(native_specs):
    exports, _, _, _, _ = make_exports(native_specs)
    full = exports["full"]
    for count in (0, 1, 4, full.storage_block_size):
        assert sum(s.length for s in full.page_spans([0], valid_tokens=[count])) == full.page_bytes
    for pages in ([-1], [full.num_pages], [True]):
        with pytest.raises(ValueError):
            full.page_spans(pages)
    for counts in ([full.storage_block_size + 1], [-1], [True], []):
        with pytest.raises(ValueError):
            full.page_spans([0], valid_tokens=counts)
    assert full.page_spans([]) == ()
    with pytest.raises(ValueError, match="contiguous"):
        CacheComponent.from_tensor("holes", torch.zeros(3, 4, 5)[:, :, ::2])
    with pytest.raises(ValueError, match="overlap"):
        CacheComponent.from_tensor("overlap", torch.zeros(1, 4).expand(3, -1))


def test_geometry_identity_ignores_pool_capacity_and_addresses(native_specs):
    first, *_ = make_exports(native_specs, blocks=3, offset=4096)
    second, *_ = make_exports(native_specs, blocks=9, offset=8192)
    assert {name: export.geometry() for name, export in first.items()} == {
        name: export.geometry() for name, export in second.items()}


def test_registered_region_uses_aligned_raw_slice_not_unaligned_owner():
    alignment = 2 * 1024 * 1024
    owner = torch.empty(alignment + 512, dtype=torch.uint8)
    offset = (-owner.data_ptr()) % alignment
    raw = owner[offset:offset + 512]
    tensor = raw.as_strided((3, 32), (128, 1), offset + 16)
    component = CacheComponent.from_tensor("packed", tensor)
    export = LayerCacheExport("full", "oscar_full", 128, (component,), tensor)
    region, = registration_regions({"full": export})
    assert region.address == raw.data_ptr() and region.address % alignment == 0
    assert region.length == 16 + 2 * 128 + 32
    assert owner.data_ptr() <= region.address
    assert region.address + region.length <= owner.data_ptr() + owner.numel()
    # No silently extending registration outside the real allocation.
    with pytest.raises(ValueError, match="aligned registration"):
        registration_regions({"full": export}, alignment=1 << 63)


def test_gdn_preserves_native_mixed_dtypes_without_bf16_assumption():
    conv = torch.arange(3 * 6 * 8, dtype=torch.bfloat16).reshape(3, 6, 8)
    ssm = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)
    spec = SimpleNamespace(block_size=768, shapes=((6, 8), (4, 8)),
                           dtypes=(torch.bfloat16, torch.float32), mamba_cache_mode="align")
    exported = export_layer_cache("gdn", (conv, ssm), spec=spec)
    assert [span.length for span in exported.page_spans([2], valid_tokens=[1])] == [96, 128]
    assert [component.dtype for component in exported.components] == ["bfloat16", "float32"]
