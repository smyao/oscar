"""Archive #26/#37-49: native full-page prefix and bounded MTP rollback.

These tests establish byte/lifetime algebra on CPU, not device operator
completion. Native FullAttentionManager.cache_blocks caches floor(L/B) pages;
MTP reuses physical slots past the accepted boundary on its next forward.
"""
import pytest
import torch

from oscar_ascend.layout import HybridPageLayout, SlotLayout
from oscar_ascend.lifecycle import SnapshotLayout, causal_segments


@pytest.fixture
def snapshot():
    return SnapshotLayout(HybridPageLayout(SlotLayout(256), 15360, 393216, 768,
                                          native_page_size_bytes=801792), 64, 256, 3)


def test_target_snapshots_use_only_native_padding(snapshot):
    blocks = 7
    assert snapshot.required_bytes == 333336
    assert snapshot.page.block_size == 2304
    assert snapshot.required_bytes < 393216
    for page in range(blocks):
        begin, end = snapshot.interval(blocks, page)
        assert begin >= blocks * (15360 + 393216)
        assert end <= blocks * 801792
        for gdn_page in range(blocks):
            for left, right in (snapshot.page.conv_interval(blocks, gdn_page),
                                snapshot.page.ssm_interval(blocks, gdn_page)):
                assert end <= left or begin >= right


def test_views_alias_original_storage_and_preserve_every_gdn_byte(snapshot):
    blocks = 3
    raw = torch.full((blocks * 801792,), 91, dtype=torch.uint8)
    keys, values, tags = snapshot.views(raw, blocks)
    before = raw[:blocks * (15360 + 393216)].clone()
    keys.fill_(1)
    values.fill_(2)
    tags.fill_(-1)
    assert keys.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert values.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert tags.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    torch.testing.assert_close(raw[:before.numel()], before, rtol=0, atol=0)
    assert keys.stride(0) * keys.element_size() == 393216
    assert tags.stride(0) * tags.element_size() == 393216
    assert torch.all(keys == 1) and torch.all(values == 2) and torch.all(tags == -1)


def _publish_raw(page_values, tags, positions, *, block_size, sink, ring):
    # Independent scalar simulation of the physical ABI. CPU only; there is
    # no production import of this test. Epoch/value is included in test data.
    for logical, value in positions:
        page, offset = divmod(logical, block_size)
        if logical < sink:
            page_values[page, logical] = value
            tags[page, logical] = offset
        row = sink + offset % ring
        page_values[page, row] = value
        tags[page, row] = offset


@pytest.mark.parametrize("context", [64, 255, 256, 319, 320, 321, 2302, 2304, 2305, 4608, 10000])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_all_mtp_acceptance_lengths_preserve_committed_exact_window(snapshot, context, accepted):
    # Verify writes one known token plus three proposals; even if none of
    # the proposals is accepted, rejected writes can be at most three ahead.
    block_size = snapshot.page.block_size
    pages = (context + 7 + block_size - 1) // block_size
    values = torch.full((pages, snapshot.rows), -1, dtype=torch.int64)
    tags = torch.full_like(values, -1)
    _publish_raw(values, tags, ((p, p + 100000) for p in range(context + 4)),
                 block_size=block_size, sink=64, ring=259)
    next_context = context + 1 + accepted
    for p in list(range(min(64, next_context))) + list(range(max(64, next_context - 256), next_context)):
        page, offset = divmod(p, block_size)
        row = snapshot.row_for_position(p)
        assert tags[page, row] == offset
        assert values[page, row] == p + 100000


def test_every_cached_prefix_boundary_retains_exact_tail_after_long_prefill(snapshot):
    # Native prefix caching retains any complete block boundary, including
    # boundaries far behind the request's current recent window.
    count = 2304 * 6 + 333
    values = torch.full((7, snapshot.rows), -1, dtype=torch.int64)
    tags = torch.full_like(values, -1)
    _publish_raw(values, tags, ((p, p) for p in range(count)),
                 block_size=2304, sink=64, ring=259)
    for boundary in range(2304, count, 2304):
        for p in list(range(64)) + list(range(boundary - 256, boundary)):
            page, offset = divmod(p, 2304)
            row = snapshot.row_for_position(p)
            assert tags[page, row] == offset and values[page, row] == p


def test_recycled_page_does_not_need_a_batch_row_owner(snapshot):
    values = torch.full((1, snapshot.rows), -1, dtype=torch.int64)
    tags = torch.full_like(values, -1)
    _publish_raw(values, tags, ((p, p) for p in range(2304)),
                 block_size=2304, sink=64, ring=259)
    # Allocator frees the whole page; a new request first recomputes its own
    # context before it can read it. Old extra rows are outside its seq_len.
    _publish_raw(values, tags, ((p, p + 90000) for p in range(400)),
                 block_size=2304, sink=64, ring=259)
    for p in list(range(64)) + list(range(144, 400)):
        assert values[0, snapshot.row_for_position(p)] == p + 90000


@pytest.mark.parametrize("length", [1, 63, 64, 65, 319, 320, 321, 16384, 32768, 50000, 262144])
def test_causal_partition_has_no_duplicates_or_missing_positions(length):
    context = length - 1
    segments = causal_segments(context, length - 1, 64, 256)
    assert segments[0][0] == 0 and segments[-1][1] == length
    assert all(left <= right for left, right in segments)
    assert all(segments[i][1] == segments[i + 1][0] for i in range(3))
    assert segments[2][1] - segments[2][0] <= 256
    assert sum(right - left for left, right in segments) == length


def test_snapshot_capacity_mismatch_is_not_an_int2_window_substitution():
    page = HybridPageLayout(SlotLayout(256), 15360, 393216, 768)
    with pytest.raises(ValueError, match="exact prefix snapshots need"):
        SnapshotLayout(page, 64, 256, 3)


def test_current_chunk_is_uncompressed_without_recovering_history():
    assert causal_segments(50000, 50003, 64, 256) == (
        (0, 64), (64, 49748), (49748, 50000), (50000, 50004))
