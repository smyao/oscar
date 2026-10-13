"""Archive #34/#36/#142/#155: epoch, traversal and arena ownership contracts."""
from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch
from oscar_ascend.integration.prepare_cache import prepare_key, reuse_prepare, commit_prepare, invalidate_prepare
from oscar_ascend.integration.metadata import OscarMetadata


@pytest.mark.parametrize("later", [False, True])
def test_forward_reuses_only_prepare_keeps_every_layer_compute_store_guard(later):
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        128, 20, 16, 8, 16, later, True, True, True, _reuse_layers=True)


def metadata():
    return OscarMetadata(query_start_loc=torch.tensor([0, 4]), seq_lens=torch.tensor([100]),
        block_tables=torch.zeros(1, 8, dtype=torch.int32), slot_mapping=torch.arange(4),
        num_reqs=1, num_actual_tokens=4, max_query_len=4, max_seq_len=100, prepare_epoch=object())


def key(md, **kw):
    return prepare_key(md, tokens=4, heads=6, kv_heads=1, sink=64, recent=256, splits=1,
                       native_current=kw.get("native_current", False))


@pytest.mark.parametrize("change", ["epoch", "slot", "table", "draft", "capture", "dummy", "suppress", "partition", "overwrite", "repeat"])
def test_same_addresses_cannot_hide_a_new_epoch_or_incompatible_use(change):
    md = metadata(); workspace = SimpleNamespace(); k = key(md)
    assert not reuse_prepare(workspace, k, "l0")
    commit_prepare(workspace, k, "l0")
    assert reuse_prepare(workspace, key(md), "l1")
    if change == "epoch": md = replace(md, prepare_epoch=object())
    elif change == "slot": md = replace(md, slot_mapping=md.slot_mapping.clone())
    elif change == "table": md = replace(md, block_tables=md.block_tables.clone())
    elif change == "draft": md = replace(md, draft_index=1)
    elif change == "capture": md = replace(md, capture_origin=True)
    elif change == "dummy": md = replace(md, dummy_origin=True)
    elif change == "partition": md = replace(md, mixed_split_segment="long_suffix")
    elif change == "overwrite": invalidate_prepare(workspace)
    candidate = key(md, native_current=change == "suppress")
    assert not reuse_prepare(workspace, candidate, "l0" if change == "repeat" else "l2")


def test_current_copy_is_visible_to_passive_sampler():
    from oscar_ascend.passive_timing import _OSCAR_PHASES
    assert "current_only_copy" in _OSCAR_PHASES
