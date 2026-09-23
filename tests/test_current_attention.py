"""Archive #55-#69/#126/#129-#140: current-source host/ABI contracts.

These CPU checks cover stage, padding and error preservation. They do not
establish native FIA device completion, fused merge accuracy or throughput.
"""

from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.current_attention import (
    CurrentAttentionError, _empty_current_source_ranges,
    _populate_slot_error_status,
    current_cumulative_lengths, native_current_partial, use_native_current,
)
from oscar_ascend.integration.metadata import from_common


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@pytest.mark.parametrize("stage", [AscendAttentionState.PrefillNoCache,
                                   AscendAttentionState.PrefillCacheHit,
                                   AscendAttentionState.ChunkedPrefill])
def test_main_model_eager_prefill_and_mixed_use_exact_current(stage):
    metadata = SimpleNamespace(attn_state=stage, capture_origin=False, is_draft=False)
    assert use_native_current(metadata)
    metadata.is_draft = True  # First draft_index=0 is still a draft.
    assert not use_native_current(metadata)
    metadata.is_draft = False
    metadata.capture_origin = True
    assert not use_native_current(metadata)


@pytest.mark.parametrize("stage", [AscendAttentionState.DecodeOnly,
                                   AscendAttentionState.SpecDecoding])
def test_decode_and_spec_graph_keep_complete_cv_current(stage):
    metadata = SimpleNamespace(attn_state=stage, capture_origin=False, is_draft=False)
    assert not use_native_current(metadata)


def test_missing_native_stage_fails_instead_of_silent_route_change():
    metadata = SimpleNamespace(attn_state=None, capture_origin=False, is_draft=False)
    with pytest.raises(CurrentAttentionError, match="AscendAttentionState"):
        use_native_current(metadata)


def test_mixed_cpu_query_starts_strip_only_synthetic_padding():
    metadata = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 1, 7, 8], dtype=torch.int32),
                               num_reqs=3)
    assert current_cumulative_lengths(metadata, 7) == [1, 7]
    metadata.query_start_loc_cpu = torch.tensor([0, 1, 6, 8], dtype=torch.int32)
    with pytest.raises(CurrentAttentionError, match="actual token count"):
        current_cumulative_lengths(metadata, 7)
    metadata.query_start_loc_cpu = torch.tensor([0, 7, 6, 8], dtype=torch.int32)
    with pytest.raises(CurrentAttentionError, match="malformed"):
        current_cumulative_lengths(metadata, 7)
    metadata.query_start_loc_cpu = torch.empty(4, dtype=torch.int32, device="meta")
    with pytest.raises(CurrentAttentionError, match="CPU query-start mirror"):
        current_cumulative_lengths(metadata, 7)


def test_native_cpu_qstarts_are_assembled_once_for_all_full_builders(monkeypatch):
    from oscar_ascend.integration import current_attention
    common = SimpleNamespace(
        causal=True, query_start_loc=torch.tensor([0, 1, 7, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 7, 8], dtype=torch.int32),
        seq_lens=torch.tensor([32768, 23000], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5, 6, -1], dtype=torch.int64),
        block_table_tensor=torch.zeros((2, 4), dtype=torch.int32),
        num_reqs=3, num_actual_tokens=7, num_input_tokens=8,
        max_query_len=6, max_seq_len=32768,
        attn_state=AscendAttentionState.ChunkedPrefill,
    )
    calls = []
    original = current_attention.current_cumulative_lengths

    def counted(metadata, active_tokens):
        calls.append(active_tokens)
        return original(metadata, active_tokens)

    monkeypatch.setattr(current_attention, "current_cumulative_lengths", counted)
    first, second = from_common(common), from_common(common)
    assert calls == [7]  # No per-layer CPU .tolist or per-request loop.
    assert first.current_cumulative is second.current_cumulative == (1, 7)
    assert first.query_start_loc_cpu.device.type == "cpu"
    assert first.query_start_loc_cpu.tolist() == [0, 1, 7]


def test_source2_empty_range_preserves_noncurrent_and_bad_tasks():
    tasks = torch.zeros((3 * 1 * 3 * 2, 16), dtype=torch.int64)
    view = tasks.view(3, 1, 3, 2, 16)
    view[:, :, :, :, 3] = 10
    view[:, :, :, :, 4] = 14
    view[:, :, :, :, 7] = torch.arange(3).view(1, 1, 3, 1)
    current = view[:, :, 2]
    current[0, 0, 1, 10] = 5  # Metadata error remains an error.
    current[1, 0, 1, 4] = 9  # Inverted range remains invalid.
    before = tasks.clone()
    _empty_current_source_ranges(tasks, 3, 1, 2)
    assert torch.equal(view[:, :, :2], before.view_as(view)[:, :, :2])
    assert current[0, 0, 0, 4] == current[0, 0, 0, 3] == 10
    assert current[0, 0, 1, 4] == 14 and current[0, 0, 1, 10] == 5
    assert current[1, 0, 1, 4] == 9
    assert torch.equal(view[..., 7], before.view_as(view)[..., 7])


def test_internal_negative_slot_is_guard_error_before_native_current():
    status = torch.empty((3, 6), dtype=torch.int32)
    slots = torch.tensor([11, -1, -1], dtype=torch.int64)
    active = _populate_slot_error_status(status, slots, 2)
    assert torch.equal(active[:, 0], torch.tensor([0, 1], dtype=torch.int32))
    assert bool(torch.all(active[1] == 1))
    # Slot2 is trailing graph padding, so it never enters the active FIA batch.
    slots[1] = 12
    active = _populate_slot_error_status(status, slots, 2)
    assert bool(torch.all(active == 0))


def test_native_current_never_uses_cpu_operator_substitute():
    q = torch.zeros((1, 6, 64), dtype=torch.bfloat16)
    kv = torch.zeros((1, 1, 64), dtype=torch.bfloat16)
    with pytest.raises(CurrentAttentionError, match="same-NPU"):
        native_current_partial(q, kv, kv, [1], heads=6, kv_heads=1, scale=0.125)
