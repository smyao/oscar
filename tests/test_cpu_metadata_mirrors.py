"""Archive #34/#36/#69/#140/#142/#157-165: CPU mirror boundaries stay checked.

Meta tensors stand in for a non-CPU input without initializing an NPU. A spy
on Tensor.tolist proves malformed/cached mirrors fail before values are read.
These are host fail-closed contracts, not evidence of a production NPU fault.
"""

from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.cpu_metadata import (
    cpu_int_mirror_values,
    validate_cpu_int_mirror,
)
from oscar_ascend.integration.metadata import OscarMetadataError, from_common
from oscar_ascend.integration.mixed_decode_split import _slice_metadata


class AscendAttentionState(Enum):
    ChunkedPrefill = 1


def _common():
    starts = torch.tensor([0, 4, 7], dtype=torch.int32)
    return SimpleNamespace(
        causal=True, query_start_loc=starts, query_start_loc_cpu=starts.clone(),
        seq_lens=torch.tensor([104, 3], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([104, 3], dtype=torch.int32),
        block_table_tensor=torch.zeros((2, 2), dtype=torch.int32),
        slot_mapping=torch.arange(7, dtype=torch.int64),
        num_reqs=2, num_actual_tokens=7, num_input_tokens=7,
        max_query_len=4, max_seq_len=104, attn_state=AscendAttentionState.ChunkedPrefill,
    )


@pytest.fixture
def read_spy(monkeypatch):
    reads = []
    original = torch.Tensor.tolist

    def recorded(tensor):
        reads.append((tensor.device.type, tensor.dtype, tuple(tensor.shape)))
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", recorded)
    return reads


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_only_requested_cpu_integer_prefix_is_read(dtype, read_spy):
    values = torch.tensor([0, 4, 7, 99], dtype=dtype)
    assert validate_cpu_int_mirror(values, 3, label="mirror") is values
    assert read_spy == []
    assert cpu_int_mirror_values(values, 3, label="mirror") == (0, 4, 7)
    assert read_spy == [("cpu", dtype, (3,))]


@pytest.mark.parametrize("bad", [
    torch.empty(3, dtype=torch.int32, device="meta"),
    torch.zeros(3, dtype=torch.float32),
    torch.zeros((1, 3), dtype=torch.int32),
    torch.zeros(2, dtype=torch.int32),
    None,
    [0, 4, 7],
])
def test_invalid_mirror_fails_before_any_tensor_value_read(bad, read_spy):
    with pytest.raises(ValueError, match="mirror is missing or invalid"):
        cpu_int_mirror_values(bad, 3, label="mirror")
    assert read_spy == []


@pytest.mark.parametrize("current_only", [False, True])
def test_cached_cumulative_does_not_authorize_non_cpu_query_mirror(current_only, read_spy):
    common = _common()
    first = from_common(common, current_only=current_only)
    assert first.current_cumulative == (4, 7)
    common.query_start_loc_cpu = torch.empty(3, dtype=torch.int32, device="meta")
    read_spy.clear()
    with pytest.raises(OscarMetadataError, match="CPU query-start mirror"):
        from_common(common, current_only=current_only)
    assert read_spy == []


@pytest.mark.parametrize("bad", [
    torch.empty(2, dtype=torch.int32, device="meta"),
    torch.tensor([104.0, 3.0]),
    torch.tensor([104], dtype=torch.int32),
])
def test_current_only_upper_bound_rejects_noninteger_or_device_mirror(bad, read_spy):
    common = _common()
    from_common(common)  # Seed the cumulative cache without requiring bounds.
    common.seq_lens_cpu_upper_bound = bad
    read_spy.clear()
    with pytest.raises(OscarMetadataError, match="CPU sequence upper bounds"):
        from_common(common, current_only=True)
    assert read_spy == []


def test_mixed_slice_cannot_convert_an_unchecked_device_mirror(read_spy):
    metadata = from_common(_common(), mixed_decode_split=True)
    from dataclasses import replace
    metadata = replace(metadata, query_start_loc_cpu=torch.empty(3, dtype=torch.int32, device="meta"))
    read_spy.clear()
    with pytest.raises(ValueError, match="CPU query-start mirror"):
        _slice_metadata(metadata, 0, 1, 0, 4, 7, "short_prefix")
    assert read_spy == []
