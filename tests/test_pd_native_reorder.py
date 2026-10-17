"""Archive #34/#36/#129/#140/#142/#157-163/#165: PD request-domain seams.

Execute the pinned native reorder function unchanged, followed by production
OSCAR metadata and split planning. The small input-batch fixture implements
only the swap callback; it does not emulate attention, GDN, or NPU execution.
Every permutation checks request-owned lengths, virtual block tables and slots
against independently identified requests. This is host contract evidence,
not graph replay, numerical acceptance, or a performance measurement.
"""

from enum import Enum
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oscar_ascend.integration.metadata import from_common
from oscar_ascend.integration.mixed_decode_split import (
    _slice_metadata,
    plan_mixed_decode,
)
from test_runtime_native import definitions


class AscendAttentionState(Enum):
    ChunkedPrefill = 1


# Query length, already computed tokens, total prompt tokens, native region.
# In particular, the warm q4 row has history despite sharing a decode shape,
# and a three-token fresh request belongs after the long continuation.
REQUESTS = {
    "decode1": (1, 32768, 32768, 0),
    "decode4": (4, 65536, 32768, 0),
    "tail4": (4, 32764, 32768, 1),
    "continue512": (512, 16384, 32768, 2),
    "fresh3": (3, 0, 3, 3),
    "fresh8192": (8192, 0, 32768, 3),
}
REQUEST_INDEX = {name: index for index, name in enumerate(REQUESTS)}


def _owned_table(name):
    # Distinct physical ownership, with enough virtual128 columns for the
    # longest sequence. No two requests can accidentally share a valid slot.
    return torch.arange(520, dtype=torch.int32) + 1 + REQUEST_INDEX[name] * 1024


class Batch:
    def __init__(self, names):
        self.req_ids = list(names)
        self.num_computed_tokens_cpu = np.array([REQUESTS[name][1] for name in names])
        self.num_prompt_tokens = np.array([REQUESTS[name][2] for name in names])
        self.tables = torch.stack([_owned_table(name) for name in names])

    def swap_states(self, left, right):
        self.req_ids[left], self.req_ids[right] = self.req_ids[right], self.req_ids[left]
        for values in (self.num_computed_tokens_cpu, self.num_prompt_tokens):
            values[[left, right]] = values[[right, left]]
        self.tables[[left, right]] = self.tables[[right, left]]


def _common(batch):
    query_lengths = torch.tensor([REQUESTS[name][0] for name in batch.req_ids], dtype=torch.int32)
    starts = torch.cat((torch.zeros(1, dtype=torch.int32), query_lengths.cumsum(0))).to(torch.int32)
    seq_lens = query_lengths + torch.from_numpy(batch.num_computed_tokens_cpu).to(torch.int32)
    slots = []
    for row, name in enumerate(batch.req_ids):
        query_len, context, _, _ = REQUESTS[name]
        positions = torch.arange(context, context + query_len, dtype=torch.int64)
        slots.append(batch.tables[row, positions // 128].to(torch.int64) * 128 + positions % 128)
    tokens = int(starts[-1])
    return SimpleNamespace(
        causal=True, query_start_loc=starts, query_start_loc_cpu=starts.clone(),
        seq_lens=seq_lens, seq_lens_cpu_upper_bound=seq_lens.clone(),
        slot_mapping=torch.cat(slots), block_table_tensor=batch.tables,
        num_reqs=len(batch.req_ids), num_actual_tokens=tokens, num_input_tokens=tokens,
        max_query_len=int(query_lengths.max()), max_seq_len=int(seq_lens.max()),
        attn_state=AscendAttentionState.ChunkedPrefill,
    )


def _assert_request_domains(metadata, names):
    starts = metadata.query_start_loc.tolist()
    assert starts == metadata.query_start_loc_cpu.tolist()
    assert starts[0] == 0 and starts[-1] == metadata.num_actual_tokens
    assert metadata.num_reqs == len(names)
    for row, name in enumerate(names):
        query_len, context, _, _ = REQUESTS[name]
        begin, end = starts[row:row + 2]
        assert end - begin == query_len
        # The historical domain [0, context) and current domain
        # [context, context+query_len) retain this request's exact bounds.
        assert int(metadata.seq_lens[row]) == context + query_len
        assert int(metadata.seq_lens[row]) - (end - begin) == context
        expected_table = _owned_table(name)
        assert torch.equal(metadata.block_tables[row], expected_table)
        positions = torch.arange(context, context + query_len, dtype=torch.int64)
        expected_slots = expected_table[positions // 128].to(torch.int64) * 128 + positions % 128
        assert torch.equal(metadata.slot_mapping[begin:end], expected_slots)


@pytest.mark.parametrize("draft", [False, True], ids=["main", "first_mtp"])
def test_all_native_pd_orders_preserve_request_domains(monkeypatch, draft):
    native = definitions(
        "references/vllm/vllm/v1/attention/backends/utils.py",
        "native_pd_request_reorder", {"reorder_batch_to_split_decodes_and_prefills"},
        {"np": np}, monkeypatch,
    )
    scheduler = SimpleNamespace(num_scheduled_tokens={name: data[0] for name, data in REQUESTS.items()})
    checked = 0
    for order in permutations(REQUESTS):
        batch = Batch(order)
        native.reorder_batch_to_split_decodes_and_prefills(batch, scheduler, decode_threshold=4)
        assert [REQUESTS[name][3] for name in batch.req_ids] == [0, 0, 1, 2, 3, 3]
        assert set(batch.req_ids) == set(REQUESTS)
        assert batch.num_computed_tokens_cpu.tolist() == [REQUESTS[name][1] for name in batch.req_ids]
        assert batch.num_prompt_tokens.tolist() == [REQUESTS[name][2] for name in batch.req_ids]

        common = _common(batch)
        metadata = from_common(
            common, current_only=True, is_draft=draft,
            first_draft_current_only=draft, first_draft_current_fia=draft,
            mixed_decode_split=True,
        )
        _assert_request_domains(metadata, batch.req_ids)
        assert torch.equal(common.seq_lens_cpu_upper_bound, metadata.seq_lens)
        current = metadata.current_only_plan
        assert (current.prefix_requests, current.prefix_tokens) == (4, 521)
        assert (current.fresh_requests, current.fresh_tokens) == (2, 8195)
        assert all(REQUESTS[name][1] == 0 for name in batch.req_ids[current.prefix_requests:])
        assert current.cumulative == tuple(
            torch.tensor([REQUESTS[name][0] for name in batch.req_ids[4:]]).cumsum(0).tolist()
        )

        split = plan_mixed_decode(metadata, common.num_input_tokens)
        assert (split.prefix_requests, split.prefix_tokens) == (3, 9)
        assert batch.req_ids[split.prefix_requests] == "continue512"
        for rb, re, tb, te, segment in (
            (0, 3, 0, 9, "short_prefix"),
            (3, 6, 9, common.num_actual_tokens, "long_suffix"),
        ):
            child = _slice_metadata(metadata, rb, re, tb, te, common.num_input_tokens, segment)
            _assert_request_domains(child, batch.req_ids[rb:re])
            assert child.block_tables.data_ptr() == metadata.block_tables[rb:].data_ptr()
            assert child.slot_mapping.data_ptr() == metadata.slot_mapping[tb:].data_ptr()
            assert child.is_draft is draft and child.draft_index == 0
            assert not child.mixed_decode_split
        checked += 1
    assert checked == 720
