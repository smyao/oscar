"""Archive #34/#36/#157-159: native B32 -> 12 -> 1 metadata lifecycle.

Execute pinned native padding and MTP update methods unchanged on CPU.
These tests establish metadata/address contracts, not NPU graph replay or
attribution of the user's asynchronous stable-service error.
"""
from types import MethodType, SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.compact_later_mtp import _compact_call, should_compact_rows
from oscar_ascend.integration.draft_discard import native_draft_scope
from oscar_ascend.integration.metadata import GraphMetadataBindings, from_common
from test_compact_later_mtp import FLAGS, _case
from test_first_draft_current_only import builder as make_builder
from test_first_draft_padded_native import Common
from test_later_draft_slots import _prepare_status
from test_native_discard_pipeline import _native
from test_runtime_native import definitions


def test_native_uniform_graph_padding_drains_in_same_fixed_buffers(monkeypatch):
    modes = SimpleNamespace(FULL=object(), FULL_AND_PIECEWISE=object())
    native = definitions(
        "references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
        "native_drain_padding", set(),
        {"CUDAGraphMode": modes,
         "copy_snapshot_to_gpu": lambda buf: buf.gpu.copy_(buf.np)}, monkeypatch,
        methods=[("NPUModelRunner", "_pad_query_start_loc_for_fia")])
    starts = SimpleNamespace(np=torch.zeros(34, dtype=torch.int32),
                             gpu=torch.zeros(34, dtype=torch.int32))
    runner = SimpleNamespace(uniform_decode_query_len=4,
        compilation_config=SimpleNamespace(cudagraph_mode=modes.FULL_AND_PIECEWISE),
        arange_np=torch.arange(34, dtype=torch.int32))
    lengths = torch.zeros(32, dtype=torch.int32)
    table = torch.zeros(32, 264, dtype=torch.int32)
    slots = torch.full((128,), -1, dtype=torch.int64)
    common = Common(query_start_loc=starts.gpu[:33], seq_lens=lengths,
        block_table_tensor=table, slot_mapping=slots, num_reqs=32,
        num_actual_tokens=128, num_input_tokens=128, max_query_len=4,
        max_seq_len=33000)
    bindings = GraphMetadataBindings()
    captured = bindings.bind(from_common(common, capture_origin=True))
    for size in (32, 12, 1):
        # Native _prepare_inputs publishes the active prefix; the real native
        # padding method below must replace stale larger-batch qstart tails.
        starts.np[:size + 1] = torch.arange(size + 1, dtype=torch.int32) * 4
        padded = native._pad_query_start_loc_for_fia(
            runner, starts, 128, 32, size, modes.FULL, 32)
        assert padded == 32
        # Match the native _build_attention_metadata padding writes.
        slots[:size * 4] = torch.arange(size * 4)
        slots[size * 4:].fill_(-1)
        lengths[:size].fill_(33000)
        lengths[size:].zero_()
        table[size:].zero_()
        common.num_actual_tokens = size * 4
        actual = bindings.bind(from_common(common))
        assert torch.equal(actual.query_start_loc, torch.arange(33) * 4)
        assert actual.slot_mapping[size * 4:].eq(-1).all()
        assert captured.query_start_loc.data_ptr() == actual.query_start_loc.data_ptr()
        assert captured.seq_lens.data_ptr() == actual.seq_lens.data_ptr()
        assert captured.slot_mapping.data_ptr() == actual.slot_mapping.data_ptr()
        assert not should_compact_rows(size * 4, size)


@pytest.mark.parametrize("input_capacity", [128, 1024])
def test_native_later_draft_drain_refreshes_tail_and_compact_rows(monkeypatch, input_capacity):
    native, states, modes = _native(monkeypatch)
    builder = make_builder(monkeypatch)
    model, context, values, public_call, seen = _case(monkeypatch)
    runner = SimpleNamespace(pcp_manager=None,
        discard_request_mask=SimpleNamespace(gpu=torch.ones(32, dtype=torch.bool)),
        attn_state=states.SpecDecoding, decode_token_per_req=4)
    proposer = SimpleNamespace(runner=runner, pcp_size=1, dcp_size=1, method="mtp",
        arange=torch.arange(input_capacity, dtype=torch.int32),
        token_arange_np=torch.arange(input_capacity, dtype=torch.int32), uses_mrope=True,
        has_gdn=True, kernel_block_size=128, max_model_len=262144, use_compress=False,
        slot_mapping_group=[torch.full((input_capacity,), -1, dtype=torch.int32) for _ in range(3)],
        seq_lens_group=[torch.zeros(32, dtype=torch.int32) for _ in range(3)],
        query_start_loc_group=[torch.zeros(34, dtype=torch.int32) for _ in range(3)])
    proposer.shallow_copy_metadata = MethodType(native.shallow_copy_metadata, proposer)
    group = SimpleNamespace(get_metadata_builder=lambda: builder)
    addresses = {}
    for size in (32, 12, 1):
        # Survivors are reordered, like the native input-batch compaction.
        ids = torch.arange(32 - size, 32).flip(0)
        runner.input_batch = SimpleNamespace(num_reqs=size)
        runner.discard_request_mask.gpu[:size].zero_()
        query_lengths = torch.full((size,), 4, dtype=torch.int32)
        draft_counts = torch.full((size,), 3, dtype=torch.int32)
        counts = (1 + ids % 4).clone()
        if input_capacity == 1024:
            # A legal mixed prefill/decode first pass has N1024, then its
            # later drafts retain N1024 while only B leading rows are live.
            # Do not invent an eager 1024-row padding envelope for pure q4.
            query_lengths[-1] = input_capacity - 4 * (size - 1)
            draft_counts[-1] = 0
            counts[-1] = 1
        runner.attn_state = states.ChunkedPrefill if input_capacity == 1024 else states.SpecDecoding
        prefilling = torch.zeros(size, dtype=torch.bool)
        if input_capacity == 1024:
            prefilling[-1] = True
        starts = torch.cat((torch.zeros(1, dtype=torch.int32), query_lengths.cumsum(0))).to(torch.int32)
        runner.actual_seq_lengths_q = starts[1:].tolist()
        lengths = (32768 + ids * 4).to(torch.int32)
        table = (ids[:, None] * 528 + torch.arange(264)[None, :]).to(torch.int32)
        positions = torch.full((input_capacity,), 99999, dtype=torch.int64)
        for row in range(size):
            positions[starts[row]:starts[row + 1]] = torch.arange(
                int(lengths[row] - query_lengths[row]), int(lengths[row]))
        common = Common(query_start_loc=starts, query_start_loc_cpu=starts.clone(),
            seq_lens=lengths.clone(), seq_lens_cpu=None, _seq_lens_cpu=lengths.clone(),
            seq_lens_cpu_upper_bound=lengths.clone(), num_reqs=size,
            num_actual_tokens=int(starts[-1]), num_input_tokens=input_capacity,
            max_query_len=int(query_lengths.max()),
            num_computed_tokens_cpu=None, _num_computed_tokens_cpu=None,
            block_table_tensor=table, slot_mapping=proposer.slot_mapping_group[0],
            positions=positions.clone(), positions_cpu=None, attn_state=runner.attn_state,
            decode_token_per_req=4, is_prefilling=prefilling,
            max_seq_len=int(lengths.max()))
        spec = SimpleNamespace(cu_num_draft_tokens=draft_counts.cumsum(0))
        previous, _, anchors, _ = native.prepare_inputs_padded(proposer, common, spec, counts)
        used = positions[anchors].repeat(3, 1)
        with native_draft_scope(proposer):
            for index in (1, 2):
                previous, metadata = native.attn_update_stack_num_spec_norm(
                    proposer, index, None, previous, size, input_capacity, used,
                    modes.NONE, attn_group=group)
                assert metadata.query_start_loc.tolist() == list(range(size + 1))
                assert metadata.slot_mapping[size:].eq(-1).all()
                assert proposer.seq_lens_group[index][size:].eq(0).all()
                assert proposer.query_start_loc_group[index][size + 1:].eq(0).all()
                for row in range(size):
                    assert _prepare_status(int(metadata.slot_mapping[row]),
                        int(metadata.seq_lens[row]), table[row].tolist()) == 0
                signature = (metadata.slot_mapping.data_ptr(), metadata.seq_lens.data_ptr(),
                             metadata.query_start_loc.data_ptr())
                assert addresses.setdefault(index, signature) == signature
                if input_capacity == 1024:
                    context.attn_metadata = {"mtp": metadata}
                    context.num_accept_tokens = size
                    before = len(seen)
                    output = _compact_call(public_call, model, (), values, None,
                                           context, FLAGS, (1, 1, 4))
                    assert output.shape == (size, 4)
                    assert seen[before][0:1] == (size,)
                    assert context.attn_metadata["mtp"] is metadata
