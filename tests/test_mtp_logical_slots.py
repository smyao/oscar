"""Archive #157-161: separate logical KV ordinals from M-RoPE coordinates.

The pinned native later-update method is executed unchanged. A materialized
logical KV prefix is the storage oracle: reading a longer RoPE coordinate is
not allowed to fabricate missing KV. These are host contracts, not NPU proof.
"""
from types import MethodType, SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.draft_discard import (
    later_draft_lengths_copy, later_draft_slot_copy, native_draft_scope)
from test_first_draft_current_only import builder as make_builder
from test_first_draft_padded_native import Common
from test_native_discard_pipeline import _native


def _case(*, discard=False, native_padding=False):
    table = torch.tensor([[10, 12, 15]], dtype=torch.int32)
    groups = [torch.full((4,), -1, dtype=torch.int32) for _ in range(3)]
    groups[0].copy_(torch.arange(10 * 128 + 124, 10 * 128 + 128, dtype=torch.int32))
    groups[1][0] = -1 if native_padding else 12 * 128
    runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=1),
        discard_request_mask=SimpleNamespace(gpu=torch.tensor([discard])),
        pcp_manager=None, actual_seq_lengths_q=[4], attn_state=None,
        decode_token_per_req=4)
    proposer = SimpleNamespace(runner=runner, pcp_size=1, dcp_size=1, method="mtp",
        arange=torch.arange(4, dtype=torch.int32),
        token_arange_np=torch.arange(4, dtype=torch.int32),
        uses_mrope=True, has_gdn=True, kernel_block_size=128, max_model_len=262144,
        use_compress=False, slot_mapping_group=groups,
        seq_lens_group=[torch.full((1,), 128 + i, dtype=torch.int32) for i in range(3)],
        query_start_loc_group=[torch.zeros(2, dtype=torch.int32) for _ in range(3)])
    common = Common(query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([128], dtype=torch.int32), seq_lens_cpu=None,
        _seq_lens_cpu=torch.tensor([128], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([128], dtype=torch.int32),
        num_reqs=1, num_actual_tokens=4, num_input_tokens=4, max_query_len=4,
        num_computed_tokens_cpu=None, _num_computed_tokens_cpu=None,
        block_table_tensor=table, slot_mapping=groups[0],
        positions=torch.arange(124, 128), positions_cpu=None,
        attn_state=None, decode_token_per_req=4, is_prefilling=torch.tensor([False]),
        max_seq_len=128)
    return proposer, common


@pytest.mark.parametrize("rope_delta", [-17, 0, 17])
def test_native_later_mrope_preserves_materialized_logical_prefix(monkeypatch, rope_delta):
    native, states, modes = _native(monkeypatch)
    builder = make_builder(monkeypatch)
    for rejected_count in range(4):
        proposer, common = _case()
        common.attn_state = states.SpecDecoding
        proposer.runner.attn_state = states.SpecDecoding
        proposer.shallow_copy_metadata = MethodType(native.shallow_copy_metadata, proposer)
        group = SimpleNamespace(get_metadata_builder=lambda: builder)
        # First draft already wrote every scheduled logical position. Only
        # the accepted prefix is visible to subsequent draft attention.
        physical = lambda pos: int(common.block_table_tensor[0, pos // 128]) * 128 + pos % 128
        memory = {physical(pos): ("first", pos) for pos in range(128)}
        accepted = [("first", pos) for pos in range(128 - rejected_count)]
        used_positions = torch.full((3, 1), len(accepted) - 1 + rope_delta, dtype=torch.int64)
        rejected = torch.tensor([rejected_count], dtype=torch.int32)
        with native_draft_scope(proposer, common=common, rejected_tokens=rejected):
            previous = common
            for index in (1, 2):
                previous, metadata = native.attn_update_stack_num_spec_norm(
                    proposer, index, None, previous, 1, 4, used_positions,
                    modes.NONE, attn_group=group)
                native_slot = int(proposer.slot_mapping_group[index][0])
                logical_slot = int(metadata.slot_mapping[0])
                assert logical_slot == physical(len(accepted))
                assert [memory[physical(pos)] for pos in range(len(accepted))] == accepted
                assert int(previous.seq_lens[0]) == 128 + index
                assert torch.equal(metadata.seq_lens, previous.seq_lens)
                assert used_positions.tolist() == [[len(accepted) + rope_delta]] * 3
                if rope_delta == 0:
                    assert torch.equal(metadata.slot_mapping, proposer.slot_mapping_group[index])
                else:
                    assert logical_slot != native_slot
                value = ("later", index)
                memory[logical_slot] = value
                accepted.append(value)


def _later_common(proposer, original):
    return SimpleNamespace(slot_mapping=proposer.slot_mapping_group[1],
        seq_lens=proposer.seq_lens_group[1], block_table_tensor=original.block_table_tensor,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_reqs=1, num_actual_tokens=1, max_query_len=1, max_seq_len=129)


@pytest.mark.parametrize("bad_rejected", [-1, 5])
def test_invalid_live_rejection_reaches_guard_without_becoming_padding(bad_rejected):
    proposer, original = _case()
    common = _later_common(proposer, original)
    source = common.slot_mapping.clone()
    with native_draft_scope(proposer, common=original,
            rejected_tokens=torch.tensor([bad_rejected], dtype=torch.int32)):
        slots = later_draft_slot_copy(common, draft_index=1)
        lengths = later_draft_lengths_copy(common, slots, draft_index=1)
    assert int(slots[0]) >= 0
    assert int(lengths[0]) == 0  # qLen=1 gives negative context: frozen prepare error 1.
    assert torch.equal(common.slot_mapping, source)


def test_corrupt_live_page_id_reaches_guard_without_becoming_padding():
    proposer, original = _case()
    original.block_table_tensor[0, 1] = -2
    common = _later_common(proposer, original)
    with native_draft_scope(proposer, common=original,
            rejected_tokens=torch.tensor([0], dtype=torch.int32)):
        slots = later_draft_slot_copy(common, draft_index=1)
        lengths = later_draft_lengths_copy(common, slots, draft_index=1)
    assert int(slots[0]) >= 0
    assert int(lengths[0]) == 0


@pytest.mark.parametrize("discard,native_padding", [(True, False), (False, True)])
def test_explicit_discard_and_native_padding_remain_padding(discard, native_padding):
    proposer, original = _case(discard=discard, native_padding=native_padding)
    common = _later_common(proposer, original)
    rejected = torch.tensor([4 if discard else 0], dtype=torch.int32)
    with native_draft_scope(proposer, common=original, rejected_tokens=rejected):
        slots = later_draft_slot_copy(common, draft_index=1)
        later_draft_lengths_copy(common, slots, draft_index=1)
    assert slots.tolist() == [-1, -1, -1, -1]


@pytest.mark.parametrize("positional", [False, True])
def test_proposer_hook_binds_this_calls_common_and_rejection_arguments(positional):
    from oscar_ascend import plugin

    class NativeProposer:
        method = "mtp"

        # Keep the pinned native positional order, including rejection at 15.
        def _propose(self, target_token_ids, target_positions, target_hidden_states,
                next_token_ids, token_indices_to_sample, common_attn_metadata,
                target_model_batch_desc, sampling_metadata, mm_embed_inputs=None,
                req_scheduled_tokens=None, long_seq_metadata=None, num_prefill_reqs=0,
                num_decode_reqs=0, scheduler_output=None, num_scheduled_tokens=0,
                num_rejected_tokens_gpu=None):
            common = _later_common(self, common_attn_metadata)
            slots = later_draft_slot_copy(common, draft_index=1)
            lengths = later_draft_lengths_copy(common, slots, draft_index=1)
            return slots, lengths

        def dummy_run(self):
            return None

    base, common = _case()
    proposer = NativeProposer()
    proposer.__dict__.update(vars(base))
    rejected = torch.tensor([3], dtype=torch.int32)
    original = NativeProposer._propose
    try:
        plugin._patch_mtp_proposer(SimpleNamespace(AscendSpecDecodeBaseProposer=NativeProposer))
        if positional:
            slots, lengths = proposer._propose(
                None, None, None, None, None, common, None, None,
                None, None, None, 0, 0, None, 4, rejected)
        else:
            slots, lengths = proposer._propose(
                target_token_ids=None, target_positions=None, target_hidden_states=None,
                next_token_ids=None, token_indices_to_sample=None,
                common_attn_metadata=common, target_model_batch_desc=None,
                sampling_metadata=None, num_rejected_tokens_gpu=rejected)
        assert slots.tolist() == [10 * 128 + 125, -1, -1, -1]
        assert lengths.tolist() == [129]
    finally:
        plugin.unregister()
    assert NativeProposer._propose is original
