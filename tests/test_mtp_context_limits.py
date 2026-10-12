"""Archive #34/#37-49/#128/#142/#156/#157: native MTP boundary replay on CPU.

The real Ascend method is executed unchanged. The final slot/table lookup is
a scalar model of checked prepare, not an AscendC or NPU execution claim.
"""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.metadata import from_common, prepare_context_lengths
from test_runtime_native import definitions


def replay_native_drafts(monkeypatch, *, limit, contexts, accepted, block_size):
    native = definitions(
        "references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "native_mtp_context_limit", set(),
        {"torch": torch, "CUDAGraphMode": SimpleNamespace(FULL="full"),
         "AscendAttentionState": SimpleNamespace(SpecDecoding="spec", ChunkedPrefill="chunk"),
         "PADDING_SLOT_ID": -1}, monkeypatch,
        methods=[("AscendSpecDecodeBaseProposer", "attn_update_stack_num_spec_norm")])
    monkeypatch.setattr(torch, "from_numpy", lambda values: values.clone())
    batch, padded = len(contexts), 4 * len(contexts)
    ratio = block_size // 128
    pages = (limit + 3 + block_size - 1) // block_size
    # Reverse physical page order and retain extra virtual columns: lookup
    # cannot assume physical order or use an unbounded table scan.
    table_row = [page * ratio + column for page in range(pages, 0, -1)
                 for column in range(ratio)]
    table = torch.tensor([table_row] * batch, dtype=torch.int32)

    class Builder:
        def build_for_drafting(self, common, draft_index):
            return from_common(common, is_draft=True, draft_index=draft_index)

    proposer = SimpleNamespace(
        shallow_copy_metadata=copy.copy, arange=torch.arange(batch + 1, dtype=torch.int32),
        token_arange_np=torch.arange(batch + 1, dtype=torch.int32), method="mtp",
        uses_mrope=False, max_model_len=limit, runner=SimpleNamespace(), has_gdn=True,
        kernel_block_size=128, block_size=128, pcp_size=1, use_compress=False,
        slot_mapping_group=[torch.full((padded,), -1, dtype=torch.int32) for _ in range(3)],
        seq_lens_group=[torch.zeros(batch, dtype=torch.int32) for _ in range(3)],
        query_start_loc_group=[torch.zeros(batch + 1, dtype=torch.int32) for _ in range(3)])
    starts = torch.arange(0, padded + 1, 4, dtype=torch.int32)
    common = SimpleNamespace(
        query_start_loc=starts, query_start_loc_cpu=starts.clone(),
        seq_lens=torch.tensor([base + 4 for base in contexts], dtype=torch.int32),
        seq_lens_cpu=None, _seq_lens_cpu=None, num_computed_tokens_cpu=None,
        positions=torch.tensor(contexts, dtype=torch.int32), block_table_tensor=table,
        slot_mapping=torch.zeros(padded, dtype=torch.int32), num_reqs=batch,
        num_actual_tokens=padded, num_input_tokens=padded, max_query_len=4,
        max_seq_len=0, causal=True)
    positions = torch.tensor([base + keep for base, keep in zip(contexts, accepted)], dtype=torch.int32)
    group = SimpleNamespace(get_metadata_builder=lambda: Builder())
    for step in (1, 2):
        common, metadata = native.attn_update_stack_num_spec_norm(
            proposer, draft_index=step, old_attn_metadata=None, old_common_metadata=common,
            batch_size=batch, input_batch_size=padded, used_update_positions=positions,
            aclgraph_runtime_mode="none", attn_group=group)
        yield metadata


def checked_contexts(metadata, lengths=None):
    """Scalar interpretation of the production slotContext search/guards."""
    lengths = metadata.seq_lens if lengths is None else lengths
    result = []
    for row in range(metadata.num_reqs):
        slot = int(metadata.slot_mapping[row])
        if slot < 0:
            result.append(-1)
            continue
        upper = int(lengths[row])
        table = metadata.block_tables[row, :min(metadata.block_tables.shape[1], (upper + 127) // 128)]
        matches = (table == slot // 128).nonzero().flatten().tolist()
        position = matches[0] * 128 + slot % 128 if len(matches) == 1 else -1
        if len(matches) != 1 or not 0 <= position < upper:
            raise ValueError(f"prepare status=5 row={row} slot={slot} seq_len={upper} matches={matches}")
        result.append(position)
    return result


@pytest.mark.parametrize("block_size", [2304, 2816])
@pytest.mark.parametrize("limit", [4096, 262144])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_native_mtp_rejection_at_model_limit_keeps_valid_slots(monkeypatch, block_size, limit, accepted):
    # One near-limit request shares a padded batch with a short request.
    for metadata in replay_native_drafts(monkeypatch, limit=limit,
            contexts=[limit - 4, 100], accepted=[accepted, 0], block_size=block_size):
        expected = limit - 4 + accepted + metadata.draft_index
        if expected >= limit:
            expected = -1
        original = metadata.seq_lens.clone()
        if expected >= 0:
            with pytest.raises(ValueError, match="prepare status=5"):
                checked_contexts(metadata)
        lengths = prepare_context_lengths(metadata, limit)
        assert checked_contexts(metadata, lengths) == [expected, 100 + metadata.draft_index]
        assert torch.equal(metadata.seq_lens, original)
        assert lengths[1] == original[1]


@pytest.mark.parametrize("base", [1, 63, 255, 2302, 24624, 32765, 49997, 262139])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_continuous_two_drafts_retain_normal_context_and_padding(monkeypatch, base, accepted):
    for metadata in replay_native_drafts(monkeypatch, limit=262144,
            contexts=[base, 100], accepted=[accepted, 1], block_size=2304):
        lengths = prepare_context_lengths(metadata, 262144)
        expected = [base + accepted + metadata.draft_index, 101 + metadata.draft_index]
        expected = [p if p < 262144 else -1 for p in expected]
        assert checked_contexts(metadata, lengths) == expected
        assert (metadata.slot_mapping[metadata.num_reqs:] == -1).all()
        # Normal native upper bounds survive byte-for-byte. This includes
        # the 24624-token shape from #156, without claiming its NPU replay.
        if int(metadata.seq_lens[0]) > metadata.draft_index:
            assert torch.equal(lengths, metadata.seq_lens)


def test_bound_repair_remains_traceable_and_does_not_weaken_slot_guards(monkeypatch):
    metadata = next(replay_native_drafts(monkeypatch, limit=262144,
        contexts=[262140, 100], accepted=[0, 0], block_size=2304))
    from torch.fx.experimental.proxy_tensor import make_fx
    graph = make_fx(lambda lengths: prepare_context_lengths(replace(metadata, seq_lens=lengths), 262144))(
        metadata.seq_lens)
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)
    assert graph(torch.tensor([1, 105], dtype=torch.int32)).tolist() == [262144, 105]
    assert graph(torch.tensor([2, 106], dtype=torch.int32)).tolist() == [2, 106]
    assert graph(torch.tensor([0, -1], dtype=torch.int32)).tolist() == [0, -1]
    repaired = prepare_context_lengths(metadata, 262144)
    table = metadata.block_tables.clone()
    slot = int(metadata.slot_mapping[0])
    column = (table[0] == slot // 128).nonzero()[0, 0]
    table[0, column] = -1
    with pytest.raises(ValueError, match="status=5"):
        checked_contexts(replace(metadata, block_tables=table), repaired)
    table[0, column] = slot // 128
    table[0, 0] = slot // 128
    with pytest.raises(ValueError, match="status=5"):
        checked_contexts(replace(metadata, block_tables=table), repaired)


def test_main_and_first_draft_keep_native_length_storage(monkeypatch):
    metadata = next(replay_native_drafts(monkeypatch, limit=4096,
        contexts=[4092], accepted=[0], block_size=2304))
    for main in (True, False):
        row = replace(metadata, is_draft=not main, draft_index=0)
        assert prepare_context_lengths(row, None).data_ptr() == row.seq_lens.data_ptr()
    with pytest.raises(ValueError, match="model limit"):
        prepare_context_lengths(metadata, None)
