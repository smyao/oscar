"""Archive #140/#142/#148/#155: padded native first-MTP CPU contracts.

Execute pinned prepare_inputs_padded, set_inputs_first_pass and _set_positions
unchanged. The CPU branch establishes rejection and metadata propagation, not
Triton/NPU completion, native FIA accuracy, graph replay or model quality.
"""
from enum import Enum
from types import MethodType, SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.metadata import from_common
from test_runtime_native import definitions


class AscendAttentionState(Enum):
    ChunkedPrefill = 1


class Common(SimpleNamespace):
    def __init__(self, **kwargs):
        # CommonAttentionMetadata.causal has the native decoder default True.
        kwargs.setdefault("causal", True)
        super().__init__(**kwargs)

    def batch_size(self):
        return self.num_reqs


@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("async_spec", [False, True])
def test_native_padded_reject_zero_to_three_preserves_fresh_seven(
        monkeypatch, mrope, async_spec):
    native = definitions(
        "references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "native_padded_first_mtp", set(),
        {"torch": torch, "HAS_TRITON": False, "AscendCommonAttentionMetadata": Common},
        monkeypatch, methods=[
            ("AscendSpecDecodeBaseProposer", "prepare_inputs_padded"),
            ("AscendSpecDecodeBaseProposer", "set_inputs_first_pass")])
    positions_native = definitions(
        "references/vllm/vllm/v1/spec_decode/llm_base_proposer.py",
        "native_padded_first_mtp_positions", set(), {"torch": torch}, monkeypatch,
        methods=[("SpecDecodeBaseProposer", "_set_positions")])
    starts = torch.tensor([0, 4, 8, 12, 16, 23], dtype=torch.int32)
    upper = torch.tensor([104, 204, 304, 404, 7], dtype=torch.int32)
    logical_positions = torch.cat([
        *(torch.arange(begin, begin + 4) for begin in (100, 200, 300, 400)),
        torch.arange(7)])
    state = AscendAttentionState.ChunkedPrefill
    common = Common(
        causal=True, query_start_loc=starts, query_start_loc_cpu=starts.clone(),
        seq_lens=upper.clone(), seq_lens_cpu=None if async_spec else upper.clone(),
        _seq_lens_cpu=upper.clone(), seq_lens_cpu_upper_bound=upper,
        num_reqs=5, num_actual_tokens=23, num_input_tokens=23,
        block_table_tensor=torch.zeros(5, 8, dtype=torch.int32),
        slot_mapping=logical_positions.clone(), positions=logical_positions,
        positions_cpu=None, num_computed_tokens_cpu=None,
        _num_computed_tokens_cpu=None, attn_state=state, decode_token_per_req=4,
        is_prefilling=torch.tensor([False, False, False, False, True]),
        max_query_len=7, max_seq_len=404)
    metadata_fields = (
        "query_start_loc", "query_start_loc_cpu", "seq_lens",
        "_seq_lens_cpu", "seq_lens_cpu_upper_bound", "slot_mapping", "positions")
    before = {name: getattr(common, name).clone() for name in metadata_fields}
    proposer = SimpleNamespace(
        arange=torch.arange(32), pcp_size=1, needs_extra_input_slots=False,
        runner=SimpleNamespace(pcp_manager=None, actual_seq_lengths_q=starts[1:].tolist(),
                               attn_state=state, decode_token_per_req=4),
        input_ids=torch.full((23,), -99, dtype=torch.int32),
        hidden_states=torch.empty(23, 2), positions=torch.empty(23, dtype=torch.int64),
        mrope_positions=torch.empty(3, 23, dtype=torch.int64),
        uses_mrope=mrope, uses_xdrope_dim=0, draft_uses_xdrope_dim=0,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(uses_mrope=mrope)))
    proposer._set_positions = MethodType(positions_native._set_positions, proposer)
    spec = SimpleNamespace(cu_num_draft_tokens=torch.tensor([3, 6, 9, 12, 12]))
    prepared, token_indices, sample, rejected = native.prepare_inputs_padded(
        proposer, common, spec, torch.tensor([4, 3, 2, 1, 1]))
    assert rejected.tolist() == [0, 1, 2, 3, 0]
    assert sample.tolist() == [3, 6, 9, 12, 22]
    assert token_indices.tolist() == list(range(23))
    for name in metadata_fields:
        assert getattr(prepared, name) is getattr(common, name)
    assert prepared.seq_lens_cpu is common.seq_lens_cpu
    assert prepared.num_actual_tokens == 23

    inputs = torch.arange(1000, 1023, dtype=torch.int32)
    next_ids = torch.arange(9000, 9005, dtype=torch.int32)
    positions = (torch.stack((logical_positions, logical_positions + 1000,
                              logical_positions + 2000)) if mrope else logical_positions)
    hidden = torch.arange(46, dtype=torch.float32).view(23, 2)
    count, actual_sample, returned, cp_args = native.set_inputs_first_pass(
        proposer, inputs[token_indices], next_ids, positions, hidden[token_indices],
        sample, prepared, rejected)
    assert count == 23 and returned is prepared and actual_sample is sample
    assert cp_args is None
    assert torch.equal(proposer.hidden_states, hidden)
    assert torch.equal(proposer.mrope_positions if mrope else proposer.positions, positions)
    assert proposer.input_ids[sample].tolist() == next_ids.tolist()
    assert proposer.input_ids[16:].tolist() == [1017, 1018, 1019, 1020, 1021, 1022, 9004]
    for name, value in before.items():
        assert torch.equal(getattr(returned, name), value)

    metadata = from_common(
        returned, is_draft=True, current_only=True, first_draft_current_only=True)
    plan = metadata.current_only_plan
    assert (plan.prefix_requests, plan.prefix_tokens) == (4, 16)
    assert (plan.fresh_requests, plan.fresh_tokens, plan.cumulative) == (1, 7, (7,))
    assert metadata.first_draft_current_only and metadata.draft_index == 0
