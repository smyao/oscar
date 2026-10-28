"""Archive #157-163: native four-token prefill fold must use committed GDN state.

Execute the pinned native fold and prefill state-store statements unchanged.
This is host address/ownership evidence, not NPU or whole-service acceptance.
Native GDN arithmetic and native reference files remain untouched.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.layout import HybridPageLayout, SlotLayout
from oscar_ascend.ops.reference import encode_kv
from test_runtime_native import definitions

ROOT = Path(__file__).resolve().parents[1]
BUILDER = "references/vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py"
GDN = "references/vllm-ascend/vllm_ascend/ops/gdn.py"


def native_fold(monkeypatch, prefilling, query_lengths, lengths, mask, accepted):
    native = definitions(BUILDER, "native_gdn_fold_regression", set(),
        {"torch": torch}, monkeypatch,
        methods=[("AscendGDNAttentionMetadataBuilder", "_fold_spec_sized_prefill_chunks_into_spec")])
    starts = torch.cat((torch.zeros(1, dtype=torch.int32),
                        torch.tensor(query_lengths, dtype=torch.int32).cumsum(0))).to(torch.int32)
    common = SimpleNamespace(is_prefilling=torch.tensor(prefilling),
        query_start_loc_cpu=starts,
        seq_lens_cpu_upper_bound=torch.tensor(lengths, dtype=torch.int32))
    return native._fold_spec_sized_prefill_chunks_into_spec(
        SimpleNamespace(num_spec=3), common, torch.tensor(mask),
        torch.tensor(accepted, dtype=torch.int32))


def _native_prefill_store(states, indices, final_state):
    # Use the actual native assignment, so this witness cannot invent writes to
    # speculative state columns that native prefill never performs.
    path = ROOT / GDN
    tree = ast.parse(path.read_text())
    assignment = next(node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name) and target.value.id == "ssm_state"
                and isinstance(target.slice, ast.Name) and target.slice.id == "prefill_state_indices"
                for target in node.targets))
    namespace = {"ssm_state": states, "prefill_state_indices": indices,
                 "last_recurrent_state": final_state}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                 str(path), "exec"), namespace)


def _recurrent_read_indices(table, accepted, lengths):
    # Source-bound index algebra of unchanged AscendC Process(), which reads
    # ssmStateIndices[seq0 + acceptedTokenNum - 1]. No numerical substitute.
    text = (ROOT / "references/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h").read_text()
    assert "stateTokenIdx = seq0 + acceptedTokenNum - 1" in text
    assert "stateOffset = ssmStateIndicesGm_.GetValue(stateTokenIdx)" in text
    starts = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)[:-1]))
    return table.flatten()[starts + accepted - 1]


def test_actual_native_fold_overwrites_recurrent_initial_index(monkeypatch):
    mask, accepted = native_fold(monkeypatch,
        [True, True, False, True], [4, 4, 4, 3], [32768, 4, 32900, 30003],
        [False, False, True, False], [1, 1, 2, 1])
    assert mask.tolist() == [True, False, True, False]
    assert accepted.tolist() == [4, 1, 2, 1]
    # Existing true spec rows retain their native count. Cold first chunks and
    # qLen3 prefills are not folded; only the warm qLen4 prefill changes to 4.


def test_legal_int2_reused_page_exposes_folded_uninitialized_ssm(monkeypatch):
    page = HybridPageLayout(SlotLayout(256), 30720, 393216, 262144,
                            native_page_size_bytes=817152,
                            native_mamba_cache_mode="none")
    count = 9
    raw = torch.zeros(count * page.page_size_bytes, dtype=torch.uint8)
    states = raw[count * page.conv_bytes:count * (page.conv_bytes + page.ssm_bytes)]
    states = states.view(torch.bfloat16).view(count, -1)
    # A real finite INT2 quantizer input gives code bytes ff 7f. Reinterpreting
    # a prior FULL page as BF16 SSM therefore produces NaN, without injecting
    # invalid quantizer metadata or nonfinite model inputs.
    vector = torch.zeros(256)
    vector[:7] = 3
    vector[7] = 1
    encoded = encode_kv(vector, vector)
    assert encoded[:2].tolist() == [255, 127]
    table = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    off = page.address(count, 4, 0)
    raw[off:off + encoded.numel()] = encoded
    assert torch.isnan(states[4, 0])
    committed = torch.full((2, states.shape[1]), 0.25, dtype=torch.bfloat16)
    _native_prefill_store(states, table[:, 0].long(), committed)
    assert torch.isfinite(states[1]).all() and torch.isnan(states[4, 0])
    _, accepted = native_fold(monkeypatch, [True, False], [4, 4], [32768, 32772],
                              [False, True], [1, 2])
    old_indices = _recurrent_read_indices(table, accepted, torch.tensor([4, 4]))
    assert old_indices.tolist() == [4, 6]
    assert torch.isnan(states[old_indices[0], 0])
    # The immediate initial-state selector should be checkpoint zero. This
    # alone is NOT a fix: the final test shows the next-decode commit gap.
    recurrent_accepted = accepted.clone()
    recurrent_accepted[0] = 1
    new_indices = _recurrent_read_indices(table, recurrent_accepted, torch.tensor([4, 4]))
    assert new_indices.tolist() == [1, 6]
    assert torch.equal(states[new_indices[0]], committed[0])
    assert accepted.tolist() == [4, 2]


def test_native_alignment_avoids_one_specific_fixed_prompt_tail(monkeypatch):
    # A 32K prompt follows a four-token decode row in a 16K budget. Its first
    # chunk already materialized 16K. Native none permits 16380 + 4, whereas
    # native align rounds the same interior chunk to 128-token alignment.
    # This interior example does not prove short tails impossible after the
    # last cache boundary; test_align_prefill_state_contract covers that case.
    scheduler = definitions("references/vllm/vllm/v1/core/sched/scheduler.py",
        "native_gdn_fold_scheduler", set(), {}, monkeypatch,
        methods=[("Scheduler", "_mamba_block_aligned_split")])
    runner = SimpleNamespace(cache_config=SimpleNamespace(block_size=128), use_eagle=False)
    request = SimpleNamespace(num_computed_tokens=16384, num_prompt_tokens=32768, num_tokens=32768)
    remaining_budget = 16384 - 4
    none_chunk = min(request.num_tokens - request.num_computed_tokens, remaining_budget)
    align_chunk = scheduler._mamba_block_aligned_split(runner, request, none_chunk)
    assert none_chunk == 16380 and align_chunk == 16256
    none_tail = request.num_tokens - request.num_computed_tokens - none_chunk
    align_tail = request.num_tokens - request.num_computed_tokens - align_chunk
    assert (none_tail, align_tail) == (4, 128)
    assert native_fold(monkeypatch, [True], [none_tail], [32768], [False], [1])[0].item()
    assert not native_fold(monkeypatch, [True], [align_tail], [32768], [False], [1])[0].item()


def test_folded_prefill_conv_checkpoint_is_also_column_zero(monkeypatch):
    # Actual native prefill WriteBackState places width-1 final tokens at
    # state positions 0..width-2, regardless of speculative stateLen=6.
    source = (ROOT / "references/vllm-ascend/csrc/moe/causal_conv1d/op_kernel/causal_conv1d.h").read_text()
    fn = source.split("CAUSAL_CONV1D_CLASS::WriteBackState(int32_t", 1)[1].split(
        "CAUSAL_CONV1D_CLASS::WriteBackStateSpec", 1)[0]
    assert "pos < (width - 1)" in fn
    assert "stateBaseOffset + static_cast<int64_t>(pos) * dim" in fn
    conv = torch.full((6, 2), -42.0, dtype=torch.bfloat16)
    committed = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.bfloat16)
    for pos in range(4 - 1):
        conv[pos].copy_(committed[pos])
    _, accepted = native_fold(monkeypatch, [True], [4], [32768], [False], [1])
    # Native ProcessDefault uses accepted-1 as the starting history offset.
    assert "stateTokenOffset = accepted - 1" in source
    old_offset = int(accepted[0]) - 1
    assert old_offset == 3
    assert not torch.equal(conv[old_offset:old_offset + 3], committed)
    assert torch.equal(conv[:3], committed)


def test_changing_folded_count_alone_leaves_wrong_next_decode_checkpoint():
    # Mark the four exact native recurrent write destinations with distinct
    # per-token checkpoint identities. Native first decode after prompt samples
    # one token, so its accepted count is1 and it reads destination column0.
    source = (ROOT / "references/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h").read_text()
    assert "ssmStateIndicesGm_.GetValue(seq_i)" in source
    table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    folded_states = torch.zeros(5, 1, dtype=torch.bfloat16)
    checkpoints = torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.bfloat16)
    for seq_i in range(4):
        folded_states[table.flatten()[seq_i]] = checkpoints[seq_i]
    next_index = _recurrent_read_indices(table, torch.tensor([1]), torch.tensor([4]))
    assert folded_states[next_index].item() == 1  # Should start from checkpoint4.
    # Keeping this chunk on the native prefill path commits the final state
    # into column0, satisfying the following decode's unchanged accepted1 ABI.
    prefill_states = torch.zeros_like(folded_states)
    _native_prefill_store(prefill_states, table[:, 0].long(), checkpoints[-1:])
    assert prefill_states[next_index].item() == 4


def _owner(*, apc=False, mode="none", method="mtp"):
    cache = SimpleNamespace(enable_prefix_caching=apc, mamba_cache_mode=mode)
    speculative = SimpleNamespace(method=method, num_speculative_tokens=3)
    return SimpleNamespace(vllm_config=SimpleNamespace(cache_config=cache,
        speculative_config=speculative), cache_config=cache,
        speculative_config=speculative, num_spec_tokens=3, num_spec=3)


def _fold_fixture(monkeypatch):
    native = definitions(BUILDER, "native_gdn_fixed_fold_regression", set(),
        {"torch": torch}, monkeypatch,
        methods=[("AscendGDNAttentionMetadataBuilder", "_fold_spec_sized_prefill_chunks_into_spec")])
    common = SimpleNamespace(is_prefilling=torch.tensor([True, False]),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([32768, 32900], dtype=torch.int32))
    return native._fold_spec_sized_prefill_chunks_into_spec, common


def test_fixed_none_keeps_native_prefill_and_commits_next_decode_state(monkeypatch):
    from oscar_ascend.integration.prefill_state_contract import keep_prefill_state
    original, common = _fold_fixture(monkeypatch)
    builder = _owner()
    mask, counts = torch.tensor([False, True]), torch.tensor([1, 2], dtype=torch.int32)
    old_mask, old_counts = original(builder, common, mask, counts)
    assert old_mask.tolist() == [True, True] and old_counts.tolist() == [4, 2]
    fixed_mask, fixed_counts = keep_prefill_state(builder, original, common, mask, counts)
    assert fixed_mask is mask and fixed_counts is counts
    assert fixed_mask.tolist() == [False, True] and fixed_counts.tolist() == [1, 2]
    # Only the genuine prefill row uses the unchanged native prefill assignment.
    # The existing speculative decode row retains its original accepted count2.
    table = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    states = torch.full((9, 1), float("nan"), dtype=torch.bfloat16)
    final_prefill_state = torch.tensor([[4.]], dtype=torch.bfloat16)
    _native_prefill_store(states, table[~fixed_mask, 0].long(), final_prefill_state)
    next_index = _recurrent_read_indices(table[:1], torch.tensor([1]), torch.tensor([4]))
    assert torch.equal(states[next_index], final_prefill_state)
    assert torch.isnan(states[2:5]).all()  # No speculative pages were fabricated.


@pytest.mark.parametrize("apc,mode", [(True, "align"), (True, "none"), (False, "align")])
def test_prefix_enabled_or_align_preserves_real_prefill_state(monkeypatch, apc, mode):
    from oscar_ascend.integration.prefill_state_contract import keep_prefill_state
    original, common = _fold_fixture(monkeypatch)
    builder = _owner(apc=apc, mode=mode)
    mask, counts = torch.tensor([False, True]), torch.tensor([1, 2], dtype=torch.int32)
    old_mask, old_counts = original(builder, common, mask, counts)
    assert old_mask.tolist() == [True, True] and old_counts.tolist() == [4, 2]
    observed = keep_prefill_state(builder, original, common, mask, counts)
    assert observed[0] is mask and observed[1] is counts
    assert mask.tolist() == [False, True] and counts.tolist() == [1, 2]


@pytest.mark.parametrize("computed,prompts,scheduled,max_tokens,uniform,expected", [
    ([32764, 32800], [32768, 32768], [4, 4], 4, None, True),
    ([0, 32800], [32768, 32768], [4, 4], 4, None, False),
    ([32768, 32800], [32768, 32768], [4, 4], 4, None, False),
    ([32764, 32800], [32768, 32768], [3, 4], 4, None, False),
    ([32764, 32800], [32768, 32768], [4, 16], 16, None, False),
    ([32764, 32800], [32768, 32768], [4, 4], 4, True, False),
])
@pytest.mark.parametrize("mode", ["none", "align"])
def test_only_live_prefill_four_token_step_requires_eager(
        computed, prompts, scheduled, max_tokens, uniform, expected, mode):
    from oscar_ascend.integration.prefill_state_contract import force_prefill_eager
    runner = _owner(mode=mode)
    runner.input_batch = SimpleNamespace(num_reqs=2,
        num_computed_tokens_cpu=torch.tensor(computed),
        num_prompt_tokens=torch.tensor(prompts))
    assert force_prefill_eager(runner, 2, torch.tensor(scheduled), max_tokens, uniform) is expected
    runner.cache_config.enable_prefix_caching = True
    assert force_prefill_eager(runner, 2, torch.tensor(scheduled), max_tokens, uniform) is expected


@pytest.mark.parametrize("mode,method", [("all", "mtp"), ("align", "eagle"), ("none", "eagle")])
def test_unrelated_state_protocols_keep_original_fold(monkeypatch, mode, method):
    from oscar_ascend.integration.prefill_state_contract import keep_prefill_state
    original, common = _fold_fixture(monkeypatch)
    builder = _owner(apc=True, mode=mode, method=method)
    mask, counts = torch.tensor([False, True]), torch.tensor([1, 2], dtype=torch.int32)
    expected = original(builder, common, mask, counts)
    observed = keep_prefill_state(builder, original, common, mask, counts)
    assert all(torch.equal(a, b) for a, b in zip(expected, observed))


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("drafts", [1, 2, 3, 7])
def test_prefill_contract_uses_speculation_protocol_width_not_input_length(monkeypatch, mode, drafts):
    from oscar_ascend.integration.prefill_state_contract import keep_prefill_state, force_prefill_eager
    original, common = _fold_fixture(monkeypatch)
    width = drafts + 1
    owner = _owner(apc=True, mode=mode)
    owner.num_spec = drafts
    owner.speculative_config.num_speculative_tokens = drafts
    common.query_start_loc_cpu = torch.tensor([0, width, 2 * width], dtype=torch.int32)
    mask, counts = torch.tensor([False, True]), torch.tensor([1, 1], dtype=torch.int32)
    assert original(owner, common, mask, counts)[1].tolist() == [width, 1]
    actual = keep_prefill_state(owner, original, common, mask, counts)
    assert actual[0] is mask and actual[1] is counts
    owner.input_batch = SimpleNamespace(num_computed_tokens_cpu=torch.tensor([32768 - width, 32900]),
                                       num_prompt_tokens=torch.tensor([32768, 32768]))
    assert force_prefill_eager(owner, 2, torch.tensor([width, width]), width)
    owner.input_batch.num_computed_tokens_cpu[0] = 32768
    assert not force_prefill_eager(owner, 2, torch.tensor([width, width]), width)
