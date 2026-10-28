"""Archive #157-163/#171/#174: real APC tail chunks retain prefill state.

Execute unchanged native scheduler, align preprocessing, block-table gather,
fold and prefill state publication. State values are synthetic CPU witnesses;
this is not device/model acceptance. D.4: no historical KV materialization,
no new copy/page; only metadata routing of actual warm prompt work changes.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend import plugin
from oscar_ascend.integration.prefill_state_contract import keep_prefill_state
from oscar_ascend.integration.prefix_checkpoint import split_prefix_checkpoint
from oscar_ascend.integration.runtime_api import clear_runtime
from test_gdn_folded_initial_state import _native_prefill_store, _recurrent_read_indices
from test_prefix_checkpoint_retention import _setup
from test_prefill_state_contract_hooks import _native_modules, _register, _runner, Mode
from test_runtime_native import definitions
from test_universal_bf16_regression_protection import selected_operator


@pytest.fixture(autouse=True)
def plugin_cleanup(monkeypatch):
    plugin.unregister()
    clear_runtime()
    yield
    plugin.unregister()
    clear_runtime()


def native_chain(monkeypatch):
    cache, scheduler, native_split, interface, _ = _setup(monkeypatch, 200, 'lease')
    utils = definitions('references/vllm/vllm/v1/attention/backends/utils.py',
        'align_prefill_native_utils', {'mamba_get_block_table_tensor'},
        {'torch': torch, 'MambaSpec': interface.MambaSpec}, monkeypatch)
    fold = definitions('references/vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py',
        'align_prefill_native_fold', set(), {'torch': torch}, monkeypatch,
        methods=[('AscendGDNAttentionMetadataBuilder', '_fold_spec_sized_prefill_chunks_into_spec')])
    return cache, scheduler, native_split, interface, utils, fold._fold_spec_sized_prefill_chunks_into_spec


def tail_schedule(scheduler, native_split, prompt, *, hit=0):
    req = NS(request_id='tail', num_prompt_tokens=prompt, num_tokens=prompt,
             num_computed_tokens=hit, has_encoder_inputs=False)
    # Inside native's final uncached tail, other requests can consume enough
    # of the global16384 budget to leave this request four prompt tokens.
    block = scheduler.cache_config.block_size
    last_native = max(prompt - prompt % block - block, 0)
    trace = []
    while req.num_computed_tokens < prompt:
        remaining = prompt - req.num_computed_tokens
        budget = (remaining - 4 if req.num_computed_tokens >= last_native and remaining > 4
                  else 16384)
        budget = min(16384, budget)
        proposed = min(remaining, budget)
        count = split_prefix_checkpoint(scheduler, req, proposed)
        if count is None:
            count = native_split(scheduler, req, proposed)
        assert 0 < count <= proposed <= 16384
        trace.append({'before': req.num_computed_tokens, 'available_budget': budget, 'scheduled': count})
        req.num_computed_tokens += count
    return trace


def state_case(monkeypatch, context):
    cache, _, _, _, utils, original_fold = native_chain(monkeypatch)
    spec = cache.single_type_managers[1].kv_cache_spec
    # Every logical state block has a distinct real physical ID in this
    # bounded address witness, including the three original speculative pages.
    size = (context + 8 + spec.block_size - 1) // spec.block_size + 6
    physical = torch.arange(1, size + 1, dtype=torch.int32)[None]
    state = torch.full((size + 1, 1), -42., dtype=torch.bfloat16)
    conv = torch.full((size + 1, 6, 1), -42., dtype=torch.bfloat16)
    before = utils.mamba_get_block_table_tensor(physical, torch.tensor([context]), spec, 'align')
    _native_prefill_store(state, before[:, 0].long(), torch.tensor([[.25]], dtype=torch.bfloat16))
    conv[before[0, 0], :3] = .25
    copy_log = []
    copies = definitions('references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py',
        'align_prefill_native_copy_specs', {'get_conv_copy_spec', 'get_temporal_copy_spec'},
        {'MambaCopySpec': NS, 'is_conv_state_dim_first': lambda: False}, monkeypatch)

    def collect(_bufs, _config, _funcs, _groups, source_index, dest_index,
                accepted_bias, request, _forward):
        # Actual native preprocessing chooses these indices/bias; this CPU
        # hook represents its existing copy, never additional production work.
        assert accepted_bias == 0
        dst = request.block_ids[1][dest_index]
        for tensor, copy_spec in ((state, copies.get_temporal_copy_spec),
                                  (conv, copies.get_conv_copy_spec)):
            selection = copy_spec(tensor, request.block_ids[1], source_index, accepted_bias + 1)
            offset = (selection.start_addr - tensor.data_ptr()) // tensor.element_size()
            selected = tensor.view(-1)[offset:offset + selection.num_elements].clone()
            tensor[dst].view(-1)[:selection.num_elements].copy_(selected)
        copy_log.append((source_index, dest_index, accepted_bias))

    prep = definitions('references/vllm-ascend/vllm_ascend/patch/worker/patch_mamba_utils.py',
        'align_prefill_native_preprocess', {'preprocess_mamba'},
        {'itertools': itertools, 'cdiv': lambda n, d: (n + d - 1) // d,
         'mamba_utils': NS(collect_mamba_copy_meta=collect)}, monkeypatch).preprocess_mamba
    req = NS(num_computed_tokens=context, block_ids=[physical[0].tolist()] * 4)
    batch = NS(req_ids=['tail'], num_accepted_tokens_cpu=[1])
    scheduled = NS(finished_req_ids=set(), preempted_req_ids=set(),
        scheduled_cached_reqs=NS(resumed_req_ids=set()), num_scheduled_tokens={'tail': 4})
    state_idx = {'tail': (context - 1) // spec.block_size}
    buffers = NS(mamba_group_ids=[1, 2, 3], mamba_spec=spec, offset=0)
    prep(scheduled, None, NS(enable_prefix_caching=True, mamba_cache_mode='align'),
         state_idx, batch, {'tail': req}, {}, (), buffers)
    current = utils.mamba_get_block_table_tensor(physical, torch.tensor([context + 4]), spec, 'align')
    assert batch.num_accepted_tokens_cpu == [1]
    assert state[current[0, 0]].item() == .25
    assert state[current[0, 3]].item() == -42
    common = NS(is_prefilling=torch.tensor([True]),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([context + 4], dtype=torch.int32))
    builder = NS(num_spec=3, vllm_config=NS(
        cache_config=NS(enable_prefix_caching=True, mamba_cache_mode='align'),
        speculative_config=NS(method='mtp', num_speculative_tokens=3)))
    mask, accepted = torch.tensor([False]), torch.tensor([1], dtype=torch.int32)
    old_mask, old_accepted = original_fold(builder, common, mask, accepted)
    fixed_mask, fixed_accepted = keep_prefill_state(builder, original_fold, common, mask, accepted)
    old_index = _recurrent_read_indices(current, old_accepted, torch.tensor([4]))
    assert old_mask.tolist() == [True] and old_accepted.tolist() == [4]
    assert state[old_index[0]].item() == -42
    assert fixed_mask is mask and fixed_accepted is accepted
    assert fixed_mask.tolist() == [False] and fixed_accepted.tolist() == [1]
    # Keep the normal native prefill final-state write. The next decode uses
    # accepted1; no scratch state is fabricated to make accepted4 look valid.
    committed = torch.tensor([[4.]], dtype=torch.bfloat16)
    _native_prefill_store(state, current[:, 0].long(), committed)
    conv[current[0, 0], :3] = torch.tensor([[2.], [3.], [4.]], dtype=torch.bfloat16)
    req.num_computed_tokens = context + 4
    before_next = list(copy_log)
    prep(scheduled, None, NS(enable_prefix_caching=True, mamba_cache_mode='align'),
         state_idx, batch, {'tail': req}, {}, (), buffers)
    following = utils.mamba_get_block_table_tensor(physical, torch.tensor([context + 8]), spec, 'align')
    next_index = _recurrent_read_indices(following, torch.tensor([1]), torch.tensor([4]))
    assert torch.equal(state[next_index], committed)
    assert conv[following[0, 0], :3, 0].tolist() == [2., 3., 4.]
    return {'context': context, 'warm_tokens': 4, 'native_align_table': current.tolist(),
        'old_read_page': int(old_index[0]), 'fixed_commit_page': int(current[0, 0]),
        'old_read_value': -42., 'valid_initial_value': .25,
        'copies_before_tail': before_next, 'copies_total_including_next_decode': copy_log,
        'next_decode_initial_value': float(state[next_index].item())}


@pytest.mark.parametrize('prompt', [772, 4610, 5000, 19972, 32768, 65536])
def test_align_native_scheduler_can_leave_warm_four_token_prompt_tail(monkeypatch, prompt):
    _, scheduler, native_split, *_ = native_chain(monkeypatch)
    trace = tail_schedule(scheduler, native_split, prompt)
    assert trace[-1]['scheduled'] == 4
    assert trace[-1]['before'] == prompt - 4
    assert sum(row['scheduled'] for row in trace) == prompt


@pytest.mark.parametrize('context', [768, 4606, 4996, 19968, 32764, 65532])
def test_align_preprocess_fold_and_next_decode_state_are_coherent(monkeypatch, context):
    result = state_case(monkeypatch, context)
    assert result['old_read_page'] != result['fixed_commit_page']
    assert result['next_decode_initial_value'] == 4.


def test_actual_prefix_hit_can_also_reach_warm_four_tail(monkeypatch):
    cache, scheduler, native_split, *_ = native_chain(monkeypatch)
    adopt = definitions('references/vllm/vllm/v1/core/kv_cache_coordinator.py',
        'align_prefill_native_adopt', set(), {}, monkeypatch,
        methods=[('KVCacheCoordinator', 'allocate_new_computed_blocks')]).allocate_new_computed_blocks
    monkeypatch.setattr(type(cache), 'allocate_new_computed_blocks', adopt, raising=False)
    req = NS(request_id='cached', num_prompt_tokens=5000, num_tokens=5000,
        num_computed_tokens=0, has_encoder_inputs=False,
        block_hashes=[i.to_bytes(8, 'big') for i in range(5000 // cache.hash_block_size)])
    for count in (2304, 1536, 1160):
        cache.new_step_starts()
        old, end = req.num_computed_tokens, req.num_computed_tokens + count
        cache.remove_skipped_blocks(req.request_id, old)
        cache.allocate_new_blocks(req.request_id, end + 3, end, 0)
        cache.cache_blocks(req, end)
        req.num_computed_tokens = end
    cache.new_step_starts()
    blocks, hit = cache.find_longest_cache_hit(req.block_hashes, req.num_tokens - 1)
    assert hit == 2304
    cache.allocate_new_computed_blocks('repeat', blocks, hit, 0)
    trace = tail_schedule(scheduler, native_split, req.num_prompt_tokens, hit=hit)
    assert [row['scheduled'] for row in trace] == [1536, 1156, 4]
    assert cache.single_type_managers[0].req_to_blocks['repeat'][0] is blocks[0][0]
    cache.free('repeat')
    cache.free('cached')
    assert cache.block_pool.get_num_free_blocks() == 199


@pytest.mark.parametrize('other_length,expected_op', [
    (4, 'attention_cv_official_decode_out'),
    (512, 'attention_cv_official_cluster4_out'),
    (8192, 'attention_cv_official_cluster16_out'),
])
def test_align_hook_preserves_true_decode_and_mixed_attention_routes(monkeypatch, other_length, expected_op):
    runner_mod, builder_mod = _native_modules(monkeypatch)
    runner, dispatches = _runner(runner_mod.NPUModelRunner, prefix=True, mode='align',
                                  computed=(4996, 6000))
    runner.input_batch.num_prompt_tokens[:] = torch.tensor([5000, 6000 + (other_length if other_length > 4 else 0)])
    runner.input_batch.num_prompt_tokens_cpu[:] = runner.input_batch.num_prompt_tokens
    scheduled = torch.tensor([4, other_length])
    def execute():
        return runner._determine_batch_execution_and_padding(
            num_tokens=4 + other_length, num_reqs=2, num_scheduled_tokens_np=scheduled,
            max_num_scheduled_tokens=other_length, use_cascade_attn=False)
    raw_mode = execute()[0]
    assert raw_mode is (Mode.FULL if other_length == 4 else Mode.NONE)
    dispatches.clear()
    _register(monkeypatch)
    assert execute()[0] is Mode.NONE
    if other_length == 4:
        assert dispatches == []  # Existing native force_eager choice.
    else:
        assert len(dispatches) == 1  # No new eager override for already mixed steps.
    builder = builder_mod.AscendGDNAttentionMetadataBuilder()
    builder.vllm_config, builder.num_spec = runner.vllm_config, 3
    common = NS(is_prefilling=torch.tensor([True, other_length > 4]),
        seq_lens_cpu_upper_bound=torch.tensor([5000, 6000 + other_length], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 4 + other_length], dtype=torch.int32))
    mask = torch.tensor([False, other_length == 4])
    counts = torch.tensor([1, 2 if other_length == 4 else 1], dtype=torch.int32)
    fixed = builder._fold_spec_sized_prefill_chunks_into_spec(common, mask, counts)
    assert fixed[0] is mask and fixed[1] is counts
    assert selected_operator(4 + other_length, other_length) == expected_op
    # Same fixed input buffers later contain genuine decode: graph resumes.
    if other_length == 4:
        runner.input_batch.num_computed_tokens_cpu[0] = 5000
        assert execute()[0] is Mode.FULL
