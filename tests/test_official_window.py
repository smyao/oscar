"""Archive #37-49/#86/#127/#128: independent official tier/state contracts.

Cases derive from pinned OSCAR allocation.py, unified_kv_pool.py and the
flush-plan kernel, not from the current Ascend reader implementation. These
are host integer/state tests; they do not qualify NPU, APC restore or MTP.
"""
import ast
from dataclasses import asdict
from pathlib import Path
import pytest

from oscar_ascend.official_window import (
    DecodeWindowState, WindowGeometry, extend_layout, plan_decode_step,
    prefix_match_cap,
)


@pytest.mark.parametrize("context,length,final,expected", [
    (0, 1001, True, (64, 256, 681, 688, 7)),
    (0, 1001, False, (64, 0, 937, 944, 263)),
    (1000, 1, True, (0, 1, 0, 0, 262)),
    (16384, 128, True, (0, 128, 0, 0, 135)),
    (32, 100, True, (32, 68, 0, 0, 195)),
    (0, 23, True, (23, 0, 0, 0, 263)),
    (1000, 300, True, (0, 256, 44, 48, 7)),
    (1000, 300, False, (0, 0, 300, 304, 263)),
])
def test_official_extend_assignment_uses_actual_new_tail_not_absolute_page_floor(context,length,final,expected):
    layout = extend_layout(context, length, is_final_chunk=final)
    assert layout.counts() == expected
    assert layout.current_range == (context, context + length)
    assert layout.hp_prefix_count + layout.hp_recent_count + layout.quant_count == length


def test_prior_nonfinal_chunk_is_not_promoted_when_final_chunk_is_short():
    first = extend_layout(0, 16384, is_final_chunk=False)
    last = extend_layout(16384, 128, is_final_chunk=True)
    assert first.hp_recent_count == 0
    assert last.new_recent_start == 16384
    assert last.hp_recent_count == 128  # Not the nominal R=256.


def test_flush_phase_is_relative_to_initial_tail_not_absolute_token_multiple():
    state = DecodeWindowState(1001, 745)
    for _ in range(7):
        step = plan_decode_step(state)
        assert step.history_end == 745 and step.demoted_range == (745,745)
        state = step.next_state
    assert state.context == 1008 and state.recent_count == 263
    step = plan_decode_step(state)
    assert step.demoted_range == (745,753)
    assert step.next_state.context == 1009 and step.next_state.recent_count == 256
    # Neither cut is quantum aligned; only allocated pages/flush widths are.
    assert step.history_end % 8 == 1


def test_same_context_can_have_different_persistent_boundaries():
    first = plan_decode_step(DecodeWindowState(1000,744)).next_state
    second = DecodeWindowState(1001,745)
    assert first.context == second.context == 1001
    assert first.history_end != second.history_end
    assert first.flush_counter != second.flush_counter


def test_small_final_chunk_seeds_countdown_from_actual_hp_occupancy():
    state = DecodeWindowState(1001,1000)
    for _ in range(262):
        step = plan_decode_step(state)
        assert step.demoted_range == (1000,1000)
        state = step.next_state
    step = plan_decode_step(state)
    assert step.demoted_range == (1000,1008)
    assert step.next_state.recent_count == 256


def test_short_prompt_does_not_invent_more_prefix_slots_during_decode():
    state = DecodeWindowState(23,23)
    for _ in range(263):
        step = plan_decode_step(state)
        assert step.next_state.sink_end == 23
        state = step.next_state
    step = plan_decode_step(state)
    assert step.demoted_range == (23,31)
    assert step.next_state.recent_count == 256


def test_snapshot_roundtrip_preserves_boundary_and_phase_not_only_context():
    original = DecodeWindowState(1008,745)
    record = asdict(original)
    record["geometry"] = WindowGeometry(**record["geometry"])
    restored = DecodeWindowState(**record)
    assert plan_decode_step(restored) == plan_decode_step(original)
    assert plan_decode_step(restored).demoted_range == (745,753)


def test_retention_allowance_does_not_change_nominal_policy_or_flush_quantum():
    geometry = WindowGeometry()
    assert geometry.recent_tokens == 256 and geometry.quantum == 8
    assert geometry.ring_tokens == 263
    assert geometry.retention_capacity(3) == 266
    # #37-49: up to three materialized writes beyond the committed endpoint
    # cannot overwrite any of its possible 263 retained recent positions.
    for committed in (264,1001,2816,5632):
        keep = range(committed - geometry.ring_tokens, committed)
        capacity = geometry.retention_capacity(3)
        protected = {position % capacity for position in keep}
        rejected = {(committed + offset) % capacity for offset in range(3)}
        assert protected.isdisjoint(rejected)


@pytest.mark.parametrize("length,cap", [(0,0),(7,0),(23,16),(64,64),(300,64),(328,64),(1001,736)])
def test_official_radix_cap_excludes_request_owned_recent_ring(length,cap):
    assert prefix_match_cap(length) == cap


def test_decode_reference_matches_pinned_host_countdown_over_long_accepted_stream():
    # Literal countdown behavior from unified_kv_pool.advance_flush_counters;
    # the flush kernel demotes [C-R-(N_Q-1), C-R+1) before the next write.
    for initial_recent in (0,1,128,255,256,263):
        state = DecodeWindowState(1024,1024-initial_recent)
        countdown = 263-initial_recent
        for _ in range(1024):
            old_context = state.context
            due = countdown == 0
            countdown = 7 if due else countdown-1
            step = plan_decode_step(state)
            if due:
                assert step.demoted_range == (old_context-263,old_context-255)
            else:
                assert step.demoted_range[0] == step.demoted_range[1]
            state = step.next_state
            assert state.flush_counter == countdown
            assert 0 <= state.recent_count <= 263


# Pure integer helpers extracted unchanged from the official Apache-2.0 source.
# Source: FutureMLS-Lab/OSCAR 38680852, python/sglang/srt/mem_cache/allocation.py
# Full-file SHA256: 1be60073e3af259aeb40ff6495a0e91d8e59741af3971963bd4750b01e8d132e
UPSTREAM_WINDOW_SOURCE = 'def _mixed_window_lengths(\n    seq_len: int, hp_prefix_tokens: int, hp_recent_tokens: int\n) -> tuple[int, int, int]:\n    prefix_len = min(seq_len, hp_prefix_tokens)\n    recent_len = min(max(seq_len - prefix_len, 0), hp_recent_tokens)\n    quant_len = seq_len - prefix_len - recent_len\n    return prefix_len, recent_len, quant_len\n\ndef _mixed_extend_layout_counts(\n    pre_len: int,\n    seq_len: int,\n    hp_prefix_tokens: int,\n    hp_recent_tokens: int,\n    n_q: int,\n    is_final_chunk: bool = True,\n) -> tuple[int, int, int, int, int]:\n    """Return per-request mixed-KV extend counts.\n\n    Default (final chunk) layout: ``[HP-prefix][quant-middle][HP-recent]``.\n    Non-final chunks collapse to ``[HP-prefix][quant-all]`` and skip\n    HP-recent -- chunk N\'s trailing positions are never the request\'s final\n    HP-recent window, so allocating HP-recent there only wraps the\n    per-request ring and recycles slots whose K/V is still referenced via\n    ``req_to_token`` at earlier chunk N positions.\n    """\n    if is_final_chunk:\n        prefix_keep, recent_keep, _ = _mixed_window_lengths(\n            seq_len, hp_prefix_tokens, hp_recent_tokens\n        )\n        recent_start = seq_len - recent_keep\n        hp_prefix_count = max(0, min(prefix_keep, seq_len) - pre_len)\n        quant_count = max(0, recent_start - max(pre_len, prefix_keep))\n        hp_recent_count = max(0, seq_len - max(pre_len, recent_start))\n    else:\n        prefix_keep = min(seq_len, hp_prefix_tokens)\n        hp_prefix_count = max(0, min(prefix_keep, seq_len) - pre_len)\n        quant_count = max(0, seq_len - max(pre_len, prefix_keep))\n        hp_recent_count = 0\n    quant_alloc_count = ceil_align(quant_count, n_q)\n\n    # Per-request flush counter: count steps until this request\'s next flush.\n    # ``H_0`` is the current HP-recent size after this extend chunk is admitted.\n    # For non-final chunks we skip HP-recent entirely (``hp_recent_count=0``),\n    # and for chunked-prefill final chunks the actual HP-recent count is\n    # ``hp_recent_count`` (positions newly admitted in this chunk), NOT the\n    # ``hp_recent_tokens`` target. If we use the target as ``H_0`` while the\n    # actual count is small, ``counter_init`` underestimates how many decode\n    # steps must elapse before the flush kernel\'s demote range\n    # ``[seq_len - hp_recent - flush_overflow : seq_len - hp_recent]`` falls\n    # entirely inside HP-recent. Early flushes hit the chunk-1 quant region\n    # (valid=0 for all 8 slots -> whole-page return). Worse, at the *last*\n    # premature flush before the demote range crosses into HP-recent, the\n    # 8-position window straddles the boundary: 3-5 valid=0 and 3-5 valid=1\n    # -> page partially used + partially returned. The used slots get written\n    # to ``req_to_token``; the returned slots go to the free pool. When the\n    # request finishes, ``cache_finished_req`` frees the used slots -> same\n    # page added to the free pool TWICE. That is the +N quant-page duplicate\n    # the strict idle leak check trips on.\n    h0_total = hp_recent_count\n    counter_init = max(0, (hp_recent_tokens + n_q - 1) - h0_total)\n    return (\n        hp_prefix_count,\n        hp_recent_count,\n        quant_count,\n        quant_alloc_count,\n        counter_init,\n    )'

def test_extend_counts_match_pinned_official_function():
    tree = ast.parse(UPSTREAM_WINDOW_SOURCE)
    wanted = {"_mixed_window_lengths", "_mixed_extend_layout_counts"}
    body = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    assert len(body) == 2
    namespace = {"ceil_align": lambda value,quantum: (value+quantum-1)//quantum*quantum}
    exec(compile(ast.Module(body=body,type_ignores=[]),"pinned-official-window-helpers","exec"),namespace)
    official = namespace["_mixed_extend_layout_counts"]
    for context in (0,1,63,64,65,257,1000,16384):
        for length in (0,1,3,4,7,8,9,127,256,257,16384):
            for final in (False,True):
                assert extend_layout(context,length,is_final_chunk=final).counts() == official(
                    context,context+length,64,256,8,is_final_chunk=final)


@pytest.mark.parametrize("make", [
    lambda: WindowGeometry(recent_tokens=255),
    lambda: WindowGeometry(quantum=0),
    lambda: WindowGeometry(sink_tokens=True),
    lambda: extend_layout(-1,1,is_final_chunk=True),
    lambda: extend_layout(0,1,is_final_chunk=1),
    lambda: DecodeWindowState(1000,736),
    lambda: DecodeWindowState(1000,1001),
    lambda: DecodeWindowState(1000,63),
    lambda: WindowGeometry().retention_capacity(-1),
])
def test_invalid_geometry_or_unrestorable_tier_state_fails_closed(make):
    with pytest.raises(ValueError):
        make()
