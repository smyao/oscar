"""Host reference for the pinned official HP/INT2 tier lifecycle, not a route.

Archive #37-49: committed/materialized state and rejection retention are
distinct. #86: MTP is not a target prefill. #127/#128: the official INT2
quantum is not a native FULL physical-page or metadata-block size.

Source: OSCAR 38680852e52c10bd6b9e9a64ed9bc815e742859b, allocation.py
_mixed_extend_layout_counts/_alloc_for_decode_mixed, unified_kv_pool.py
compute_recent_ring_size/advance_flush_counters, and gpu_flush_int2.py
_flush_plan_kernel. Non-final prompt chunks store no HP-recent tokens.
The official mixed allocator does not support speculative decoding; this
module deliberately exposes single-token decode transitions, not a guessed
multi-token serving policy. No production dispatch or native state is changed.

D.4: bounded host integer algebra only. No history scans/materialization,
tensor allocation, device synchronization, or per-token production callbacks;
the 6499.8-6655.1 ms historical dequant and ~725 ms prepare paths are absent.
"""
from __future__ import annotations

from dataclasses import dataclass


OFFICIAL_COMMIT = "38680852e52c10bd6b9e9a64ed9bc815e742859b"
BF16_QUANTUM = 8


def _nonnegative(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class WindowGeometry:
    sink_tokens: int = 64
    recent_tokens: int = 256
    quantum: int = BF16_QUANTUM

    def __post_init__(self):
        _nonnegative("sink_tokens", self.sink_tokens)
        _nonnegative("recent_tokens", self.recent_tokens)
        if type(self.quantum) is not int or self.quantum <= 0:
            raise ValueError("quantum must be a positive integer")
        if self.recent_tokens <= 0:
            raise ValueError("this HP-recent reference requires recent_tokens > 0")
        if self.sink_tokens % self.quantum or self.recent_tokens % self.quantum:
            raise ValueError("official sink/recent windows must be quantum-aligned")

    @property
    def ring_tokens(self) -> int:
        """Official HP occupancy bound; nominal recent remains unchanged."""
        return self.recent_tokens + self.quantum - 1

    def retention_capacity(self, speculative_tokens: int = 0) -> int:
        """Local storage bound for extra writes beyond committed context.

        The additive speculative allowance extends this project's existing
        snapshot rollback proof. It is storage retention, not a claim that
        the official allocator supports MTP or uses a larger recent policy.
        """
        return self.ring_tokens + _nonnegative("speculative_tokens", speculative_tokens)


@dataclass(frozen=True)
class ExtendLayout:
    context: int
    current_length: int
    is_final_chunk: bool
    hp_prefix_count: int
    hp_recent_count: int
    quant_count: int
    quant_alloc_count: int
    flush_counter_init: int

    @property
    def sequence_length(self) -> int:
        return self.context + self.current_length

    @property
    def current_range(self) -> tuple[int, int]:
        """All current keys remain exact for this prefill's attention."""
        return self.context, self.sequence_length

    @property
    def new_recent_start(self) -> int:
        """First newly admitted HP-recent position, or sequence end if empty."""
        return self.sequence_length - self.hp_recent_count

    def counts(self) -> tuple[int, int, int, int, int]:
        return (self.hp_prefix_count, self.hp_recent_count, self.quant_count,
                self.quant_alloc_count, self.flush_counter_init)


def extend_layout(context: int, current_length: int, *, is_final_chunk: bool,
                  geometry: WindowGeometry = WindowGeometry()) -> ExtendLayout:
    """Official storage assignment for *new* tokens in one prompt chunk.

    The caller determines finality from sequence_end >= original_prompt_len.
    Cached tiers are not reconstructed from C/L: prefix reuse or earlier
    chunks retain their actual stored representation. For attention, the
    entire current chunk is exact regardless of its eventual stored tier.
    Only allocation size is rounded to quantum; the logical cut is not.
    """
    _nonnegative("context", context)
    _nonnegative("current_length", current_length)
    if type(is_final_chunk) is not bool:
        raise ValueError("is_final_chunk must be an explicit boolean")
    end = context + current_length
    prefix_keep = min(end, geometry.sink_tokens)
    prefix_count = max(0, prefix_keep - context)
    if is_final_chunk:
        recent_keep = min(end - prefix_keep, geometry.recent_tokens)
        recent_start = end - recent_keep
        quant_count = max(0, recent_start - max(context, prefix_keep))
        recent_count = max(0, end - max(context, recent_start))
    else:
        quant_count = max(0, end - max(context, prefix_keep))
        recent_count = 0
    allocated = ((quant_count + geometry.quantum - 1) // geometry.quantum) * geometry.quantum
    return ExtendLayout(context, current_length, is_final_chunk, prefix_count,
                        recent_count, quant_count, allocated,
                        geometry.ring_tokens - recent_count)


@dataclass(frozen=True)
class DecodeWindowState:
    """Actual persistent tier boundary; serialize it with any exact snapshot.

    [0,sink_end) is sink, [sink_end,history_end) is INT2, and
    [history_end,C) is request-owned HP-recent. history_end cannot be inferred
    from C alone because the final prompt chunk may contribute fewer than R
    tokens, and flushes are anchored to that initial HP occupancy.
    """
    context: int
    history_end: int
    geometry: WindowGeometry = WindowGeometry()
    sink_end: int | None = None

    def __post_init__(self):
        _nonnegative("context", self.context)
        _nonnegative("history_end", self.history_end)
        if self.sink_end is None:
            object.__setattr__(self, "sink_end", min(self.geometry.sink_tokens, self.context))
        _nonnegative("sink_end", self.sink_end)
        if not self.sink_end <= min(self.geometry.sink_tokens, self.context):
            raise ValueError("sink_end exceeds the materialized HP-prefix")
        if not self.sink_end <= self.history_end <= self.context:
            raise ValueError("history_end must separate sink/history from HP-recent")
        if self.recent_count > self.geometry.ring_tokens:
            raise ValueError("HP-recent occupancy exceeds the official ring")

    @property
    def recent_count(self) -> int:
        return self.context - self.history_end

    @property
    def flush_counter(self) -> int:
        return self.geometry.ring_tokens - self.recent_count


@dataclass(frozen=True)
class DecodeStep:
    history_end: int
    demoted_range: tuple[int, int]
    next_state: DecodeWindowState


def plan_decode_step(state: DecodeWindowState) -> DecodeStep:
    """Official q1: flush before the new token is written/read by attention.

    This reference describes one logical decode step. Calling it repeatedly
    models a serial accepted-token stream, not an official speculative frame.
    A native MTP integration must separately choose its tentative-view policy
    and commit only accepted positions; rejected writes cannot advance state.
    """
    end = state.history_end
    if state.flush_counter == 0:
        end += state.geometry.quantum
    return DecodeStep(end, (state.history_end, end),
                      DecodeWindowState(state.context + 1, end, state.geometry, state.sink_end))


def prefix_match_cap(key_length: int, *, geometry: WindowGeometry = WindowGeometry()) -> int:
    """Official radix tier cap, in its INT2 quantum (not native page size).

    This is a source reference, not a request to cap native vLLM APC hits.
    A vLLM adapter that preserves larger native hits must restore the exact
    tier state and raw snapshot under its own validated ownership protocol.
    """
    _nonnegative("key_length", key_length)
    cap = min(key_length, max(geometry.sink_tokens, key_length - geometry.ring_tokens))
    return cap // geometry.quantum * geometry.quantum
