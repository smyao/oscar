"""Archive #170/#172/#175/#176: bounded, offline-only shared OptR staging.

The usual capture is staged exactly once, in the original head/request order.
Larger captures are traversed repeatedly in that same order, splitting only
queries; complete K/V histories and each query's real frame masks are kept.
Query-count weighting preserves the upstream mean, rather than introducing a
mean over chunks. Streaming costs additional transfers and fixed-side work.

These are fixed tensor-cache budgets, not claims about allocator or autograd
peak memory. The streaming consumer must delete its row and loss references
before requesting the next row, so two device rows never overlap in Python.
Use a plain loop: enumerate can retain its previously yielded tuple even
after the row variable is deleted, defeating that one-row lifetime.
No serving code or accelerator-specific memory-management API is involved.
"""
from __future__ import annotations

import torch

from .optr_sequential import prepare_samples, _key_probabilities


RESIDENT_CACHE_BYTES = 8 * 1024 ** 3
STREAMED_ROW_CACHE_BYTES = 512 * 1024 ** 2


def sample_cache_bytes(sample, *, queries=None, include_weight=True):
    """Exact byte count of the tensors retained by prepare_samples, from shape.

    FP32 Q/K/V and three teacher tensors, two Boolean masks, and one W_O.
    The caller authenticates sample values/dtypes before reaching the fitter;
    this estimator deliberately requires no tensor allocation or device read.
    """
    count, group, dim = sample["q"].shape
    queries = count if queries is None else queries
    if type(queries) is not int or not 0 <= queries <= count:
        raise ValueError("OptR cache estimate requires a valid query count")
    length = sample["k"].shape[0]
    fixed = 4 * (sample["k"].numel() + sample["v"].numel())
    if include_weight:
        fixed += sample["wo"].numel() * sample["wo"].element_size()
    per_query = 4 * group * dim + 2 * length + 12 * group * length
    return fixed + queries * per_query


def _resident_bytes(train_heads, held_heads):
    total, weights = 0, set()
    for heads in (train_heads, held_heads):
        for samples in heads:
            for sample in samples:
                identity = id(sample["wo"])
                total += sample_cache_bytes(sample, include_weight=identity not in weights)
                weights.add(identity)
    return total


def _freeze_value_row(row, rotation, policy):
    with torch.no_grad():
        row["selected_k_quant_p"] = _key_probabilities(rotation, row, policy).masked_fill(row["exact"], 0)
    for key in ("teacher_p", "teacher_logp", "teacher_logits", "q", "k", "exact", "causal"):
        del row[key]


class ResidentRows:
    """Original all-resident cache, including one shared W_O cache per fit."""

    def __init__(self, heads, mean, policy, device, weights, split, emit):
        self.rows, self.policy, self.device = [], policy, device
        self.value_ready = False
        for head, samples in enumerate(heads):
            staged = prepare_samples(samples, mean[head], policy, device, weights=weights)
            weight = 1. / (len(heads) * len(samples))
            self.rows.extend((row, weight) for row in staged)
            emit("head_staged", head=head, split=split, samples=len(samples))

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def prepare_value(self, rotation):
        if self.value_ready:
            raise ValueError("OptR fixed K probabilities were already prepared")
        rotation = rotation.detach().to(self.device)
        for row, _ in self.rows:
            _freeze_value_row(row, rotation, self.policy)
        self.value_ready = True

    set_phase_v = prepare_value


def _stream_plan(heads, mean, limit):
    plan, largest = [], 0
    for head, samples in enumerate(heads):
        for sample in samples:
            count = sample["q"].shape[0]
            fixed = sample_cache_bytes(sample, queries=0)
            per_query = sample_cache_bytes(sample, queries=1) - fixed
            capacity = min(count, (limit - fixed) // per_query)
            if capacity < 1:
                raise ValueError(
                    f"OptR streamed cache cannot fit head {head} request {sample['request_id']!r}: "
                    f"complete K/V/W_O plus one query require {fixed + per_query} bytes, "
                    f"exceeding the {limit}-byte fixed row budget; histories are never truncated")
            for start in range(0, count, capacity):
                stop = min(start + capacity, count)
                weight = (stop - start) / (len(heads) * len(samples) * count)
                plan.append((sample, mean[head], start, stop, weight))
                largest = max(largest, fixed + (stop - start) * per_query)
    return plan, largest


class StreamedRows:
    """Repeatable one-row device staging over immutable CPU capture tensors."""

    def __init__(self, plan, policy, device, weights):
        self.plan, self.policy, self.device, self.weights = plan, policy, device, weights
        self.selected_k = None

    def __len__(self):
        return len(self.plan)

    def __iter__(self):
        for sample, mean, start, stop, weight in self.plan:
            # Keep only the current W_O, while reusing it across this head's
            # requests/chunks. All-head caches would defeat the row budget.
            if id(sample["wo"]) not in self.weights:
                self.weights.clear()
            chunk = {name: sample[name] for name in ("k", "v", "wo", "request_id")}
            chunk.update({name: sample[name][start:stop] for name in ("q", "positions", "contexts")})
            row = prepare_samples([chunk], mean, self.policy, self.device, weights=self.weights)[0]
            if self.selected_k is not None:
                _freeze_value_row(row, self.selected_k, self.policy)
            yield row, weight
            # The consumer must likewise delete its references before next().
            del row, chunk

    def prepare_value(self, rotation):
        if self.selected_k is not None:
            raise ValueError("OptR fixed K probabilities were already prepared")
        # Snapshot once: subsequent caller mutation or gradients cannot alter K.
        self.selected_k = rotation.detach().to(self.device).clone()

    set_phase_v = prepare_value


def prepare_shared_rows(train_heads, held_heads, mean, policy, device, emit):
    """Return repeatable training/held-out rows and an explicit resource record.

    Below 8 GiB, preserve the prior full-row preparation calls and arithmetic.
    Above that threshold, preflight every chunk before any device staging and
    retain at most one fixed row (512 MiB) plus the small selected-K matrix.
    Working tensors, device allocator overhead and model capture are separate.
    """
    resident_bytes = _resident_bytes(train_heads, held_heads)
    streamed = resident_bytes > RESIDENT_CACHE_BYTES
    weights = {}
    if streamed:
        train_plan, train_largest = _stream_plan(train_heads, mean, STREAMED_ROW_CACHE_BYTES)
        held_plan, held_largest = _stream_plan(held_heads, mean, STREAMED_ROW_CACHE_BYTES)
        train = StreamedRows(train_plan, policy, device, weights)
        held = StreamedRows(held_plan, policy, device, weights)
        largest = max(train_largest, held_largest)
    else:
        train = ResidentRows(train_heads, mean, policy, device, weights, "train", emit)
        held = ResidentRows(held_heads, mean, policy, device, weights, "validation", emit)
        largest = max(sample_cache_bytes(s) for heads in (train_heads, held_heads)
                      for samples in heads for s in samples)
    record = {
        "mode": "streamed" if streamed else "resident",
        "estimated_resident_fixed_bytes": resident_bytes,
        "resident_fixed_budget_bytes": RESIDENT_CACHE_BYTES,
        "streamed_row_fixed_budget_bytes": STREAMED_ROW_CACHE_BYTES,
        "largest_staged_fixed_row_bytes": largest,
        "train_rows": len(train), "validation_rows": len(held),
        "query_chunk_weighting": "chunk_queries/full_request_queries",
        "extra_transfers_per_iteration": streamed,
        "teacher_recomputed_per_iteration": streamed,
        "selected_k_probabilities_recomputed_per_v_iteration": streamed,
        "budget_scope": "fixed_tensor_cache_excludes_autograd_working_memory_and_allocator",
    }
    emit("staging_resources", **record)
    return train, held, record
