"""Archive #27/#34/#36/#37-49/#157/#158: preserve native request discard semantics.

Native model_runner_v1.py:1125-1147 publishes discard_request_mask in current
input_batch order before proposing; llm_base_proposer.py:1535-1666 retains that
order with one token/request on later steps. Only this explicit mask permits
padding. A slot/table error is never evidence that a live request was discarded.
Copies occur in metadata construction, before eager execution or graph replay;
fixed per-draft destinations preserve captured addresses. No device readback,
block-table scan, KV data operation, or per-layer tensor work is introduced.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from .runtime_api import OscarReadinessError


@dataclass(frozen=True)
class _DraftScope:
    proposer: object
    mask: object
    requests: int
    dummy: bool


_scope = ContextVar("oscar_native_draft_discard", default=None)


@contextmanager
def native_draft_scope(proposer, *, dummy=False):
    """Keep each native proposal's mask/order within its synchronous call.

    The mask is the native fixed device buffer, not a cached derived mask.
    Metadata construction enqueues the copy/mask on the native current stream,
    before the proposal runs and before the next input preparation updates it.
    Dummy construction never consumes stale real-request mask contents.
    """
    runner = proposer.runner
    requests = 0 if dummy else runner.input_batch.num_reqs
    mask = None if dummy else getattr(getattr(runner, "discard_request_mask", None), "gpu", None)
    if not dummy:
        if (type(requests) is not int or requests < 0 or not torch.is_tensor(mask)
                or mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() < requests):
            raise OscarReadinessError("native MTP discard mask does not match the current input batch")
        if getattr(proposer, "pcp_size", 1) != 1 or getattr(proposer, "dcp_size", 1) != 1:
            raise OscarReadinessError("native MTP discard bridge requires request order without context parallelism")
    token = _scope.set(_DraftScope(proposer, mask, requests, dummy))
    try:
        yield
    finally:
        _scope.reset(token)


def later_draft_slot_copy(common, *, draft_index=None, capture=False, require_scope=False):
    """Return a fixed OSCAR slot copy with only native-discarded rows padded.

    Capture builders do not receive a draft_index. Identify their source by
    its native persistent slot group address, which is also used by replay.
    First draft and main-model slots always retain their original address.
    All live and trailing-padding bytes survive, even when they are malformed;
    prepare/status_guard remains responsible for rejecting those bytes.
    """
    scope = _scope.get()
    source = common.slot_mapping
    if require_scope and scope is None and draft_index != 0 and not capture:
        raise OscarReadinessError(
            "OSCAR native MTP discard bridge missing: later-draft metadata was built "
            "outside the hooked proposer scope")
    if scope is None or draft_index == 0:
        return source
    groups = scope.proposer.slot_mapping_group
    index = next((i for i, group in enumerate(groups)
                  if source.data_ptr() == group.data_ptr()), None)
    if index is None or (draft_index is not None and index != draft_index):
        raise OscarReadinessError("later MTP slots are not the native draft group's fixed buffer")
    if index == 0:
        return source
    native = groups[index]
    if (source.ndim != 1 or source.stride() != (1,)
            or source.dtype not in (torch.int32, torch.int64)
            or source.dtype != native.dtype or source.device != native.device
            or source.numel() > native.numel()):
        raise OscarReadinessError("later MTP slot view does not match its native allocation")
    if not scope.dummy and not capture:
        if (common.max_query_len != 1 or common.num_actual_tokens != scope.requests
                or common.num_reqs < scope.requests or source.numel() < scope.requests
                or scope.mask.device != source.device):
            raise OscarReadinessError("later MTP discard rows do not match native one-token request order")
    copies = getattr(scope.proposer, "_oscar_discard_slot_copies", None)
    if copies is None:
        copies = {}
        scope.proposer._oscar_discard_slot_copies = copies
    entry = copies.get(index)
    if entry is None:
        entry = (native, torch.empty_like(native))
        copies[index] = entry
    original, destination = entry
    if (original.data_ptr() != native.data_ptr() or original.shape != native.shape
            or original.dtype != native.dtype or original.device != native.device):
        raise OscarReadinessError("native MTP slot allocation changed after OSCAR binding")
    result = destination[:source.numel()]
    result.copy_(source)
    if not scope.dummy and not capture:
        result[:scope.requests].masked_fill_(scope.mask[:scope.requests], -1)
        if not getattr(scope.proposer, "_oscar_discard_bridge_logged", False):
            # Host-only execution evidence; the native mask stays on device.
            # Enqueue is not a claim that device execution has completed.
            print("[oscar] MTP_DISCARD_BRIDGE "
                  f"proposer={type(scope.proposer).__name__} method=mtp "
                  f"requests={scope.requests} mask_capacity={scope.mask.numel()} "
                  f"draft_index={index} source=native.discard_request_mask "
                  "binding=queued", flush=True)
            scope.proposer._oscar_discard_bridge_logged = True
    return result


def later_draft_lengths_copy(common, slots, *, draft_index):
    """Keep valid accepted positions visible when native MTP wraps lens to 1.

    Native llm_base_proposer.py:1608 wraps its optimistic (rejected rows kept)
    length when it exceeds max_model_len, while padding the slot only when
    the accepted position itself exceeds the limit. Those conditions differ.
    The original first-pass device length plus draft_index identifies this
    explicit wrap. Restore only its scanning upper bound, never infer discard
    or accepted position from a bad slot or an mRoPE coordinate. Each proposal
    refreshes the fixed destination before executing its later draft model.
    """
    scope = _scope.get()
    source = common.seq_lens
    if scope is None or scope.dummy or draft_index == 0:
        return source
    proposer = scope.proposer
    limit = getattr(proposer, "max_model_len", None)
    groups = getattr(proposer, "seq_lens_group", None)
    if (type(limit) is not int or limit <= 0 or groups is None
            or draft_index >= len(groups)):
        raise OscarReadinessError("native MTP model limit or fixed length groups are missing")
    base, native = groups[0], groups[draft_index]
    if (source.data_ptr() != native.data_ptr() or source.ndim != 1 or source.dtype != torch.int32
            or source.device != slots.device or source.numel() < scope.requests
            or base.dtype != source.dtype or base.device != source.device
            or base.numel() < scope.requests):
        raise OscarReadinessError("native MTP length groups do not match this proposal")
    copies = getattr(proposer, "_oscar_later_length_copies", None)
    if copies is None:
        copies = {}
        proposer._oscar_later_length_copies = copies
    entry = copies.get(draft_index)
    if entry is None:
        entry = (native, torch.empty_like(native))
        copies[draft_index] = entry
    original, destination = entry
    if (original.data_ptr() != native.data_ptr() or original.shape != native.shape
            or original.dtype != native.dtype or original.device != native.device):
        raise OscarReadinessError("native MTP length allocation changed after OSCAR binding")
    result = destination[:source.numel()]
    result.copy_(source)
    # Native NPUModelRunner._build_attention_metadata publishes the maximum
    # CPU optimistic sequence length (or the full model limit for capture).
    # It bounds the corrected first-pass device lengths. Its scalar survives
    # native shallow-copy later metadata unchanged. Preserve the destination
    # address but avoid extra device predicates when this bound proves no wrap.
    upper = getattr(common, "max_seq_len", 0)
    if type(upper) is int and 0 < upper and upper + draft_index <= limit:
        return result
    size = scope.requests
    before_wrap = base[:size] + draft_index
    wrapped_live = ((source[:size] == 1) & (before_wrap > limit)
                    & (slots[:size] >= 0) & ~scope.mask[:size])
    result[:size].copy_(torch.where(wrapped_live, before_wrap.clamp(max=limit), source[:size]))
    return result
