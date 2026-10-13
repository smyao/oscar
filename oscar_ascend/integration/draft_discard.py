"""Archive #27/#34/#36/#37-49/#157: preserve native request discard semantics.

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


def later_draft_slot_copy(common, *, draft_index=None, capture=False):
    """Return a fixed OSCAR slot copy with only native-discarded rows padded.

    Capture builders do not receive a draft_index. Identify their source by
    its native persistent slot group address, which is also used by replay.
    First draft and main-model slots always retain their original address.
    All live and trailing-padding bytes survive, even when they are malformed;
    prepare/status_guard remains responsible for rejecting those bytes.
    """
    scope = _scope.get()
    source = common.slot_mapping
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
    return result
