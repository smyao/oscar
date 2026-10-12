"""Archive #27/#94/#95/#125/#157: opt-in, same-stream first-error evidence.

No host inspection of device tensors, synchronization or numerical fallback.
The diagnostic op snapshots this small packet BEFORE the unchanged device Trap.
Static context labels describe the recorded callsite, not live request lengths.
"""
import json
import os

import torch


def enabled():
    value = os.environ.get("OSCAR_STATUS_DIAGNOSTICS", "0")
    if value not in ("0", "1"):
        raise ValueError("OSCAR_STATUS_DIAGNOSTICS must be 0 or 1")
    return value == "1"


def _gather(values, index, missing):
    if values is None or values.numel() == 0:
        return missing
    flat = values.reshape(-1)
    valid = (index >= 0) & (index < flat.numel())
    selected = flat.index_select(0, index.clamp(0, flat.numel() - 1).reshape(1))[0]
    return torch.where(valid, selected.to(torch.int64), missing)


def make_packet(statuses, *, heads=1, kv_heads=1, source_splits=1, tasks=None,
                slots=None, positions=None, starts=None, seq_lens=None,
                block_tables=None, block_size=0, attention_layout=False, flat_heads=1):
    """4 x 32 int64; reductions cover EVERY status, not only the first 1024.

    Columns 0:16: index/code/count/size/token/head/lane/slot/position/task_index/
    query_begin/query_end/seq_len/virtual_block/expected_slot/physical_page.
    Columns 16:32: raw task row. Non-attention rows use a representative source0
    task only for request context. Flat guards deliberately have no task mapping.
    """
    if (len(statuses) != 4 or heads <= 0 or kv_heads <= 0 or heads % kv_heads or
            source_splits <= 0 or flat_heads <= 0):
        raise ValueError("diagnostic packet requires four groups and positive geometry")
    rows = []
    for group, status in enumerate(statuses):
        flat = status.reshape(-1)
        missing = torch.full((), -1, dtype=torch.int64, device=status.device)
        size = torch.full_like(missing, flat.numel())
        if flat.numel():
            bad = flat != 0
            first = torch.where(bad, torch.arange(flat.numel(), device=status.device,
                                                 dtype=torch.int64), size).amin()
            index = torch.where(first < size, first, missing)
            code = torch.where(first < size, _gather(flat, index, missing), size * 0)
            count = bad.to(torch.int64).sum()
        else:
            index, code, count = missing, size, size
        token = head = lane = task_index = missing
        row = torch.full((16,), -1, dtype=torch.int64, device=status.device)
        if attention_layout:
            if group == 0:
                task_index = index // 2
                lane = index % 2
            else:
                width = kv_heads if group == 3 else heads
                token, head = index // width, index % width
                kh = head if group == 3 else head // (heads // kv_heads)
                task_index = (token * kv_heads + kh) * 3 * source_splits
            if tasks is not None and tasks.shape[0]:
                valid = (index >= 0) & (task_index >= 0) & (task_index < tasks.shape[0])
                selected = tasks.index_select(0, task_index.clamp(0, tasks.shape[0]-1).reshape(1))[0]
                row = torch.where(valid, selected.to(torch.int64), row)
                if group == 0:
                    token, head = row[0], row[2]
        elif slots is not None:
            # Flat padding checks have one status per supplied slot.
            token, head = index // flat_heads, index % flat_heads
        token = torch.where(index >= 0, token, missing)
        slot = _gather(slots, token, missing)
        position = _gather(positions, token, missing)
        request = row[5]
        begin = _gather(starts, request, missing)
        end = _gather(starts, torch.where(request >= 0, request + 1, missing), missing)
        length = _gather(seq_lens, request, missing)
        virtual_block = expected_slot = missing
        if block_tables is not None and block_tables.numel():
            # Later MTP seq_lens may include rejected tokens. Prefer checked
            # prepare's repaired position/context, NOT seq_len - query_len.
            # Error 5 means slot-context lookup failed: do not invent a position.
            logical = torch.where(position >= 0, position, row[8] + token - row[9])
            known = ((position >= 0) | ((row[10] != 5) & (row[8] >= 0) & (row[9] >= 0)))
            column = logical // 128
            valid = (known & (token >= 0) & (request >= 0) & (request < block_tables.shape[0]) &
                     (column >= 0) & (column < block_tables.shape[1]))
            offset = request * block_tables.shape[1] + column
            virtual_block = torch.where(valid, _gather(block_tables, offset, missing), missing)
            expected_slot = torch.where(valid & (virtual_block >= 0),
                                        virtual_block * 128 + logical % 128, missing)
        page = torch.where(slot >= 0, slot // block_size, missing) if block_size > 0 else missing
        prefix = torch.stack((index, code, count, size, token, head, lane, slot, position,
                              task_index, begin, end, length, virtual_block, expected_slot, page))
        rows.append(torch.cat((prefix, row)))
    return torch.stack(rows).contiguous()


def status_guard(ops, attention, rotate, merge, store, *, context=None, **geometry):
    if not enabled():
        return ops.status_guard(attention, rotate, merge, store)
    if not hasattr(ops, "status_guard_diagnostic"):
        raise RuntimeError("OSCAR_STATUS_DIAGNOSTICS requires rebuilding the OSCAR extension")
    packet = make_packet((attention, rotate, merge, store), **geometry)
    # Only callsite constants go into this label. Replay-varying evidence lives
    # in the device packet, including seq_lens, table entries and slot_mapping.
    label = json.dumps(context or {}, sort_keys=True, separators=(",", ":"))
    return ops.status_guard_diagnostic(attention, rotate, merge, store, packet, label)
