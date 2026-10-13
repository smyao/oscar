# 档案 #157：later-draft 死行 slot 净化必须与 prepare 内核 slotContext 解析逐位一致。
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(scope="module")
def mask_fn():
    backend = types.ModuleType("vllm.v1.attention.backend")
    backend.AttentionImpl = type("AttentionImpl", (), {})
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/integration/impl.py"
    with patch.dict(sys.modules, {backend.__name__: backend}):
        spec = importlib.util.spec_from_file_location("oscar_ascend.integration._draft_mask_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module._unresolvable_later_draft_slots



def _reference_dead(slots, seq_lens, block_tables, num_reqs):
    """Pure-Python mirror of attention_tasks.cpp slotContext: matches==1 and 0<=position<lens."""
    dead = []
    cols = block_tables.shape[1]
    for token in range(slots.numel()):
        slot = int(slots[token])
        if slot < 0:
            dead.append(False)
            continue
        r = min(token, num_reqs - 1)
        lens = int(seq_lens[r])
        valid = min(cols, (lens + 127) // 128)
        vblock = slot // 128
        matches = [c for c in range(valid) if int(block_tables[r][c]) == vblock]
        position = (matches[0] if matches else 0) * 128 + slot % 128
        dead.append(not (len(matches) == 1 and 0 <= position < lens))
    return torch.tensor(dead, dtype=torch.bool)


def _check(mask_fn, slots, seq_lens, block_tables, num_reqs):
    mask = mask_fn(slots, seq_lens, block_tables, num_reqs)
    expected = _reference_dead(slots, seq_lens, block_tables, num_reqs)
    assert mask is not None and torch.equal(mask.cpu(), expected)


def test_live_rows_resolve_and_dead_rows_mask(mask_fn):
    # Two requests; request 0 has two virtual128 pages, request 1 has one.
    block_tables = torch.tensor([[10, 11, 0, 0], [22, 0, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([200, 90], dtype=torch.int32)
    # live: token0 slot in request0 page10 first column; token1 dead: slot points
    # at request1's absent second page region (no matching column).
    slots = torch.tensor([10 * 128 + 5, 23 * 128 + 7], dtype=torch.int64)
    _check(mask_fn, slots, seq_lens, block_tables, 2)
    dead = mask_fn(slots, seq_lens, block_tables, 2)
    assert dead.tolist() == [False, True]


def test_padding_negative_slots_never_masked_and_empty_inputs(mask_fn):
    block_tables = torch.tensor([[10, 11]], dtype=torch.int32)
    seq_lens = torch.tensor([200], dtype=torch.int32)
    slots = torch.tensor([-1, 10 * 128 + 3], dtype=torch.int64)
    _check(mask_fn, slots, seq_lens, block_tables, 1)
    dead = mask_fn(slots, seq_lens, block_tables, 1)
    assert dead.tolist() == [False, False]
    assert mask_fn(torch.empty(0, dtype=torch.int64),
                                           seq_lens, block_tables, 1) is None
    assert mask_fn(slots, seq_lens, block_tables, 0) is None


def test_duplicate_block_and_position_beyond_lens_mask(mask_fn):
    # Same virtual block twice in the valid prefix is not a unique match.
    block_tables = torch.tensor([[10, 10]], dtype=torch.int32)
    seq_lens = torch.tensor([200], dtype=torch.int32)
    slots = torch.tensor([10 * 128 + 1], dtype=torch.int64)
    _check(mask_fn, slots, seq_lens, block_tables, 1)
    assert mask_fn(slots, seq_lens, block_tables, 1).tolist() == [True]
    # Position beyond the retained sequence length fails the bound.
    block_tables = torch.tensor([[10, 11]], dtype=torch.int32)
    seq_lens = torch.tensor([130], dtype=torch.int32)  # valid columns = 2
    slots = torch.tensor([11 * 128 + 5], dtype=torch.int64)  # position 133 > 130
    _check(mask_fn, slots, seq_lens, block_tables, 1)
    assert mask_fn(slots, seq_lens, block_tables, 1).tolist() == [True]
