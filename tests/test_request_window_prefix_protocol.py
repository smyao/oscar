"""Archive #26/#37-49/#127/#128: independent host checkpoint protocol tests.

No serving integration, device copy, AscendC, graph or NPU evidence is produced.
"""
from dataclasses import replace

import pytest

from oscar_ascend.request_window_storage import RequestWindowLayout
from oscar_ascend.request_windows import WindowLease
from tools.experiments.request_window_prefix_protocol import (
    PrefixCheckpointKey, ExactPrefixCheckpoint, ExactPrefixDirectory,
    restore_exact_rows, select_joint_checkpoint_hit, dense_checkpoint_ledger,
)


def layout():
    return RequestWindowLayout(8, 64, 256, 3, 1, 256, 262144)


def checkpoint(end=4608, generation=1):
    shape = layout()
    key = PrefixCheckpointKey(b"model-layer-rotation-striped-v1", b"native-prefix-chain", 7, generation, end)
    rows = {p: ((p % 65536).to_bytes(2, "little") * 256,
                ((65535-p) % 65536).to_bytes(2, "little") * 256)
            for span in shape.retained_ranges(end) for p in span}
    return ExactPrefixCheckpoint.capture(shape, key, rows, committed_end=end,
                                         physical_block_tokens=4608), rows


def test_forks_restore_bf16_bits_into_distinct_lease_epochs():
    record, raw = checkpoint()
    shape = layout()
    left = restore_exact_rows(record, shape, WindowLease(1, 40))
    right = restore_exact_rows(record, shape, WindowLease(2, 41))
    assert len(left) == len(right) == 323
    for position in record.positions:
        row = shape.row_for_position(position)
        assert left[row][:2] == right[row][:2] == raw[position]
        assert shape.decode_tag(left[row][2]) == (40, position)
        assert shape.decode_tag(right[row][2]) == (41, position)
    # A divergent current write in one request cannot change immutable source
    # or the other request's restored destination.
    before = dict(right)
    left[shape.row_for_position(4608)] = (b"new", b"value", shape.encode_tag(40, 4608))
    assert right == before and record.keys == tuple(raw[p][0] for p in record.positions)


def test_publish_requires_every_exact_row_and_committed_boundary():
    record, raw = checkpoint()
    for committed in (4604, 4605, 4606, 4607):
        with pytest.raises(ValueError, match="committed"):
            ExactPrefixCheckpoint.capture(layout(), record.key, raw,
                committed_end=committed, physical_block_tokens=4608)
    missing = dict(raw); del missing[4607]
    with pytest.raises(KeyError):
        ExactPrefixCheckpoint.capture(layout(), record.key, missing,
            committed_end=4608, physical_block_tokens=4608)
    with pytest.raises(ValueError, match="complete"):
        ExactPrefixCheckpoint.capture(layout(), replace(record.key, logical_end=4607), raw,
            committed_end=4608, physical_block_tokens=4608)


def test_page_reuse_and_reset_cannot_invalidate_inflight_restore():
    record, _ = checkpoint()
    directory = ExactPrefixDirectory(1)
    directory.publish(record)
    directory.acquire(record.key); directory.acquire(record.key)
    with pytest.raises(ValueError, match="in-flight"):
        directory.evict(record.key)
    with pytest.raises(ValueError, match="reset"):
        directory.reset()
    directory.release(record.key)
    assert directory.acquire(record.key) is record
    directory.release(record.key); directory.release(record.key)
    directory.evict(record.key)
    fresh, _ = checkpoint(generation=2)
    directory.publish(fresh)
    with pytest.raises(KeyError):
        directory.acquire(record.key)
    assert directory.acquire(fresh.key) is fresh
    directory.release(fresh.key)
    directory.reset()
    assert not directory.records


def test_checkpoint_scope_hash_and_generation_must_all_match():
    record, _ = checkpoint()
    directory = ExactPrefixDirectory(1); directory.publish(record)
    for key in (replace(record.key, scope=b"other-rotation"),
                replace(record.key, block_hash=b"other-prefix"),
                replace(record.key, allocation_epoch=2)):
        with pytest.raises(KeyError):
            directory.acquire(key)
    corrupted = replace(record, keys=(b"corrupt", *record.keys[1:]))
    with pytest.raises(ValueError, match="immutable"):
        directory.publish(corrupted)


def test_sparse_full_checkpoint_and_gdn_must_reach_the_same_fixed_point():
    calls=[]
    native_joint_hits={4608, 13824}  # GDN lacks state at the 9216 checkpoint.
    def native(bound):
        calls.append(bound)
        return max((n for n in native_joint_hits if n<=bound), default=0)
    # Layer2 lacks 13824. Native 13824 -> FULL checkpoint9216 -> native4608.
    hit=select_joint_checkpoint_hit(18000,native,[{4608,9216,13824},{4608,9216}])
    assert hit==4608 and calls==[18000,9216]
    # A native larger hit does not imply all smaller FULL windows exist.
    assert select_joint_checkpoint_hit(18000,native,[{9216},{9216}])==0
    assert select_joint_checkpoint_hit(18000,native,[set(),{4608}])==0


def test_dense_prefix_checkpoints_cannot_be_reported_as_sixfold_capacity():
    report=dense_checkpoint_ledger(page_bytes=817152,native_block_tokens=768,
        compressed_block_tokens=4608,layout=layout())
    assert report['checkpoint_bytes_per_page']==333336
    assert report['nominal_full_ratio_without_checkpoints']==6
    assert report['full_ratio_dense_checkpoint_upper_bound']==pytest.approx(4.261593341260404)
