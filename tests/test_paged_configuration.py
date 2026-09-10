"""Keep unvalidated Ascend tile experiments out of the default serving path."""

import pytest

from oscar_ascend.kernels.paged_attention import grouped_mtp_splits, paged_block_kv


def test_default_uses_npu_validated_tile(monkeypatch):
    monkeypatch.delenv("OSCAR_ASCEND_PAGED_BLOCK_KV", raising=False)
    assert paged_block_kv() == 4


def test_larger_tile_requires_explicit_configuration(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_PAGED_BLOCK_KV", "32")
    assert paged_block_kv() == 32


def test_unsupported_tile_is_rejected(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_PAGED_BLOCK_KV", "12")
    with pytest.raises(ValueError, match="must be"):
        paged_block_kv()


@pytest.mark.parametrize(
    ("length", "expected"),
    [(1, 1), (1024, 1), (1025, 2), (2048, 2), (16384, 16), (24579, 32)],
)
def test_grouped_mtp_splits_bound_long_programs(monkeypatch, length, expected):
    monkeypatch.delenv("OSCAR_ASCEND_GROUPED_MTP_TARGET_KV_PER_SPLIT", raising=False)
    monkeypatch.delenv("OSCAR_ASCEND_GROUPED_MTP_MAX_SPLITS", raising=False)
    assert grouped_mtp_splits(length) == expected


def test_grouped_mtp_splits_respect_cap(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_GROUPED_MTP_TARGET_KV_PER_SPLIT", "256")
    monkeypatch.setenv("OSCAR_ASCEND_GROUPED_MTP_MAX_SPLITS", "32")
    assert grouped_mtp_splits(24579) == 32


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OSCAR_ASCEND_GROUPED_MTP_TARGET_KV_PER_SPLIT", "0"),
        ("OSCAR_ASCEND_GROUPED_MTP_MAX_SPLITS", "12"),
    ],
)
def test_grouped_mtp_split_configuration_is_validated(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="OSCAR_ASCEND_GROUPED_MTP"):
        grouped_mtp_splits(16384)
