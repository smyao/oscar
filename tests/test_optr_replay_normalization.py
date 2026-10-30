"""Offline normalization must follow deployed FP32 reciprocal then multiply.

The expected values below follow the explicit Muls(acc, 1.0F/sum) in
attention_cv_official_{decode,cluster4,cluster16}.cpp. These are CPU arithmetic
witnesses, not bitwise NPU, full-model quality, or performance acceptance.
"""
import pytest
import torch

from oscar_ascend.optr_calibration import _tiles, replay_sample


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_partial_normalization_uses_source_reciprocal_multiply_order():
    query = torch.zeros(1, 6, 256)
    key = torch.zeros(7, 256)
    value = key.clone()
    value[0, 0], value[1, 0] = 9.4375, -.00390625
    # All inputs are exactly BF16. Zero scores make P=1 and sum=7, hence
    # acc[0]=9.43359375 without any GEMM/reduction uncertainty in this example.
    assert torch.equal(value, value.bfloat16().float())
    actual, lse = _tiles(query, key, value, torch.tensor([6]), torch.arange(7))
    expected = torch.tensor(9.43359375) * (torch.tensor(1.) / torch.tensor(7.))
    assert float(expected) == 1.3476563692092896
    assert torch.equal(actual[0, :, 0], expected.expand(6))
    assert not torch.count_nonzero(actual[..., 1:])
    assert torch.isfinite(lse).all()
    # Division rounds exactly onto a BF16 midpoint; multiplication by the
    # already-rounded reciprocal falls above it and changes the BF16 result.
    assert float(expected.bfloat16()) == 1.3515625
    assert float((torch.tensor(9.43359375) / 7).bfloat16()) == 1.34375


def test_reciprocal_order_survives_real_d256_gqa6_window_merge_and_inverse():
    dim, group = 256, 6
    query = torch.zeros(1, group, dim, dtype=torch.bfloat16)
    key = torch.zeros(4, dim, dtype=torch.bfloat16)
    value = key.clone()
    value[1, 0], value[2, 0] = 1., 1. / 256
    sample = {
        "request_id": "normalization_arithmetic_witness",
        "q": query, "k": key, "v": value,
        "gate": torch.ones_like(query), "teacher": torch.zeros_like(query),
        "wo": torch.zeros(group, 1, dim),
        "positions": torch.tensor([3]), "contexts": torch.tensor([1]),
        "current_ends": torch.tensor([4]), "history_splits": torch.tensor([1]),
    }
    identity = torch.eye(dim)
    actual = replay_sample(identity, identity, torch.zeros(dim), sample,
                           {"replay_partition": "captured_runtime_frames"})
    # One zero-valued exact-window row and three current rows: the current
    # partial has sum=3 and accumulator=1+1/256. Production merge computes
    # normalized LSE weights first, then the final BF16 inverse rounds once.
    lse = torch.tensor([1., 3.]).log()
    weights = (lse - lse.max()).exp()
    weights /= weights.sum()
    accumulator = torch.tensor(1. + 1. / 256)
    source_order = accumulator * (torch.tensor(1.) / 3) * weights[1]
    old_order = (accumulator / 3) * weights[1]
    assert float(source_order.bfloat16()) == .251953125
    assert float(old_order.bfloat16()) == .25
    expected = torch.zeros_like(actual)
    expected[..., 0] = source_order.bfloat16().float()
    assert torch.equal(actual, expected)


def test_empty_source_stays_zero_with_negative_infinite_lse():
    query = torch.zeros(1, 6, 256)
    empty = torch.zeros(0, 256)
    actual, lse = _tiles(query, empty, empty, torch.tensor([0]),
                         torch.empty(0, dtype=torch.int64))
    assert not torch.count_nonzero(actual)
    assert torch.isneginf(lse).all()
