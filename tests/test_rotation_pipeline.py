"""CPU semantic tests for rotation constants; CANN/device evidence is separate.

Archive G27/#13-22/#37-49/#111 and PR oscar_attn.py:235-243: exact raw
window rows, FP32 rotated vectors, true absolute percentile interpolation.
These tests compare the butterfly and order-statistic design to the independent
torch oracle. They do not pretend to execute or validate an AscendC binary.
"""
from pathlib import Path

import pytest
import torch

from oscar_ascend.ops.reference import rotate_clip
from oscar_ascend.rotations import (
    build_hadamard_artifact, prepare_device_rotations, validate_artifact,
)


LAYER = "model.layers.3.self_attn.attn"


def _artifact(dim=64):
    return build_hadamard_artifact(layer_names=[LAYER], head_dim=dim,
        model_fingerprint="explicit-test-model", device="cpu", testing=True)


def _butterfly(x, coefficient):
    """Mathematical Sylvester decomposition, independent of CANN APIs."""
    dimension = x.shape[-1]
    result = x.float().clone()
    width = 1
    while width < dimension:
        groups = result.reshape(*result.shape[:-1], -1, 2, width)
        low, high = groups[..., 0, :].clone(), groups[..., 1, :].clone()
        groups[..., 0, :] = low + high
        groups[..., 1, :] = low - high
        width *= 2
    return result * coefficient


@pytest.mark.parametrize("dimension", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hadamard_butterfly_retains_fp32_rotation_semantics(dimension, dtype):
    artifact = _artifact(dimension)
    rotation = artifact["layers"][LAYER]["Rk"]
    source = torch.randn(9, 2, dimension, generator=torch.Generator().manual_seed(71)).to(dtype)
    actual = _butterfly(source, rotation[0, 0])
    expected = rotate_clip(source, rotation)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("dimension", [64, 128, 256])
def test_prepared_transposes_match_forward_and_inverse_equations(dimension):
    artifact = _artifact(dimension)
    matrices = prepare_device_rotations(artifact, device="cpu", testing=True)[LAYER]
    entry = artifact["layers"][LAYER]
    assert matrices.hadamard
    assert matrices.key_transposed.is_contiguous()
    assert matrices.value_transposed.is_contiguous()
    assert matrices.inverse_value_transposed.is_contiguous()
    torch.testing.assert_close(matrices.key_transposed.T, entry["Rk"], atol=0, rtol=0)
    torch.testing.assert_close(matrices.value_transposed.T, entry["Rv"], atol=0, rtol=0)
    torch.testing.assert_close(matrices.inverse_value_transposed.T, entry["Rv"].T, atol=0, rtol=0)


def test_calibrated_nonsymmetric_matrix_uses_correct_inverse_orientation():
    artifact = _artifact()
    rotation = torch.linalg.qr(torch.randn(64, 64, generator=torch.Generator().manual_seed(133))).Q
    artifact["objective"] = "qqt_sst_r_h_pbr"
    artifact["layers"][LAYER]["Rv"] = rotation
    valid = validate_artifact(artifact, layer_names=[LAYER], head_dim=64,
        model_fingerprint="explicit-test-model", device="cpu", testing=True)
    matrices = prepare_device_rotations(valid, device="cpu", testing=True)[LAYER]
    assert not matrices.hadamard
    vector = torch.randn(4, 64, generator=torch.Generator().manual_seed(8))
    rotated = vector @ matrices.value_transposed.T
    torch.testing.assert_close(rotated @ matrices.inverse_value_transposed.T, vector,
                               atol=2e-6, rtol=2e-6)


def test_forged_hadamard_label_is_rejected_before_fastpath_selection():
    artifact = _artifact()
    artifact["layers"][LAYER]["Rk"] = torch.eye(64)
    with pytest.raises(ValueError, match="exact Sylvester pattern"):
        prepare_device_rotations(artifact, device="cpu", testing=True)


def test_cpu_rotation_preparation_requires_explicit_testing():
    with pytest.raises(RuntimeError, match="requires NPU"):
        prepare_device_rotations(_artifact(), device="cpu")


@pytest.mark.parametrize("dimension", [64, 128, 256])
@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.5, 0.92, 0.96, 1.0])
def test_adjacent_sorted_absolute_values_match_pr_quantile(dimension, ratio):
    # Repeated magnitudes and signed outliers exercise genuine order statistics;
    # clipping at max*ratio or iterative floating threshold search is different.
    x = torch.linspace(-3.2, 2.7, dimension).repeat(3, 1)
    x[0, -1] = 100
    x[1, :dimension//2] = -0.75
    x[2] = 0.3
    if ratio == 0:
        expected = x
    else:
        values = x.abs().sort(dim=-1).values
        rank = torch.tensor(ratio, dtype=torch.float32) * (dimension - 1)
        low, high = int(rank), min(int(rank) + 1, dimension - 1)
        fraction = rank - low
        threshold = torch.lerp(values[:, low], values[:, high], fraction).unsqueeze(-1)
        expected = x.clamp(-threshold, threshold)
    oracle = rotate_clip(x, torch.eye(dimension), ratio)
    # Quantile's fused interpolation and the explicit lerp can differ by one
    # FP32 rounding; use the analytic two-rounding bound, not a fitted gate.
    rounding_bound = 2 * torch.finfo(torch.float32).eps * expected.abs().clamp_min(1)
    assert bool(((oracle - expected).abs() <= rounding_bound).all())


@pytest.mark.parametrize("recent", [1, 4, 259])
@pytest.mark.parametrize("start,count", [(0, 17), (64, 127), (1533, 809), (2177, 6145)])
def test_recent_winner_lookahead_has_exactly_one_writer_per_page_slot(recent, start, count):
    # Native mapping grants each writable page at most one contiguous interval.
    # Its arithmetic permits an O(1) lookahead instead of a history/token scan.
    physical = 2816
    slots = list(range(start, start + count))
    winners = {}
    for token, slot in enumerate(slots):
        page, in_page = divmod(slot, physical)
        later = slots[token + recent] if token + recent < len(slots) else -1
        keep = not (later >= 0 and later // physical == page and later == slot + recent)
        if keep:
            key = (page, in_page % recent)
            assert key not in winners
            winners[key] = slot
    expected = {}
    for slot in slots:
        page, in_page = divmod(slot, physical)
        expected[page, in_page % recent] = slot
    assert winners == expected


def test_rotation_kernel_keeps_cpu_simulator_and_production_launch_separate():
    # #91: source compiling does not prove execution. The CANN simulator must
    # include the same kernel body, while direct NPU launches stay out of it.
    source = (Path(__file__).parents[1] / "csrc/kernels/rotate_clip_store.cpp").read_text()
    guard = source.index("#ifndef ASCENDC_CPU_DEBUG")
    assert source.index("void oscar_rotate_clip_store_kernel") < guard
    assert source.index("<<<cores,nullptr,stream>>>") > guard


def test_masked_nan_and_sink_only_goldens_preserve_exact_contract(tmp_path):
    # G27/#12/#37-49: inspect independent expectations consumed by both the
    # actual CANN debugger and opt-in real NPU tests; this is not kernel proof.
    from tools.generate_rotation_cpu_cases import generate
    cases = generate(tmp_path)
    assert len(cases) == 26
    def tensor(directory, name, dtype):
        return torch.frombuffer(bytearray((directory / f"{name}.bin").read_bytes()), dtype=dtype)
    for dim in (64, 128, 256):
        for hadamard in (0, 1):
            directory = tmp_path / f"rotate_masked_{dim}_{hadamard}"
            slots = tensor(directory, "slots", torch.int64)
            expected = tensor(directory, "expected_output", torch.float32).reshape(13, 2, dim)
            status = tensor(directory, "expected_status", torch.int32).reshape(13, 2)
            assert bool((expected[slots < 0] == 0).all())
            assert bool((status[slots < 0] == 0).all())
            assert status[10, 0] == 2 and bool(torch.isnan(expected[10, 0]).all())
    directory = tmp_path / "rotate_store_sink_only"
    fields = (directory / "shape.txt").read_text().split()
    n, heads, dim, _, _, blocks, _, _, ks, vs, ts, sink, recent = map(int, fields[:13])
    assert n < sink and blocks == 1
    for name in ("expected_raw_key", "expected_raw_value"):
        raw = tensor(directory, name, torch.uint8)
        assert bool((raw[sink * heads * dim * 2:] == 173).all())
    tags = tensor(directory, "expected_tags", torch.uint8)
    assert bool((tags[sink * 8:] == 173).all())
