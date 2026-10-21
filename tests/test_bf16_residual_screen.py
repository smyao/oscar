"""Independent CPU arithmetic evidence; no device/production qualification."""
import json

import pytest
import torch

from tools.experiments import bf16_residual_screen as screen

ACCEPTANCE = json.loads((screen.ROOT / 'configs/acceptance.json').read_text())


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_residual_planes_recover_bf16_cancellation_information():
    x = torch.tensor([256., -255.5, 1.00390625, -1.00390625, 0.])
    hi, lo = screen.split_bf16(x)
    assert torch.equal(hi, hi.bfloat16().float())
    assert torch.equal(lo, lo.bfloat16().float())
    assert not torch.equal(hi, x)
    assert torch.equal(hi + lo, x)


@pytest.mark.parametrize('terms', [3, 4])
def test_explicit_cross_products_and_precision(terms):
    a = torch.tensor([[1.00390625, -2.0078125]])
    b = torch.tensor([[3.01171875], [4.015625]])
    ah, al = screen.split_bf16(a)
    bh, bl = screen.split_bf16(b)
    expected = (ah @ bh + ah @ bl) + al @ bh
    if terms == 4:
        expected = expected + al @ bl
    actual = screen.residual_product('ij,jk->ik', a, b, terms)
    assert torch.equal(actual, expected)
    assert (actual - a @ b).abs().max() < (ah @ bh - a @ b).abs().max()
    if terms == 4:
        assert torch.equal(actual, a @ b)


def test_precise_sources_keep_qk_lse_and_compensate_p():
    generator = torch.Generator().manual_seed(3)
    q = torch.randn((4, 6, 256), generator=generator).bfloat16()
    k = torch.randn((37, 1, 256), generator=generator).bfloat16()
    v = torch.randn(k.shape, generator=generator).bfloat16()
    mask = torch.ones((4, 37), dtype=torch.bool)
    gold = screen.online(q, k, v, mask, 'fp32', history=False)
    a = screen.online(q, k, v, mask, 'residual3', history=False)
    b = screen.online(q, k, v, mask, 'residual4', history=False)
    assert torch.equal(gold.lse, a.lse)
    assert torch.equal(a.output, b.output)
    assert screen.compare(a.output, gold.output, ACCEPTANCE['fused_attention'])['passed']


@pytest.mark.parametrize('case', screen.CASES, ids=lambda c: c.name)
def test_fixed_cases_pass_every_unchanged_gate(case):
    row = screen.evaluate(case, ACCEPTANCE['fused_attention'], ACCEPTANCE['prefill_window']['relative_l2_lt'])
    assert row['baseline_passed']
    assert row['input_bytes_unchanged']
    for stage in ('residual3', 'residual4'):
        for key, metric in row['stages'][stage].items():
            assert metric['passed'] if isinstance(metric, dict) else metric, (stage, key, metric)


def test_subset_does_not_qualify_and_report_cannot_overwrite(tmp_path):
    row = screen.run((screen.CASES[1],))
    assert row['status'] == 'cpu_screen_incomplete'
    assert row['returncode'] == 2
    assert not row['production_qualified']
    path = tmp_path / 'old.json'
    path.write_text('{"old":true}')
    with pytest.raises(FileExistsError):
        screen.main(['--output', str(path)])
    assert json.loads(path.read_text()) == {'old': True}


def test_residual_metadata_sweep_keeps_approximation_boundary_explicit():
    row = screen.validate_residual_split()
    assert row['status'] == 'cpu_residual_split_passed'
    assert row['levels_checked'] == 2539488
    assert row['codes_checked_per_pair'] == 4
    assert row['fp32_reconstruction_not_exact_elements'] > 0
    assert row['metadata_cartesian_product_exhaustive'] is False
    assert row['device_completion'] == 'not_run'
