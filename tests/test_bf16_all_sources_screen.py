"""Archive #13-22/#126/#148/#154/#155: CPU arithmetic/protocol diagnostics only."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from oscar_ascend.ops import reference
from tools.experiments import bf16_all_sources_screen as screen,bf16_direct_decode as decode
from tools.striped_fixture import canonical_slots_to_striped

ACCEPTANCE=json.loads((screen.ROOT/'configs/acceptance.json').read_text())


@pytest.fixture(autouse=True)
def single_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_four_level_lookup_matches_full_code_and_metadata_sweeps():
    report=decode.validate()
    assert report['status']=='cpu_direct_decode_passed'
    assert report['packed_words']==65536 and report['levels_checked']==2539488
    assert report['exact_halfway_values_checked']==142980
    assert report['metadata_cartesian_product_exhaustive'] is False
    assert report['device_completion']=='not_run'


def test_canonical_and_striped_lookup_match_unchanged_codec_bf16_cast():
    generator=torch.Generator().manual_seed(1731)
    k=torch.randn((7,2,256),generator=generator);v=torch.randn(k.shape,generator=generator)
    raw=reference.encode_kv(k,v);gold=tuple(x.bfloat16() for x in reference.decode_kv(raw,256))
    canonical=decode.decode_canonical_kv(raw)
    striped=canonical_slots_to_striped(raw).reshape(-1,136)
    for i,(start,meta) in enumerate(((0,128),(64,132))):
        words=striped[:,start:start+64:2].int()|(striped[:,start+1:start+64:2].int()<<8)
        sb=striped[:,meta].int()|(striped[:,meta+1].int()<<8)
        zb=striped[:,meta+2].int()|(striped[:,meta+3].int()<<8)
        actual=decode.decode_words(words,sb.to(torch.int16).view(torch.float16),zb.to(torch.int16).view(torch.float16)).view(7,2,256)
        assert torch.equal(actual.view(torch.int16),gold[i].view(torch.int16))
        assert torch.equal(canonical[i].view(torch.int16),gold[i].view(torch.int16))


@pytest.mark.parametrize('scale_bits,zero_bits',[(0,0),(0x8000,0),(0xbc00,0),(0x7c00,0),
    (0xfc00,0),(0x7e00,0),(0xfe00,0),(0x3c00,0x7c00),(0x3c00,0xfc00),(0x3c00,0x7e00),(0x3c00,0xfe00)])
def test_live_invalid_metadata_errors_but_padded_poison_is_harmless(scale_bits,zero_bits):
    scale=torch.tensor([scale_bits],dtype=torch.int32).to(torch.int16).view(torch.float16)
    zero=torch.tensor([zero_bits],dtype=torch.int32).to(torch.int16).view(torch.float16)
    with pytest.raises(ValueError,match='error3'):decode.four_levels(scale,zero)
    padded=decode.decode_words(torch.full((1,32),0xffff,dtype=torch.int32),scale,zero,live=torch.tensor([False]))
    assert torch.equal(padded.view(torch.int16),torch.zeros((1,256),dtype=torch.int16))


def test_halfway_nearest_even_neighbors_negative_ties_and_signed_zero():
    # Valid half inputs around BF16 boundaries. code0 covers positive and
    # negative ties; code1..3 also exercise the non-fused metadata arithmetic.
    zeros=torch.tensor([1.0029296875,1.00390625,1.0048828125,1.01171875,
                        -1.00390625,-1.01171875,-0.0],dtype=torch.float16)
    scales=torch.full_like(zeros,2**-24)
    actual=decode.four_levels(scales,zeros)
    oracle=decode.independent_level_bits(scales.view(torch.int16).numpy().view(np.uint16),zeros.view(torch.int16).numpy().view(np.uint16))
    assert np.array_equal(actual.view(torch.int16).numpy().view(np.uint16),oracle)
    assert actual[:6,0].float().tolist()==[1.,1.,1.0078125,1.015625,-1.,-1.015625]
    assert actual.view(torch.int16)[-1,0].item()==0


def test_every_source_uses_bf16_p_while_retaining_fp32_softmax_denominator():
    generator=torch.Generator().manual_seed(3)
    q=torch.randn((4,6,256),generator=generator).bfloat16()
    k=torch.randn((37,1,256),generator=generator).bfloat16()
    v=torch.randn(k.shape,generator=generator).bfloat16()
    mask=torch.ones((4,37),dtype=torch.bool)
    baseline=screen.online(q,k,v,mask,'fp32');full=screen.online(q,k,v,mask,'full')
    assert torch.equal(baseline.lse,full.lse)  # exact BF16 Q/K; FP32 exp/sum
    assert not torch.equal(baseline.output.view(torch.int32),full.output.view(torch.int32))
    assert screen.compare(full.output,baseline.output,ACCEPTANCE['fused_attention'])['passed']


def test_unified_screen_retains_analytic_failure_and_independent_oracle():
    row=screen.evaluate(screen.CASES[0],ACCEPTANCE['fused_attention'])
    assert row['baseline_passed'] and not row['candidate_passed']
    assert row['operand_contract']==decode.CONTRACT and row['source12_bitwise_required'] is False
    failure=row['stages']['full']['output']['first_failure']
    assert failure['actual']==0.0 and failure['expected']==pytest.approx(-.021484375,abs=1e-7)
    assert row['stages']['full']['direct_history_decode_identity']
    assert row['stages']['full']['precise_bf16_input_values_preserved']


def test_empty_history_still_executes_changed_current_arithmetic():
    row=screen.evaluate(screen.CASES[1],ACCEPTANCE['fused_attention'])
    assert row['baseline_passed'] and row['candidate_passed']
    assert row['stages']['full']['output']['max_abs']>0
    assert row['stages']['q_only']['output']['max_abs']<row['stages']['full']['output']['max_abs']
    assert 'unchanged_source1_source2_bits' not in row['stages']['full']


def test_passing_subset_and_historical_contract_cannot_qualify_new_screen():
    from tools import probe_bf16_history as probe
    subset=screen.run((screen.CASES[1],))
    assert subset['status']=='cpu_screen_incomplete' and subset['returncode']==2
    historical=json.loads((screen.ROOT/'reports/bf16_history_stage1_20261016.json').read_text())
    assert 'operand_contract' in probe.cpu_screen_identity_errors(historical)


def test_experiment_output_paths_cannot_replace_historical_receipts(tmp_path):
    path=tmp_path/'existing.json';path.write_text('{"status":"old_failure"}')
    for main in (screen.main,decode.main):
        with pytest.raises(FileExistsError):main(['--output',str(path)])
        assert json.loads(path.read_text())['status']=='old_failure'
