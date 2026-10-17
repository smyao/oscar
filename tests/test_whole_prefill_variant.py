"""Archive #34/#36/#163: retired scheduling cannot change stable/native budgets."""
from copy import deepcopy
import pytest
from tools.serving_variants import variant_config,WHOLE_PREFILL_SCHEDULER


def configuration():
    import json
    from pathlib import Path
    result=json.loads(Path('configs/target.json').read_text())
    result.update(experimental_current_only=True,experimental_first_mtp_current_fia=True,
                  experimental_whole_prefill=True)
    return result


def test_retired_policy_is_idempotent_and_keeps_user_explicit_budget():
    original=configuration();snapshot=deepcopy(original)
    candidate=variant_config(original,'candidate')
    assert candidate['max_num_batched_tokens']==16384
    assert 'scheduler_cls' not in candidate
    assert candidate['experimental_whole_prefill'] is False
    assert variant_config(candidate,'candidate')==candidate
    for mode in ('native',):
        restored=variant_config(candidate,mode)
        assert restored['max_num_batched_tokens']==original['max_num_batched_tokens']==16384
        assert 'scheduler_cls' not in restored
        assert restored['experimental_whole_prefill'] is False
    with pytest.raises(ValueError, match='baseline FP32 OSCAR production attention is retired'):
        variant_config(candidate,'baseline')
    assert original==snapshot


@pytest.mark.parametrize('missing',['experimental_current_only','experimental_first_mtp_current_fia'])
def test_candidate_restores_complete_current_bundle_without_enabling_whole_prefill(missing):
    config=configuration();config[missing]=False
    selected=variant_config(config,None)
    assert selected[missing] is True
    assert selected['experimental_whole_prefill'] is False
    assert selected['max_num_batched_tokens']==16384
    assert 'scheduler_cls' not in selected
    assert config[missing] is False


def test_explicit_other_scheduler_not_overwritten():
    config=configuration();config['scheduler_cls']='user.CustomScheduler'
    with pytest.raises(ValueError,match='conflicts'):
        variant_config(config,'candidate')
