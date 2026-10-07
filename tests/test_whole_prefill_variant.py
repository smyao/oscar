"""Whole-prompt policy is explicit and cannot leak into a native comparison."""
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


def test_generated_policy_is_idempotent_and_restores_native_budget():
    original=configuration();snapshot=deepcopy(original)
    candidate=variant_config(original,'candidate')
    assert candidate['max_num_batched_tokens']==32768
    assert candidate['scheduler_cls']==WHOLE_PREFILL_SCHEDULER
    assert variant_config(candidate,'candidate')==candidate
    for mode in ('native','baseline'):
        restored=variant_config(candidate,mode)
        assert restored['max_num_batched_tokens']==original['max_num_batched_tokens']==16384
        assert 'scheduler_cls' not in restored
        assert restored['experimental_whole_prefill'] is False
    assert original==snapshot


@pytest.mark.parametrize('missing',['experimental_current_only','experimental_first_mtp_current_fia'])
def test_no_quadratic_first_mtp_regression_from_budget_only(missing):
    config=configuration();config[missing]=False
    with pytest.raises(ValueError,match='requires'):
        variant_config(config,None)


def test_explicit_other_scheduler_not_overwritten():
    config=configuration();config['scheduler_cls']='user.CustomScheduler'
    with pytest.raises(ValueError,match='conflicts'):
        variant_config(config,'candidate')
