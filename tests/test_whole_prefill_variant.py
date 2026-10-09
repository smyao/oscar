"""The candidate leaves vLLM token budgeting and scheduling at native defaults."""
from copy import deepcopy
from tools.serving_variants import variant_config


def configuration():
    import json
    from pathlib import Path
    result=json.loads(Path('configs/target.json').read_text())
    return result


def test_candidate_does_not_generate_a_token_budget_or_scheduler():
    original=configuration();snapshot=deepcopy(original)
    candidate=variant_config(original,'candidate')
    assert 'max_num_batched_tokens' not in candidate
    assert 'scheduler_cls' not in candidate
    assert variant_config(candidate,'candidate')==candidate
    for mode in ('native','baseline'):
        restored=variant_config(candidate,mode)
        assert 'max_num_batched_tokens' not in restored
        assert 'scheduler_cls' not in restored
    assert original==snapshot


def test_explicit_user_scheduler_is_not_overwritten():
    config=configuration();config['scheduler_cls']='user.CustomScheduler'
    assert variant_config(config,'candidate')['scheduler_cls']=='user.CustomScheduler'
