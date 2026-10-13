"""User 2026-10-13 stable policy; archive #34/#36/#148: actual shared launch args."""
import json
from pathlib import Path
import pytest
from tools import install_serve, observe_serve
from tools.serving_variants import variant_config
from tools.target_cli import serve_argv

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('entry', [install_serve.main, observe_serve.main])
def test_candidate_one_key_has_explicit_stable_policy(entry, capsys):
    assert entry(['--variant', 'candidate', '--plan']) == 0
    plan = json.loads(capsys.readouterr().out)
    argv = plan['vllm_argv']
    assert '--no-enable-prefix-caching' in argv
    assert '--enable-prefix-caching' not in argv
    assert argv[argv.index('--max-num-batched-tokens') + 1] == '16384'
    assert '--scheduler-cls' not in argv
    assert '--enforce-eager' not in argv
    selected = plan['runtime_selection']
    assert selected['token_budget_source'] == 'explicit'
    assert selected['max_num_batched_tokens'] == 16384
    assert selected['prefix_caching'] is False
    assert selected['optimizations']['mdl_decode'] is True
    assert selected['optimizations']['c4_member_pipeline'] is True


def test_explicit_native_override_and_rear_cards(capsys):
    assert install_serve.main(['--variant', 'candidate', '--rear-cards', '--native-token-budget', '--plan']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert '--max-num-batched-tokens' not in plan['vllm_argv']
    assert plan['target_devices'] == [4, 5, 6, 7]
    assert plan['port'] == 7878
    assert '--no-enable-prefix-caching' in plan['vllm_argv']

@pytest.mark.parametrize('budget', [0, -1, True, '16384', None])
def test_invalid_explicit_budget_is_rejected(budget):
    config = json.loads((ROOT / 'configs/target.json').read_text())
    config['max_num_batched_tokens'] = budget
    with pytest.raises(ValueError):
        serve_argv(config)
