"""Archive #34/#36/#148/#155: preserve the user-qualified stable launch.

The fixture was captured from f96820f1. The 2026-10-18 user authorization changes
model, public model name, activation dtype, weight quantization and the old
checkpoint-specific YaRN override (official Qwen3.8 uses default RoPE).
Every other native argument remains checked against the original fixture.
"""
import json
from pathlib import Path

import pytest
from tools import install_serve, observe_serve

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / 'reports/stable_32k_64k_20261015/frozen_launch.json').read_text())


def authorized_model_arguments(argv):
    result = list(argv)
    result[1] = '/softwarePlatform/c00879303/Qwen3.8-27B'
    result[result.index('--served-model-name') + 1] = 'qwen3.8'
    start = result.index('--quantization')
    result[start:start + 2] = ['--dtype', 'bfloat16']
    result[result.index("--hf-overrides")+1] = "{}"
    return result


@pytest.mark.parametrize('name,entry', [('direct', install_serve.main), ('observe', observe_serve.main)])
@pytest.mark.parametrize('prefix', [False, True])
def test_complete_stable_vllm_launch_is_preserved(name, entry, prefix, capsys):
    options = ['--variant', 'candidate', '--plan']
    if prefix:
        options.append('--enable-prefix-caching')
    assert entry(options) == 0
    actual = json.loads(capsys.readouterr().out)
    expected = FROZEN['cases'][f'{name}_prefix_{str(prefix).lower()}']
    assert actual['vllm_argv'] == authorized_model_arguments(expected['argv'])
    assert actual.get('target_devices', actual.get('devices')) == expected['devices']
    argv = actual['vllm_argv']
    assert int(argv[argv.index('--port') + 1]) == expected['port']
