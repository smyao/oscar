"""Archive #34/#36/#148/#155: preserve the user-qualified stable launch.

The fixture was captured from f96820f1. The 2026-10-18 user authorization changes
model, public model name, activation dtype, weight quantization and the old
checkpoint-specific YaRN override (official Qwen3.8 uses default RoPE).
The 2026-11-09 user-authorized offload extension adds only the explicit
KV-transfer argument for the active candidate. Native controls exclude it.
The 2026-11-10 user authorization changes the default HTTP port to 7878.
The 2026-11-11 user authorization enables asynchronous KV loads by default.
Every other native argument remains checked against the original fixture.
"""
import json
from pathlib import Path

import pytest
from tools import install_serve, observe_serve

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / 'reports/stable_32k_64k_20261015/frozen_launch.json').read_text())
AUTHORIZED_KV_OFFLOAD = {"enabled": True, "backend": "memcache",
                         "load_async": True, "max_batch_objects": 256, "timeout_seconds": 60}
AUTHORIZED_HTTP_PORT = 7878
AUTHORIZED_KV_TRANSFER = {
    "kv_connector": "OscarAscendStoreConnector",
    "kv_connector_module_path": "oscar_ascend.offload.connector",
    "kv_role": "kv_both", "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": {"enabled": True, "backend": "memcache", "use_layerwise": False,
                                   "load_async": True, "max_batch_objects": 256, "timeout_seconds": 60},
}


def without_authorized_kv_transfer(argv):
    """Compare native math argv after checking the exact approved extension."""
    result = list(argv)
    assert result.count("--kv-transfer-config") == 1
    index = result.index("--kv-transfer-config")
    assert json.loads(result[index + 1]) == AUTHORIZED_KV_TRANSFER
    del result[index:index + 2]
    return result


def authorized_model_arguments(argv, *, offload=True):
    result = list(argv)
    result[result.index('--port') + 1] = str(AUTHORIZED_HTTP_PORT)
    result[1] = '/softwarePlatform/models_directory/Qwen3.8-27B'
    result[result.index('--served-model-name') + 1] = 'qwen3.8'
    start = result.index('--quantization')
    result[start:start + 2] = ['--dtype', 'bfloat16']
    result[result.index("--hf-overrides")+1] = "{}"
    if offload:
        result += ["--kv-transfer-config", json.dumps(AUTHORIZED_KV_TRANSFER, separators=(",", ":"))]
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
    assert int(argv[argv.index('--port') + 1]) == AUTHORIZED_HTTP_PORT
