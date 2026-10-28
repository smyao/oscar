"""Archive #163/#172/#174: protect accepted work while fixing prompt semantics.

Execute pinned native split/dispatch contracts on synthetic CPU metadata.
These traces bound which steps change; they are not the user's benchmark trace
and do not claim NPU graph replay, model accuracy or elapsed-time acceptance.
No model/NPU connection, timing, source mutation, or launch knob is used.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.integration.dummy_context import native_dummy_run
from oscar_ascend.integration.prefill_state_contract import force_prefill_eager
from oscar_ascend.integration.prefix_checkpoint import split_prefix_checkpoint
from test_prefix_checkpoint_retention import _setup
from test_prefill_state_contract_hooks import _native_modules, _register, _runner, _invoke, Mode
from test_universal_bf16_regression_protection import selected_operator
from tools import install_serve, observe_serve

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = '614c73982d762365e20c7362adc6f9350e465218'


def prompt_trace(monkeypatch, prompt_tokens, budgets, *, initial=0):
    """One request gets these residuals of the unchanged global16K budget.

    Production splitter and actual native tail splitter execute unchanged.
    Other request work is represented only by the residual budget, so this is
    not a full scheduler/capacity or benchmark replay.
    """
    _, scheduler, native_split, _, _ = _setup(monkeypatch, 795, 'control')
    request = NS(request_id='trace', num_prompt_tokens=prompt_tokens,
                 num_tokens=prompt_tokens, num_computed_tokens=initial,
                 has_encoder_inputs=False)
    runner = NS(cache_config=scheduler.cache_config,
                speculative_config=NS(method='mtp', num_speculative_tokens=3))
    rows = []
    while request.num_computed_tokens < prompt_tokens:
        index = len(rows)
        assert index < 256
        budget = budgets[index % len(budgets)]
        assert 0 < budget <= 16384
        before = request.num_computed_tokens
        proposed = min(budget, prompt_tokens - before)
        count = split_prefix_checkpoint(scheduler, request, proposed)
        count = native_split(scheduler, request, proposed) if count is None else count
        assert 0 < count <= budget
        runner.input_batch = NS(num_computed_tokens_cpu=torch.tensor([before]),
                                num_prompt_tokens=torch.tensor([prompt_tokens]))
        changes_graph = force_prefill_eager(runner, 1, torch.tensor([count]), count)
        # Expected semantic condition is independent of prompt/context size.
        assert changes_graph is (before > 0 and count == 4)
        rows.append(dict(context=before, scheduled=count, available_budget=budget,
                         corrected_warm_prefill=changes_graph))
        request.num_computed_tokens += count
    assert sum(row['scheduled'] for row in rows) == prompt_tokens - initial
    return rows


@pytest.mark.parametrize('prompt', [4096, 8192, 16384, 32768, 65536])
@pytest.mark.parametrize('budget', [16384, 16380, 16260])
def test_fixed_length_residual_decode_budget_keeps_existing_prefill_work(monkeypatch, prompt, budget):
    # 0,1,31 preceding q4 decode requests in a16K budget; BS32 upper bound.
    rows = prompt_trace(monkeypatch, prompt, [budget])
    assert not any(row['corrected_warm_prefill'] for row in rows)
    assert all(row['scheduled'] <= 16384 for row in rows)


@pytest.mark.parametrize('prompt', [19968, 19969, 19972, 20000, 20740])
def test_variable_twenty_k_prompt_trace_preserves_every_nonaffected_step(monkeypatch, prompt):
    rows = prompt_trace(monkeypatch, prompt, [16384, 16260, 2304, 768, 4])
    assert all(row['corrected_warm_prefill'] == (row['scheduled'] == 4 and row['context'] > 0)
               for row in rows)


@pytest.mark.parametrize('prompt', [19972, 32768, 65536])
def test_real_warm_four_prompt_tokens_are_corrected_at_any_input_length(monkeypatch, prompt):
    # A genuine cached/noncached continuation may get an arbitrary residual
    # after the last state-caching boundary. Never protect the wrong state read
    # just because the complete prompt happens to be32K or64K.
    rows = prompt_trace(monkeypatch, prompt, [4], initial=prompt - 4)
    assert rows == [dict(context=prompt - 4, scheduled=4, available_budget=4,
                        corrected_warm_prefill=True)]


def test_full_production_cann_tree_is_identical_to_accepted_performance_base():
    result = subprocess.run(
        ['git', 'diff', '--exit-code', ACCEPTED, '--', 'csrc/kernels', 'csrc/include',
         'csrc/*.cpp', 'csrc/CMakeLists.txt', 'cmake'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('context', [64, 768, 20000, 32768, 65536, 131072])
def test_real_native_decode_graph_and_cv_operator_remain_at_all_contexts(monkeypatch, context):
    from oscar_ascend import plugin
    from oscar_ascend.integration.runtime_api import clear_runtime
    plugin.unregister()
    clear_runtime()
    try:
        runner_mod, _ = _native_modules(monkeypatch)
        runner, calls = _runner(runner_mod.NPUModelRunner, prefix=True, mode='align',
                                computed=(context, context + 4))
        runner.input_batch.num_prompt_tokens[:] = context
        runner.input_batch.num_prompt_tokens_cpu[:] = context
        native = _invoke(runner)
        _register(monkeypatch)
        observed = _invoke(runner)
        assert native == observed
        assert observed[0] is Mode.FULL and all(c['uniform_decode'] for c in calls)
        assert selected_operator(128, 4) == 'attention_cv_official_decode_out'
        assert selected_operator(128, 1, later_draft=True) == 'attention_cv_official_decode_q1_out'
        with native_dummy_run():
            assert _invoke(runner) == native
        assert _invoke(runner, force_uniform_decode=True) == native
    finally:
        plugin.unregister()
        clear_runtime()


@pytest.mark.parametrize('entry', [install_serve.main, observe_serve.main])
def test_fixed_candidate_direct_and_probe_keep_frozen_launch_and_math(entry, capsys):
    assert entry(['--variant', 'candidate', '--enable-prefix-caching', '--plan']) == 0
    report = json.loads(capsys.readouterr().out)
    argv = report['vllm_argv']
    assert argv[argv.index('--max-num-batched-tokens') + 1] == '16384'
    assert '--enable-prefix-caching' in argv
    spec = json.loads(argv[argv.index('--speculative_config') + 1])
    assert spec['method'] == 'qwen3_5_mtp' and spec['num_speculative_tokens'] == 3
    assert report['runtime_selection']['oscar_math_profile'] == 'oscar_bf16_attention_v4'
    assert report['runtime_selection']['rotation_variant'] == 'perhead_center'
    assert report['runtime_selection']['aligned_c16_operator'] == 'attention_cv_official_cluster16_out'
    assert argv[argv.index('--port') + 1] == '8989'
    assert report['target_devices' if entry is install_serve.main else 'devices'] == [0, 1, 2, 3]
