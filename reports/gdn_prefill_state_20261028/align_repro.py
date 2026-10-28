"""Archive #157-163/#171/#174: native align tail-state contract witness.

No reference or production source is edited by this script. It executes the
native CPU scheduling/address contracts and synthetic tensor state identities.
D.4: no historical dequantization, device/VM run, performance test or new copy.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]

import pytest
from oscar_ascend import plugin
from oscar_ascend.integration.runtime_api import clear_runtime
from test_align_prefill_state_contract import (
    native_chain, tail_schedule, state_case,
    test_actual_prefix_hit_can_also_reach_warm_four_tail,
    test_align_hook_preserves_true_decode_and_mixed_attention_routes,
)

SOURCES = (
    'oscar_ascend/integration/prefill_state_contract.py',
    'oscar_ascend/plugin.py',
    'oscar_ascend/integration/prefix_checkpoint.py',
    'references/vllm/vllm/v1/core/sched/scheduler.py',
    'references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py',
    'references/vllm/vllm/v1/attention/backends/utils.py',
    'references/vllm-ascend/vllm_ascend/patch/worker/patch_mamba_utils.py',
    'references/vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py',
    'references/vllm-ascend/vllm_ascend/ops/gdn.py',
    'references/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h',
    'references/vllm-ascend/csrc/moe/causal_conv1d/op_kernel/causal_conv1d.h',
    'tests/test_align_prefill_state_contract.py',
)


def isolated(fn):
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            return fn(monkeypatch)
        finally:
            # Restore external hooks before the native harness restores its
            # temporary classes; this matches production ownership ordering.
            plugin.unregister()
            clear_runtime()


def main():
    report = {
        'backend': 'pinned_native_host_AST_and_synthetic_CPU_state_addresses',
        'NPU': 'not_run', 'model_quality': 'not_run', 'performance': 'not_run',
        'fixed_global_token_budget': 16384,
        'limits': [
            'Request lengths and remaining batch budgets are synthetic valid inputs, not a user traffic replay.',
            'State payloads .25/-42/4 are finite ownership witnesses, not complete Qwen hidden states.',
            'Existing native preprocess copy is represented by a CPU copy callback; no NPU copy was executed.',
            'The native prefill SSM assignment and native speculative initial-state index are executed/read unchanged.',
            'This proves a state-protocol defect and its host repair, not the share of LongBench score/runtime loss.',
        ],
        'tail_schedules': [], 'state_cases': [], 'mixed_routes': [],
    }
    for prompt in (772, 4610, 5000, 19972, 32768, 65536):
        def schedule(mp):
            _, scheduler, native_split, *_ = native_chain(mp)
            return tail_schedule(scheduler, native_split, prompt)
        trace = isolated(schedule)
        assert trace[-1]['scheduled'] == 4
        report['tail_schedules'].append({'prompt': prompt, 'trace': trace})
        report['state_cases'].append(isolated(lambda mp: state_case(mp, prompt - 4)))
    isolated(test_actual_prefix_hit_can_also_reach_warm_four_tail)
    report['actual_native_prefix_hit'] = {'hit_tokens': 2304, 'remaining_chunks': [1536, 1156, 4],
                                        'all_references_released': True}
    for other, operator in ((4, 'attention_cv_official_decode_out'),
                            (512, 'attention_cv_official_cluster4_out'),
                            (8192, 'attention_cv_official_cluster16_out')):
        isolated(lambda mp: test_align_hook_preserves_true_decode_and_mixed_attention_routes(mp, other, operator))
        report['mixed_routes'].append({'query_lengths': [4, other], 'attention_operator': operator,
            'prefill_mask_and_accepted_preserved': True, 'later_true_decode_restores_graph': other == 4})
    report['source_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}
    report['source_sha256'][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = Path(__file__).with_suffix('.json')
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'passed', 'tail_cases': 6, 'state_cases': 6, 'mixed_routes': 3,
                      'prefix_hit': 2304, 'report': str(destination)}))


if __name__ == '__main__':
    main()
