"""Execute current metadata/planner code on CPU; archive #148/#155/#163/#167.

The tensors are CPU fixtures. Timings describe this Mac's Python planning,
not accelerator execution, graph replay, or the complete model step.
"""
from __future__ import annotations

import argparse
import cProfile
import copy
from enum import Enum
import hashlib
import json
from pathlib import Path
import platform
import pstats
import statistics
import time
from types import SimpleNamespace

import torch

from oscar_ascend.integration.metadata import from_common
from oscar_ascend.integration.mixed_decode_split import plan_mixed_decode
from oscar_ascend.ops.cv_dispatch import select_cv_op
from tools.serving_variants import variant_config

ROOT = Path(__file__).resolve().parents[2]
AscendAttentionState = Enum('AscendAttentionState', 'SpecDecoding ChunkedPrefill')
BATCHES = (1, 8, 20, 31, 32, 33, 46, 64, 127)


def fixture(batch: int, kind: str):
    if kind not in ('pure_target', 'first_draft', 'continued_mixed', 'fresh_mixed'):
        raise ValueError(kind)
    mixed = kind.endswith('mixed')
    if batch <= 0 or batch + int(mixed) > 128:
        raise ValueError('request count exceeds current native limit')
    lengths = [4] * batch
    if mixed:
        lengths.append(16384 - 4 * batch)
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    upper = torch.tensor([32768 + x for x in lengths], dtype=torch.int32)
    if kind == 'fresh_mixed':
        upper[-1] = lengths[-1]
    n = sum(lengths)
    return SimpleNamespace(
        causal=True, query_start_loc=starts, query_start_loc_cpu=starts,
        seq_lens=upper, seq_lens_cpu_upper_bound=upper,
        slot_mapping=torch.arange(n, dtype=torch.int64),
        block_table_tensor=torch.zeros((len(lengths), 512), dtype=torch.int32),
        num_reqs=len(lengths), num_actual_tokens=n, num_input_tokens=n,
        max_query_len=max(lengths), max_seq_len=int(upper.max()),
        attn_state=(AscendAttentionState.ChunkedPrefill if mixed else
                    AscendAttentionState.SpecDecoding))


def run_step(common, kind: str, config, *, layers=16):
    # Native model_runner_v1._build_attn_group_metadata builds once per
    # attention group, then assigns that same object to all group layers.
    current = copy.copy(common)
    current.__dict__.pop('_oscar_current_cumulative', None)
    flags = {k: config['experimental_' + k] for k in
             ('fast_unpack', 'mixed_cv', 'striped_cache', 'decode_bundle',
              'c4_member_pipeline', 'mdl_decode', 'bf16_history')}
    md = from_common(current, capture_origin=kind == 'pure_target',
                     is_draft=kind == 'first_draft', current_only=True,
                     first_draft_current_only=True, first_draft_current_fia=True,
                     mixed_decode_split=True)
    result = None
    for _ in range(layers):
        suffix = md.current_only_plan
        split = None if suffix else plan_mixed_decode(md, common.num_input_tokens)
        # Mirrors the priority in impl.forward; this profiles decisions only,
        # not child dispatch, tensor copies, guards, native FIA or CV.
        prefix_tokens = (suffix.prefix_tokens if suffix else split.prefix_tokens if split else None)
        qmax = 4 if prefix_tokens is not None else common.max_query_len
        op = select_cv_op(16, 6, 1, prefix_tokens or common.num_input_tokens, qmax, **flags)
        result = {'current_suffix': suffix is not None, 'mixed_split': split is not None,
                  'short_prefix_tokens': prefix_tokens, 'short_request_operator': op,
                  'cpu_cumulative_cached': hasattr(current, '_oscar_current_cumulative')}
    return result


def profile(*, loops=100, repeats=5):
    if loops < 1 or repeats < 1:
        raise ValueError('positive loop counts required')
    config = variant_config(json.loads((ROOT/'configs/target.json').read_text()), 'candidate')
    torch.set_num_threads(1)
    rows = []
    for kind in ('pure_target', 'first_draft', 'continued_mixed', 'fresh_mixed'):
        for batch in BATCHES:
            common = fixture(batch, kind)
            layers = 1 if kind == 'first_draft' else 16
            for _ in range(10):
                run_step(common, kind, config, layers=layers)
            elapsed = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                for _ in range(loops):
                    result = run_step(common, kind, config, layers=layers)
                elapsed.append((time.perf_counter_ns()-start)/loops/1e3)
            rows.append({'kind': kind, 'short_requests': batch, 'layers': layers,
                         'actual_tokens': common.num_actual_tokens,
                         'median_host_us': statistics.median(elapsed),
                         'minimum_host_us': min(elapsed), 'maximum_host_us': max(elapsed),
                         'repeat_host_us': elapsed, 'route': result})
    profiler = cProfile.Profile()
    common = fixture(46, 'continued_mixed')
    profiler.enable()
    for _ in range(loops):
        run_step(common, 'continued_mixed', config)
    profiler.disable()
    stats = pstats.Stats(profiler)
    functions = []
    for (path, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        if '/oscar_ascend/' in path:
            functions.append({'path': str(Path(path).relative_to(ROOT)), 'line': line,
                              'function': name, 'calls': calls, 'self_seconds': own,
                              'cumulative_seconds': cumulative})
    sources = ['oscar_ascend/integration/metadata.py', 'oscar_ascend/integration/impl.py',
               'oscar_ascend/integration/current_only_plan.py',
               'oscar_ascend/integration/mixed_decode_split.py',
               'oscar_ascend/ops/cv_dispatch.py', 'tools/serving_variants.py',
               'references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py', __file__]
    return {'evidence_kind': 'executed_CPU_Python_profile_not_NPU_or_service_timing',
            'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                            'torch': torch.__version__, 'torch_threads': torch.get_num_threads()},
            'loops': loops, 'repeats': repeats, 'rows': rows,
            'cprofile': sorted(functions, key=lambda x: x['cumulative_seconds'], reverse=True),
            'sources': [{'path': str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else p,
                         'sha256': hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sources],
            'limits': ['No target-device timing or CPU-to-NPU scale conversion.',
                       'Metadata and dispatch decisions only; no actual attention execution.',
                       'Pure target simulates capture-time metadata/routing, not replay-time host cost. Per-layer Python forward may be bypassed during replay.',
                       'One from_common per FULL group; first draft one layer, target fixtures 16 routing decisions. Guard binding/child dispatch are excluded.',
                       'All shapes are conditional local 6-query/1-KV/D256 fixture.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/batch_host_profile_20261019.json')
    parser.add_argument('--loops', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    report = profile(loops=args.loops, repeats=args.repeats)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Executed {len(report["rows"])} host profile cases; report={args.output}')


if __name__ == '__main__':
    main()
