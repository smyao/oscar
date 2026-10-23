"""Archive #13-20/#126/#145/#148/#164: actual-window masked-tail boundaries.

Two local CPU cases only. Reuse the independently derived four-category dense
FP64 formula, not the production task or attention oracle. D.4: bounded CV
correctness, with no NPU/model-quality/performance claim or kernel changes.
"""
from pathlib import Path
import argparse
import json
import math

import torch

import analytic_case as base


def configure():
    base.CONTEXT = 944
    base.RECENT = 256
    base.CUT = 689


def make(root, family, history_only=False):
    configure()
    base.make(root, family)
    folder, output = root / family / 'fixture', root / family / 'output'
    definition = json.loads((root / family / 'definition.json').read_text())
    definition.update(context=944, recent=256, cut=689, source_splits=1,
                      history_candidate_tile_counts=[256, 256, 256, 112],
                      history_tile_visible_counts=[256, 256, 113, 0])
    if family == 'q4':
        splits, active, n = 4, 4, 5
        shape = (folder / 'shape.txt').read_text().split()
        shape[14] = str(splits)
        (folder / 'shape.txt').write_text(' '.join(shape) + '\n')
        partial = torch.zeros(n, base.HQ, 3 * splits, base.D)
        lse = torch.full((n, base.HQ, 3 * splits), -math.inf)
        history = list(range(base.SINK, base.CONTEXT))
        window = list(range(base.SINK)) + list(range(base.CUT, base.CONTEXT))
        current = list(range(base.CONTEXT, base.CONTEXT + active))
        for source, candidates in enumerate((history, window, current)):
            width = (len(candidates) + splits - 1) // splits
            for part in range(splits):
                candidate_part = candidates[part * width:(part + 1) * width]
                for token in range(active):
                    visible = [position for position in candidate_part
                               if (source != 0 or position < base.CUT)
                               and position <= base.CONTEXT + token]
                    for head in range(base.HQ):
                        gamma = base.GAMMAS[(token + head) % len(base.GAMMAS)]
                        result, logsum = base.closed_form(visible, gamma)
                        partial[token, head, source * splits + part] = torch.tensor(result)
                        lse[token, head, source * splits + part] = logsum
        base.dump(output / 'official_expected_partial.bin', partial)
        base.dump(output / 'official_expected_lse.bin', lse)
        definition.update(source_splits=4, fully_masked_history_split=3,
                          fully_masked_history_interval=[724, 944],
                          history_split_visible_counts=[220, 220, 185, 0])
    if history_only:
        assert family == 'c16'
        (folder / 'cluster_flags.txt').write_text('1 1 -1 -1\n')
        n = base.ACTIVE[family] + 1
        partial = torch.frombuffer(bytearray((output / 'official_expected_partial.bin').read_bytes()), dtype=torch.float32).reshape(n, base.HQ, 3, base.D)
        lse = torch.frombuffer(bytearray((output / 'official_expected_lse.bin').read_bytes()), dtype=torch.float32).reshape(n, base.HQ, 3)
        partial[:, :, 1:] = 0
        lse[:, :, 1:] = -math.inf
        for name, value in {'partial': partial, 'lse': lse,
                            'output': partial[:, :, 0], 'output_lse': lse[:, :, 0]}.items():
            base.dump(output / ('official_expected_' + name + '.bin'), value)
        definition.update(isolated_history_only=True,
                          source1_source2='empty KV ranges; real kernel retains leaders and publishes zero/-inf',
                          full_R256_case='separate r3 timed out at unchanged 180s; not validated by this isolated case')
    (root / family / 'definition.json').write_text(json.dumps(definition, indent=2) + '\n')


def check(root, family, history_only=False):
    configure()
    if family == 'c16':
        base.check(root, family)
        path = root / family / 'analytic_check.json'
        result = json.loads(path.read_text())
        result.update(context=944, recent=256, source_splits=1,
                      trailing_fully_masked_history_tile_checked=True)
        if history_only:
            output = root / family / 'output'
            n = base.ACTIVE[family] + 1
            partial = torch.frombuffer(bytearray((output / 'partial.bin').read_bytes()), dtype=torch.float32).reshape(n, base.HQ, 3, base.D)
            lse = torch.frombuffer(bytearray((output / 'lse.bin').read_bytes()), dtype=torch.float32).reshape(n, base.HQ, 3)
            assert torch.count_nonzero(partial[:, :, 1:].contiguous().view(torch.int32)) == 0
            assert bool(torch.isneginf(lse[:, :, 1:]).all())
            result.update(isolated_history_only=True, future_current_mask_checked=False,
                          source1_source2_empty_ranges_exact_zero_and_negative_infinity=True,
                          full_R256_case='not_completed; independent r3 180s timeout retained')
    else:
        output = root / family / 'output'
        def read(name, shape):
            return torch.frombuffer(bytearray((output / (name + '.bin')).read_bytes()), dtype=torch.float32).reshape(shape)
        errors = {}
        for name, shape in [('partial', (5, 6, 12, 256)), ('lse', (5, 6, 12)),
                            ('output', (5, 6, 256)), ('output_lse', (5, 6))]:
            actual, expected = read(name, shape), read('official_expected_' + name, shape)
            finite = torch.isfinite(expected)
            assert torch.equal(torch.isneginf(actual), torch.isneginf(expected)), name
            delta = (actual[finite] - expected[finite]).abs()
            assert bool(torch.isfinite(delta).all()), name
            assert bool((delta <= .005 + .005 * expected[finite].abs()).all()), name
            errors[name] = float(delta.max())
        partial = read('partial', (5, 6, 12, 256))
        lse = read('lse', (5, 6, 12))
        assert torch.count_nonzero(partial[:4, :, 3].contiguous().view(torch.int32)) == 0
        assert bool(torch.isneginf(lse[:4, :, 3]).all())
        assert torch.count_nonzero(partial[-1].contiguous().view(torch.int32)) == 0
        assert bool(torch.isneginf(lse[-1]).all())
        assert errors['lse'] < 2e-5 and errors['output_lse'] < 2e-5
        result = {'family': family, 'status': 'passed', 'context': 944, 'recent': 256,
                  'source_splits': 4, 'fully_masked_history_split': 3,
                  'fully_masked_history_interval': [724, 944],
                  'empty_split_exact_positive_zero_and_negative_infinity': True,
                  'real_merge_with_empty_history_split_checked': True,
                  'padding_exact_positive_zero_and_negative_infinity': True,
                  'dense_fp64_errors': errors, 'npu': 'not_run',
                  'model_quality': 'not_run', 'performance': 'not_run'}
        path = root / family / 'analytic_check.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('make', 'check'))
    parser.add_argument('root', type=Path)
    parser.add_argument('family', choices=('c16', 'q4'))
    parser.add_argument('--history-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    (make if args.action == 'make' else check)(args.root, args.family, args.history_only)
