"""Archive #13-20/#126/#145/#148/#164: independent three-tile math case.

D.4: exercise bounded reader/online softmax/PV/merge in official CPU debug;
never a full-history production restore or a timing/model-quality experiment.
Expected attention is an FP64 sum over four value categories. It imports no
production attention implementation, task oracle, codec oracle or old golden.
Exact 0/3 codes and power-of-two scales eliminate quantization uncertainty.
"""
from pathlib import Path
import argparse
import json
import math
import struct

import torch

D, HK, HQ = 256, 1, 6
CONTEXT, SINK, RECENT, SPEC = 704, 64, 16, 3
CUT = CONTEXT + 1 - RECENT
GAMMAS = (0., .125, .25, .5, 1., -.125)
ACTIVE = {'q4': 4, 'c4': 85, 'c16': 337}


def dump(path, value):
    path.write_bytes(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))


def key_and_category(position):
    if position < SINK:
        return 0., 3
    if position < 320:
        return 0., 0
    if position < 576:
        return 1.5, 1
    if position < CUT:
        return 3., 2
    if position < CONTEXT:
        return 3., 3
    index = (position - CONTEXT) % 4
    return (0., 1.5, 3., 6.)[index], index


def closed_form(positions, gamma):
    """Dense softmax math, without online recurrence or implementation tasks."""
    counts = {}
    for position in positions:
        level, category = key_and_category(position)
        score = 4. * gamma * level  # 64 equal dimensions / sqrt(256).
        counts[(score, category)] = counts.get((score, category), 0) + 1
    if not counts:
        return [0.] * D, -math.inf
    maximum = max(score for score, _ in counts)
    masses = [0.] * 4
    for (score, category), count in counts.items():
        masses[category] += count * math.exp(score - maximum)
    denominator = sum(masses)
    return [3. * masses[dim // 64] / denominator for dim in range(D)], maximum + math.log(denominator)


def packed_row(position):
    level, category = key_and_category(position)
    # The striped reader reconstructs eight 32-column planes from each u16.
    # The first 64 dimensions are exactly code3 when the level is nonzero.
    key_word = 0x000f if level else 0
    value_word = 0x000f << (4 * category)
    scale = level / 3. if level else struct.unpack('<f', struct.pack('<f', 1e-8))[0] / 3.
    return struct.pack('<32H', *([key_word] * 32)) + struct.pack('<32H', *([value_word] * 32)) + struct.pack('<4f', scale, -0., 1., -0.)


def make(root, family):
    folder, output = root / family / 'fixture', root / family / 'output'
    folder.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    active = ACTIVE[family]
    n, total = active + 1, CONTEXT + active + 1
    block, prefix, splits = 128, 32, 1
    blocks = (total + block - 1) // block
    stride, ring, window_rows = block * 144, RECENT + SPEC, SINK + RECENT + SPEC
    keys = torch.zeros(total, HK, D, dtype=torch.bfloat16)
    values = torch.zeros_like(keys)
    for position in range(total - 1):
        level, category = key_and_category(position)
        keys[position, :, :64] = level
        values[position, :, category * 64:(category + 1) * 64] = 3.
    keys[-1] = float('nan'); values[-1] = float('nan')
    q = torch.zeros(n, HQ, D, dtype=torch.bfloat16)
    for token in range(active):
        for head in range(HQ):
            q[token, head, :64] = GAMMAS[(token + head) % len(GAMMAS)]
    q[-1] = float('nan')
    qr = q.clone(); qr[-1] = 0
    empty_raw = torch.full((prefix + blocks * stride,), 0x85, dtype=torch.uint8)
    raw_bytes = bytearray(bytes(empty_raw.tolist()))
    for position in range(total - 1):
        offset = prefix + position * 144
        raw_bytes[offset:offset + 144] = packed_row(position)
    raw = torch.tensor(list(raw_bytes), dtype=torch.uint8)
    shape = (blocks, window_rows, HK, D)
    empty_window = torch.full((blocks * window_rows * HK * D * 2,), 0x85, dtype=torch.uint8).view(torch.bfloat16).reshape(shape)
    wk, wv = empty_window.clone(), empty_window.clone()
    empty_tags = torch.full(shape[:2], -1, dtype=torch.int64)
    tags = empty_tags.clone()
    # Decode harness executes its writer before attention (four live current
    # rows fit ring R+3). Cluster harness reads the pre-forward prefix snapshot.
    store_end = total - 1 if family == 'q4' else CONTEXT
    for position in range(store_end):
        page, within = divmod(position, block)
        row = position if position < SINK else SINK + within % ring
        wk[page, row], wv[page, row], tags[page, row] = keys[position], values[position], within
    slots = torch.arange(CONTEXT, CONTEXT + n, dtype=torch.int64); slots[-1] = -1
    files = {'q': q, 'qr': qr, 'ck': keys[CONTEXT:].clone(), 'cv': values[CONTEXT:].clone(),
             'raw_striped': empty_raw if family == 'q4' else raw,
             'wk': empty_window if family == 'q4' else wk,
             'wv': empty_window if family == 'q4' else wv,
             'tags': empty_tags if family == 'q4' else tags,
             'table': torch.arange(blocks, dtype=torch.int32)[None],
             'starts': torch.tensor([0, active], dtype=torch.int32),
             'lens': torch.tensor([CONTEXT + active], dtype=torch.int32),
             'slots': slots, 'expected_positions': slots,
             'required_status_checks': torch.empty((0, 3), dtype=torch.int64)}
    if family == 'q4':
        store_slots = torch.arange(total, dtype=torch.int64); store_slots[-1] = -1
        files.update(store_rk=keys, store_rv=values, store_k=keys, store_v=values,
                     store_slots=store_slots, store_positions=torch.arange(total), qr_bf16=q,
                     store_expected_raw=raw, store_expected_wk=wk,
                     store_expected_wv=wv, store_expected_tags=tags)
        (folder / 'store_tokens.txt').write_text(str(total) + '\n')
    else:
        (folder / 'cluster_flags.txt').write_text('0 1 -1 -1\n')
    for name, value in files.items():
        dump(folder / (name + '.bin'), value)
    (folder / 'case_flags.txt').write_text('0\n')
    dims = (n, HQ, HK, D, CONTEXT, SINK, RECENT, SPEC, block, blocks, prefix, stride, 2, 1, splits, blocks)
    (folder / 'shape.txt').write_text(' '.join(map(str, dims)) + '\n')
    partial = torch.zeros(n, HQ, 3, D, dtype=torch.float32)
    lse = torch.full((n, HQ, 3), -math.inf, dtype=torch.float32)
    merged = torch.zeros(n, HQ, D, dtype=torch.float32)
    merged_lse = torch.full((n, HQ), -math.inf, dtype=torch.float32)
    domains = (list(range(SINK, CUT)), list(range(SINK)) + list(range(CUT, CONTEXT)))
    for token in range(active):
        for head in range(HQ):
            gamma = GAMMAS[(token + head) % len(GAMMAS)]
            for source, positions in enumerate((*domains, range(CONTEXT, CONTEXT + token + 1))):
                result, logsum = closed_form(positions, gamma)
                partial[token, head, source] = torch.tensor(result)
                lse[token, head, source] = logsum
            result, logsum = closed_form(range(CONTEXT + token + 1), gamma)
            merged[token, head], merged_lse[token, head] = torch.tensor(result), logsum
    for name, value in {'partial': partial, 'lse': lse, 'output': merged, 'output_lse': merged_lse}.items():
        dump(output / ('official_expected_' + name + '.bin'), value)
    dump(output / 'rv_bf16.bin', torch.eye(D, dtype=torch.bfloat16))
    (root / family / 'definition.json').write_text(json.dumps({
        'family': family, 'active_queries': active, 'head_dim': D, 'q_heads': HQ,
        'history_candidate_tokens': CONTEXT - SINK, 'visible_history_tokens': CUT - SINK,
        'history_tile_visible_counts': [256, 256, CUT - 576], 'query_gammas': GAMMAS,
        'history_base_logits': [0, 6, 12], 'current_base_logits': [0, 6, 12, 24],
        'golden': 'independent scalar FP64 dense four-category softmax; no production/oracle imports',
        'rounding_limit': 'existing frozen v4 per-element 0.005+0.005*abs(expected); extra tight history checks for nonnegative gamma',
        'model_quality': 'not_run', 'npu': 'not_run', 'performance': 'not_run'}, indent=2) + '\n')


def check(root, family):
    output = root / family / 'output'
    active, n = ACTIVE[family], ACTIVE[family] + 1
    def read(name, shape):
        return torch.frombuffer(bytearray((output / (name + '.bin')).read_bytes()), dtype=torch.float32).reshape(shape)
    actual_p = read('partial', (n, HQ, 3, D))
    expected_p = read('official_expected_partial', actual_p.shape)
    actual_lse = read('lse', (n, HQ, 3))
    expected_lse = read('official_expected_lse', actual_lse.shape)
    errors = {}
    for name, shape in [('partial', (n, HQ, 3, D)), ('lse', (n, HQ, 3)),
                        ('output', (n, HQ, D)), ('output_lse', (n, HQ))]:
        actual, expected = read(name, shape), read('official_expected_' + name, shape)
        finite = torch.isfinite(expected)
        assert torch.equal(torch.isneginf(actual), torch.isneginf(expected)), name
        delta = (actual[finite] - expected[finite]).abs()
        assert bool(torch.isfinite(delta).all()), name
        assert bool((delta <= .005 + .005 * expected[finite].abs()).all()), name
        errors[name] = float(delta.max())
    tight = []
    for token in range(active):
        for head in range(HQ):
            if GAMMAS[(token + head) % len(GAMMAS)] >= 0:
                # Every visible tile is uniform and its maximum never drops:
                # P=1 exactly, so these rows isolate alpha/acc without P RNE.
                tight.append(float((actual_p[token, head, 0] - expected_p[token, head, 0]).abs().max()))
    assert max(tight) < 2e-5, ('history_alpha_acc_error', max(tight))
    assert errors['lse'] < 2e-5 and errors['output_lse'] < 2e-5
    assert torch.count_nonzero(actual_p[-1]) == 0
    assert bool(torch.isneginf(actual_lse[-1]).all())
    result = {'family': family, 'status': 'passed', 'dense_fp64_errors': errors,
              'history_without_probability_rounding_max_abs': max(tight),
              'padding_exact_zero_and_negative_infinity': True,
              'expected_full_history_lse_gamma_zero': math.log(CUT - SINK),
              'future_current_mask_checked': True,
              'npu': 'not_run', 'model_quality': 'not_run', 'performance': 'not_run'}
    (root / family / 'analytic_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('make', 'check'))
    parser.add_argument('root', type=Path)
    parser.add_argument('family', choices=tuple(ACTIVE))
    args = parser.parse_args()
    torch.set_num_threads(1)
    (make if args.action == 'make' else check)(args.root, args.family)
