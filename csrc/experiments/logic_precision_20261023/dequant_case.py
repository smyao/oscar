"""Archive #13-20/#148/#164: a narrow-range K decoder precision witness.

Actual writer bytes and constant-within-source attention have independent
analytic expectations. No old oracle imports, NPU calls or timing experiment.
D.4: four historical rows only, never production full-history restoration.
"""
from pathlib import Path
import argparse
import json
import math
import struct

import torch


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def dump(path, value):
    path.write_bytes(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))


def make(root, mode):
    folder, out = root / f'mode{mode}/fixture', root / f'mode{mode}/output'
    folder.mkdir(parents=True, exist_ok=False); out.mkdir(parents=True, exist_ok=False)
    n, active, total, d, hq, hk = 2, 1, 6, 256, 6, 1
    context, sink, recent, spec, block, blocks, prefix, stride = 4, 0, 1, 3, 128, 1, 32, 128 * 144
    key = torch.ones(total, hk, d, dtype=torch.bfloat16)
    key[:, :, 128:] = 1.0078125
    value = torch.ones_like(key); value[context] = -1
    key[-1] = float('nan'); value[-1] = float('nan')
    q = torch.ones(n, hq, d, dtype=torch.bfloat16); q[:, :, 128:] = -1; q[-1] = float('nan')
    qr = q.clone(); qr[-1] = 0
    empty_raw = torch.full((prefix + stride,), 0x85, dtype=torch.uint8)
    raw_bytes = bytearray(empty_raw.tolist())
    k_scale, k_zero = f32(.0078125 / 3), -384.
    v_scale = f32(f32(1e-8) / 3)
    for position in range(total - 1):
        v = -1. if position == context else 1.
        row = struct.pack('<32H', *([0xff00] * 32)) + bytes(64)
        row += struct.pack('<4f', k_scale, k_zero, v_scale, f32(-v / v_scale))
        raw_bytes[prefix + position * 144:prefix + (position + 1) * 144] = row
    expected_raw = torch.tensor(list(raw_bytes), dtype=torch.uint8)
    empty_window = torch.full((blocks * 4 * hk * d * 2,), 0x85, dtype=torch.uint8).view(torch.bfloat16).reshape(blocks, 4, hk, d)
    wk, wv = empty_window.clone(), empty_window.clone()
    empty_tags = torch.full((blocks, 4), -1, dtype=torch.int64); tags = empty_tags.clone()
    for position in range(total - 1):
        row = position % 4
        wk[0, row], wv[0, row], tags[0, row] = key[position], value[position], position
    slots = torch.tensor([context, -1], dtype=torch.int64)
    store_slots = torch.arange(total, dtype=torch.int64); store_slots[-1] = -1
    files = {'q': q, 'qr': qr, 'ck': key[context:], 'cv': value[context:],
             'raw_striped': empty_raw, 'wk': empty_window, 'wv': empty_window, 'tags': empty_tags,
             'table': torch.zeros(1, 1, dtype=torch.int32), 'starts': torch.tensor([0, active], dtype=torch.int32),
             'lens': torch.tensor([context + active], dtype=torch.int32), 'slots': slots,
             'expected_positions': slots, 'required_status_checks': torch.empty((0, 3), dtype=torch.int64),
             'store_rk': key, 'store_rv': value, 'store_k': key, 'store_v': value,
             'store_slots': store_slots, 'store_positions': torch.arange(total), 'qr_bf16': q,
             'store_expected_raw': expected_raw, 'store_expected_wk': wk,
             'store_expected_wv': wv, 'store_expected_tags': tags}
    for name, tensor in files.items():
        dump(folder / (name + '.bin'), tensor)
    (folder / 'store_tokens.txt').write_text(str(total) + '\n')
    (folder / 'case_flags.txt').write_text('0\n')
    shape = (n, hq, hk, d, context, sink, recent, spec, block, blocks, prefix, stride, 2, 1, 1, 1)
    (folder / 'shape.txt').write_text(' '.join(map(str, shape)) + '\n')
    partial = torch.zeros(n, hq, 3, d)
    partial[0, :, 0] = 1.; partial[0, :, 2] = -1.
    lse = torch.full((n, hq, 3), -math.inf)
    history_score = -.0625 if mode == 0 else -.125
    current_score = -.0625
    lse[0, :, 0] = math.log(context) + history_score
    lse[0, :, 2] = current_score
    history_mass = context * math.exp(history_score - current_score)
    expected_output = (history_mass - 1) / (history_mass + 1)
    merged = torch.zeros(n, hq, d); merged[0] = expected_output
    merged_lse = torch.full((n, hq), -math.inf)
    merged_lse[0] = current_score + math.log(history_mass + 1)
    for name, tensor in {'partial': partial, 'lse': lse, 'output': merged, 'output_lse': merged_lse}.items():
        dump(out / ('official_expected_' + name + '.bin'), tensor)
    dump(out / 'rv_bf16.bin', torch.eye(d, dtype=torch.bfloat16))
    (root / f'mode{mode}/definition.json').write_text(json.dumps({
        'mode': mode, 'context': context, 'sink': sink, 'recent': recent, 'active_queries': active,
        'k_input_halves': [1., 1.0078125], 'q_halves': [1., -1.],
        'history_v': 1., 'current_v': -1., 'k_scale_f32': k_scale, 'k_zero_f32': k_zero,
        'k_codes_halves': [0, 3], 'history_k_decoded_halves': [1., 1.0078125 if mode == 0 else 1.015625],
        'history_logit': history_score, 'current_logit': current_score,
        'analytic_merged_output': expected_output, 'exact_original_attention': .6,
        'probability_rounding': 'none: identical logits within every nonempty source, P=1',
        'scope': 'actual writer/reader/FP32 merge before final inverse; synthetic numerical witness',
        'model_quality': 'not_run', 'npu': 'not_run', 'performance': 'not_run'}, indent=2) + '\n')


def check(root, mode):
    out = root / f'mode{mode}/output'
    definition = json.loads((root / f'mode{mode}/definition.json').read_text())
    errors = {}
    for name, shape in [('partial', (2, 6, 3, 256)), ('lse', (2, 6, 3)),
                        ('output', (2, 6, 256)), ('output_lse', (2, 6))]:
        def read(prefix):
            return torch.frombuffer(bytearray((out / (prefix + name + '.bin')).read_bytes()), dtype=torch.float32).reshape(shape)
        actual, expected = read(''), read('official_expected_')
        finite = torch.isfinite(expected)
        assert torch.equal(torch.isneginf(actual), torch.isneginf(expected)), name
        delta = (actual[finite] - expected[finite]).abs()
        assert bool(torch.isfinite(delta).all()) and float(delta.max()) < 2e-6, name
        errors[name] = float(delta.max())
    actual = torch.frombuffer(bytearray((out / 'output.bin').read_bytes()), dtype=torch.float32).reshape(2, 6, 256)
    result = {'mode': mode, 'status': 'passed', 'actual_merged_output': float(actual[0, 0, 0]),
              'analytic_merged_output': definition['analytic_merged_output'],
              'absolute_error_to_unquantized_attention': abs(float(actual[0, 0, 0]) - .6),
              'errors_to_own_mode_analytic': errors, 'writer_bytes_verified_by_executor': True,
              'probability_rounding': 'none', 'model_quality': 'not_run', 'npu': 'not_run', 'performance': 'not_run'}
    (root / f'mode{mode}/analytic_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('make', 'check'))
    parser.add_argument('root', type=Path); parser.add_argument('mode', type=int, choices=(0, 1))
    args = parser.parse_args(); torch.set_num_threads(1)
    (make if args.action == 'make' else check)(args.root, args.mode)
