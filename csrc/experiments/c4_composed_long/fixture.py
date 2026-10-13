"""Archive #126/#129/#140/#145/#148/#154-156/D.4: full Q336/C1341 fixture.

All1008 task rows retained. Only production's healthy source2 end is emptied;
native current FIA is excluded from this complete-CV scope on BOTH arms.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import probe_history_reuse as reuse
from tools.striped_fixture import raw_to_striped
from csrc.experiments.window_batch_decode.fixture import _bytes
from csrc.experiments.c4_phase_profile.full_partition_budget import prepared_formula, ownership
from csrc.experiments.c4_composed_long.generate import sha, pair_sources

TOKENS, CONTEXT = 336, 1341


def export(path):
    import torch
    from oscar_ascend.ops.reference import encode_kv, decode_kv, attention
    if path.exists(): raise ValueError('fresh fixture output required')
    acceptance = json.loads((ROOT / 'configs/acceptance.json').read_text())
    if acceptance.get('frozen_before_measurement') is not True or acceptance['fused_attention'] != {'atol': .005, 'rtol': .005}:
        raise ValueError('frozen full oracle contract changed')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    logical = {}
    def observe(request, begin, q, k, v, ck, cv): logical[request] = (k.clone(), v.clone())
    spec = reuse.Shape('c4_composed_long', (TOKENS,), (CONTEXT,), 256, 1, 1, True,
                       padded_tokens=TOKENS, recent_tokens=256)
    fixture = reuse.make_fixture(torch, spec, on_request=observe)
    tensors = fixture['cpu']; tensors['raw'] = raw_to_striped(fixture)
    k, v = logical[0]; rk = reuse._hadamard(torch, 256)
    dk, dv = decode_kv(encode_kv(k.float() @ rk, v.float() @ tensors['rv']), 256)
    dk, dv = dk @ rk.T, dv @ tensors['rv'].T
    expected = torch.empty((TOKENS, 6, 256)); lse = torch.empty((TOKENS, 6))
    for query in range(TOKENS):
        cut = min(CONTEXT, max(64, CONTEXT + query + 1 - 256))
        kk, vv = k.float().clone(), v.float().clone()
        kk[64:cut], vv[64:cut] = dk[64:cut], dv[64:cut]
        result = attention(tensors['q'][query:query + 1], kk, vv, scale=.0625, causal=False)
        expected[query], lse[query] = result.output[0], result.lse[0]
    tensors.update(expected_output=expected, expected_lse=lse,
                   expected_positions=torch.arange(CONTEXT, CONTEXT + TOKENS, dtype=torch.int64))
    attrs = [TOKENS, 6, 1, 256, 1, tensors['table'].shape[1], 1008, reuse.BLOCK_TOKENS,
             fixture['blocks'], reuse.PREFIX, fixture['stride'], 323 * 256, 323, 64, 256, 3, 1]
    witness = b''.join(struct.pack('<16q', *row) for row in prepared_formula(TOKENS, CONTEXT))
    path.mkdir(parents=True); hashes = {}
    for name, value in tensors.items():
        data = _bytes(value); (path / (name + '.bin')).write_bytes(data); hashes[name] = sha(data)
    (path / 'expected_tasks.bin').write_bytes(witness); hashes['expected_tasks'] = sha(witness)
    (path / 'shape.txt').write_text(' '.join(map(str, attrs)) + '\n')
    (path / 'bad_position.txt').write_text('-1\n'); (path / 'tolerance.txt').write_text('0.005 0.005\n')
    meta = {'scope': 'complete Q336/C1341/R1/Hq6/Hkv1/D256/S1/20Cube source0+1; all1008 tasks, legal source2 suppression',
        'attrs': attrs, 'context': CONTEXT, 'bad_position': -1, 'input_sha256': hashes,
        'oracle_queries': TOKENS, 'valid_numeric_oracle': True, 'pair_sources': pair_sources(),
        'declared_deadlines': {'CPU_each_arm_s': 120, 'CAModel_each_arm_s': 900},
        'minimum_complete_CV_speedup': 1.15, 'host_work_partition': ownership(TOKENS, CONTEXT),
        'task_witness_status': 'source_formula_only_until_exact_real_CPU_prepare_comparison',
        'NPU': 'not_run', 'CAModel': 'not_run', 'CPU': 'not_run', 'full_model': 'not_run'}
    (path / 'fixture.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); meta = export(args.output)
    print(json.dumps({'output': str(args.output), 'context': CONTEXT, 'oracle_queries': meta['oracle_queries'],
                      'execution': 'host_oracle_only'}))
