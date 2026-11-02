"""Archive #86/#170/#172/#174: CPU-only baseline-substitution counterexample.

No user activation/model or NPU/performance evidence. Both actual fitters see
the same deterministic BF16 QKV rows. The deployed CPU replay is unchanged.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from oscar_ascend.rotation_calibration import OfflineLayerAccumulator, fit_checkpoint_pair, checkpoint_pair_artifact
from oscar_ascend.optr_oscar_seed import BaseSeedAccumulator, fit_seed_artifact, seed_profile
from oscar_ascend.rotations import calibration_cache_identity
from oscar_ascend.optr_policy import optr_policy
from oscar_ascend.optr_artifacts import nonworse
from oscar_ascend.optr_traces import aggregate_layer_error
from tools.prepare_optr_rotations import refine_layer
from oscar_ascend.optr_artifacts import independent_test_summary

torch.set_num_threads(1)
NAME = 'model.layers.3.self_attn.attn'
D, H, G, TRAIN = 64, 2, 2, 512
MODEL = hashlib.sha256(b'synthetic-cpu-counterexample-no-model').hexdigest()
CONFIG = dict(rotation_variant='perhead_center', rotation_calibration_tokens=TRAIN,
              mamba_cache_dtype='bfloat16', mamba_ssm_cache_dtype='bfloat16')
PROFILE = calibration_cache_identity(CONFIG, MODEL)


def qkv(seed, n, head):
    rng = torch.Generator().manual_seed(seed)
    mean = (torch.linspace(-5., 5., D) * (1 if head == 0 else -1)).reshape(1, 1, D)
    q = torch.randn(n, G, D, generator=rng).bfloat16()
    k = (.3 * torch.randn(n, 1, D, generator=rng) + mean).bfloat16()
    v = torch.randn(n, 1, D, generator=rng).bfloat16()
    return q, k, v


def fit():
    rank_data = {'center': [], 'shared': []}
    for h in range(H):
        q, k, v = qkv(11 + h, TRAIN, h)
        for label, kind in (('center', OfflineLayerAccumulator), ('shared', BaseSeedAccumulator)):
            collector = kind(head_dim=D, local_kv_heads=1, local_q_heads=G,
                             token_budget=TRAIN, testing=True)
            collector.observe(q, k, v)
            rank_data[label].append(dict(rank=h, tp_size=H, global_q_heads=G*H,
                global_kv_heads=H, prompt_sha256='same-training-data', model_fingerprint=MODEL,
                layers={3: collector.export_payload()}))
    k, v = fit_checkpoint_pair(rank_data['center'], layer_names=[NAME], head_dim=D,
        model_path='synthetic-no-model', model_fingerprint=MODEL, global_q_heads=G*H,
        global_kv_heads=H, tp_size=H, prompt_sha256='same-training-data', max_tokens=TRAIN,
        calibration_profile=PROFILE, testing=True)
    center = checkpoint_pair_artifact(k, v, layer_names=[NAME], head_dim=D,
        model_fingerprint=MODEL, calibration_profile=PROFILE, testing=True)
    shared = fit_seed_artifact(rank_data['shared'], seed_profile(PROFILE), layer_names=[NAME],
        head_dim=D, global_q_heads=G*H, global_kv_heads=H, tp_size=H,
        prompt_sha256='same-training-data', testing=True)
    return center['layers'][NAME], shared['layers'][NAME]


def samples(split, seed):
    result = []
    n = 620
    positions = torch.tensor([511, n-1])
    for h in range(H):
        q, k, v = qkv(seed + h, n, h)
        query, key, value = q[positions], k[:, 0], v[:, 0]
        scores = torch.einsum('mgd,ld->mgl', query.float(), key.float()) / D**.5
        scores.masked_fill_(torch.arange(n)[None, None] > positions[:, None, None], -torch.inf)
        teacher = torch.einsum('mgl,ld->mgd', scores.softmax(-1), value.float()).bfloat16()
        rng = torch.Generator().manual_seed(300+h)
        sample = dict(request_id=split, q=query, k=key, v=value,
            gate=torch.ones_like(query), teacher=teacher,
            positions=positions, contexts=torch.tensor([0, 512]),
            current_ends=torch.tensor([512, n]), frame_ids=torch.tensor([0, 1]),
            history_splits=torch.ones(2, dtype=torch.int64), token_ids=torch.arange(n),
            wo=(.1*torch.randn(G, 32, D, generator=rng)).bfloat16().float(),
            sample_sha256=hashlib.sha256(f'{split}:{h}:{seed}'.encode()).hexdigest())
        result.append([sample])
    return result


def main():
    center, shared = fit()
    policy = optr_policy({})
    policy['replay_partition'] = 'single_split'
    output = {'kind': 'synthetic_cpu_baseline_substitution', 'dim': D, 'kv_heads': H,
              'training_rows_per_head': TRAIN, 'window': [policy['sink_tokens'], policy['recent_tokens']],
              'no_model_npu_or_speed_claim': True, 'splits': {}}
    for split, seed in (('validation', 25), ('test', 36)):
        rows = samples(split, seed)
        a = aggregate_layer_error(center, rows, policy)
        b = aggregate_layer_error(shared, rows, policy)
        gate = nonworse(b, copy.deepcopy(b), require_frames=True, require_terminal=True,
                       require_queries=True, require_projection=True)
        result = {'recipe3': a, 'shared_seed': b, 'v9_unchanged_seed_nonworse': gate}
        result['shared_to_recipe3_sse_ratio'] = b['squared_error'] / a['squared_error']
        output['splits'][split] = result
        assert gate
        assert b['squared_error'] > a['squared_error']
    fitting_samples = {split: samples(split, seed) for split, seed in
                       (('train', 47), ('validation', 25), ('test', 36))}
    selected, record = refine_layer(NAME, shared, fitting_samples, optr_policy({}),
                                   progress=lambda **row: None)
    output['actual_default_80_step_refinement'] = record
    output['actual_default_80_step_test_status'] = independent_test_summary({NAME: record})
    for split, seed in (('validation', 25), ('test', 36)):
        metric = aggregate_layer_error(selected, samples(split, seed), policy)
        output['splits'][split]['actual_refined'] = metric
        output['splits'][split]['refined_to_recipe3_sse_ratio'] = metric['squared_error'] / output['splits'][split]['recipe3']['squared_error']
    # Diagnostic counterexample search only: the matrices above remain fixed.
    # This is not a new fitting policy or a claim of independent evaluation.
    counterexamples = []
    for diagnostic_seed in range(50, 80):
        rows = samples('diagnostic-test', diagnostic_seed)
        metrics = {name: aggregate_layer_error(constants, rows, policy)
                   for name, constants in (('recipe3', center), ('seed', shared), ('selected', selected))}
        passes = nonworse(metrics['seed'], metrics['selected'], require_frames=True,
                         require_terminal=True, require_queries=True, require_projection=True)
        if passes and metrics['selected']['squared_error'] > metrics['recipe3']['squared_error']:
            counterexamples.append({'synthetic_seed': diagnostic_seed,
                'all_v9_nonregression_checks_pass': True,
                'selected_to_recipe3_sse_ratio': metrics['selected']['squared_error'] / metrics['recipe3']['squared_error'],
                'metrics': metrics})
            break
    output['passed_gate_but_worse_than_recipe3'] = counterexamples
    output['center_mean_absmax'] = float(center['k_mean'].abs().max())
    output['shared_mean_nonzeros'] = int(shared['k_mean'].count_nonzero())
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'splits': {split: {k: value[k] for k in
            ('shared_to_recipe3_sse_ratio', 'v9_unchanged_seed_nonworse', 'refined_to_recipe3_sse_ratio')}
            for split, value in output['splits'].items()},
            'actual_fitter_accepted': record['accepted'],
            'actual_fitter_test': output['actual_default_80_step_test_status'],
            'passed_but_worse': [{k:v for k,v in row.items() if k != 'metrics'} for row in counterexamples]}, indent=2))


if __name__ == '__main__':
    main()
