"""Archive #13-22/#126/#148/#154/#155: BF16 high/low residual CPU numerical screen.

D.4: all three CV sources use BF16 Q/K/V/P operands, online softmax and
merge stay FP32, history remains tiled INT2. The readonly independent
original-space OSCAR oracle and frozen tolerances are unchanged. CPU-only
arithmetic diagnostics do not certify CANN, NPU, graphs or model quality.
Historical source0-only screen/reports remain separate and unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import torch
from oscar_ascend.ops.reference import AttentionResult,attention,merge_attention
from tools.phase import atomic_json
from .bf16_history_screen import (Case,CASES,STAGES,make_fixture,fingerprint,task_ranges,
                                  visible,compare,relative_l2,codec_cast_counterexample,KV_TILE)
from .bf16_direct_decode import decode_canonical_kv,validate as validate_decode,metadata_sweep
import numpy as np

STAGES = ("fp32", "residual3", "residual4")
CONTRACT = "BF16 high/low QKV history; BF16 exact QKV precise; high/low P all sources; FP32 softmax, accumulation and merge"


def split_bf16(value):
    value = value.float()
    high = value.bfloat16().float()
    low = (value - high).bfloat16().float()
    return high, low


def validate_residual_split():
    """All codes, each valid FP16 coordinate crossed with fixed anchor strata.

    Two BF16 planes approximate FP32; this does not assert exact recovery for
    every metadata pair. The full Cartesian product is deliberately not claimed.
    """
    scales, zeros = metadata_sweep()
    checked = 0
    unequal = 0
    max_abs = 0.0
    for first in range(0, len(scales), 8192):
        scale = torch.from_numpy(scales[first:first+8192].view(np.float16).copy()).float()
        zero = torch.from_numpy(zeros[first:first+8192].view(np.float16).copy()).float()
        x = scale[:, None] * torch.arange(4).float() + zero[:, None]
        high, low = split_bf16(x)
        error = ((high + low) - x).abs()
        bound = (x - high).abs() / 256 + torch.finfo(torch.float32).tiny
        if not bool(torch.isfinite(high + low).all()) or bool((error > bound).any()):
            raise RuntimeError('BF16 residual split exceeds nearest-rounding error bound')
        checked += x.numel()
        unequal += int((error != 0).sum())
        max_abs = max(max_abs, float(error.max()))
    return {'contract': CONTRACT, 'status': 'cpu_residual_split_passed', 'returncode': 0,
            'levels_checked': checked, 'metadata_pairs': len(scales),
            'codes_checked_per_pair': 4, 'metadata_cartesian_product_exhaustive': False,
            'fp32_reconstruction_not_exact_elements': unequal, 'max_abs_error': max_abs,
            'error_bound': 'abs(FP32-high)/256 + FP32_min_normal',
            'device_completion': 'not_run', 'production_qualified': False}


def residual_product(equation, left, right, terms):
    if terms not in (3, 4):
        raise ValueError("residual product requires three or four terms")
    lh, ll = split_bf16(left)
    rh, rl = split_bf16(right)
    result = torch.einsum(equation, lh, rh)
    result = result + torch.einsum(equation, lh, rl)
    result = result + torch.einsum(equation, ll, rh)
    if terms == 4:
        result = result + torch.einsum(equation, ll, rl)
    return result


def online(query, key, value, mask, stage, *, history=True):
    """Explicit BF16-valued products, each accumulated in FP32 on CPU.

    Exact sources do not introduce QKV rounding or redundant zero-residual
    products: QK is one product; PV is P_high V + P_low V. This is arithmetic
    screening, not a claim about Ascend Cube accumulation or scheduling.
    """
    if stage not in STAGES:
        raise ValueError("unknown numerical stage")
    nq, hq, dim = query.shape
    mapping = torch.arange(hq) // (hq // key.shape[1])
    maximum = torch.full((nq, hq), -torch.inf)
    total = torch.zeros((nq, hq))
    acc = torch.zeros((nq, hq, dim))
    terms = 3 if stage == "residual3" else 4
    for start in range(0, len(key), KV_TILE):
        k = key[start:start + KV_TILE].float().index_select(1, mapping)
        v = value[start:start + KV_TILE].float().index_select(1, mapping)
        if stage == "fp32" or not history:
            scores = torch.einsum("qhd,khd->qhk", query.float(), k)
        else:
            scores = residual_product("qhd,khd->qhk", query, k, terms)
        scores = scores * dim**-0.5
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("non-finite score before masking")
        scores = scores.masked_fill(~mask[:, None, start:start + KV_TILE], -torch.inf)
        updated = torch.maximum(maximum, scores.amax(-1))
        safe = torch.where(torch.isfinite(updated), updated, torch.zeros_like(updated))
        alpha = torch.exp(maximum - safe)
        p32 = torch.exp(scores - safe.unsqueeze(-1))
        if stage == "fp32":
            pv = torch.einsum("qhk,khd->qhd", p32, v)
        elif history:
            pv = residual_product("qhk,khd->qhd", p32, v, terms)
        else:
            ph, pl = split_bf16(p32)
            pv = torch.einsum("qhk,khd->qhd", ph, v) + torch.einsum("qhk,khd->qhd", pl, v)
        acc = acc * alpha.unsqueeze(-1) + pv
        total = total * alpha + p32.sum(-1)
        maximum = updated
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    return AttentionResult(acc / safe_total.unsqueeze(-1),
                           torch.where(total > 0, maximum + torch.log(safe_total), maximum))

ROOT=Path(__file__).resolve().parents[2]
SOURCE_PATHS=("configs/acceptance.json","oscar_ascend/ops/reference.py",
    "tools/experiments/bf16_history_screen.py","tools/experiments/bf16_all_sources_screen.py",
    "tools/experiments/bf16_direct_decode.py","tools/experiments/bf16_residual_screen.py","tools/probe_history_reuse.py",
    "csrc/kernels/attention_tasks.cpp","csrc/kernels/attention_cv_mdl_decode.cpp",
    "csrc/kernels/attention_cv_member_pipeline_cluster4.cpp","csrc/kernels/rotate_clip_store_striped.cpp")

def evaluate(case: Case, tolerance, window_limit=None):
    if window_limit is None:
        window_limit = json.loads((ROOT / "configs/acceptance.json").read_text())["prefill_window"]["relative_l2_lt"]
    f = make_fixture(case)
    q, k, v, rk, rv = (f[name] for name in ("q", "k", "v", "rk", "rv"))
    direct_k,direct_v=decode_canonical_kv(f["raw"])
    direct_identity=(torch.equal(direct_k.view(torch.int16),f["dk"].bfloat16().view(torch.int16))
                     and torch.equal(direct_v.view(torch.int16),f["dv"].bfloat16().view(torch.int16)))
    qr = q.float() @ rk
    original_k, original_v = f["dk"] @ rk.T, f["dv"] @ rv.T
    n, h, d = q.shape
    def empty_parts():
        return [AttentionResult(torch.zeros((n, h, d)), torch.full((n, h), -torch.inf))
                for _ in range(3 * case.splits)]
    gold_parts = empty_parts()
    candidates = {stage: empty_parts() for stage in STAGES}
    ranges = []
    # The frozen task generator uses M128/GQA, including the MDL dispatch.
    qtile = 128 // (h // case.kv_heads)
    for begin in range(0, n, qtile):
        end = min(n, begin + qtile)
        qpos = torch.arange(case.context + begin, case.context + end)
        for source, split, positions in task_ranges(case, begin, end - begin):
            index = source * case.splits + split
            mask = visible(case, qpos, positions, source)
            ranges.append({"query_begin": begin, "query_end": end, "source": source,
                           "split": split, "candidate_count": len(positions),
                           "first_position": int(positions[0]) if len(positions) else None,
                           "last_position": int(positions[-1]) if len(positions) else None,
                           "visible_per_query": mask.sum(-1).tolist()})
            ok, ov = ((original_k[positions], original_v[positions]) if source == 0
                      else (k[positions].float(), v[positions].float()))
            for row in range(begin, end):
                selected = mask[row - begin]
                expected = attention(q[row:row+1], ok[selected], ov[selected], causal=False)
                gold_parts[index].output[row] = expected.output[0]
                gold_parts[index].lse[row] = expected.lse[0]
            cq, ck, cv = ((qr[begin:end], f["dk"][positions], f["dv"][positions])
                          if source == 0 else (q[begin:end].float(), ok, ov))
            for stage in STAGES:
                result = online(cq, ck, cv, mask, stage, history=source == 0)
                candidates[stage][index].output[begin:end] = result.output @ rv.T if source == 0 else result.output
                candidates[stage][index].lse[begin:end] = result.lse
    # A separate dense original-space oracle detects missing/duplicated sources
    # or incorrect split/window masks in the candidate and segmented reference.
    dense_output, dense_lse = [], []
    for row in range(n):
        position = case.context + row
        history = ((torch.arange(case.context) >= case.sink)
                   & (torch.arange(case.context) < max(case.sink, position + 1 - case.recent)))
        dk = torch.where(history[:, None, None], original_k, k[:case.context].float())
        dv = torch.where(history[:, None, None], original_v, v[:case.context].float())
        result = attention(q[row:row+1], torch.cat((dk, k[case.context:position+1].float())),
                           torch.cat((dv, v[case.context:position+1].float())), causal=False)
        dense_output.append(result.output)
        dense_lse.append(result.lse)
    gold = AttentionResult(torch.cat(dense_output), torch.cat(dense_lse))
    segmented = merge_attention(gold_parts)
    partition = {"output": compare(segmented.output, gold.output, tolerance),
                 "lse": compare(segmented.lse, gold.lse, tolerance)}
    stats = {}
    for stage, parts in candidates.items():
        result = merge_attention(parts)
        stats[stage] = {
            "output": compare(result.output, gold.output, tolerance),
            "lse": compare(result.lse, gold.lse, tolerance),
            # Match the existing probe: do not round the independent expected.
            "final_bf16_output": compare(result.output.bfloat16().float(), gold.output, tolerance),
            "partial_output": compare(torch.stack([p.output for p in parts]),
                                      torch.stack([p.output for p in gold_parts]), tolerance),
            "partial_lse": compare(torch.stack([p.lse for p in parts]),
                                   torch.stack([p.lse for p in gold_parts]), tolerance,
                                   allow_negative_infinity=True),
            "prefill_window_relative_l2": relative_l2(result.output, gold.output, window_limit),
            "precise_bf16_input_values_preserved": bool(torch.equal(k,k.float().bfloat16()) and torch.equal(v,v.float().bfloat16())),
            "direct_history_decode_identity": direct_identity,
        }
    def passed(metrics):
        return all(value["passed"] if isinstance(value, dict) else value for value in metrics.values())
    baseline_passed = f["codec_identity"] and passed(partition) and passed(stats["fp32"])
    full_passed = passed(stats["residual4"])
    return {"case": asdict(case), "input_sha256": f["input_sha256"],
            "input_bytes_unchanged": f["input_sha256"] == fingerprint({name: f[name] for name in ("q", "k", "v", "rk", "rv", "raw")}),
            "codec_oracle_identity": f["codec_identity"], "query_tile": qtile,
            "all_queries_and_heads_checked": True, "partial_index_order": "source*split_count+split,query,head,dimension",
            "task_ranges": ranges, "partition_vs_dense": partition, "stages": stats,
            "operand_contract":CONTRACT,"source12_bitwise_required":False,
            "source12_output_change_expected":"BF16 P high/low reconstruction changes PV even for exact BF16 Q/K/V",
            "baseline_passed": baseline_passed, "candidate_passed": full_passed,
            "residual3_passed": passed(stats["residual3"]), "residual4_passed": full_passed,
            "status": ("baseline_failed_inconclusive" if not baseline_passed else
                       "cpu_candidate_passed" if full_passed else "cpu_candidate_failed")}


def run(cases=CASES,progress=None):
    acceptance=json.loads((ROOT/'configs/acceptance.json').read_text());cases=tuple(cases)
    if not cases:raise ValueError('at least one predeclared case is required')
    complete=cases==CASES;rows=[]
    direct=validate_decode()
    residual_split=validate_residual_split()
    for case in cases:
        row=evaluate(case,acceptance['fused_attention'],acceptance['prefill_window']['relative_l2_lt'])
        rows.append(row)
        if progress:progress({'case':case.name,'status':row['status'],
            'output_failed':row['stages']['residual4']['output']['failed_elements'],
            'partial_failed':row['stages']['residual4']['partial_output']['failed_elements']})
    passed=all(row['baseline_passed'] and row['candidate_passed'] and row['input_bytes_unchanged'] for row in rows)
    return {'schema_version':2,'operand_contract':CONTRACT,
        'status':'cpu_screen_failed' if not passed else 'cpu_screen_passed_not_qualified' if complete else 'cpu_screen_incomplete',
        'returncode':0 if passed and complete else 2,'production_qualified':False,
        'required_cases_complete':complete,'required_cases':[asdict(case) for case in CASES],
        'scope':'CPU residual3/residual4 BF16-valued products; not A2/CANN emulation',
        'cost':{'history_qk_products':{'residual3':3,'residual4':4},'history_pv_products':{'residual3':3,'residual4':4},'precise_qk_products':1,'precise_pv_products':2,'history_operand_planes_vs_single_bf16':2,'performance_established':False},
        'environment':{'torch':torch.__version__,'platform':platform.platform(),'threads':torch.get_num_threads()},
        'source_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_PATHS},
        'frozen_attention_gate':acceptance['fused_attention'],'frozen_prefill_window_gate':acceptance['prefill_window'],
        'model_quality':acceptance['model_quality'],'direct_decode':direct,'residual_split':residual_split,
        'codec_cast_counterexample':codec_cast_counterexample(acceptance['store_dequant']),
        'cases':rows,'next_stage':'numerical_gate_failed_no_performance_qualification' if not passed else 'device_design_allowed' if complete else 'incomplete_screen',
        'not_run':['CANN_build','official_CPU_debug','NPU_completion','physical_pages_and_tags','status_lanes',
                   'C4_C16_atomics','graph_capture','graph_replay','MTP_GDN_service','model_quality','performance'],
        'failure_policy':'Do not change tolerance or oracle; numerical failure prohibits performance qualification.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('refusing to replace an earlier numerical report')
    torch.set_num_threads(1)
    report=run(progress=lambda row:print(json.dumps(row),flush=True))
    atomic_json(args.output,report)
    print(json.dumps({'status':report['status'],'returncode':report['returncode'],'report':str(args.output)}),flush=True)
    return report['returncode']


if __name__=='__main__':raise SystemExit(main())
