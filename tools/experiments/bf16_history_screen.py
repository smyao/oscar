"""Archive #126/#145/#148/#154: stage-1 CPU falsification, never serving code.

D.4: this studies fused history dequant/QK/PV. The old 6.5 s full-history
restore is forbidden in production. Only this independent CPU experiment
materializes oracle tensors; the proposed device algorithm uses KV256 tiles.
No CANN compilation, device completion, graph or performance is inferred here.
Keep the oracle/acceptance unchanged, including failed and empty source rows.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import torch

from oscar_ascend.ops.reference import (
    AttentionResult, attention, decode_kv, dequantize_int2, encode_kv,
    merge_attention, quantize_int2, rotate_clip,
)
from tools.phase import atomic_json
from tools.probe_history_reuse import _hadamard

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("fp32", "kv_only", "q_only", "p_only", "full")
KV_TILE = 256


@dataclass(frozen=True)
class Case:
    name: str
    qlen: int
    context: int
    kv_heads: int = 1
    splits: int = 1
    sink: int = 64
    recent: int = 256
    seed: int = 1931
    rotation: str = "hadamard"
    value_amplitude: float = 1.0
    cancellation: bool = False


# Fixed before measurement, not chosen by candidate error or timing. q84/q168
# exercise multiple 21-query task groups, not C4/C16 scheduling or atomics.
CASES = (
    Case("identity_exact_cancellation", 1, 1023, rotation="identity", cancellation=True),
    Case("empty_history_q4", 4, 0),
    Case("frontier_q1_hkv2", 1, 319, kv_heads=2, splits=3),
    Case("frontier_q2_hkv2", 2, 320, kv_heads=2, splits=3),
    Case("kv_tail_q3", 3, 575, splits=3),
    Case("kv_tail_q4", 4, 577, splits=3),
    Case("hadamard_large_values", 4, 1025, value_amplitude=128.0),
    Case("ordinary_qr", 4, 641, splits=3, rotation="qr"),
    Case("groups_q84", 84, 641, recent=32),
    Case("groups_q168", 168, 641, recent=32, splits=3),
    Case("long_history_16k", 4, 16384, splits=3),
)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compare(actual, expected, tolerance, *, allow_negative_infinity=False):
    """Elementwise frozen gate; matching empty LSE is the only infinity allowed."""
    if actual.shape != expected.shape:
        raise ValueError("comparison shape mismatch")
    a, b = actual.float(), expected.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    empty = (torch.isneginf(a) & torch.isneginf(b)
             if allow_negative_infinity else torch.zeros_like(finite))
    difference = torch.where(finite, (a - b).abs(), torch.zeros_like(a))
    allowed = tolerance["atol"] + tolerance["rtol"] * b.abs()
    failed = (~finite & ~empty) | (finite & (difference > allowed))
    relative_mask = finite & (b != 0)
    relative = difference[relative_mask] / b[relative_mask].abs()
    first = failed.nonzero()[0].tolist() if bool(failed.any()) else None
    def scalar(value):
        return float(value) if bool(torch.isfinite(value)) else str(float(value))
    return {
        "passed": not bool(failed.any()), "elements": a.numel(),
        "failed_elements": int(failed.sum()),
        "nonfinite_mismatches": int((~finite & ~empty).sum()),
        "matching_empty_lse": int(empty.sum()),
        "max_abs": float(difference.max()) if a.numel() else 0.0,
        "max_rel_nonzero_reference": float(relative.max()) if relative.numel() else None,
        "zero_reference_nonzero_error": int((finite & (b == 0) & (difference != 0)).sum()),
        "first_failure": (None if first is None else {
            "index": first, "actual": scalar(a[tuple(first)]),
            "expected": scalar(b[tuple(first)]), "allowed": scalar(allowed[tuple(first)]),
            "abs_error": scalar((a[tuple(first)] - b[tuple(first)]).abs()),
        }),
    }


def make_fixture(case: Case):
    if (case.qlen < 1 or case.context < 0 or case.kv_heads < 1
            or case.splits < 1 or case.sink < 0 or case.recent < 1):
        raise ValueError("invalid CPU fixture geometry")
    generator = torch.Generator().manual_seed(case.seed)
    hq, d = 6 * case.kv_heads, 256
    q = torch.randn((case.qlen, hq, d), generator=generator).bfloat16()
    k = torch.randn((case.context + case.qlen, case.kv_heads, d), generator=generator).bfloat16()
    v = (torch.randn(k.shape, generator=generator) * case.value_amplitude).bfloat16()
    if case.rotation == "identity":
        rk, rv = torch.eye(d), torch.eye(d)
    elif case.rotation == "hadamard":
        rk = _hadamard(torch, d)
        rv = rk.clone()
    elif case.rotation == "qr":
        rk = torch.linalg.qr(torch.randn((d, d), generator=generator)).Q.contiguous()
        rv = torch.linalg.qr(torch.randn((d, d), generator=generator)).Q.contiguous()
    else:
        raise ValueError("unknown rotation")
    if case.cancellation:
        if (case.qlen, case.context, case.rotation) != (1, 1023, "identity"):
            raise ValueError("analytic cancellation fixture must retain its declared geometry")
        q.zero_()
        k[:] = torch.tensor([0, 1, 2, 3], dtype=torch.bfloat16).repeat(64)
        vector = torch.tensor([2048, 2112, 2176, 2304], dtype=torch.bfloat16).repeat(64)
        v[::2] = vector
        v[1::2] = -vector
    if case.context:
        kr, vr = rotate_clip(k[:case.context], rk), rotate_clip(v[:case.context], rv)
        raw = encode_kv(kr, vr)
        dk, dv = decode_kv(raw, d)
        # Test-only codec identity, not a claim about a new NPU store kernel.
        codec_identity = (torch.equal(dk, dequantize_int2(quantize_int2(kr)))
                          and torch.equal(dv, dequantize_int2(quantize_int2(vr))))
    else:
        raw = torch.empty((0, case.kv_heads, 136), dtype=torch.uint8)
        dk = torch.empty((0, case.kv_heads, d))
        dv = torch.empty_like(dk)
        codec_identity = True
    tensors = {"q": q, "k": k, "v": v, "rk": rk, "rv": rv, "raw": raw}
    return {**tensors, "dk": dk, "dv": dv,
            "codec_identity": codec_identity, "input_sha256": fingerprint(tensors)}


def task_ranges(case: Case, begin: int, count: int):
    """Logical ranges from attention_tasks.cpp:135-146, no physical-page emulation."""
    c = case.context
    tail = min(c, max(case.sink, c + begin + 1 - case.recent))
    sources = (
        torch.arange(min(case.sink, c), min(c, max(case.sink, c + begin + count - case.recent))),
        torch.cat((torch.arange(min(case.sink, c)), torch.arange(tail, c))),
        torch.arange(c, c + begin + count),
    )
    for source, positions in enumerate(sources):
        width = (len(positions) + case.splits - 1) // case.splits
        for split in range(case.splits):
            yield source, split, positions[split * width:(split + 1) * width]


def visible(case: Case, query_positions, positions, source):
    causal = positions[None, :] <= query_positions[:, None]
    cut = torch.maximum(torch.tensor(case.sink), query_positions + 1 - case.recent)
    if source == 0:
        return causal & (positions[None, :] >= case.sink) & (positions[None, :] < cut[:, None])
    if source == 1:
        return causal & ((positions[None, :] < case.sink) | (positions[None, :] >= cut[:, None]))
    return causal


def online(query, key, value, mask, stage):
    """CPU FP32 arithmetic on BF16-valued operands, NOT A2 accumulation emulation."""
    if stage not in STAGES:
        raise ValueError("unknown numerical stage")
    nq, hq, dim = query.shape
    mapping = torch.arange(hq) // (hq // key.shape[1])
    q = query.bfloat16().float() if stage in {"q_only", "full"} else query.float()
    maximum = torch.full((nq, hq), -torch.inf)
    total = torch.zeros((nq, hq))
    acc = torch.zeros((nq, hq, dim))
    for start in range(0, len(key), KV_TILE):
        k, v = key[start:start + KV_TILE].float(), value[start:start + KV_TILE].float()
        if stage in {"kv_only", "full"}:
            k, v = k.bfloat16().float(), v.bfloat16().float()
        scores = torch.einsum("qhd,khd->qhk", q, k.index_select(1, mapping)) * dim**-0.5
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("non-finite score before masking")
        scores = scores.masked_fill(~mask[:, None, start:start + KV_TILE], -torch.inf)
        updated = torch.maximum(maximum, scores.amax(-1))
        safe = torch.where(torch.isfinite(updated), updated, torch.zeros_like(updated))
        alpha = torch.exp(maximum - safe)
        p32 = torch.exp(scores - safe.unsqueeze(-1))
        p = p32.bfloat16().float() if stage in {"p_only", "full"} else p32
        acc = acc * alpha.unsqueeze(-1) + torch.einsum("qhk,khd->qhd", p, v.index_select(1, mapping))
        total = total * alpha + p32.sum(-1)  # Always the unrounded FP32 denominator.
        maximum = updated
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    return AttentionResult(acc / safe_total.unsqueeze(-1),
                           torch.where(total > 0, maximum + torch.log(safe_total), maximum))


def relative_l2(actual, expected, limit):
    difference = torch.linalg.vector_norm((actual - expected).double())
    denominator = torch.linalg.vector_norm(expected.double())
    ratio = float(difference / denominator) if denominator > 0 else (0.0 if difference == 0 else float("inf"))
    return {"passed": ratio < limit, "relative_l2": ratio if ratio != float("inf") else "inf",
            "strict_upper_bound": limit}


def evaluate(case: Case, tolerance, window_limit=None):
    if window_limit is None:
        window_limit = json.loads((ROOT / "configs/acceptance.json").read_text())["prefill_window"]["relative_l2_lt"]
    f = make_fixture(case)
    q, k, v, rk, rv = (f[name] for name in ("q", "k", "v", "rk", "rv"))
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
            exact = online(cq, ck, cv, mask, "fp32") if source != 0 else None
            for stage in STAGES:
                result = exact if exact is not None else online(cq, ck, cv, mask, stage)
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
            "unchanged_source1_source2_bits": all(
                torch.equal(p.output.view(torch.int32), base.output.view(torch.int32))
                and torch.equal(p.lse.view(torch.int32), base.lse.view(torch.int32))
                for p, base in zip(parts[case.splits:], candidates["fp32"][case.splits:])),
        }
    def passed(metrics):
        return all(value["passed"] if isinstance(value, dict) else value for value in metrics.values())
    baseline_passed = f["codec_identity"] and passed(partition) and passed(stats["fp32"])
    full_passed = passed(stats["full"])
    return {"case": asdict(case), "input_sha256": f["input_sha256"],
            "input_bytes_unchanged": f["input_sha256"] == fingerprint({name: f[name] for name in ("q", "k", "v", "rk", "rv", "raw")}),
            "codec_oracle_identity": f["codec_identity"], "query_tile": qtile,
            "all_queries_and_heads_checked": True, "partial_index_order": "source*split_count+split,query,head,dimension",
            "task_ranges": ranges, "partition_vs_dense": partition, "stages": stats,
            "baseline_passed": baseline_passed, "candidate_passed": full_passed,
            "status": ("baseline_failed_inconclusive" if not baseline_passed else
                       "cpu_candidate_passed" if full_passed else "cpu_candidate_failed")}


def codec_cast_counterexample(tolerance):
    x = torch.tensor([16, 16.5, 17, 18], dtype=torch.bfloat16).repeat(64)
    packed = quantize_int2(x)
    restored = dequantize_int2(packed)
    return {"scope": "BF16 cast versus FP32 codec reconstruction, not an attention verdict",
            "scale": float(packed.scale), "zero": float(packed.zero),
            "fp32_levels": restored[:4].tolist(), "bf16_levels": restored.bfloat16().float()[:4].tolist(),
            "cast_vs_store_tolerance": compare(restored.bfloat16().float(), restored, tolerance)}


def run(cases=CASES, progress=None):
    acceptance = json.loads((ROOT / "configs/acceptance.json").read_text())
    cases = tuple(cases)
    # A passing subset (in particular empty history) must not unlock design.
    # Compare complete specifications, not user-controlled names alone.
    complete = cases == CASES
    rows = []
    for case in cases:
        row = evaluate(case, acceptance["fused_attention"], acceptance["prefill_window"]["relative_l2_lt"])
        rows.append(row)
        if progress:
            progress({"case": case.name, "status": row["status"],
                      "output_failed": row["stages"]["full"]["output"]["failed_elements"],
                      "partial_failed": row["stages"]["full"]["partial_output"]["failed_elements"]})
    if not rows:
        raise ValueError("at least one predeclared case is required")
    passed = all(row["baseline_passed"] and row["candidate_passed"] and row["input_bytes_unchanged"] for row in rows)
    sources = ("configs/acceptance.json", "oscar_ascend/ops/reference.py",
               "tools/experiments/bf16_history_screen.py", "tools/probe_history_reuse.py",
               "csrc/kernels/attention_tasks.cpp", "csrc/kernels/attention_cv_mdl_decode.cpp",
               "csrc/kernels/attention_cv_member_pipeline_cluster4.cpp",
               "csrc/kernels/rotate_clip_store_striped.cpp")
    return {"schema_version": 1, "status": ("cpu_screen_failed" if not passed else
                "cpu_screen_passed_not_qualified" if complete else "cpu_screen_incomplete"),
            "returncode": 0 if passed and complete else 2, "production_qualified": False,
            "required_cases_complete": complete, "required_cases": [asdict(case) for case in CASES],
            "scope": "CPU FP32 arithmetic with BF16-valued operands; not A2/CANN emulation",
            "environment": {"torch": torch.__version__, "platform": platform.platform(),
                            "threads": torch.get_num_threads()},
            "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            check=True, capture_output=True, text=True).stdout.strip(),
            "source_sha256": {name: _hash((ROOT / name).read_bytes()) for name in sources},
            "frozen_attention_gate": acceptance["fused_attention"],
            "frozen_prefill_window_gate": acceptance["prefill_window"],
            "model_quality": acceptance["model_quality"],
            "codec_cast_counterexample": codec_cast_counterexample(acceptance["store_dequant"]),
            "cases": rows, "next_stage": ("stop_production_integration" if not passed else
                "device_design_allowed" if complete else "incomplete_screen"),
            "not_run": ["CANN_build", "official_CPU_debug", "NPU_completion", "physical_pages_and_tags",
                        "status_lanes", "C4_C16_atomics", "graph_capture", "graph_replay", "MTP_GDN_service",
                        "model_quality", "performance"],
            "failure_policy": "Do not change tolerance, oracle, serving defaults or enable an unqualified BF16 kernel."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = run(progress=lambda row: print(json.dumps(row), flush=True))
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "returncode": result["returncode"],
                      "report": str(args.output), "production_qualified": False}), flush=True)
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
