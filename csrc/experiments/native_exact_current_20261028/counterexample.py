"""Archive #13-20/#126/#145/#164/#170: exact-current rotation counterexample.

D.4: this tests a proven zero-history source, not history restoration. It
uses the actual recipe3 fitter on disclosed synthetic CPU observations and
independent original-space FP64 dense attention with BF16 interfaces. No
NPU, model inference, timing or user benchmark is run. Production functions
are called only to compare the four writer input tensors, not to emulate an
AscendC writer. A local numerical result cannot certify LongBench accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from oscar_ascend.integration.official_math import rotate_inputs, store_rotated
from oscar_ascend.rotation_calibration import (
    OFFICIAL_COMMIT, OfflineLayerAccumulator, fit_checkpoint_pair,
)

ROOT = Path(__file__).resolve().parents[3]
D, TP, CAL_ROWS = 256, 4, 512
CAL_SEED, QUERY_SEED_BASE = 28102026, 1000
LAYER = "model.layers.3.self_attn.attn"


def tensor_sha(tensor):
    return hashlib.sha256(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())).hexdigest()


def gram_error(matrix):
    return float((matrix.float().T @ matrix.float() - torch.eye(D)).abs().max())


def fitted_rotations():
    """Run production recipe3 fitting; no hand-picked rotation or clip search."""
    payloads = []
    for rank in range(TP):
        rng = torch.Generator().manual_seed(CAL_SEED + rank)
        q = (torch.randn(CAL_ROWS, 6, D, generator=rng) * torch.linspace(.5, 1.5, D)).bfloat16()
        k = (torch.randn(CAL_ROWS, 1, D, generator=rng) + torch.linspace(-1, 1, D)).bfloat16()
        v = (torch.randn(CAL_ROWS, 1, D, generator=rng) * torch.linspace(.7, 1.3, D)).bfloat16()
        accumulator = OfflineLayerAccumulator(head_dim=D, local_kv_heads=1,
            local_q_heads=6, token_budget=CAL_ROWS, testing=True)
        accumulator.observe(q, k, v)
        payloads.append({"rank": rank, "tp_size": TP, "global_q_heads": 24,
            "global_kv_heads": 4, "prompt_sha256": "synthetic-exact-current-audit",
            "model_fingerprint": "synthetic-exact-current-audit",
            "layers": {3: accumulator.export_payload()}})
    profile = {"official_commit": OFFICIAL_COMMIT,
        "model_fingerprint": "synthetic-exact-current-audit", "recipe_version": 3,
        "rotation_variant": "perhead_center", "max_tokens": CAL_ROWS,
        "model_execution": {"dtype": "bfloat16", "mamba_cache_dtype": "bfloat16",
                            "mamba_ssm_cache_dtype": "bfloat16"}}
    key, value = fit_checkpoint_pair(payloads, layer_names=[LAYER], head_dim=D,
        model_path="/synthetic-exact-current-audit", model_fingerprint="synthetic-exact-current-audit",
        global_q_heads=24, global_kv_heads=4, tp_size=TP,
        prompt_sha256="synthetic-exact-current-audit", max_tokens=CAL_ROWS,
        calibration_profile=profile, testing=True)
    return key["layers"][3]["rotation"], value["layers"][3]["rotation"], key["layers"][3]["k_mean"]


def original_space_attention(query, key, value):
    """Independent dense algebra; not the v4 oracle or an NPU FIA claim."""
    count = query.shape[0]
    scores = torch.einsum("thd,shd->hts", query.double(), key.expand(-1, 6, -1).double()) / math.sqrt(D)
    scores.masked_fill_(torch.ones(count, count, dtype=torch.bool).triu(1), -torch.inf)
    return torch.einsum("hts,shd->thd", scores.softmax(-1), value.expand(-1, 6, -1).double()).bfloat16()


class RecordingOps:
    """Record actual producer inputs; no claim to execute device validation."""
    def copy_validate_bf16_query_out(self, source, slots, target, status, cores):
        target.copy_(source)
        status.zero_()

    def store_aligned_int2_out(self, key, value, exact_key, exact_value, *unchanged):
        self.saved = tuple(x.clone() for x in (key, value, exact_key, exact_value))


def writer_inputs_equal(query, key, value, rk, rv, mean):
    count = len(query)
    workspace = NS(query_rot_bf16=torch.empty_like(query), query_rot=torch.empty_like(query),
        query_input=torch.empty_like(query), key_rot_bf16=torch.empty_like(key),
        value_rot_bf16=torch.empty_like(value), centered_key_bf16=torch.empty_like(key),
        rotate_status=torch.empty(count, 6, dtype=torch.int32),
        store_status=torch.empty(count, 1, dtype=torch.int32), geometry=NS(cube_cores=20))
    state = NS(workspace=workspace, rotation_k_bf16=rk, rotation_v_bf16=rv,
        key_mean_bf16=mean, cache_format="striped_rotated_fp32_v3", key_clip_ratio=None,
        raw=None, window_key=None, window_value=None, window_tags=None,
        spec=NS(block_size=2304, conv_bytes=0, ssm_bytes=0), num_blocks=1,
        snapshots=NS(sink_tokens=64, ring_tokens=259))
    ops, slots = RecordingOps(), torch.arange(count)
    _, old_key, old_value = rotate_inputs(ops, state, query, key, value, slots, current_only=True)
    store_rotated(ops, state, key, value, slots, slots, {}, rotated_kv=(old_key, old_value))
    previous = ops.saved
    store_rotated(ops, state, key, value, slots, slots, {}, rotated_kv=None)
    names = ("quantized_key_input", "quantized_value_input", "exact_key_input", "exact_value_input")
    return {name: {"equal": bool(torch.equal(before, after)), "before_sha256": tensor_sha(before),
                   "after_sha256": tensor_sha(after)}
            for name, before, after in zip(names, previous, ops.saved)}


def evaluate_case(count, rotations):
    all_rk, all_rv, all_mean = rotations
    rk, rv, mean = all_rk[0].bfloat16(), all_rv[0].bfloat16(), all_mean[0].bfloat16()
    rng = torch.Generator().manual_seed(QUERY_SEED_BASE + count)
    query = torch.randn(count, 6, D, generator=rng).bfloat16()
    key = torch.randn(count, 1, D, generator=rng).bfloat16()
    value = torch.randn(count, 1, D, generator=rng).bfloat16()
    native = original_space_attention(query, key, value)
    rotated_query = (query @ rk).bfloat16()
    rotated_key = ((key - mean).bfloat16() @ rk).bfloat16()
    rotated_value = (value @ rv).bfloat16()
    old = (original_space_attention(rotated_query, rotated_key, rotated_value) @ rv.T).bfloat16()
    error = old.float() - native.float()
    result = {"tokens": count, "seed": QUERY_SEED_BASE + count,
        "max_abs_old_vs_original_space": float(error.abs().max()),
        "relative_l2_old_vs_original_space": float(error.norm() / native.float().norm()),
        "different_elements": int((old != native).sum()), "total_elements": old.numel(),
        "input_sha256": {name: tensor_sha(tensor) for name, tensor in (("q", query), ("k", key), ("v", value))},
        "writer_inputs": writer_inputs_equal(query, key, value, rk, rv, mean)}
    assert all(row["equal"] for row in result["writer_inputs"].values())
    if count == 1:
        # The singleton mathematical attention weight is exactly1. This
        # counterexample needs neither an INT2 approximation nor softmax error.
        result["original_space_equals_v_exact"] = bool(torch.equal(native, value.expand(-1, 6, -1)))
        assert result["original_space_equals_v_exact"] and result["different_elements"] > 0
    return result


def run():
    torch.set_num_threads(1)
    rotations = fitted_rotations()
    all_rk, all_rv, all_mean = rotations
    geometry = {"global_query_heads": 24, "global_kv_heads": 4, "tp": TP, "head_dim": D}
    report = {"status": "passed", "scope": "actual recipe3 fitter plus synthetic CPU algebra, not user activations",
        "calibration": {"rows_per_rank": CAL_ROWS, "rank_seed": "28102026 + rank", "recipe": 3,
                        "rotation_variant": "perhead_center", "geometry": geometry},
        "rotations": [{"head": head,
            "rk_fp32_gram_max": gram_error(all_rk[head]), "rv_fp32_gram_max": gram_error(all_rv[head]),
            "rk_bf16_gram_max": gram_error(all_rk[head].bfloat16()), "rv_bf16_gram_max": gram_error(all_rv[head].bfloat16()),
            "rk_sha256": tensor_sha(all_rk[head]), "rv_sha256": tensor_sha(all_rv[head]),
            "mean_sha256": tensor_sha(all_mean[head])} for head in range(TP)],
        "cases": [evaluate_case(count, rotations) for count in (1, 33, 257)],
        "operation_ledger": {"per_rank": {"query_heads": 6, "kv_heads": 1, "head_dim": D},
            "removed_matmuls": ["Q @ Rk", "BF16 attention output @ Rv.T"],
            "removed_matmul_flops_per_token": 4 * 6 * D * D,
            "removed_query_head_bf16_copies": 0,
            "output_copy": "Preserve the original-size N*Hq*D BF16 copy into contiguous preallocated storage; native FIA does not guarantee output contiguity.",
            "preserved": ["raw-Q finite validation/copy", "same native FIA shape/mask/scale", "key centering",
                          "K/V rotations", "INT2 writer/clip", "rotated exact snapshot bytes", "fixed output/LSE validation"]},
        "limits": ["No NPU FIA or model quality measurement; FP64 dense attention isolates the coordinate-transform loss.",
                   "Writer equality is for one layer with the same incoming K/V; downstream model K/V may legitimately change.",
                   "New exact-current policy intentionally differs from v4 cold rotation; not a v4 lifecycle defect.",
                   "Deleted operations prove a smaller source workload, not a measured runtime gain."],
        "npu": "not_run", "performance": "not_run", "model_quality": "not_run"}
    report["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
        "oscar_ascend/rotation_calibration.py", "oscar_ascend/integration/official_math.py",
        "csrc/experiments/native_exact_current_20261028/counterexample.py")}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "writer_tensors_exact": 12, "report": str(args.output)}))
