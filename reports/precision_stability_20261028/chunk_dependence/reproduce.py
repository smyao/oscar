"""Archive #37-49/#126/#145/#170-172: explicit chunk-dependence evidence.

Read-only production/native checkpoint replay plus an independent CPU codec
and attention oracle. No model, NPU, timing or production path is executed.
The deliberately constructed inputs are not LongBench examples or a claim
that this mechanism uniquely explains the user's accuracy or run variance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def reproduce() -> dict:
    import torch
    from tools.experiments.prefix_checkpoint_audit import run_case, source_provenance
    from tools.official_reference import pack_kv, unpack_kv, attention_tiles, merge_and_inverse

    torch.set_num_threads(1)
    schedules = {}
    for label, budget in (("full_remaining_budget", 16384), ("other_requests_leave_768", 768)):
        row = run_case(prompt_tokens=5000, mtp=True, arm="oscar_fixed",
                       remaining_budgets=(budget,))
        assert row["global_token_budget"] == 16384
        assert row["status"] == "completed"
        assert row["symbolic_state_identity_violations"] == []
        assert row["repeat_prompt_hit_tokens"] == 2304
        schedules[label] = row

    position, dim, context, sink, recent = 2000, 256, 1536, 64, 256
    cut = max(sink, context + 1 - recent)
    # Same pre-rotation Q/K/V in both legal schedules. Identity rotation and
    # zero key mean are a valid mathematical special case; they isolate tier
    # choice from centering, rotation error and calibration quality.
    key = torch.zeros((position + 1, 1, dim), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    query = torch.zeros((1, 6, dim), dtype=torch.bfloat16)
    key[64, 0, :13] = 4
    query[:, :, :13] = 4
    value[64, 0, :128] = .2
    value[64, 0, 128:192] = -1
    value[64, 0, 192:] = 1
    rotation = torch.eye(dim, dtype=torch.bfloat16)

    all_exact = merge_and_inverse([attention_tiles(query, key, value)], rotation)
    exact_positions = torch.cat((torch.arange(sink), torch.arange(cut, context)))

    def split_result(history_key, history_value):
        return merge_and_inverse([
            attention_tiles(query, history_key[sink:cut], history_value[sink:cut]),
            attention_tiles(query, key[exact_positions], value[exact_positions]),
            attention_tiles(query, key[context:], value[context:]),
        ], rotation)

    # The independent all-BF16 control keeps the identical three source
    # boundaries/merge order, changing only whether history is quantized.
    split_exact = split_result(key, value)
    packed = pack_kv(key[:context], value[:context], k_clip=.96, v_clip=.92)
    decoded_key, decoded_value = unpack_kv(packed, dim, mode=0)
    split_quantized = split_result(decoded_key, decoded_value)
    assert packed.shape[-1] == 144
    assert torch.equal(all_exact.output, split_exact.output)
    assert torch.isfinite(all_exact.output).all() and torch.isfinite(split_quantized.output).all()
    difference = (all_exact.output.float() - split_quantized.output.float()).abs()
    assert float(difference.max()) == .1328125
    assert int((all_exact.output != split_quantized.output).sum()) == 768

    paths = (
        "csrc/kernels/attention_official_tasks.cpp",
        "csrc/kernels/attention_cv_official_cluster16.cpp",
        "csrc/kernels/attention_cv_official_cluster4.cpp",
        "csrc/kernels/attention_cv_official_decode.cpp",
        "csrc/kernels/store_aligned_int2.cpp",
        "tools/official_reference.py",
        "reports/precision_stability_20261028/chunk_dependence/reproduce.py",
    )
    return {
        "scope": "synthetic CPU numerical counterexample and production/pinned-native host contracts",
        "git_head_at_execution": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_state": "working-tree source hashes; HEAD alone is not the evidence identity",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "checkpoint_provenance": source_provenance(),
        "schedules": schedules,
        "synthetic_inputs": {
            "prompt_tokens": 5000, "query_position": position, "Hq": 6, "Hkv": 1, "D": dim,
            "query_nonzero": "first13 dimensions=4 in each query head",
            "key_nonzero": "token64 first13 dimensions=4; all other entries=0",
            "value_nonzero": "token64 dims0:128=BF16(.2),128:192=-1,192:256=1; all others=0",
            "rotation": "identity", "key_mean": "zero", "k_clip": .96, "v_clip": .92,
            "codec": "144B striped FP32 metadata, mode0", "attention": "frozen independent v4 KV256 oracle",
        },
        "query_tier_difference": {
            "large_chunk": {"context": 0, "query_in_chunk": position, "selected_token64": "exact_current"},
            "small_chunk": {"context": context, "query_in_chunk": position-context,
                            "fixed_cut": cut, "selected_token64": "INT2_history"},
        },
        "results": {
            "raw_selected_value": float(value[64, 0, 0]),
            "dequantized_selected_value": float(decoded_value[64, 0, 0]),
            "all_exact_output_first": float(all_exact.output[0, 0, 0]),
            "split_all_exact_output_first": float(split_exact.output[0, 0, 0]),
            "split_quantized_output_first": float(split_quantized.output[0, 0, 0]),
            "all_exact_source_split_ablation_bitwise_equal": True,
            "all_exact_source_split_ablation_max_abs": 0.,
            "quantized_tier_difference_max_abs": float(difference.max()),
            "quantized_tier_different_elements": int((all_exact.output != split_quantized.output).sum()),
            "all_outputs_finite": True,
        },
        "interpretation": [
            "Legal remaining budgets change the precision tier of the same prior token for the same query.",
            "The unchanged all-BF16 source-split ablation isolates this difference from splitting/merge rounding.",
            "Both schedule replays produce a queryable2304 joint checkpoint for the same synthetic prompt hashes.",
            "The attention output difference can feed later layers' KV/GDN state; this script does not execute a full model or GDN payload.",
            "No physical-page contamination, shared-page overwrite or device race is demonstrated by this example.",
            "Equal budgets/cache seed paths may produce identical outputs; not every cold/hot cache run must differ.",
            "No attribution fraction, LongBench result, one-point quality claim or performance claim follows from this synthetic counterexample.",
        ],
        "NPU": "not_run", "model_quality": "not_run", "performance": "not_run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    report = reproduce()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(args.output), **report["results"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
