"""Archive #139/#150/#151/#154/#159: source-derived cost model, not NPU timing.

Separate effective model operations, persistent bytes and temporary GM traffic.
Do not sum nested timings, convert simulator ticks to device ms, or equate GM
traffic with HBM traffic. This standalone stdlib tool never starts a service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GIB = 2**30


def evaluate(batch=32, context=32768, hbm_gbs=819.2, fp32_tflops=70.0):
    # Geometry agrees with current production-shaped fixtures. Model checkpoint
    # itself is remote; nominal 27B is a coarse linear-layer proxy, not a count
    # of active quantized matrices or a measurement of weight traffic.
    hq, hk, dim, full_layers, linear_layers = 6, 1, 256, 16, 48
    exact = min(context, 320)
    history = context - exact
    visits = batch * history * hk
    packed = visits * (2 * dim // 4 + 8)
    window = batch * exact * hk * 2 * dim * 2
    # One FP32 K/V publication and one consumption. Actual Cube rereads,
    # score/P/PV/Q, padding, metadata and exact-window temporaries add traffic.
    temporary = visits * 2 * dim * 4 * 2
    f1 = 4 * batch * history * hq * dim
    passes = full_layers + 3  # Target 16, same one-layer MTP model 3 times.
    weighted_queries = full_layers * 4 + 4 + 1 + 1
    nominal_weights = 27e9 / 4
    source_files = [
        "configs/target.json", "tools/serving_variants.py",
        "oscar_ascend/ops/cv_dispatch.py", "oscar_ascend/integration/impl.py",
        "oscar_ascend/layout.py", "csrc/kernels/attention_cv_mdl_decode.cpp",
        "csrc/kernels/attention_striped_unpack_padded.h",
        "references/vllm/vllm/model_executor/models/qwen3_5_mtp.py",
        "references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py",
    ]
    result = {
        "kind": "analytical_work_model_not_benchmark",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
        "scope": {"per_rank": True, "tp": 4, "actual_decode_batch": batch,
                  "context": context, "query_heads_per_rank": hq, "kv_heads_per_rank": hk,
                  "head_dim": dim, "target_full_layers": full_layers,
                  "target_gdn_layers": linear_layers, "mtp_layer_calls": 3,
                  "mtp_queries": [4, 1, 1], "target_queries": 4,
                  "history_tokens": history, "exact_tokens": exact,
                  "assumptions": "uniform active decode; all three MTP calls; independent request histories; no diagnostic host sync"},
        "hardware_reference_not_measured": {
            "hbm_GB_s": hbm_gbs, "fp32_TFLOP_s": fp32_tflops,
            "source": "https://www.hiascend.com/s/ascendstatic/lst/files/pdf/Flying_Product_Catalogue.pdf",
            "qualification": "A2 catalogue reference, not measured 910B4 kernel throughput; no INT8 peak inferred",
        },
        "single_full_layer_per_rank": {
            "packed_history_MiB": packed / 2**20, "exact_window_MiB": window / 2**20,
            "temporary_KV_logical_GiB_lower_bound": temporary / GIB,
            "q4_useful_history_GFLOP": f1 * 4 / 1e9,
            "q1_useful_history_GFLOP": f1 / 1e9,
            "q4_FLOP_per_packed_byte": f1 * 4 / packed,
            "q1_FLOP_per_packed_byte": f1 / packed,
            "q4_FLOP_per_temporary_byte": f1 * 4 / temporary,
            "q1_FLOP_per_temporary_byte": f1 / temporary,
            "q4_ideal_fp32_ms": f1 * 4 / (fp32_tflops * 1e12) * 1000,
            "q1_ideal_fp32_ms": f1 / (fp32_tflops * 1e12) * 1000,
            "packed_history_ideal_HBM_ms": packed / (hbm_gbs * 1e9) * 1000,
        },
        "target_plus_MTP_attention_per_rank": {
            "history_scans": passes, "query_weight_sum": weighted_queries,
            "packed_history_GiB": packed * passes / GIB,
            "exact_window_GiB": window * passes / GIB,
            "temporary_KV_logical_GiB_lower_bound": temporary * passes / GIB,
            "useful_history_TFLOP": f1 * weighted_queries / 1e12,
            "attention_math_full_context_TFLOP": 4 * batch * context * hq * dim * weighted_queries / 1e12,
        },
        "temporary_HBM_fraction_sensitivity": [],
        "nominal_linear_weight_proxy": {
            "parameter_proxy": 27e9, "weight_bytes_per_rank_if_all_W8": nominal_weights,
            "target_verify_INT8_TOP": 2 * nominal_weights * batch * 4 / 1e12,
            "target_verify_OP_per_weight_byte": 2 * batch * 4,
            "no_MTP_decode_OP_per_weight_byte": 2 * batch,
            "ideal_weight_read_ms": nominal_weights / (hbm_gbs * 1e9) * 1000,
            "warning": "excludes separate MTP weights/heads, vision inactivity, BF16 exceptions, activations and communication; 9.76 GiB allocated weights in supplied log is NOT per-step bytes",
        },
        "prefill": {
            "total_input_tokens": batch * context,
            "nominal_total_linear_TOP_per_rank": 2 * nominal_weights * batch * context / 1e12,
            "causal_FULL_attention_TFLOP_per_rank": 2 * batch * hq * dim * context * (context + 1) * full_layers / 1e12,
            "linear_weight_only_intensity_by_scheduled_tokens": {str(n): 2 * n for n in (128, 768, 1536, 2048)},
            "qualification": "total-work proxy, not one engine step; APC reduces uncached work; current continuation CV and current FIA have different precision/traffic",
        },
        "GDN": {
            "current_state_MiB": batch * linear_layers * 12 * 128 * 128 * 2 / 2**20,
            "one_logical_state_read_write_GiB": batch * linear_layers * 12 * 128 * 128 * 2 * 2 / GIB,
            "qualification": "one current state per request/layer; not full APC allocation, q4 speculative checkpoint traffic or measured HBM; no linear context-length history scan",
        },
        "excluded_from_lower_bounds": ["tile rounding and Cube padding", "score/P/PV/Q/rotation traffic",
            "Vector instructions and scalar issue", "TP/HCCL", "layer/pipe dependencies",
            "GDN and MTP dense/LM-head costs", "APC checkpoint store/restore", "launch and scheduling overhead"],
        "verdict": "Decode is not established as saturated HBM or saturated Cube. Weight reuse favors memory sensitivity; OSCAR adds substantial bounded FP32 publication/consumption and pipeline latency. Prefill dense GEMMs have much higher compute intensity.",
    }
    for fraction in (0, .05, .1, .25, 1):
        nbytes = passes * (packed + window + fraction * temporary)
        result["temporary_HBM_fraction_sensitivity"].append({
            "fraction": fraction, "selected_attention_HBM_proxy_GB": nbytes / 1e9,
            "HBM_service_lower_bound_ms": nbytes / (hbm_gbs * 1e9) * 1000,
            "not_total_step_time": True,
        })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--hbm-gbs", type=float, default=819.2)
    parser.add_argument("--fp32-tflops", type=float, default=70)
    args = parser.parse_args()
    if args.batch <= 0 or args.context <= 320 or not args.hbm_gbs > 0 or not args.fp32_tflops > 0:
        parser.error("positive batch/rates and context > 320 required")
    print(json.dumps(evaluate(args.batch, args.context, args.hbm_gbs, args.fp32_tflops), indent=2))
