"""Archive #86/#97/#148: immutable offline refinement policy and identity.

OptR adds no serving operations. The policy identifies calibration and the
existing v4/group-D/FP32-metadata reader; it never changes that reader.
"""
from __future__ import annotations

import copy
import math

RECIPE_VERSION = 5
OBJECTIVE = "optr_post_gate_output_v1"
UPSTREAM_COMMIT = "66d46561b9bdf5d715f591e0f5845a4cbd927dee"
POLICY_VERSION = "optr_native_gate_v1"


def optr_policy(config):
    """Return a portable policy. Research overrides cannot relax acceptance."""
    policy = {
        "policy_version": POLICY_VERSION,
        "capture_lengths": {"train": [4096], "validation": [4096, 32768, 65536],
                            "test": [4096, 32768, 65536]},
        "decode_tokens": 8, "query_samples_per_frame": 8,
        "steps": 16, "lr": 0.002, "eval_every": 4, "query_batch_size": 8,
        "kv_tile_size": 256, "history_splits": 1,
        "validation_partitions": ["single_split", "captured_runtime_frames"],
        "sink_tokens": int(config.get("sink_tokens", 64)),
        "recent_tokens": int(config.get("recent_tokens", 256)),
        "k_clip": float(config.get("k_clip_ratio", .96)),
        "v_clip": float(config.get("v_clip_ratio", .92)),
        "group_size": "head_dim",
        "math_contract": "oscar_bf16_attention_v4_fp32_metadata",
        "selection": "joint_all_heads_each_request_nonworse",
        "test_policy": "independent_accept_or_fail_no_reselection",
        "execution": "cpu_offline_native_npu_traces",
        "source_plan": "disjoint_records_packed_calibration_only_v1",
        "sampling_chunk_tokens": config.get("max_num_batched_tokens", 16384),
        "tensor_parallel_size": config.get("tensor_parallel_size", 4),
        "max_model_len": config.get("max_model_len", 262144),
        "attention_splits": config.get("attention_splits", 1),
        "history_token_capacity": config.get("history_token_capacity"),
        "cube_cores": config.get("cube_cores"),
        "compilation_config": copy.deepcopy(config.get("compilation_config", {})),
    }
    overrides = config.get("optr_calibration", {})
    if not isinstance(overrides, dict):
        raise ValueError("optr_calibration must be an object")
    allowed = {"capture_lengths", "decode_tokens", "query_samples_per_frame",
               "steps", "lr", "eval_every", "query_batch_size"}
    if set(overrides) - allowed:
        raise ValueError("unknown or frozen OptR policy field")
    policy.update(copy.deepcopy(overrides))
    for key, upper in (("decode_tokens", 64), ("query_samples_per_frame", 64),
                       ("steps", 160), ("eval_every", 160), ("query_batch_size", 64)):
        value = policy[key]
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"OptR {key} must be an integer in [1,{upper}]")
    if type(policy["lr"]) not in (float, int) or not math.isfinite(policy["lr"]) or not 0 < policy["lr"] <= .1:
        raise ValueError("OptR learning rate must be finite and in (0,.1]")
    lengths = policy["capture_lengths"]
    if not isinstance(lengths, dict) or set(lengths) != {"train", "validation", "test"}:
        raise ValueError("OptR requires separate train, validation and test lengths")
    for split, values in lengths.items():
        if (not isinstance(values, list) or not values or len(values) > 8 or
                any(type(n) is not int or n < 1 or n > 65536 for n in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"invalid OptR {split} capture lengths")
    if lengths["validation"] != lengths["test"]:
        raise ValueError("OptR independent test must cover the validation length buckets")
    if policy["sink_tokens"] < 0 or policy["recent_tokens"] < 1:
        raise ValueError("invalid OptR exact window")
    for key in ("sampling_chunk_tokens", "tensor_parallel_size", "max_model_len"):
        if type(policy[key]) is not int or policy[key] <= 0:
            raise ValueError(f"OptR {key} must be a positive integer")
    if any(not math.isfinite(policy[k]) or not 0 < policy[k] <= 1 for k in ("k_clip", "v_clip")):
        raise ValueError("OptR clipping must be finite and in (0,1]")
    return policy


def validate_refinement_config(config):
    refinement = config.get("rotation_refinement")
    if refinement not in (None, "optr"):
        raise ValueError("rotation_refinement must be omitted or optr")
    if refinement == "optr":
        if (config.get("use_native_token_budget", False) or
                type(config.get("max_num_batched_tokens")) is not int or
                config["max_num_batched_tokens"] <= 0):
            raise ValueError("OptR requires an explicit serving token budget")
        if (config.get("rotation_method", "qqt_sst") != "qqt_sst" or
                config.get("rotation_variant") != "perhead_center" or
                config.get("rotation_k_clip") is not None):
            raise ValueError("OptR requires recipe3 perhead_center and fixed scalar clips")
        if (not config.get("experimental_official_math") or
                config.get("oscar_dequant_policy") != "fp32_metadata"):
            raise ValueError("OptR requires the existing v4 FP32 metadata serving path")
        if config.get("quantization") is not None:
            raise ValueError("OptR native gate/W_O calibration currently requires BF16 weights")
        optr_policy(config)


def refinement_profile(base_profile, config):
    validate_refinement_config(config)
    return {**base_profile, "recipe_version": RECIPE_VERSION, "objective": OBJECTIVE,
            "accumulator_dtype": "float32",
            "rotation_refinement": "optr", "refinement_upstream_commit": UPSTREAM_COMMIT,
            "refinement_policy": optr_policy(config),
            "base_calibration_profile": copy.deepcopy(base_profile)}
