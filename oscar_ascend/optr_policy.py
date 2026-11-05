"""Archive #86/#97/#148/#170/#173/#175/#176/#177: offline policy and identity.

OptR adds no serving operations. The policy identifies calibration and the
existing v4/group-D/FP32-metadata reader; it never changes that reader.
"""
from __future__ import annotations

import copy
import math

RECIPE_VERSION = 5
OBJECTIVE = "optr_post_gate_output_v1"
UPSTREAM_COMMIT = "66d46561b9bdf5d715f591e0f5845a4cbd927dee"
POLICY_VERSION = "optr_oscar_shared_candidates_ascend_v9"
RECIPE3_POLICY_VERSION = "optr_recipe3_centered_candidates_ascend_v10"
RECIPE3_REPAIR_POLICY_VERSION = "optr_recipe3_trust_region_ascend_v11"
RECIPE3_REPAIR = "single_head_trust_region_v1"
RECIPE3_NATIVE_POLICY_VERSION = "optr_recipe3_native_residual_ascend_v12"
RECIPE3_NATIVE_OPTIMIZER = "joint_native_residual_v1"
RECIPE3_NATIVE_CANDIDATE_SELECTION = "recipe3_native_residual_train_then_validation_v1"
RECIPE3_NATIVE_MAX_CANDIDATES = 8
PROJECTION_VIEWS = ("fp32", "bf16_output", "bf16_rank_output")
DEPLOYMENT_CANDIDATE_SELECTION = "fixed_k_existing_v_checkpoints_first_nonworse"
MAX_DEPLOYMENT_CANDIDATES = 6
RECIPE3_DEPLOYMENT_CANDIDATE_SELECTION = "perhead_existing_kv_checkpoints_best_nonworse"
RECIPE3_MAX_DEPLOYMENT_CANDIDATES = 12
RECIPE3_PUBLICATION_POLICY = "recipe3_validation_and_independent_test_required_v1"
# Historical v9 authorization allowed experimental publication after reporting
# independent-test regressions. The new recipe3 candidate requires both gates;
# retain this identifier only for explicitly selected legacy artifacts.
EXPERIMENTAL_PUBLICATION_POLICY = "validation_gate_independent_test_report_v1"


def optr_policy(config):
    """Return a portable policy. Research overrides cannot relax acceptance."""
    if config.get("optr_baseline") not in (None, "recipe3"):
        raise ValueError("optr_baseline must be omitted or recipe3")
    repair = config.get("optr_recipe3_repair")
    if repair not in (None, RECIPE3_REPAIR) or (repair is not None and config.get("optr_baseline") != "recipe3"):
        raise ValueError("optr_recipe3_repair requires recipe3 and its fixed single-head trust-region strategy")
    optimizer = config.get("optr_recipe3_optimizer")
    if optimizer not in (None, RECIPE3_NATIVE_OPTIMIZER) or (
            optimizer is not None and config.get("optr_baseline") != "recipe3"):
        raise ValueError("optr_recipe3_optimizer requires recipe3 and its fixed native-residual strategy")
    if optimizer is not None and repair is not None:
        raise ValueError("optr_recipe3_optimizer and optr_recipe3_repair are mutually exclusive")
    policy = {
        "policy_version": POLICY_VERSION,
        "capture_lengths": {"train": [4096, 32768, 65536], "validation": [4096, 32768, 65536],
                            "test": [4096, 32768, 65536]},
        "decode_tokens": 8, "query_samples_per_frame": 8,
        "steps": 80, "lr": 0.02, "eval_every": 20, "query_batch_size": 8,
        "fitting_objective": "upstream_fp32_independent_gqa_kl_output_k_then_conditional_v",
        "fitted_grouping": "layer",
        "lambda_output": 1.0,
        "kv_tile_size": 256, "history_splits": 1,
        "validation_partitions": ["single_split", "captured_runtime_frames"],
        "validation_projection_views": list(PROJECTION_VIEWS),
        "sink_tokens": int(config.get("sink_tokens", 64)),
        "recent_tokens": int(config.get("recent_tokens", 256)),
        "k_clip": float(config.get("k_clip_ratio", .96)),
        "v_clip": float(config.get("v_clip_ratio", .92)),
        "group_size": "head_dim",
        "math_contract": "oscar_bf16_attention_v4_fp32_metadata",
        "selection": "shared_layer_each_query_and_projection_nonworse",
        "deployment_candidate_selection": DEPLOYMENT_CANDIDATE_SELECTION,
        "max_deployment_candidates": MAX_DEPLOYMENT_CANDIDATES,
        "test_policy": "independent_accept_or_fail_no_reselection",
        "execution": "npu_fp32_proxy_cpu_orthogonal_and_deployment_validation",
        "source_plan": "disjoint_records_packed_calibration_only_v1",
        "sampling_chunk_tokens": config.get("max_num_batched_tokens", 16384),
        "tensor_parallel_size": config.get("tensor_parallel_size", 4),
        "max_model_len": config.get("max_model_len", 262144),
        "attention_splits": config.get("attention_splits", 1),
        "history_token_capacity": config.get("history_token_capacity"),
        "cube_cores": config.get("cube_cores"),
        "compilation_config": copy.deepcopy(config.get("compilation_config", {})),
    }
    if config.get("optr_baseline") == "recipe3":
        policy.update(
            policy_version=RECIPE3_POLICY_VERSION,
            fitted_grouping="layer_kv_head",
            selection="recipe3_layer_each_query_and_projection_nonworse",
            deployment_candidate_selection=RECIPE3_DEPLOYMENT_CANDIDATE_SELECTION,
            max_deployment_candidates=RECIPE3_MAX_DEPLOYMENT_CANDIDATES,
            source_plan="recipe3_excluded_records_independent_buckets_v1")
    if repair is not None:
        # Archive #176: reuse fitted directions, with independently versioned
        # bounded deployment corrections; never relabel an old completed fit.
        from .optr_trust_region import REPAIR_POLICY
        policy.update(policy_version=RECIPE3_REPAIR_POLICY_VERSION,
                      recipe3_repair=copy.deepcopy(REPAIR_POLICY))
    if optimizer is not None:
        from .optr_native_residual import NATIVE_RESIDUAL_POLICY
        policy.update(policy_version=RECIPE3_NATIVE_POLICY_VERSION,
                      native_residual=copy.deepcopy(NATIVE_RESIDUAL_POLICY),
                      fitting_objective="joint_native_bf16_postgate_query_residual",
                      execution="cpu_native_residual_constrained_selection",
                      deployment_candidate_selection=RECIPE3_NATIVE_CANDIDATE_SELECTION,
                      max_deployment_candidates=RECIPE3_NATIVE_MAX_CANDIDATES,
                      steps=1, eval_every=1)
    overrides = config.get("optr_calibration", {})
    if not isinstance(overrides, dict):
        raise ValueError("optr_calibration must be an object")
    allowed = {"capture_lengths", "decode_tokens", "query_samples_per_frame",
               "steps", "lr", "eval_every", "query_batch_size"}
    if set(overrides) - allowed:
        raise ValueError("unknown or frozen OptR policy field")
    if optimizer is not None and ("lr" in overrides or any(
            key in overrides and (type(overrides[key]) is not int or overrides[key] != 1)
            for key in ("steps", "eval_every"))):
        raise ValueError("native-residual OptR has a frozen direction/radius budget, not optimizer-step overrides")
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


def legacy_capture_policy(config, version="v1"):
    """Reconstruct an exact historical policy, never relabel old coverage.

    Both earlier fitters captured one 4K training request by default. A new
    default must not silently describe those seven requests as today's nine.
    Explicit research overrides remain part of the historical identity.
    """
    if version not in ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"):
        raise ValueError("unknown historical OptR capture policy")
    # Historical identities always describe their original shared seed. A
    # recipe3 caller may inspect them, but cannot relabel them as new captures.
    historical_config = {key: value for key, value in config.items()
                         if key not in ("optr_baseline", "optr_recipe3_repair", "optr_recipe3_optimizer")}
    policy = optr_policy(historical_config)
    overrides = config.get("optr_calibration", {})
    policy.pop("deployment_candidate_selection")
    policy.pop("max_deployment_candidates")
    if version == "v8":
        policy["policy_version"] = "optr_oscar_shared_projection_ascend_v8"
        return policy
    policy.pop("validation_projection_views")
    policy["selection"] = "shared_layer_each_request_frame_and_terminal_nonworse"
    if version == "v7":
        policy["policy_version"] = "optr_oscar_shared_train_seed_ascend_v7"
        return policy
    if version == "v6":
        policy["policy_version"] = "optr_oscar_shared_upstream_ascend_v6"
        return policy
    policy.pop("fitted_grouping")
    policy["selection"] = "joint_all_heads_each_request_frame_and_terminal_nonworse"
    if version == "v5":
        policy["policy_version"] = "optr_oscar_base_upstream_ascend_v5"
        return policy
    if version == "v4":
        policy.update(policy_version="optr_joint_gqa_bf16_ascend_v4",
                      fitting_objective="sequential_native_k_joint_gqa_bf16_gate_conditional_v_v3")
        return policy
    if version == "v3":
        policy.update(policy_version="optr_frame_bf16_ascend_v3",
                      fitting_objective="sequential_frame_bf16_kl_output_k_then_conditional_v_gate_v2",
                      selection="joint_all_heads_each_request_and_frame_nonworse")
        return policy
    policy.update(policy_version="optr_sequential_ascend_v2",
                  fitting_objective="sequential_kl_output_k_then_conditional_v_gate_v1",
                  selection="joint_all_heads_each_request_nonworse")
    policy["capture_lengths"] = copy.deepcopy(overrides.get("capture_lengths", {
        "train": [4096], "validation": [4096, 32768, 65536], "test": [4096, 32768, 65536]}))
    if version == "v1":
        policy.pop("fitting_objective")
        policy.pop("lambda_output")
        policy.update(policy_version="optr_native_gate_v1", execution="cpu_offline_native_npu_traces")
        for key, old in (("steps", 16), ("lr", .002), ("eval_every", 4)):
            policy[key] = overrides.get(key, old)
    return policy


def legacy_capture_profile(profile, config, version="v1"):
    from .optr_oscar_seed import legacy_base_profile, legacy_seed_profile
    previous = copy.deepcopy(profile)
    base = previous.get("base_calibration_profile")
    if base is not None:
        # v5/v6 used the unpartitioned shared seed. Its raw native traces
        # remain reusable, but must never be relabelled as train-only seed.
        if version in ("v5", "v6"):
            previous["base_calibration_profile"] = legacy_seed_profile(base)
        elif version not in ("v7", "v8"):
            previous["base_calibration_profile"] = legacy_base_profile(base)
    return {**previous, "refinement_policy": legacy_capture_policy(config, version)}


def compatible_capture_profiles(profile, config):
    """Historical raw captures reusable only with identical sampling coverage.

    The trace reader still verifies every request, source record, rank, frame
    and tensor hash. This only enumerates authenticated historical profiles;
    it grants no permission to reuse old rotations or fitted checkpoints.
    """
    if config.get("optr_baseline") == "recipe3":
        # Recipe3 calibration records must be excluded before fresh independent
        # length buckets are captured. Old v9 traces lack that evidence.
        if config.get("optr_recipe3_optimizer") == RECIPE3_NATIVE_OPTIMIZER:
            # Only sampling is reused. Reconstruct both historical policies
            # exactly; their default optimizer budget stays 80, not v12's one direction.
            old_config = {key: value for key, value in config.items()
                          if key not in ("optr_recipe3_optimizer", "optr_recipe3_repair")}
            historical = (old_config, {**old_config, "optr_recipe3_repair": RECIPE3_REPAIR})
            return [{**copy.deepcopy(profile), "refinement_policy": optr_policy(previous)}
                    for previous in historical]
        if config.get("optr_recipe3_repair") is None:
            return []
        old_config = {key: value for key, value in config.items() if key != "optr_recipe3_repair"}
        return [{**copy.deepcopy(profile), "refinement_policy": optr_policy(old_config)}]
    current = optr_policy(config)
    sampling = ("capture_lengths", "decode_tokens", "query_samples_per_frame")
    candidates = [legacy_capture_profile(profile, config, version) for version in ("v8", "v7", "v6", "v5", "v4", "v3", "v2", "v1")]
    return [candidate for candidate in candidates if all(
        candidate["refinement_policy"][key] == current[key] for key in sampling)]


def validate_refinement_config(config):
    refinement = config.get("rotation_refinement")
    if refinement not in (None, "optr"):
        raise ValueError("rotation_refinement must be omitted or optr")
    if config.get("optr_baseline") not in (None, "recipe3"):
        raise ValueError("optr_baseline must be omitted or recipe3")
    if config.get("optr_baseline") is not None and refinement != "optr":
        raise ValueError("optr_baseline requires OptR refinement")
    if config.get("optr_recipe3_repair") is not None and refinement != "optr":
        raise ValueError("optr_recipe3_repair requires OptR refinement")
    if config.get("optr_recipe3_optimizer") is not None and refinement != "optr":
        raise ValueError("optr_recipe3_optimizer requires OptR refinement")
    if refinement == "optr":
        if (config.get("use_native_token_budget", False) or
                type(config.get("max_num_batched_tokens")) is not int or
                config["max_num_batched_tokens"] <= 0):
            raise ValueError("OptR requires an explicit serving token budget")
        if (config.get("rotation_method", "qqt_sst") != "qqt_sst" or
                config.get("rotation_variant") != "perhead_center" or
                config.get("rotation_k_clip") is not None):
            raise ValueError("OptR requires the per-head artifact layout and fixed scalar clips")
        if (not config.get("experimental_official_math") or
                config.get("oscar_dequant_policy") != "fp32_metadata"):
            raise ValueError("OptR requires the existing v4 FP32 metadata serving path")
        if config.get("quantization") is not None:
            raise ValueError("OptR native gate/W_O calibration currently requires BF16 weights")
        optr_policy(config)


def refinement_profile(base_profile, config):
    from .optr_oscar_seed import seed_profile
    validate_refinement_config(config)
    return {**base_profile, "recipe_version": RECIPE_VERSION, "objective": OBJECTIVE,
            "accumulator_dtype": "float32",
            "rotation_refinement": "optr", "refinement_upstream_commit": UPSTREAM_COMMIT,
            "refinement_policy": optr_policy(config),
            "base_calibration_profile": (copy.deepcopy(base_profile)
                if config.get("optr_baseline") == "recipe3" else seed_profile(base_profile))}
