"""Archive #170/#175/#176/#177: certify native-residual proposals, never proxies.

This is preparation/load-time evidence validation. No serving math or weaker
validation/test rule is introduced, and finite traces do not certify accuracy.
"""
from __future__ import annotations


POLICY_VERSION = "optr_recipe3_native_residual_ascend_v12"
CANDIDATE_STRATEGY = "recipe3_native_residual_train_then_validation_v1"


def validate_policy(policy):
    from .optr_native_residual import NATIVE_RESIDUAL_POLICY
    if (policy.get("policy_version") != POLICY_VERSION or
            policy.get("native_residual") != NATIVE_RESIDUAL_POLICY or
            policy.get("fitted_grouping") != "layer_kv_head" or
            policy.get("fitting_objective") != "joint_native_bf16_postgate_query_residual" or
            policy.get("execution") != "cpu_native_residual_constrained_selection" or
            policy.get("deployment_candidate_selection") != CANDIDATE_STRATEGY or
            type(policy.get("max_deployment_candidates")) is not int or
            policy["max_deployment_candidates"] != 8 or
            policy.get("validation_partitions") != ["single_split", "captured_runtime_frames"] or
            policy.get("validation_projection_views") != ["fp32", "bf16_output", "bf16_rank_output"] or
            policy.get("selection") != "recipe3_layer_each_query_and_projection_nonworse" or
            policy.get("source_plan") != "recipe3_excluded_records_independent_buckets_v1" or
            policy.get("test_policy") != "independent_accept_or_fail_no_reselection" or
            policy.get("steps") != 1 or policy.get("eval_every") != 1 or
            "recipe3_repair" in policy):
        raise ValueError("recipe3 native-residual policy differs from its fixed objective and gates")


def validate_native_record(name, record, policy, heads):
    from .optr_native_residual import validate_decision
    validate_policy(policy)
    decision = record.get("native_residual_selection")
    if (not isinstance(decision, dict) or
            decision.get("accepted") is not record.get("accepted") or
            decision.get("base_constants_sha256") != record.get("base_constants_sha256") or
            decision.get("selected_constants_sha256") != record.get("selected_constants_sha256") or
            record.get("recipe3_baseline") != {"recipe_version": 3, "rotation_variant": "perhead_center",
                "constants_sha256": record.get("base_constants_sha256")} or
            any(key in record for key in ("shared_layer_selection", "recipe3_repair_selection", "source_fit_provenance"))):
        raise ValueError("recipe3 native-residual record does not bind the actual baseline/selection")
    validate_decision(name, record["validation"]["baseline"], record["validation"]["selected"],
                      decision, heads, policy)
    # All three partitions of the data have independent requests and content.
    metrics = [decision["train_baseline"], record["validation"]["baseline"], record["test"]["baseline"]]
    ids = [{row["request_id"] for row in metric["requests"]} for metric in metrics]
    hashes = [{row["sample_sha256"] for row in metric["requests"]} for metric in metrics]
    if any(ids[i] & ids[j] or hashes[i] & hashes[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("recipe3 native-residual train/validation/test evidence overlaps")

