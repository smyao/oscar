"""Recipe3 OptR validation/independent-test gate; offline preparation only.

Archive #86/#148/#170/#172/#174: preserve the actual per-head recipe3 means
and constants as the comparison baseline. All existing candidate checkpoints
face the same strict BF16 deployment replay; test never selects a candidate.
A failed independent test is persisted and raised, never published as passed.
"""
from __future__ import annotations

import copy
import math

import torch

from oscar_ascend.optr_artifacts import matrix_digest, nonworse
from oscar_ascend.optr_checkpoint import fitting_implementation_digest
from oscar_ascend.optr_policy import PROJECTION_VIEWS, RECIPE3_POLICY_VERSION
from oscar_ascend.optr_recipe3_fit import (
    CANDIDATE_STRATEGY, MAX_CANDIDATES, checkpoint_steps, fit_perhead,
    validate_perhead_pair,
)


def _validate_result(result, baseline, policy):
    """Validate saved proposals before replay, including unchanged K/V sides."""
    if not isinstance(result, dict) or not isinstance(result.get("decision"), dict):
        raise ValueError("invalid completed OptR recipe3 checkpoint")
    heads, _ = validate_perhead_pair(result.get("Rk"), result.get("Rv"), baseline["k_mean"])
    decision = result["decision"]
    if (decision.get("policy") != policy or decision.get("fitted_grouping") != "layer_kv_head" or
            decision.get("deployment_candidate_selection") != CANDIDATE_STRATEGY or
            type(decision.get("accepted")) is not bool):
        raise ValueError("OptR recipe3 checkpoint policy/strategy differs")
    if not decision["accepted"] and any(not torch.equal(result[key], baseline[key]) for key in ("Rk", "Rv")):
        raise ValueError("rejected OptR recipe3 proxy changed its primary matrices")
    candidates = result.get("candidates")
    if (not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES or
            decision.get("candidate_count") != len(candidates)):
        raise ValueError("OptR recipe3 candidate count differs")
    expected_steps = [0, *checkpoint_steps(policy)]
    if decision.get("candidate_steps") != expected_steps:
        raise ValueError("OptR recipe3 candidate schedule differs")
    seen_ids, seen_pairs = set(), set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError("invalid OptR recipe3 candidate")
        key, value = candidate.get("Rk"), candidate.get("Rv")
        validate_perhead_pair(key, value, baseline["k_mean"])
        identity, phase, step = (candidate.get(field) for field in ("id", "phase", "step"))
        if index == 0:
            if (identity != "primary" or phase != "primary" or step is not None or
                    any(not torch.equal(candidate[field], result[field]) for field in ("Rk", "Rv"))):
                raise ValueError("OptR recipe3 primary candidate differs")
        else:
            if (phase not in ("k", "v") or type(step) is not int or step not in expected_steps or
                    identity != f"{phase}_step_{step}"):
                raise ValueError("OptR recipe3 candidate identity/step differs")
            fixed = baseline["Rv"] if phase == "k" else result["Rk"]
            if not torch.equal(value if phase == "k" else key, fixed):
                raise ValueError("OptR recipe3 candidate changed its fixed side")
            if step == 0 and not torch.equal(key if phase == "k" else value,
                                             baseline["Rk" if phase == "k" else "Rv"]):
                raise ValueError("OptR recipe3 initial checkpoint differs from baseline")
            losses = candidate.get("proxy_losses")
            if (not isinstance(losses, list) or len(losses) != heads or
                    any(type(loss) not in (int, float) or not math.isfinite(loss) for loss in losses)):
                raise ValueError("OptR recipe3 checkpoint proxy losses are invalid")
        pair = matrix_digest({"candidate": {**baseline, "Rk": key, "Rv": value}})
        if identity in seen_ids or pair in seen_pairs:
            raise ValueError("OptR recipe3 candidates contain duplicates")
        seen_ids.add(identity)
        seen_pairs.add(pair)
    return result


def refine_layer(name, baseline, samples, policy, *, fitter=None, checkpoints=None, progress=None,
                 fit_device="cpu"):
    """Select the best validation-passing proposal, then require a passing test.

    The signature matches prepare_optr_rotations.refine_layer for dispatch.
    Function-local imports allow that orchestrator to forward here without a
    module import cycle. FitCheckpoints binds the authenticated trace/profile
    identity; the payload additionally binds this baseline, policy and code.
    """
    from tools.prepare_optr_rotations import (
        IndependentTestRegression, _metrics, _progress, _report_independent_test_failure,
    )

    validate_perhead_pair(baseline["Rk"], baseline["Rv"], baseline["k_mean"])
    if (policy.get("policy_version") != RECIPE3_POLICY_VERSION or
            policy.get("fitted_grouping") != "layer_kv_head" or
            policy.get("deployment_candidate_selection") != CANDIDATE_STRATEGY or
            type(policy.get("max_deployment_candidates")) is not int or
            policy["max_deployment_candidates"] != MAX_CANDIDATES):
        raise ValueError("OptR recipe3 requires its versioned per-head candidate policy")
    if (policy.get("validation_projection_views") != list(PROJECTION_VIEWS) or
            policy.get("validation_partitions") != ["single_split", "captured_runtime_frames"]):
        raise ValueError("OptR recipe3 requires both partitions and all frozen projection views")
    if set(samples) != {"train", "validation", "test"}:
        raise ValueError("OptR recipe3 requires independent train/validation/test splits")
    split_ids = {split: {sample["request_id"] for head in rows for sample in head}
                 for split, rows in samples.items()}
    if any(split_ids[left] & split_ids[right]
           for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("OptR recipe3 fitting and independent test request identities overlap")
    emit = _progress if progress is None else progress
    binding = {"strategy": CANDIDATE_STRATEGY, "policy": copy.deepcopy(policy),
               "base_constants_sha256": matrix_digest({name: baseline}),
               "fitting_implementation_sha256": fitting_implementation_digest()}
    owner = f"layer-{baseline['layer_id']}-shared"  # Existing authenticated owner ABI.
    emit(layer=name, phase="fitting", fitted_grouping="layer_kv_head")
    saved = checkpoints.load(owner) if checkpoints else None
    if saved is None:
        options = {"policy": policy}
        if fitter is None:
            options.update(progress=lambda row: emit(layer=name, **row), device=fit_device)
        result = (fit_perhead if fitter is None else fitter)(baseline["Rk"], baseline["Rv"],
            baseline["k_mean"], samples["train"], samples["validation"], **options)
        _validate_result(result, baseline, policy)
        saved = {"recipe3_binding": binding, "result": result}
        if checkpoints:
            checkpoints.save(owner, saved)
        emit(layer=name, phase="perhead_fit_saved")
    else:
        if saved.get("recipe3_binding") != binding:
            raise ValueError("OptR recipe3 resumed checkpoint baseline/policy/implementation differs")
        result = _validate_result(saved.get("result"), baseline, policy)
        emit(layer=name, phase="perhead_fit_resumed")

    def measure(constants, split, candidate):
        emit(layer=name, phase="joint_output_check", split=split, candidate=candidate)
        options = {} if progress is None else {"progress": lambda **row: emit(
            layer=name, phase="joint_output_check", split=split, candidate=candidate, **row)}
        return _metrics(constants, samples[split], policy, **options)

    acceptance = dict(require_frames=True, require_terminal=True, require_queries=True, require_projection=True)
    before = measure(baseline, "validation", "baseline")
    selected, after, selected_id = baseline, before, "seed"
    evaluated = []
    for candidate in result["candidates"]:
        constants = {**baseline, "Rk": candidate["Rk"], "Rv": candidate["Rv"]}
        digest = matrix_digest({name: constants})
        measured = (before if digest == binding["base_constants_sha256"] else
                    measure(constants, "validation", candidate["id"]))
        passes = nonworse(before, measured, **acceptance) and measured["squared_error"] < before["squared_error"]
        evaluated.append({"candidate": candidate["id"], "phase": candidate["phase"], "step": candidate["step"],
                          "constants_sha256": digest, "accepted": passes, "validation": measured})
        if passes and measured["squared_error"] < after["squared_error"]:
            selected, after, selected_id = constants, measured, candidate["id"]
        emit(layer=name, phase="joint_output_check", split="validation",
             event="existing_recipe3_candidate_checked", candidate=candidate["id"], accepted=passes)
    accepted = selected_id != "seed"
    selection = {"method": "recipe3_perhead_existing_kv_checkpoints_v1", "accepted": accepted,
                 "candidate_order": "all_existing_checkpoints_best_validation_squared_error",
                 "test_used_for_selection": False, "max_candidates": MAX_CANDIDATES,
                 "selected_candidate": selected_id, "evaluated_candidates": evaluated,
                 "proposal": evaluated[0]["validation"]}
    emit(layer=name, phase="joint_output_check", split="validation", event="recipe3_layer_selected",
         accepted=accepted, fitted_grouping="layer_kv_head", candidate=selected_id)
    test_before = measure(baseline, "test", "baseline")
    test_after = measure(selected, "test", "selected") if accepted else test_before
    record = {"accepted": accepted, "shared_decision": result["decision"],
              "base_constants_sha256": binding["base_constants_sha256"],
              "selected_constants_sha256": matrix_digest({name: selected}),
              "recipe3_baseline": {"recipe_version": 3, "rotation_variant": "perhead_center",
                                   "constants_sha256": binding["base_constants_sha256"]},
              "validation": {"baseline": before, "selected": after},
              "test": {"baseline": test_before, "selected": test_after},
              "shared_layer_selection": selection}
    if not nonworse(test_before, test_after, **acceptance):
        report = _report_independent_test_failure(name, baseline["layer_id"], record, policy,
            checkpoints, emit, publication_allowed=False)
        raise IndependentTestRegression(f"OptR {name}: independent test regressed; artifact not published", report)
    emit(layer=name, phase="checked", accepted=accepted, independent_test="passed",
         validation_before=before["normalized_error"], validation_after=after["normalized_error"],
         independent_test_before=test_before["normalized_error"], independent_test_after=test_after["normalized_error"])
    return selected, record
