"""Archive #170/#175/#176/#177: native-residual selection locked before test.

Use the authentic recipe3 matrices/means and existing native captures. Only
the new constrained train/validation selection may create new coefficients;
neither old proxy results nor test data can choose them.
"""
from __future__ import annotations

import copy

from oscar_ascend.optr_artifacts import matrix_digest, nonworse
from oscar_ascend.optr_checkpoint import fitting_implementation_digest
from oscar_ascend.optr_native_residual_certificate import validate_policy


def refine_layer(name, baseline, samples, policy, *, fitter=None, checkpoints=None, progress=None,
                 fit_device="cpu", source_checkpoint=None):
    from oscar_ascend.optr_native_residual import select_native_residual, validate_selection
    from oscar_ascend.optr_recipe3_fit import validate_perhead_pair
    from tools.prepare_optr_rotations import (
        IndependentTestRegression, _metrics, _progress, _report_independent_test_failure,
    )
    validate_policy(policy)
    validate_perhead_pair(baseline["Rk"], baseline["Rv"], baseline["k_mean"])
    if fitter is not None or source_checkpoint is not None or fit_device != "cpu":
        raise ValueError("native-residual preparation cannot substitute proxy fitting or an accelerator replay")
    if set(samples) != {"train", "validation", "test"}:
        raise ValueError("native-residual preparation requires independent train/validation/test")
    ids = []
    for split in ("train", "validation", "test"):
        heads = samples[split]
        if not isinstance(heads, list) or not heads or any(not head for head in heads):
            raise ValueError("native-residual split has no samples")
        if any(sample.get("split", split) != split for head in heads for sample in head):
            raise ValueError("native-residual sample is in the wrong split")
        ids.append({sample["request_id"] for head in heads for sample in head})
    if any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("native-residual train/validation/test requests overlap")
    emit = _progress if progress is None else progress
    binding = {"strategy": policy["deployment_candidate_selection"], "policy": copy.deepcopy(policy),
               "base_constants_sha256": matrix_digest({name: baseline}),
               "fitting_implementation_sha256": fitting_implementation_digest()}
    owner = f"layer-{baseline['layer_id']}-shared"
    saved = checkpoints.load(owner) if checkpoints else None
    if saved is None:
        selected, before, after, decision, proposal = select_native_residual(
            name, baseline, samples["train"], samples["validation"], policy, progress=emit)
        validate_selection(name, baseline, selected, before, after, decision, proposal, policy)
        saved = {"native_residual_binding": binding, "selected": selected, "before": before,
                 "after": after, "decision": decision, "proposal": proposal}
        if checkpoints:
            checkpoints.save(owner, saved)
        emit(layer=name, phase="native_residual_locked_before_test")
    else:
        if saved.get("native_residual_binding") != binding:
            raise ValueError("native-residual checkpoint baseline/policy/implementation differs")
        selected, before, after, decision, proposal = (saved.get(key) for key in
            ("selected", "before", "after", "decision", "proposal"))
        validate_selection(name, baseline, selected, before, after, decision, proposal, policy)
        emit(layer=name, phase="native_residual_lock_resumed", test_reselection=False)
    accepted = decision["accepted"]
    def measure(constants, candidate):
        emit(layer=name, phase="joint_output_check", split="test", candidate=candidate)
        return _metrics(constants, samples["test"], policy, progress=lambda **row: emit(
            layer=name, phase="joint_output_check", split="test", candidate=candidate, **row))
    test_before = measure(baseline, "baseline")
    test_after = measure(selected, "selected") if accepted else test_before
    record = {"accepted": accepted, "base_constants_sha256": binding["base_constants_sha256"],
              "selected_constants_sha256": matrix_digest({name: selected}),
              "recipe3_baseline": {"recipe_version": 3, "rotation_variant": "perhead_center",
                                   "constants_sha256": binding["base_constants_sha256"]},
              "validation": {"baseline": before, "selected": after},
              "test": {"baseline": test_before, "selected": test_after},
              "native_residual_selection": decision}
    if not nonworse(test_before, test_after, require_frames=True, require_terminal=True,
                    require_queries=True, require_projection=True):
        report = _report_independent_test_failure(name, baseline["layer_id"], record, policy,
            checkpoints, emit, publication_allowed=False)
        raise IndependentTestRegression(f"OptR {name}: independent test regressed; artifact not published", report)
    emit(layer=name, phase="checked", accepted=accepted, independent_test="passed",
         validation_before=before["normalized_error"], validation_after=after["normalized_error"],
         independent_test_before=test_before["normalized_error"], independent_test_after=test_after["normalized_error"])
    return selected, record
