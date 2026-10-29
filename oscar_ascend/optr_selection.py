"""Archive #148/#170/#173: select offline head combinations by joint output.

Independent head proposals can interfere after gate/W_O and TP summation.
Replay each seed/proposal head once per partition, then enumerate the at most
16 combinations using only small projected query outputs. The unchanged
runtime receives ordinary constant matrices, never this selector or traces.
"""
from __future__ import annotations

import torch

from .optr_artifacts import nonworse
from .optr_calibration import project_output, replay_sample
from .optr_traces import aggregate_layer_error


def _merge_partitions(metrics, partitions):
    rows = []
    for metric, partition in zip(metrics, partitions):
        rows.extend({**row, "request_id": row["request_id"] + ":" + partition,
                     "partition": partition} for row in metric["requests"])
    error = sum(row["squared_error"] for row in rows)
    energy = sum(row["teacher_energy"] for row in rows)
    return {"squared_error": error, "teacher_energy": energy,
            "normalized_error": error / max(energy, 1e-20), "requests": rows}


def select_head_subset(baseline, trial, samples, policy, *, progress=None):
    """Return constants, seed/selected metrics and a validation-only decision.

    The final independent test is intentionally absent from this interface.
    Every admissible combination must improve total SSE and preserve every
    request/frame in every declared validation partition. Equal SSE prefers
    fewer changed heads, followed by the lower deterministic bit mask.
    """
    mean = baseline.get("k_mean")
    other_mean = trial.get("k_mean")
    if (not isinstance(mean, torch.Tensor) or mean.ndim != 2 or
            not isinstance(other_mean, torch.Tensor) or
            not 1 <= mean.shape[0] <= 4 or not torch.equal(mean, other_mean)):
        raise ValueError("OptR subset selection requires at most four heads and the same K mean")
    heads, dim = mean.shape
    for constants in (baseline, trial):
        for key in ("Rk", "Rv"):
            value = constants.get(key)
            if (not isinstance(value, torch.Tensor) or value.shape != (heads, dim, dim) or
                    value.dtype != torch.float32 or value.device.type != "cpu"):
                raise ValueError("OptR subset rotation geometry/dtype differs")
    if not isinstance(samples, list) or len(samples) != heads:
        raise ValueError("OptR subset samples must cover every head")
    partitions = policy["validation_partitions"]
    if not partitions or len(set(partitions)) != len(partitions):
        raise ValueError("OptR subset validation partitions must be nonempty and unique")
    changed = [head for head in range(heads) if any(
        not torch.equal(baseline[key][head], trial[key][head]) for key in ("Rk", "Rv"))]
    masks = range(1 << len(changed))
    metrics = {mask: [] for mask in masks}
    chosen = {mask: {head for bit, head in enumerate(changed) if mask & (1 << bit)} for mask in masks}

    def emit(event, **fields):
        if progress is not None:
            progress(event=event, **fields)

    with torch.no_grad():
        teachers = [[project_output(sample["teacher"], sample) for sample in head] for head in samples]
        for partition in partitions:
            replay_policy = {**policy, "replay_partition": partition}
            outputs = {"baseline": [], "trial": []}
            for head in range(heads):
                for candidate, constants in (("baseline", baseline), ("trial", trial)):
                    if candidate == "trial" and head not in changed:
                        outputs[candidate].append(outputs["baseline"][head])
                        continue
                    projected = [project_output(replay_sample(
                        constants["Rk"][head], constants["Rv"][head], mean[head], sample,
                        replay_policy), sample) for sample in samples[head]]
                    outputs[candidate].append(projected)
                    emit("subset_head_replayed", partition=partition, candidate=candidate, head=head)
            for mask in masks:
                projected = [outputs["trial" if head in chosen[mask] else "baseline"][head]
                             for head in range(heads)]
                metrics[mask].append(aggregate_layer_error(
                    baseline, samples, replay_policy, projected_outputs=projected,
                    projected_teachers=teachers))
            emit("subset_partition_checked", partition=partition, combinations=len(metrics))
            # Do not retain projected candidates across partitions or any
            # full-history K/V tensors beyond an individual replay call.
            del outputs, projected

    combined = {mask: _merge_partitions(metrics[mask], partitions) for mask in masks}
    before = combined[0]
    nonworse(before, before, require_frames=True)
    admissible = [mask for mask in masks if mask and
                  combined[mask]["squared_error"] < before["squared_error"] and
                  nonworse(before, combined[mask], require_frames=True)]
    selected_mask = min(admissible, key=lambda mask: (
        combined[mask]["squared_error"], len(chosen[mask]), mask)) if admissible else 0
    selected = baseline
    if selected_mask:
        selected = {**baseline, "Rk": baseline["Rk"].clone(), "Rv": baseline["Rv"].clone()}
        for head in chosen[selected_mask]:
            for key in ("Rk", "Rv"):
                selected[key][head] = trial[key][head]
    record = {"method": "exhaustive_head_subset_validation_v1", "selected_mask": selected_mask,
              "proposed_heads": changed, "selected_heads": sorted(chosen[selected_mask]),
              "combinations": len(metrics), "admissible_combinations": len(admissible),
              "test_used_for_selection": False,
              "candidate_errors": [{"mask": mask, "heads": sorted(chosen[mask]),
                                    "squared_error": combined[mask]["squared_error"],
                                    "accepted": mask in admissible} for mask in masks]}
    emit("subset_selected", **record)
    return selected, before, combined[selected_mask], record
