"""Archive #170/#173/#176: retain already evaluated shared V checkpoints.

The official K->V trajectory is unchanged. Only CPU D-by-D matrices are
retained, with the selected K fixed. No new optimizer steps, interpolation,
per-head mixtures, or test-driven selection are introduced.
"""
from __future__ import annotations

import hashlib
import math

import torch


def rotation_digest(matrix):
    return hashlib.sha256(matrix.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class ValueCandidates:
    """Keep the initial V and at most five best existing proxy checkpoints."""

    def __init__(self, policy):
        self.limit = policy["max_deployment_candidates"]
        self.initial = None
        self.improved = []

    def observe(self, *, step, rotation, proxy_loss):
        if not math.isfinite(proxy_loss):
            raise ValueError("nonfinite OptR candidate proxy")
        if step == 0:
            if self.initial is not None:
                raise ValueError("OptR V candidate baseline already recorded")
            self.initial = {"step": 0, "proxy_loss": proxy_loss, "Rv": rotation.detach().cpu().clone()}
            return
        if self.initial is None:
            raise ValueError("OptR V candidate requires the initial proxy")
        # Sort scalar metadata first; do not even copy a matrix outside the
        # fixed shortlist. The optimizer owns no references to these clones.
        order = sorted([*self.improved, {"step": step, "proxy_loss": proxy_loss}],
                       key=lambda row: (row["proxy_loss"], row["step"]))[:self.limit - 1]
        for row in order:
            if "Rv" not in row:
                row["Rv"] = rotation.detach().cpu().clone()
        self.improved = order

    def payload(self, selected_key):
        if self.initial is None:
            raise ValueError("OptR V candidate baseline is missing")
        return {"fixed_key_sha256": rotation_digest(selected_key[0]),
                "values": [self.initial, *self.improved]}


def validate_candidates(payload, result, baseline, policy):
    """Bind resumable candidates to the same K, seed V and scored schedule."""
    from .optr_calibration import _matrices
    if not isinstance(payload, dict) or payload.get("fixed_key_sha256") != rotation_digest(result["Rk"][0]):
        raise ValueError("OptR V candidates changed their fixed K")
    decision = result["decision"]
    if decision.get("policy") != policy:
        raise ValueError("OptR V candidate fitting policy differs")
    phases = decision.get("phases")
    if (not isinstance(phases, list) or any(not isinstance(row, dict) for row in phases)
            or [row.get("phase") for row in phases] != ["k", "v"]):
        raise ValueError("OptR V candidates require completed K then V phases")
    phase = phases[1]
    initial = phase.get("baseline_proxy_loss")
    if type(initial) not in (int, float) or not math.isfinite(initial):
        raise ValueError("OptR V candidate baseline proxy is invalid")
    schedule = [step + 1 for step in range(policy["steps"])
                if step % policy["eval_every"] == 0 or step == policy["steps"] - 1]
    scored = phase.get("candidates")
    if (not isinstance(scored, list) or any(not isinstance(row, dict) for row in scored)
            or [row.get("step") for row in scored] != schedule):
        raise ValueError("OptR V candidate validation schedule differs")
    if any(type(row.get("proxy_loss")) not in (int, float) or not math.isfinite(row["proxy_loss"])
           for row in scored):
        raise ValueError("OptR V candidate proxy must be finite")
    expected = [{"step": 0, "proxy_loss": initial}, *sorted(
        scored,
        key=lambda row: (row["proxy_loss"], row["step"]))[:policy["max_deployment_candidates"] - 1]]
    values = payload.get("values")
    if not isinstance(values, list) or len(values) != len(expected):
        raise ValueError("OptR V candidate coverage differs")
    for value, reference in zip(values, expected):
        if (not isinstance(value, dict) or type(value.get("step")) is not int or
                any(value.get(key) != reference[key] for key in ("step", "proxy_loss"))):
            raise ValueError("OptR V candidate step/proxy differs")
        _matrices(result["Rk"][0], value.get("Rv"), baseline["k_mean"][0])
    if not torch.equal(values[0]["Rv"], baseline["Rv"][0]):
        raise ValueError("OptR initial V candidate differs from OSCAR seed")
    best = min(values, key=lambda row: (row["proxy_loss"], row["step"]))
    if (phase.get("selected_proxy_loss") != best["proxy_loss"] or
            phase.get("accepted") is not (best["proxy_loss"] < initial) or
            not torch.equal(result["Rv"][0], best["Rv"])):
        raise ValueError("OptR primary V is not its original proxy winner")
    return values


def alternatives(values, primary):
    """Primary always has precedence; try each distinct V at most once."""
    seen = {rotation_digest(primary[0])}
    for row in sorted(values, key=lambda row: (row["proxy_loss"], row["step"])):
        digest = rotation_digest(row["Rv"])
        if digest not in seen:
            seen.add(digest)
            yield row
