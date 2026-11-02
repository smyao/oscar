"""Offline OptR corrections around the unchanged per-head recipe3 seed.

Archive #86/#170/#172/#174: preserve the aligned FULL QKV and per-head key
means. This changes only offline constants and candidate recall, never the
serving arithmetic, clip, windows or kernels. Upstream K-then-V losses and
the all-KV-head backward divisor remain in optr_sequential.fit_head.

Each head runs serially, so at most one head's train/validation FP32 histories
and autograd products are staged on the fitting device. That head's histories
are still resident: this does not claim a fixed device-memory bound. Candidate
matrices alone are saved on CPU. All candidates require the caller's existing
joint-head BF16 deployment gate; proxy acceptance is not model acceptance.
"""
from __future__ import annotations

import torch

from .optr_calibration import _matrices, _sample
from .optr_candidates import rotation_digest
from .optr_sequential import fit_head


MAX_CANDIDATES = 12
CHECKPOINTS_PER_PHASE = 5
CANDIDATE_STRATEGY = "perhead_existing_kv_checkpoints_best_nonworse"


def checkpoint_steps(policy):
    """A bounded subset of existing evaluations, with both endpoints retained.

    The default 80-step schedule retains every scored step: 1/21/41/61/80.
    Denser explicit research schedules use five evenly spaced scored steps;
    this is fixed before fitting and never depends on validation/test scores.
    """
    steps, interval = policy["steps"], policy["eval_every"]
    if type(steps) is not int or steps < 1 or type(interval) is not int or interval < 1:
        raise ValueError("OptR fitting steps/eval_every must be positive integers")
    scored = [step + 1 for step in range(steps) if step % interval == 0 or step == steps - 1]
    if len(scored) <= CHECKPOINTS_PER_PHASE:
        return scored
    return [scored[i * (len(scored) - 1) // (CHECKPOINTS_PER_PHASE - 1)]
            for i in range(CHECKPOINTS_PER_PHASE)]


def validate_perhead_pair(rk, rv, mean):
    if (not isinstance(rk, torch.Tensor) or rk.ndim != 3 or rk.shape[0] < 1 or
            not isinstance(rv, torch.Tensor) or rv.shape != rk.shape or
            not isinstance(mean, torch.Tensor) or mean.shape != rk.shape[:2]):
        raise ValueError("OptR recipe3 requires [heads,dim,dim] rotations and [heads,dim] means")
    for h in range(rk.shape[0]):
        _matrices(rk[h], rv[h], mean[h])
    return rk.shape[0], rk.shape[-1]


def fit_perhead(base_k, base_v, mean, train_heads, validation_heads, *, policy,
                progress=None, device="cpu"):
    """Return the per-head proxy winner plus <=12 whole-layer proposals.

    K alternatives pair a synchronous existing K checkpoint with the original
    recipe3 V. V alternatives retain each head's selected K and use a common
    existing V checkpoint step. No cross-product, interpolation or test-driven
    search is performed. The primary retains independently selected per-head
    proxy winners, exactly as upstream's unconstrained parameterization does.
    """
    heads, dim = validate_perhead_pair(base_k, base_v, mean)
    for split in (train_heads, validation_heads):
        if len(split) != heads or any(not samples for samples in split):
            raise ValueError("OptR recipe3 fitting needs nonempty samples for every KV head")
        identities = [s["request_id"] for s in split[0]]
        if len(set(identities)) != len(identities):
            raise ValueError("OptR recipe3 split contains duplicate request identities")
        for samples in split:
            if [s["request_id"] for s in samples] != identities:
                raise ValueError("OptR recipe3 heads must cover the same requests in order")
            for sample in samples:
                _sample(sample, dim)
    if {s["request_id"] for s in train_heads[0]} & {s["request_id"] for s in validation_heads[0]}:
        raise ValueError("training and validation request identities overlap")

    steps = [0, *checkpoint_steps(policy)]
    snapshots = {phase: {step: [] for step in steps} for phase in ("k", "v")}
    proxies = {phase: {step: [] for step in steps} for phase in ("k", "v")}
    selected_k, selected_v, decisions = [], [], []
    for head in range(heads):
        def emit(event):
            if progress is not None:
                progress({**event, "fitted_grouping": "layer_kv_head", "global_kv_head": head})

        def observe(*, phase, step, rotation, proxy_loss):
            if step in snapshots[phase]:
                snapshots[phase][step].append(rotation.detach().cpu().clone())
                proxies[phase][step].append(proxy_loss)

        rk, rv, decision = fit_head(base_k[head], base_v[head], mean[head],
            train_heads[head], validation_heads[head], policy=policy, progress=emit,
            device=device, global_kv_heads=heads, candidate_callback=observe)
        selected_k.append(rk)
        selected_v.append(rv)
        decisions.append({"global_kv_head": head, **decision})
    rk, rv = torch.stack(selected_k).contiguous(), torch.stack(selected_v).contiguous()
    validate_perhead_pair(rk, rv, mean)

    candidates, seen = [], set()

    def append(identity, phase, step, key, value, proxy_losses=None):
        digest = (rotation_digest(key), rotation_digest(value))
        if digest in seen:
            return
        seen.add(digest)
        candidates.append({"id": identity, "phase": phase, "step": step,
                           "Rk": key.detach().cpu().clone(), "Rv": value.detach().cpu().clone(),
                           "proxy_losses": proxy_losses})

    append("primary", "primary", None, rk, rv)
    for phase in ("k", "v"):
        for step in steps:
            if len(snapshots[phase][step]) != heads or len(proxies[phase][step]) != heads:
                raise ValueError("OptR recipe3 candidate checkpoint coverage is incomplete")
            matrix = torch.stack(snapshots[phase][step]).contiguous()
            append(f"{phase}_step_{step}", phase, step,
                   matrix if phase == "k" else rk,
                   base_v if phase == "k" else matrix, proxies[phase][step])
    # Independently chosen K can combine different steps, so its V-seed
    # pair need not equal a synchronous K-step proposal: the count can be 13.
    if len(candidates) > MAX_CANDIDATES:
        # The caller already evaluates the unchanged recipe3 baseline; that
        # identical K-seed/V-seed pair cannot pass its strict-improvement gate.
        candidates = [row for row in candidates if row["id"] != "k_step_0"]
    if len(candidates) > MAX_CANDIDATES:
        raise AssertionError("OptR recipe3 candidate budget exceeded")
    return {"Rk": rk, "Rv": rv, "decision": {
        "accepted": any(row["accepted"] for row in decisions), "heads": decisions,
        "policy": policy, "objective": policy["fitting_objective"],
        "fit_device": str(device), "matrix_exp_device": "cpu",
        "fitted_grouping": "layer_kv_head", "optimizer_loss_divisor": heads,
        "selection": "perhead_proxy_then_joint_deployment_gate",
        "deployment_candidate_selection": CANDIDATE_STRATEGY,
        "candidate_steps": steps, "candidate_count": len(candidates),
        "staging_resources": {"mode": "serial_head_resident", "max_resident_heads": 1,
                              "candidate_matrix_device": "cpu", "fixed_device_byte_bound": False},
        "model_quality": "not_measured", "npu_equivalence": "not_measured"},
        "candidates": candidates}
