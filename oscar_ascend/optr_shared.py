"""Archive #170/#173/#175/#176: OptR search within OSCAR's shared-layer basis.

The loss is the fixed upstream independent-GQA K/V loss from optr_sequential.
Only its parameters are tied: one R_K and one R_V for the entire layer, as in
the user-specified oscar_base. This is an explicit constrained OptR adaptation,
not a claim that upstream OptR uses shared parameters. Nothing runs in serving.
"""
from __future__ import annotations

import torch

from .optr_calibration import _matrices, _sample
from .optr_sequential import prepare_samples, phase_loss, _key_probabilities


def validate_shared_pair(rk, rv, mean):
    if (not isinstance(rk, torch.Tensor) or rk.ndim != 3 or rk.shape[0] < 1 or
            not isinstance(rv, torch.Tensor) or rv.shape != rk.shape or
            not isinstance(mean, torch.Tensor) or mean.shape != rk.shape[:2]):
        raise ValueError("OptR shared layer requires [heads,dim,dim] rotations and [heads,dim] means")
    for h in range(rk.shape[0]):
        _matrices(rk[h], rv[h], mean[h])
    if torch.count_nonzero(mean):
        raise ValueError("OptR shared OSCAR basis requires unchanged zero key means")
    if any(not torch.equal(r, r[:1].expand_as(r)) for r in (rk, rv)):
        raise ValueError("OptR shared layer rotations differ across KV heads")
    return rk.shape[0], rk.shape[-1]


def _stage(heads, mean, policy, device, weights, split, emit):
    rows = []
    for h, samples in enumerate(heads):
        staged = prepare_samples(samples, mean[h], policy, device, weights=weights)
        # Explicit head mean then request mean; no second head divisor later.
        weight = 1. / (len(heads) * len(samples))
        rows.extend((row, weight) for row in staged)
        emit("head_staged", head=h, split=split, samples=len(samples))
    return rows


def _score(rotation, rows, phase, policy, emit):
    total = 0.
    with torch.no_grad():
        for index, (row, weight) in enumerate(rows):
            loss = phase_loss(rotation, row, phase, policy)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite OptR shared held-out proxy loss")
            total += float(loss) * weight
            emit("validation_sample_done", phase=phase, sample=index + 1,
                 samples=len(rows), request_id=row["request_id"])
    return total


def backward_shared(rotation, rows, phase, policy):
    """Accumulate dL/dR one request at a time, then one chain rule through exp.

    All rows see the SAME parameter value. Detaching only the device leaf
    avoids retaining all long-history graphs; its accumulated gradient is
    explicitly sent through the original differentiable CPU->NPU copy once.
    Adam is called by the caller only after every head/request has contributed.
    """
    leaf = rotation.detach().requires_grad_()
    for row, weight in rows:
        loss = phase_loss(leaf, row, phase, policy)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite OptR shared training proxy loss")
        (loss * weight).backward()
    if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()):
        raise ValueError("nonfinite or missing OptR shared rotation gradient")
    rotation.backward(leaf.grad)


def fit_shared(base_k, base_v, mean, train_heads, validation_heads, *, policy,
               progress=None, device="cpu"):
    """One layer-wide K search, then one V search conditional on selected K."""
    heads, dim = validate_shared_pair(base_k, base_v, mean)
    for split in (train_heads, validation_heads):
        if len(split) != heads or any(not samples for samples in split):
            raise ValueError("OptR shared fitting needs nonempty samples for every KV head")
        identities = [s["request_id"] for s in split[0]]
        if len(set(identities)) != len(identities):
            raise ValueError("OptR shared split contains duplicate request identities")
        for samples in split:
            if [s["request_id"] for s in samples] != identities:
                raise ValueError("OptR shared heads must cover the same requests in order")
            for sample in samples:
                _sample(sample, dim)
    if {s["request_id"] for s in train_heads[0]} & {s["request_id"] for s in validation_heads[0]}:
        raise ValueError("training and validation request identities overlap")

    def emit(event, **fields):
        if progress is not None:
            progress({"event": event, "fitted_grouping": "layer", **fields})

    emit("staging_start", device=str(device), objective=policy["fitting_objective"])
    weights = {}
    train = _stage(train_heads, mean, policy, device, weights, "train", emit)
    held = _stage(validation_heads, mean, policy, device, weights, "validation", emit)
    rk, rv = base_k[0].clone(), base_v[0].clone()
    decisions = []
    for phase, origin in (("k", base_k[0]), ("v", base_v[0])):
        emit("phase_start", phase=phase, steps=policy["steps"])
        if phase == "v":
            with torch.no_grad():
                rotation = rk.to(device)
                for row, _ in (*train, *held):
                    row["selected_k_quant_p"] = _key_probabilities(rotation, row, policy).masked_fill(row["exact"], 0)
                    del row["teacher_p"], row["teacher_logp"], row["teacher_logits"]
                    # V uses only fixed probabilities, V and W_O. Release the
                    # staged K/query/masks rather than retain all heads twice.
                    del row["q"], row["k"], row["exact"], row["causal"]
            emit("fixed_k_probabilities_ready", phase=phase, samples=len(train) + len(held))
        baseline = best = _score(origin.to(device), held, phase, policy, emit)
        best_rotation = origin.clone()
        correction = torch.zeros_like(origin, requires_grad=True)
        optimizer = torch.optim.Adam([correction], lr=policy["lr"])
        candidates = []
        for step in range(policy["steps"]):
            optimizer.zero_grad(set_to_none=True)
            rotation = (origin @ torch.matrix_exp(correction - correction.T)).to(device)
            backward_shared(rotation, train, phase, policy)
            if correction.grad is None or not bool(torch.isfinite(correction.grad).all()):
                raise ValueError("nonfinite or missing OptR shared parameter gradient")
            optimizer.step()
            emit("step_done", phase=phase, step=step + 1, steps=policy["steps"])
            if step % policy["eval_every"] and step != policy["steps"] - 1:
                continue
            emit("validation_start", phase=phase, step=step + 1, steps=policy["steps"])
            with torch.no_grad():
                candidate = origin @ torch.matrix_exp(correction - correction.T)
                score = _score(candidate.to(device), held, phase, policy, emit)
                accepted = score < best
                candidates.append({"step": step + 1, "proxy_loss": score, "accepted": accepted})
                if accepted:
                    best, best_rotation = score, candidate.clone()
            emit("validation_done", phase=phase, step=step + 1, steps=policy["steps"],
                 proxy_loss=score, accepted=accepted)
        keep = best < baseline
        selected = best_rotation if keep else origin.clone()
        if phase == "k":
            rk = selected
        else:
            rv = selected
        decisions.append({"phase": phase, "baseline_proxy_loss": baseline,
                          "selected_proxy_loss": best, "accepted": keep, "candidates": candidates})
        emit("phase_done", phase=phase, proxy_loss=best, accepted=keep)
    # Keep the existing artifact/TP ABI: materialize identical copies offline.
    rk = rk.expand(heads, dim, dim).clone().contiguous()
    rv = rv.expand(heads, dim, dim).clone().contiguous()
    validate_shared_pair(rk, rv, mean)
    accepted = any(row["accepted"] for row in decisions)
    emit("fit_done", accepted=accepted)
    return rk, rv, {"accepted": accepted, "phases": decisions, "policy": policy,
        "objective": policy["fitting_objective"], "fit_device": str(device),
        "matrix_exp_device": "cpu", "fitted_grouping": "layer",
        "selection": "shared_layer_proxy_then_joint_deployment_gate",
        "loss_reduction": "mean_kv_heads_mean_requests_mean_gqa_queries_hidden",
        "model_quality": "not_measured", "npu_equivalence": "not_measured"}
