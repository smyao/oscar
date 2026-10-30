"""Archive #170/#173/#175/#176: OptR search within OSCAR's shared-layer basis.

The loss is the fixed upstream independent-GQA K/V loss from optr_sequential.
Only its parameters are tied: one R_K and one R_V for the entire layer, as in
the user-specified oscar_base. This is an explicit constrained OptR adaptation,
not a claim that upstream OptR uses shared parameters. Nothing runs in serving.
"""
from __future__ import annotations

import torch

from .optr_calibration import _matrices, _sample
from .optr_sequential import phase_loss
from .optr_shared_cache import prepare_shared_rows


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


def _score(rotation, rows, phase, policy, emit):
    total = 0.
    with torch.no_grad():
        index = 0
        # enumerate retains its previous result tuple during next(iterator),
        # keeping a streamed device row alive while the next is allocated.
        for row, weight in rows:
            index += 1
            loss = phase_loss(rotation, row, phase, policy)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite OptR shared held-out proxy loss")
            total += float(loss) * weight
            emit("validation_sample_done", phase=phase, sample=index,
                 samples=len(rows), request_id=row["request_id"])
            del row, loss  # Release a streamed device row before staging the next.
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
        del row, loss
    if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()):
        raise ValueError("nonfinite or missing OptR shared rotation gradient")
    rotation.backward(leaf.grad)


def fit_shared(base_k, base_v, mean, train_heads, validation_heads, *, policy,
               progress=None, device="cpu", value_candidate_callback=None):
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
    train, held, resources = prepare_shared_rows(
        train_heads, validation_heads, mean, policy, device, emit)
    rk, rv = base_k[0].clone(), base_v[0].clone()
    decisions = []
    for phase, origin in (("k", base_k[0]), ("v", base_v[0])):
        emit("phase_start", phase=phase, steps=policy["steps"])
        if phase == "v":
            fixed_k = rk.to(device)
            train.prepare_value(fixed_k)
            held.prepare_value(fixed_k)
            del fixed_k
            emit("fixed_k_probabilities_ready" if resources["mode"] == "resident"
                 else "fixed_k_rotation_ready", phase=phase, samples=len(train) + len(held))
        baseline = best = _score(origin.to(device), held, phase, policy, emit)
        if phase == "v" and value_candidate_callback is not None:
            value_candidate_callback(step=0, rotation=origin, proxy_loss=baseline)
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
                if phase == "v" and value_candidate_callback is not None:
                    value_candidate_callback(step=step + 1, rotation=candidate, proxy_loss=score)
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
        "staging_resources": resources,
        "selection": "shared_layer_proxy_then_joint_deployment_gate",
        "loss_reduction": "mean_kv_heads_mean_requests_mean_gqa_queries_hidden",
        "model_quality": "not_measured", "npu_equivalence": "not_measured"}
