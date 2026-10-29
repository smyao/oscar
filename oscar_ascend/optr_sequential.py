"""Offline OptR sequential objective, adapted from upstream 66d46561b9bdf.

Archive #86/#148/#170/#172/#173/#175/#176: proposals are not deployment acceptance.
K minimizes KL + post-W_O attention-shift MSE; V minimizes value-error MSE
under the selected K probabilities, using upstream's FP32 original-space
effective K/V and separate GQA-head losses. A trace contains several forwards,
so its exact-window mask uses each captured forward's context. Native gate,
BF16 rounding and joint-head output checks belong to the independent deployment
replay in the caller; they do not redefine the published OptR optimizer.

Large FP32 products run on Ascend in production. Only the small orthogonal
parameterization/Adam remains on CPU; differentiable device copies carry its
gradient. No unsupported NPU matrix_exp or full-model optimizer is required.
This module is never used by the serving forward path.
"""
from __future__ import annotations

import torch

from .optr_calibration import _matrices, _sample


def fitting_device(config, *, testing=False, logical_index=0):
    """Select one card from this run's explicit list, after capture has exited."""
    if testing:
        return "cpu"
    import os
    from tools.target_cli import target_env
    visible = target_env(config)["ASCEND_RT_VISIBLE_DEVICES"]
    if type(logical_index) is not int or not 0 <= logical_index < len(config["devices"]):
        raise ValueError("OptR worker logical device is outside the selected cards")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") not in (None, visible):
        raise ValueError("OptR fitting device visibility differs from this run's configuration")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible
    import torch_npu  # noqa: F401; registers the NPU backend, never cuda->cpu fallback
    if not torch.npu.is_available():
        raise RuntimeError("OptR production fitting requires the configured Ascend NPU")
    # Bind before options/compile setup can initialize an accelerator context.
    # Each parallel worker must own its assigned device, including workers 1-3.
    torch.npu.set_device(logical_index)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype",
                          "ALLOW_MATMUL_HF32": "disable"})
    torch.npu.set_compile_mode(jit_compile=False)
    return f"npu:{logical_index}"


def quant_int2(value, clip):
    """Upstream FP32 four-level STE, one deployed head-wide group per row."""
    index = min(max(int(clip * value.shape[-1]), 0), value.shape[-1] - 1)
    threshold = value.detach().abs().sort(dim=-1).values[..., index:index + 1]
    clipped = torch.minimum(torch.maximum(value, -threshold), threshold)
    low = clipped.amin(-1, keepdim=True)
    scale = (clipped.amax(-1, keepdim=True) - low).clamp_min(1e-8) / 3.
    zero = -low / scale
    unrounded = clipped / scale + zero
    code = (unrounded + .5).floor().clamp(0, 3)
    return (unrounded + (code - unrounded).detach() - zero) * scale


def _probabilities(row, key):
    logits = (row["q"] @ key.T) * row["scale"]
    return logits.masked_fill(row["causal"], -torch.inf).softmax(-1)


def _key_probabilities(rotation, row, policy):
    """Upstream effective K, with the exact window of each captured forward."""
    effective = quant_int2(row["k"] @ rotation, policy["k_clip"]) @ rotation.T
    quantized_logits = (row["q"] @ effective.T) * row["scale"]
    # Selecting logits is the same rowwise exact replacement as upstream,
    # without materializing query-dependent [queries,history,dim] tensors.
    logits = torch.where(row["exact"], row["teacher_logits"], quantized_logits)
    return logits.masked_fill(row["causal"], -torch.inf).softmax(-1)


def _projected_mse(error, row):
    # Upstream squares each GQA head's post-W_O contribution separately,
    # then averages over heads, queries and hidden coordinates. Direct
    # projection preserves this formula without a rounded Gram surrogate.
    return torch.bmm(error, row["wo_t"]).square().mean()


def prepare_samples(samples, mean, policy, device, *, weights=None):
    """Stage FP32 histories once; cache fixed teacher probabilities and W_O."""
    rows = []
    weights = {} if weights is None else weights
    with torch.no_grad():
        for sample in samples:
            q = sample["q"].float().to(device).transpose(0, 1).contiguous()
            k = sample["k"].float().to(device) - mean.to(device)
            v = sample["v"].float().to(device)
            # Frame metadata is authenticated on CPU. Build masks once here;
            # no NPU int64 minimum/clamp kernels or per-step host round trips.
            positions = sample["positions"]
            contexts = sample["contexts"]
            length = k.shape[0]
            indices = torch.arange(length)
            wo = sample["wo"]
            identity = id(wo)
            if identity not in weights:
                weights[identity] = wo.transpose(1, 2).contiguous().to(device)
            # Same cached cut as the deployed reader: fixed at forward start.
            # Tokens >= context (current/future) are exact; causality masks
            # future tokens. context=0 therefore has no INT2 history.
            cuts = torch.minimum(contexts, (contexts + 1 - policy["recent_tokens"]).clamp_min(policy["sink_tokens"]))
            row = {"q": q, "k": k, "v": v, "scale": k.shape[-1] ** -.5,
                   "wo_t": weights[identity], "request_id": sample["request_id"],
                   "causal": (indices[None, :] > positions[:, None]).to(device),
                   "exact": ((indices[None, :] < policy["sink_tokens"]) |
                             (indices[None, :] >= cuts[:, None])).to(device)}
            row["teacher_logits"] = (q @ k.T) * row["scale"]
            row["teacher_p"] = row["teacher_logits"].masked_fill(row["causal"], -torch.inf).softmax(-1)
            row["teacher_logp"] = row["teacher_p"].clamp_min(1e-12).log()
            rows.append(row)
    return rows


def phase_loss(rotation, row, phase, policy):
    if phase == "k":
        p = _key_probabilities(rotation, row, policy)
        pt = row["teacher_p"]
        kl = (pt * (row["teacher_logp"] - p.clamp_min(1e-12).log())).sum(-1).mean()
        return kl + policy["lambda_output"] * _projected_mse((p - pt) @ row["v"], row)
    if phase != "v":
        raise ValueError("OptR phase must be k or v")
    effective = quant_int2(row["v"] @ rotation, policy["v_clip"]) @ rotation.T
    # Upstream exact V rows have zero error. Cache their masked probabilities
    # once; no QK/softmax or query-dependent history expansion per V step.
    return _projected_mse(row["selected_k_quant_p"] @ (effective - row["v"]), row)


def _values(rotation, rows, phase, policy, emit):
    values = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            loss = phase_loss(rotation, row, phase, policy)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite OptR held-out proxy loss")
            values.append(float(loss))
            emit("validation_sample_done", phase=phase, sample=index + 1,
                 samples=len(rows), request_id=row["request_id"])
    return sum(values) / len(values)


def fit_head(base_k, base_v, mean, train_samples, validation_samples, *, policy,
             progress=None, device="cpu", global_kv_heads=1):
    """Sequential K/V learning and best held-out selection.

    Heads run independently, with the upstream all-KV-head mean retained in
    the backward loss. Adam's epsilon makes omitting this scale observable.
    Sparse query capture and frame windows are explicit input adaptations.
    These FP32 proxy losses are not normalized deployment output errors.
    """
    if type(global_kv_heads) is not int or global_kv_heads < 1:
        raise ValueError("global_kv_heads must be a positive integer")
    dim = _matrices(base_k, base_v, mean)
    if not train_samples or not validation_samples:
        raise ValueError("nonempty separate training/validation samples required")
    for sample in (*train_samples, *validation_samples):
        _sample(sample, dim)
    if {s["request_id"] for s in train_samples} & {s["request_id"] for s in validation_samples}:
        raise ValueError("training and validation request identities overlap")
    def emit(event, **fields):
        if progress is not None:
            progress({"event": event, **fields})
    emit("staging_start", device=str(device), objective=policy["fitting_objective"])
    weights = {}  # One fixed projection for both splits of this model head.
    train = prepare_samples(train_samples, mean, policy, device, weights=weights)
    held = prepare_samples(validation_samples, mean, policy, device, weights=weights)
    rk, rv = base_k.clone(), base_v.clone()
    decisions = []
    for phase, origin in (("k", base_k), ("v", base_v)):
        emit("phase_start", phase=phase, steps=policy["steps"])
        if phase == "v":
            with torch.no_grad():
                rotation = rk.to(device)
                for row in (*train, *held):
                    row["selected_k_quant_p"] = _key_probabilities(rotation, row, policy).masked_fill(row["exact"], 0)
                    # No K quantization, QK or softmax in any V optimization step.
                    del row["teacher_p"], row["teacher_logp"], row["teacher_logits"]
            emit("fixed_k_probabilities_ready", phase=phase, samples=len(train) + len(held))
        baseline = best = _values(origin.to(device), held, phase, policy, emit)
        best_rotation = origin.clone()
        correction = torch.zeros_like(origin, requires_grad=True)  # intentionally CPU
        optimizer = torch.optim.Adam([correction], lr=policy["lr"])
        candidates = []
        for step in range(policy["steps"]):
            optimizer.zero_grad(set_to_none=True)
            # One matrix_exp per step, independent of query microbatch count.
            rotation = (origin @ torch.matrix_exp(correction - correction.T)).to(device)
            loss = torch.stack([phase_loss(rotation, row, phase, policy) for row in train]).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite OptR training proxy loss")
            (loss / global_kv_heads).backward()
            if correction.grad is None or not bool(torch.isfinite(correction.grad).all()):
                raise ValueError("nonfinite or missing OptR rotation gradient")
            optimizer.step()
            emit("step_done", phase=phase, step=step + 1, steps=policy["steps"])
            # Upstream checks zero-based steps 0,20,40,60 and the last step.
            if step % policy["eval_every"] and step != policy["steps"] - 1:
                continue
            emit("validation_start", phase=phase, step=step + 1, steps=policy["steps"])
            with torch.no_grad():
                candidate = origin @ torch.matrix_exp(correction - correction.T)
                score = _values(candidate.to(device), held, phase, policy, emit)
                accepted = score < best
                candidates.append({"step": step + 1, "proxy_loss": score, "accepted": accepted})
                if accepted:
                    best, best_rotation = score, candidate.clone()
            emit("validation_done", phase=phase, step=step + 1, steps=policy["steps"],
                 proxy_loss=score, accepted=accepted)
        keep = best < baseline  # official zero margin; not a deployment gate
        selected = best_rotation if keep else origin.clone()
        if phase == "k":
            rk = selected
        else:
            rv = selected
        decisions.append({"phase": phase, "baseline_proxy_loss": baseline,
                          "selected_proxy_loss": best, "accepted": keep, "candidates": candidates})
        emit("phase_done", phase=phase, proxy_loss=best, accepted=keep)
    _matrices(rk, rv, mean)
    accepted = any(row["accepted"] for row in decisions)
    emit("fit_done", accepted=accepted)
    return rk.contiguous(), rv.contiguous(), {
        "accepted": accepted, "phases": decisions, "policy": policy,
        "objective": policy["fitting_objective"], "fit_device": str(device),
        "matrix_exp_device": "cpu", "selection": "upstream_separate_head_proxy_then_joint_deployment_gate",
        "optimizer_loss_divisor": global_kv_heads,
        "model_quality": "not_measured", "npu_equivalence": "not_measured"}
