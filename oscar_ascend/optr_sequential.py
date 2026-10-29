"""Offline OptR sequential objective, adapted from upstream 66d46561b9bdf.

Archive #86/#148/#170/#172/#173: proposals are not deployment acceptance.
K minimizes KL + post-W_O attention-shift MSE; V minimizes value-error MSE
under the selected K probabilities. Native captured gates, per-forward exact
windows and BF16 rotation/codec operands adapt the upstream proxy to Qwen/OSCAR. The independent
BF16 deployment replay remains the final validation gate in the caller.

Large FP32 products run on Ascend in production. Only the small orthogonal
parameterization/Adam remains on CPU; differentiable device copies carry its
gradient. No unsupported NPU matrix_exp or full-model optimizer is required.
This module is never used by the serving forward path.
"""
from __future__ import annotations

import torch

from .optr_calibration import _matrices, _sample


def _bf16(value):
    # RNE forward / STE backward. Finiteness is checked on the complete loss
    # and gradient, avoiding an NPU->host synchronization at every rounding.
    return value + (value.to(torch.bfloat16).float() - value).detach()


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
    index = min(int(float(torch.tensor(clip, dtype=torch.float32)) * value.shape[-1]), value.shape[-1] - 1)
    threshold = value.detach().abs().sort(dim=-1).values[..., index:index + 1]
    clipped = torch.minimum(torch.maximum(value, -threshold), threshold)
    low = clipped.amin(-1, keepdim=True)
    scale = (clipped.amax(-1, keepdim=True) - low).clamp_min(1e-8) / 3.
    zero = -low / scale
    unrounded = clipped / scale + zero
    code = (unrounded + .5).floor().clamp(0, 3)
    return (unrounded + (code - unrounded).detach() - zero) * scale


def _effective(value, rotation, exact, clip):
    approximation = quant_int2(value @ rotation, clip) @ rotation.T
    return torch.where(exact[:, None], value, approximation)


def _probabilities(row, key):
    logits = (row["q"] @ key.T) * row["scale"]
    return logits.masked_fill(row["causal"], -torch.inf).softmax(-1)


def _key_probabilities(rotation, row, policy):
    """The writer rounds centered K and Q/K rotations before QK.

    A trace contains queries from several forwards. Each query keeps the
    exact cached tail/current belonging to its own forward, irrespective of
    later rows saved in the same trace. No inverse-K shortcut across BF16.
    """
    rb = _bf16(rotation)
    query = _bf16(row["q"] @ rb)
    key = _bf16(row["k"] @ rb)
    quantized = _bf16(quant_int2(key, policy["k_clip"]))
    exact_logits = query @ key.T
    quantized_logits = query @ quantized.T
    logits = torch.where(row["exact"], exact_logits, quantized_logits) * row["scale"]
    return logits.masked_fill(row["causal"], -torch.inf).softmax(-1)


def _projected_mse(error, row):
    # Algebraically ||(error * gate) W_O^T||^2 / hidden. Retain independent
    # GQA heads here, as upstream; cross-head terms are checked by the caller.
    z = error * row["gate"]
    return ((z @ row["gram"]) * z).sum(-1).mean()


def prepare_samples(samples, mean, policy, device):
    """Stage each complete history once; cache teacher P and W_O^T W_O."""
    rows, grams = [], {}
    with torch.no_grad():
        for sample in samples:
            q = sample["q"].float().to(device).transpose(0, 1).contiguous()
            raw_key = sample["k"].float().to(device)
            k = _bf16(raw_key - _bf16(mean.to(device)))
            v = sample["v"].float().to(device)
            # Frame metadata is authenticated on CPU. Build masks once here;
            # no NPU int64 minimum/clamp kernels or per-step host round trips.
            positions = sample["positions"]
            contexts = sample["contexts"]
            length = k.shape[0]
            indices = torch.arange(length)
            wo = sample["wo"]
            identity = id(wo)
            if identity not in grams:
                weight = wo.to(device)
                grams[identity] = (weight.transpose(-1, -2) @ weight) / weight.shape[1]
            # Same cached cut as the deployed reader: fixed at forward start.
            # Tokens >= context (current/future) are exact; causality masks
            # future tokens. context=0 therefore has no INT2 history.
            cuts = torch.minimum(contexts, (contexts + 1 - policy["recent_tokens"]).clamp_min(policy["sink_tokens"]))
            row = {"q": q, "k": k, "v": v, "scale": k.shape[-1] ** -.5,
                   "gate": sample["gate"].float().to(device).transpose(0, 1).contiguous(),
                   "gram": grams[identity], "request_id": sample["request_id"],
                   "causal": (indices[None, :] > positions[:, None]).to(device),
                   "exact": ((indices[None, :] < policy["sink_tokens"]) |
                             (indices[None, :] >= cuts[:, None])).to(device)}
            row["teacher_p"] = _probabilities(row, raw_key)
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
    rb = _bf16(rotation)
    value = _bf16(row["v"] @ rb)
    quantized = _bf16(quant_int2(value, policy["v_clip"]))
    # Two cached probability partitions replace query-dependent [M,L,D]
    # materialization. No QK/softmax or frozen-side products per V step.
    output = row["selected_k_quant_p"] @ quantized + row["selected_k_exact_p"] @ value
    output = _bf16(_bf16(output) @ rb.T)
    return _projected_mse(output - row["conditional_teacher_v"], row)


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
             progress=None, device="cpu"):
    """Official sequential K/V learning and best held-out selection.

    Native gate and sparse query sampling are explicit adaptations. The
    candidate is subsequently subjected to unchanged all-head BF16 replay;
    these FP32 proxy losses must never be labelled normalized output errors.
    """
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
    train = prepare_samples(train_samples, mean, policy, device)
    held = prepare_samples(validation_samples, mean, policy, device)
    rk, rv = base_k.clone(), base_v.clone()
    decisions = []
    for phase, origin in (("k", base_k), ("v", base_v)):
        emit("phase_start", phase=phase, steps=policy["steps"])
        if phase == "v":
            with torch.no_grad():
                rotation = rk.to(device)
                for row in (*train, *held):
                    row["selected_k_p"] = _key_probabilities(rotation, row, policy)
                    row["selected_k_exact_p"] = row["selected_k_p"] * row["exact"]
                    row["selected_k_quant_p"] = row["selected_k_p"].masked_fill(row["exact"], 0)
                    row["conditional_teacher_v"] = row["selected_k_p"] @ row["v"]
                    # No K quantization, QK or softmax in any V optimization step.
                    del row["teacher_p"], row["teacher_logp"], row["selected_k_p"]
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
            loss.backward()
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
        "matrix_exp_device": "cpu", "selection": "official_heldout_proxy_then_joint_deployment_gate",
        "model_quality": "not_measured", "npu_equivalence": "not_measured"}
