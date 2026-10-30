"""Offline-only output-aware refinement of the recipe3 rotation matrices.

Archive #13-20/#126: output, codec metadata and source boundaries all matter.
Archive #86: this head-level fitter receives target FULL traces, never MTP/GDN.
Archive #145/#148/#164/#170: CPU replay and a decreasing calibration objective
are not NPU execution, model quality, or performance acceptance. This module
must not be imported by serving. It changes only offline matrix coefficients.
Archive #94/#95/#97: report real fitting boundaries and completed work through
an optional callback; never conceal a stalled fitter behind a timer heartbeat.

The sequential orthogonal correction follows the user-authorized OptR source
at 66d46561b9bdf5d715f591e0f5845a4cbd927dee. Unlike its separate K/V proxies,
both phases minimize the joint post-gate, post-W_O output discrepancy. Replay
models the current group=head_dim writer, FP32 metadata reader, BF16 operands,
KV256 online softmax and single final inverse. CPU reduction and native FIA
are not bitwise equivalents; the declared source partition is a fitting model.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any

import torch


DEFAULT_POLICY = {
    "sink_tokens": 64, "recent_tokens": 256, "k_clip": .96, "v_clip": .92,
    "steps": 16, "lr": .002, "eval_every": 4, "query_batch_size": 8,
    "kv_tile_size": 256, "history_splits": 1,
    "replay_partition": "fixed_splits",
}
REPLAY_CONTRACT = "offline_bf16_rotated_fp32_metadata_source_split_v1"


def _policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    result = {**DEFAULT_POLICY, **policy}
    for name in ("sink_tokens", "steps"):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("recent_tokens", "eval_every", "query_batch_size", "history_splits"):
        if type(result[name]) is not int or result[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if result["kv_tile_size"] != 256:
        raise ValueError("the deployed attention tile size is 256")
    if result["replay_partition"] not in ("fixed_splits", "single_split", "captured_runtime_frames"):
        raise ValueError("unknown replay_partition")
    if result["history_splits"] > 32 or (result["replay_partition"] == "single_split" and result["history_splits"] != 1):
        raise ValueError("invalid fixed source split count")
    for name in ("k_clip", "v_clip"):
        if (isinstance(result[name], bool) or not isinstance(result[name], (int, float))
                or not math.isfinite(result[name]) or not 0 < result[name] <= 1):
            raise ValueError(f"{name} must be finite and in (0,1]")
    if not isinstance(result["lr"], (int, float)) or not math.isfinite(result["lr"]) or result["lr"] <= 0:
        raise ValueError("lr must be positive and finite")
    return result


def _tensor(name: str, value: Any, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != dtype or tuple(value.shape) != shape):
        raise ValueError(f"{name} requires CPU {dtype} {shape}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")


def _matrices(rk: torch.Tensor, rv: torch.Tensor, mean: torch.Tensor) -> int:
    if not isinstance(mean, torch.Tensor) or mean.ndim != 1 or mean.numel() < 2:
        raise ValueError("mean must be a head vector of dimension >=2")
    dim = mean.numel()
    _tensor("mean", mean, torch.float32, (dim,))
    for name, matrix in (("rk", rk), ("rv", rv)):
        _tensor(name, matrix, torch.float32, (dim, dim))
        error = float((matrix.detach().T @ matrix.detach() - torch.eye(dim)).abs().max())
        if not math.isfinite(error) or error > 5e-4:
            raise ValueError(f"{name} must be orthogonal before BF16 conversion")
    return dim


def _sample(sample: Mapping[str, Any], dim: int) -> None:
    if not isinstance(sample, Mapping) or not isinstance(sample.get("request_id"), str) or not sample["request_id"]:
        raise ValueError("each trace requires an explicit nonempty request_id")
    q, k, wo = sample.get("q"), sample.get("k"), sample.get("wo")
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or q.shape[-1] != dim or min(q.shape) <= 0:
        raise ValueError("q must have nonempty [queries,GQA,dim] shape")
    if not isinstance(k, torch.Tensor) or k.ndim != 2 or k.shape[-1] != dim or k.shape[0] <= 0:
        raise ValueError("k must contain complete [history,dim] from position zero")
    m, group, _ = q.shape
    if not isinstance(wo, torch.Tensor) or wo.ndim != 3 or wo.shape[1] <= 0:
        raise ValueError("wo must be [GQA,hidden,dim]")
    for name in ("q", "gate", "teacher"):
        _tensor(name, sample.get(name), torch.bfloat16, (m, group, dim))
    for name in ("k", "v"):
        _tensor(name, sample.get(name), torch.bfloat16, tuple(k.shape))
    _tensor("wo", wo, torch.float32, (group, wo.shape[1], dim))
    for name in ("positions", "contexts"):
        _tensor(name, sample.get(name), torch.int64, (m,))
    positions, contexts = sample["positions"], sample["contexts"]
    if bool(((contexts < 0) | (contexts > positions) | (positions >= k.shape[0])).any()):
        raise ValueError("require 0 <= cached context <= query position < full history length")
    if bool(((sample["gate"] < 0) | (sample["gate"] > 1)).any()):
        raise ValueError("gate must be the captured sigmoid, in [0,1]")
    if len(set(positions.tolist())) != m:
        raise ValueError("duplicate query positions in a request trace")
    if "history_splits" in sample:
        _tensor("history_splits", sample["history_splits"], torch.int64, (m,))
        if bool(((sample["history_splits"] < 1) | (sample["history_splits"] > 32)).any()):
            raise ValueError("sample history_splits must be in [1,32]")
    if "current_ends" in sample:
        _tensor("current_ends", sample["current_ends"], torch.int64, (m,))
        if bool(((sample["current_ends"] <= positions) | (sample["current_ends"] > k.shape[0])).any()):
            raise ValueError("current_ends must be the actual forward end, after each query and within history")
    if group > 128:
        raise ValueError("GQA group exceeds the deployed query tile")


def _bf16(value: torch.Tensor) -> torch.Tensor:
    """BF16 RNE in the forward pass, identity derivative for offline fitting."""
    rounded = value.to(torch.bfloat16).float()
    if not bool(torch.isfinite(rounded).all()):
        raise ValueError("nonfinite BF16 replay operand")
    return value + (rounded - value).detach()


def quantize_dequantize(value: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    """One head-wide INT2 group, half-up codes and FP32 normalized-zero read.

    Inputs represent already-rounded BF16 rotated rows. The forward values
    match writer -> mode0 reader; the code rounding uses a straight-through
    derivative. No packing is needed for an offline differentiable objective.
    """
    if value.device.type != "cpu" or value.dtype != torch.float32 or value.ndim != 2:
        raise ValueError("quantizer requires CPU FP32 [tokens,dim]")
    if not math.isfinite(clip_ratio) or not 0 < clip_ratio <= 1 or not bool(torch.isfinite(value).all()):
        raise ValueError("finite quantizer rows and clip in (0,1] required")
    dim = value.shape[-1]
    # C++ receives a float, so freeze its binary32 percentile before indexing.
    ratio = float(torch.tensor(clip_ratio, dtype=torch.float32))
    index = min(int(ratio * dim), dim - 1)
    threshold = value.detach().abs().kthvalue(index + 1, dim=-1).values[:, None]
    clipped = torch.minimum(torch.maximum(value, -threshold), threshold)
    low, high = clipped.amin(-1, keepdim=True), clipped.amax(-1, keepdim=True)
    scale = (high - low).clamp_min(1e-8) / 3.0
    zero = -low / scale
    unrounded = clipped / scale + zero
    code = torch.floor(unrounded + .5).clamp(0, 3)
    code_ste = unrounded + (code - unrounded).detach()
    return _bf16((code_ste - zero) * scale)


def _tiles(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           positions: torch.Tensor, key_positions: torch.Tensor, *, history_end=None) -> tuple[torch.Tensor, torch.Tensor]:
    m, group, dim = q.shape
    maximum = torch.full((m, group), -math.inf)
    denominator = torch.zeros_like(maximum)
    accumulator = torch.zeros((m, group, dim))
    for begin in range(0, k.shape[0], 256):
        end = min(begin + 256, k.shape[0])
        score = torch.einsum("mgd,ld->mgl", q, k[begin:end]) * (dim ** -.5)
        if not bool(torch.isfinite(score).all()):
            raise ValueError("nonfinite attention scores before causal masking")
        visible = key_positions[None, begin:end] <= positions[:, None]
        if history_end is not None:
            visible = visible & (key_positions[None, begin:end] < history_end)
        score = score.masked_fill(~visible[:, None], -math.inf)
        new_max = torch.maximum(maximum, score.amax(-1))
        safe_max = torch.where(torch.isfinite(new_max), new_max, torch.zeros_like(new_max))
        alpha = torch.exp(maximum - safe_max)
        probability = torch.exp(score - safe_max[..., None])
        denominator = denominator * alpha + probability.sum(-1)
        accumulator = accumulator * alpha[..., None] + torch.einsum("mgl,ld->mgd", _bf16(probability), v[begin:end])
        maximum = new_max
    live = denominator > 0
    safe_sum = torch.where(live, denominator, torch.ones_like(denominator))
    output = accumulator / safe_sum[..., None]
    lse = torch.where(live, maximum + torch.log(safe_sum), torch.full_like(maximum, -math.inf))
    return output, lse


def _replay(rk: torch.Tensor, rv: torch.Tensor, mean: torch.Tensor,
            sample: Mapping[str, Any], policy: Mapping[str, Any], *, frozen=None) -> torch.Tensor:
    frozen = {} if frozen is None else frozen
    rk_b, rv_b = _bf16(rk), _bf16(rv)
    q = frozen["q"] if "q" in frozen else _bf16(sample["q"].float() @ rk_b)
    k = (frozen["k"] if "k" in frozen else
         _bf16(_bf16(sample["k"].float() - _bf16(mean)) @ rk_b))
    v = frozen["v"] if "v" in frozen else _bf16(sample["v"].float() @ rv_b)
    k_quant = frozen["k_quant"] if "k_quant" in frozen else quantize_dequantize(k, policy["k_clip"])
    v_quant = frozen["v_quant"] if "v_quant" in frozen else quantize_dequantize(v, policy["v_clip"])
    outputs = torch.empty_like(q)
    if policy["replay_partition"] == "captured_runtime_frames":
        if "history_splits" not in sample or "current_ends" not in sample:
            raise ValueError("captured runtime replay requires history_splits and current_ends")
        split_counts = sample["history_splits"]
    else:
        split_counts = torch.full_like(sample["positions"], policy["history_splits"])
    if bool((split_counts > 1).any()) and "current_ends" not in sample:
        raise ValueError("multiple source splits require captured current_ends")
    query_tile = 128 // q.shape[1]
    forward_ends = sample.get("current_ends", torch.full_like(sample["positions"], k.shape[0]))
    producer_ends = torch.minimum(forward_ends,
        sample["contexts"] + ((sample["positions"] - sample["contexts"]) // query_tile + 1) * query_tile)
    # All queries from one forward share one exact-window cut; never slide it
    # with each query or infer a new cut from the final stored history length.
    groups = sorted(set(zip(sample["contexts"].tolist(), producer_ends.tolist(), split_counts.tolist())))
    for context, producer_end, splits in groups:
        selected = torch.where((sample["contexts"] == context) & (producer_ends == producer_end) & (split_counts == splits))[0]
        qp = sample["positions"][selected]
        if context == 0:
            # Native FIA's internals are closed. This is its declared dense
            # BF16-visible oracle, not the INT2 kernel's unnormalized-P cast.
            score = torch.einsum("mgd,ld->mgl", q[selected], k[:producer_end]) * q.shape[-1] ** -.5
            score = score.masked_fill(torch.arange(producer_end)[None, None] > qp[:, None, None], -math.inf)
            current = torch.einsum("mgl,ld->mgd", score.softmax(-1), v[:producer_end])
            outputs[selected] = _bf16(_bf16(current) @ rv_b.T)
            continue
        cut = min(context, max(policy["sink_tokens"], context + 1 - policy["recent_tokens"]))
        sink_end = min(policy["sink_tokens"], context)
        # Runtime partitions the full cached history before masking its exact
        # tail. Truncating first changes every split/tile boundary at S>1.
        history = torch.arange(sink_end, context)
        window = torch.cat((torch.arange(sink_end), torch.arange(cut, context)))
        current = torch.arange(context, producer_end)
        partials, lses = [], []
        for kind, (indices, kk, vv) in enumerate(((history, k_quant, v_quant), (window, k, v), (current, k, v))):
            width = (indices.numel() + splits - 1) // splits
            for split in range(splits):
                si = indices[split * width:(split + 1) * width]
                part, lse = _tiles(q[selected], kk[si], vv[si], qp, si,
                                   history_end=cut if kind == 0 else None)
                partials.append(part)
                lses.append(lse)
        stacked_lse = torch.stack(lses)
        maximum = stacked_lse.amax(0)
        if not bool(torch.isfinite(maximum).all()):
            raise ValueError("query has no finite visible source")
        weights = torch.exp(stacked_lse - maximum)
        weights = weights / weights.sum(0)
        merged = (torch.stack(partials) * weights[..., None]).sum(0)
        outputs[selected] = _bf16(_bf16(merged) @ rv_b.T)
    return outputs


def _batches(sample: Mapping[str, Any], size: int):
    for begin in range(0, sample["q"].shape[0], size):
        piece = dict(sample)
        for name in ("q", "gate", "teacher", "positions", "contexts"):
            piece[name] = sample[name][begin:begin + size]
        if "current_ends" in sample:
            piece["current_ends"] = sample["current_ends"][begin:begin + size]
        if "history_splits" in sample:
            piece["history_splits"] = sample["history_splits"][begin:begin + size]
        yield piece


def _training_batches(samples, phase, rk, rv, mean, policy):
    """Cache the frozen side's history rotation, never a serving tensor.

    This saves repeated D-by-D transforms during offline optimization. QK
    probabilities remain microbatched rather than retaining M*G*history for
    every sample; no training attention graph survives a backward call.
    """
    batches = []
    with torch.no_grad():
        for sample in samples:
            if phase == "k":
                value = _bf16(sample["v"].float() @ _bf16(rv))
                common = {"v": value, "v_quant": quantize_dequantize(value, policy["v_clip"])}
                query = None
            else:
                key = _bf16(_bf16(sample["k"].float() - _bf16(mean)) @ _bf16(rk))
                common = {"k": key, "k_quant": quantize_dequantize(key, policy["k_clip"])}
                query = _bf16(sample["q"].float() @ _bf16(rk))
            offset = 0
            for batch in _batches(sample, policy["query_batch_size"]):
                cached = dict(common)
                size = batch["q"].shape[0]
                if query is not None:
                    cached["q"] = query[offset:offset + size]
                cached["target"] = project_output(batch["teacher"], batch)
                batches.append((batch, cached))
                offset += size
    return batches


def replay_sample(rk: torch.Tensor, rv: torch.Tensor, mean: torch.Tensor,
                  sample: Mapping[str, Any], policy: Mapping[str, Any]) -> torch.Tensor:
    """Offline attention [M,G,D] before gate, BF16 values carried as FP32.

    ``sample`` requires request_id, k/v BF16[L,D], q/gate/teacher BF16[M,G,D],
    wo FP32[G,hidden,D], positions/contexts int64[M], all CPU. ``gate`` is
    sigmoid(gate), and ``teacher`` is native attention before that gate.
    Histories start at absolute position zero and are complete, not subsampled.
    Optional current_ends int64[M] records the actual forward end; mandatory
    for multiple source splits. Optional history_splits int64[M] supplies real
    prefill/decode counts when replay_partition='captured_runtime_frames'.
    Fixed fitting uses policy history_splits. Without ends,
    a single-split sample assumes its stored history ends at the forward end.
    """
    dim = _matrices(rk, rv, mean)
    _sample(sample, dim)
    p = _policy(policy)
    # Final all-head validation previously rotated/quantized the entire 64K
    # history again for each query microbatch. Shapes and arithmetic stay
    # identical; share only the query-independent history inside this call.
    key = _bf16(_bf16(sample["k"].float() - _bf16(mean)) @ _bf16(rk))
    value = _bf16(sample["v"].float() @ _bf16(rv))
    history = {"k": key, "v": value,
               "k_quant": quantize_dequantize(key, p["k_clip"]),
               "v_quant": quantize_dequantize(value, p["v_clip"])}
    return torch.cat([_replay(rk, rv, mean, batch, p, frozen=history)
                      for batch in _batches(sample, p["query_batch_size"])])


def project_output(attention: torch.Tensor, sample: Mapping[str, Any]) -> torch.Tensor:
    """Sum all GQA query-head contributions BEFORE squaring the error.

    Gate multiplication rounds to BF16 as in native Qwen. Projection is an
    FP32 metric using the actual W_O coefficients, not native BF16 GEMM replay.
    """
    gated = _bf16(attention.float() * sample["gate"].float())
    return torch.einsum("mgd,ghd->mh", gated, sample["wo"])


def _metric_batches(samples, policy):
    """Cache invariant teacher projection with the original microbatch shape.

    Keep the same batch boundaries and FP64 scalar reduction order as the
    uncached metric. Projection over a whole sample instead can select another
    BLAS reduction and must not replace this operation for equivalence.
    """
    result = []
    with torch.no_grad():
        for sample in samples:
            batches = []
            for batch in _batches(sample, policy["query_batch_size"]):
                expected = project_output(batch["teacher"], batch)
                energy = float(expected.double().square().sum())
                batches.append((batch, expected, energy, {}))
            result.append((sample, {}, batches))
    return result


def _phase_metric_batches(prepared, phase, rk, rv, mean, policy):
    """Reuse only the phase's fixed side across validation candidates.

    Rotated histories are CPU-only and scoped to one head/phase. The variable
    side is built once per sample during each metric call, then freed; there
    is no candidate-wide cache of complete K/V pairs or attention matrices.
    """
    result = []
    with torch.no_grad():
        for sample, _, batches in prepared:
            if phase == "k":
                value = _bf16(sample["v"].float() @ _bf16(rv))
                common = {"v": value, "v_quant": quantize_dequantize(value, policy["v_clip"])}
            else:
                key = _bf16(_bf16(sample["k"].float() - _bf16(mean)) @ _bf16(rk))
                common = {"k": key, "k_quant": quantize_dequantize(key, policy["k_clip"])}
            cached_batches = []
            for batch, expected, energy, _ in batches:
                # Query projection retains the original metric microbatch
                # shape even when its K rotation is fixed for the V phase.
                cached = {} if phase == "k" else {"q": _bf16(batch["q"].float() @ _bf16(rk))}
                cached_batches.append((batch, expected, energy, cached))
            result.append((sample, common, cached_batches))
    return result


def _metric(rk, rv, mean, samples, policy, *, prepared=None, sample_progress=None) -> dict[str, Any]:
    prepared = _metric_batches(samples, policy) if prepared is None else prepared
    requests = {}
    for sample_index, (sample, fixed, batches) in enumerate(prepared):
        request = requests.setdefault(sample["request_id"], {"squared_error": 0., "teacher_energy": 0.})
        # Rowwise history transforms use the same full [L,D] shapes as before,
        # but are no longer repeated for every selected-query microbatch.
        history = dict(fixed)
        if "k" not in history:
            key = _bf16(_bf16(sample["k"].float() - _bf16(mean)) @ _bf16(rk))
            history.update(k=key, k_quant=quantize_dequantize(key, policy["k_clip"]))
        if "v" not in history:
            value = _bf16(sample["v"].float() @ _bf16(rv))
            history.update(v=value, v_quant=quantize_dequantize(value, policy["v_clip"]))
        for batch, expected, energy, cached in batches:
            actual = project_output(_replay(rk, rv, mean, batch, policy, frozen={**history, **cached}), batch)
            request["squared_error"] += float((actual - expected).double().square().sum())
            request["teacher_energy"] += energy
        if sample_progress is not None:
            sample_progress({"sample": sample_index + 1, "samples": len(prepared),
                             "request_id": sample["request_id"]})
    errors = []
    for request_id, item in requests.items():
        errors.append({"request_id": request_id, **item,
                       "normalized_error": item["squared_error"] / max(item["teacher_energy"], 1e-20)})
    error = sum(item["squared_error"] for item in errors)
    energy = sum(item["teacher_energy"] for item in errors)
    result = {"squared_error": error, "teacher_energy": energy,
              "normalized_error": error / max(energy, 1e-20), "requests": errors}
    if not all(math.isfinite(result[key]) for key in ("squared_error", "teacher_energy", "normalized_error")):
        raise ValueError("nonfinite output-aware metric")
    return result


def evaluate_head(rk: torch.Tensor, rv: torch.Tensor, mean: torch.Tensor,
                  samples: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    dim = _matrices(rk, rv, mean)
    if not samples:
        raise ValueError("output-aware evaluation requires samples")
    for sample in samples:
        _sample(sample, dim)
    with torch.no_grad():
        return _metric(rk, rv, mean, samples, _policy(policy))


def _improves(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> bool:
    """Strict total improvement and exact nonregression for every request."""
    before = {item["request_id"]: item["normalized_error"] for item in incumbent["requests"]}
    after = {item["request_id"]: item["normalized_error"] for item in candidate["requests"]}
    return (before.keys() == after.keys()
            and candidate["normalized_error"] < incumbent["normalized_error"]
            and all(after[key] <= before[key] for key in before))


def fit_head(base_k: torch.Tensor, base_v: torch.Tensor, mean: torch.Tensor,
             train_samples: Sequence[Mapping[str, Any]], validation_samples: Sequence[Mapping[str, Any]],
             *, policy: Mapping[str, Any], progress: Callable[[dict[str, Any]], None] | None = None,
             ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit K then V with fixed online operation count and a baseline candidate.

    Independent test samples must be evaluated by the caller only after this
    selection. A baseline selection is a valid reported optimizer outcome; it
    must never be presented as an accuracy improvement or hidden runtime bypass.
    ``progress`` receives actual phase/step/validation boundaries, including
    each completed validation sample. It is synchronous, optional, and absent
    from the policy/certificate; callback errors propagate to the caller.
    """
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    def emit(event, **fields):
        if progress is not None:
            progress({"event": event, **fields})

    dim = _matrices(base_k, base_v, mean)
    p = _policy(policy)
    if not train_samples or not validation_samples:
        raise ValueError("separate nonempty training and validation samples required")
    for sample in (*train_samples, *validation_samples):
        _sample(sample, dim)
    train_ids = {sample["request_id"] for sample in train_samples}
    validation_ids = {sample["request_id"] for sample in validation_samples}
    if train_ids & validation_ids:
        raise ValueError("training and validation request identities overlap")
    rk, rv = base_k.detach().clone(), base_v.detach().clone()
    emit("baseline_start")
    with torch.no_grad():
        prepared_metrics = _metric_batches(validation_samples, p)
        baseline = _metric(rk, rv, mean, validation_samples, p, prepared=prepared_metrics,
                           sample_progress=lambda fields: emit("baseline_sample_done", **fields))
        energy = sum(float(project_output(s["teacher"], s).double().square().sum()) for s in train_samples)
    emit("baseline_done", normalized_error=baseline["normalized_error"])
    normalizer = max(energy, 1e-20)
    best = baseline
    decisions = []
    for phase in ("k", "v"):
        emit("phase_start", phase=phase, steps=p["steps"])
        origin = (rk if phase == "k" else rv).detach().clone()
        correction = torch.zeros((dim, dim), requires_grad=True)
        optimizer = torch.optim.Adam([correction], lr=p["lr"])
        batches = _training_batches(train_samples, phase, rk, rv, mean, p)
        validation_batches = _phase_metric_batches(prepared_metrics, phase, rk, rv, mean, p)
        for step in range(p["steps"]):
            optimizer.zero_grad(set_to_none=True)
            for batch, cached in batches:
                # Recompute the small exp graph per query microbatch to
                # release every attention graph after gradient accumulation.
                matrix = origin @ torch.matrix_exp(correction - correction.T)
                trial_k, trial_v = (matrix, rv) if phase == "k" else (rk, matrix)
                prediction = project_output(_replay(trial_k, trial_v, mean, batch, p, frozen=cached), batch)
                target = cached["target"]
                loss = (prediction - target).square().sum() / normalizer
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite output-aware training loss")
                loss.backward()
            if correction.grad is None or not bool(torch.isfinite(correction.grad).all()):
                raise ValueError("nonfinite or missing rotation gradient")
            optimizer.step()
            emit("step_done", phase=phase, step=step + 1, steps=p["steps"])
            if (step + 1) % p["eval_every"] and step + 1 != p["steps"]:
                continue
            emit("validation_start", phase=phase, step=step + 1, steps=p["steps"])
            with torch.no_grad():
                matrix = origin @ torch.matrix_exp(correction - correction.T)
                trial_k, trial_v = (matrix, rv) if phase == "k" else (rk, matrix)
                _matrices(trial_k, trial_v, mean)
                candidate = _metric(trial_k, trial_v, mean, validation_samples, p,
                    prepared=validation_batches, sample_progress=lambda fields:
                    emit("validation_sample_done", phase=phase, step=step + 1, **fields))
                accepted = _improves(candidate, best)
                decisions.append({"phase": phase, "step": step + 1,
                                  "normalized_error": candidate["normalized_error"], "accepted": accepted})
                if accepted:
                    best = candidate
                    rk, rv = trial_k.detach().clone(), trial_v.detach().clone()
            emit("validation_done", phase=phase, step=step + 1, steps=p["steps"],
                 normalized_error=candidate["normalized_error"], accepted=accepted)
        del validation_batches, batches
        emit("phase_done", phase=phase, normalized_error=best["normalized_error"])
    emit("fit_done", accepted=_improves(best, baseline), normalized_error=best["normalized_error"])
    return rk.contiguous(), rv.contiguous(), {
        "accepted": _improves(best, baseline), "baseline": baseline, "selected": best,
        "candidates": decisions, "policy": p, "replay_contract": REPLAY_CONTRACT,
        "objective": "joint_kv_bf16_gate_fp32_wo_sum_of_gqa_heads",
        "selection": "strict_total_and_per_request_nonregression",
        "model_quality": "not_measured", "npu_equivalence": "not_measured",
    }
