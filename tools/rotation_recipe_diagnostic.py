"""Archive G19-G24/#23-25/#86/#97: offline K-recipe sample proxies only.

Reads retained capture rows, never loads a model/NPU or edits a PT/config.
Official fit_oscar2_variants.py: sym(M_q), unit mean eigenvalue, eigh, U H Pbr.
The existing production fitter inlines that composition; this small CPU-only
diagnostic repeats it without importing SGLang or changing production math.
K reconstruction errors are calibration-sample proxies, not attention loss,
LongBench scores, or a prediction that changing the recipe restores quality.
"""
from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _basis(torch, covariance):
    covariance = (covariance + covariance.T) / 2
    average = torch.trace(covariance) / covariance.shape[0]
    if not bool(torch.isfinite(average)) or float(average) <= 0:
        raise ValueError("per-head query covariance has no positive finite trace")
    covariance = covariance / average
    values, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    dim = covariance.shape[0]
    h = torch.ones((1, 1), dtype=torch.float64, device="cpu")
    while h.shape[0] < dim:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0) / math.sqrt(2)
    bits = dim.bit_length() - 1
    br = torch.tensor([int(f"{i:0{bits}b}"[::-1], 2) for i in range(dim)], device="cpu")
    permutation = torch.empty(dim, dtype=torch.long, device="cpu")
    permutation[br] = torch.argsort(values, descending=True)
    return (vectors @ h)[:, permutation].float()


def _metrics(torch, original, restored, rotation_only, covariance):
    error = restored.double() - original.double()
    def proxy(delta, centered=True):
        if centered:
            delta = delta - delta.mean(0)
        return float(((delta @ covariance) * delta).sum(-1).mean() / delta.shape[-1])
    return {"mse": float(error.square().mean()),
            "relative_l2": float((error.square().sum() / original.double().square().sum().clamp_min(1e-30)).sqrt()),
            "score_error_proxy": proxy(error), "raw_score_error_proxy": proxy(error, False),
            "rotation_only_score_error_proxy": proxy(rotation_only.double() - original.double()),
            "incremental_quantization_score_error_proxy": proxy(restored.double() - rotation_only.double())}


def compare(artifact, config=None, *, capture_manifest=None, rank_paths=None, max_samples=1024):
    """Return JSON-safe per-head V1/perhead/center diagnostics from saved rows.

    Each head uses at most 1024 equally spaced original K rows. Its mean and
    query covariance use all captured rows. Mean loading, K subtraction,
    forward/inverse transforms and outputs follow the BF16 visible sequence.
    Score proxies use M_q, not saved queries: centered error removes the
    common key-error offset whose logit shift cancels in softmax.
    """
    previous = os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    try:
        import torch
    finally:
        if previous is None:
            os.environ.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
        else:
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = previous
    from .official_reference import dequantize_rows, quantize_rows, rotate_bf16
    if not isinstance(artifact, Mapping) or type(max_samples) is not int or not 1 <= max_samples <= 1024:
        raise ValueError("artifact mapping and max_samples in [1,1024] required")
    capture = capture_manifest if capture_manifest is not None else artifact.get("capture_manifest")
    if not isinstance(capture, Mapping) or capture.get("status") != "native_post_rope_capture_complete":
        raise ValueError("retained native capture manifest/rows are unavailable")
    fingerprint = artifact.get("model_fingerprint")
    if not fingerprint or capture.get("model_fingerprint") != fingerprint:
        raise ValueError("capture and artifact model fingerprints differ")
    if capture.get('calibration_profile') != artifact.get('calibration_profile'):
        raise ValueError('capture and artifact calibration profiles differ')
    clip = float((config or {}).get('k_clip_ratio', .96))
    if not math.isfinite(clip) or not 0 <= clip <= 1:
        raise ValueError('finite K clip ratio in [0,1] required')
    layers = {entry["layer_id"]: entry for entry in artifact.get("layers", {}).values()}
    if not layers or len(layers) != len(artifact["layers"]) or set(layers) != set(capture.get("layer_ids", [])):
        raise ValueError("capture and artifact FULL layer identities differ")
    paths = list(rank_paths if rank_paths is not None else capture.get("rank_paths", []))
    tp, q_heads, kv_heads = (capture.get(key) for key in ("tp_size", "global_q_heads", "global_kv_heads"))
    if any(type(x) is not int or x <= 0 for x in (tp, q_heads, kv_heads)) or kv_heads % tp or q_heads % tp:
        raise ValueError("diagnostic requires disjoint native TP Q/KV shards")
    if len(paths) != tp:
        raise ValueError("retained capture requires every TP rank path")
    hashes = capture.get("rank_sha256")
    if hashes is not None and not isinstance(hashes, Mapping):
        raise ValueError("capture rank_sha256 must be a path-to-SHA256 mapping")
    results, seen = [], set()
    for index, supplied in enumerate(paths):
        path = Path(supplied).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"retained calibration rank rows unavailable: {path}")
        if hashes is not None:
            recorded = capture.get("rank_paths", paths)[index]
            expected = hashes.get(str(supplied), hashes.get(str(recorded)))
            if not expected or _sha256(path) != expected:
                raise ValueError(f"capture rank SHA256 mismatch: {path}")
        rank = torch.load(path, map_location="cpu", weights_only=True)
        r = rank.get("rank")
        if type(r) is not int or not 0 <= r < tp or r in seen:
            raise ValueError("retained capture has missing/duplicate TP rank identity")
        seen.add(r)
        for key, expected in (("model_fingerprint", fingerprint), ("tp_size", tp),
                              ("global_q_heads", q_heads), ("global_kv_heads", kv_heads),
                              ("prompt_sha256", capture.get("prompt_sha256"))):
            if rank.get(key) != expected:
                raise ValueError("retained rank identity differs: " + key)
        if set(rank.get("layers", {})) != set(layers):
            raise ValueError("retained rank FULL layer coverage differs")
        for layer_id, data in sorted(rank["layers"].items()):
            n, dim, hk, hq = (data[key] for key in ("count", "head_dim", "local_kv_heads", "local_q_heads"))
            if (n <= 0 or n != data["token_budget"] or n != capture["token_count"] or
                    dim not in (64, 128, 256) or (hk, hq) != (kv_heads // tp, q_heads // tp) or
                    data["gqa_ratio"] != hq // hk or hq % hk):
                raise ValueError("retained sample geometry/token coverage differs")
            gram, keys, values, shared = data["q_gram"], data["k"], data["v"], layers[layer_id]["Rk"]
            if (gram.shape != (hk, dim, dim) or gram.dtype != torch.float64 or keys.shape != (n, hk, dim)
                    or values.shape != keys.shape or shared.shape != (dim, dim)):
                raise ValueError("retained sample or V1 rotation tensor geometry differs")
            if any(x.device.type != "cpu" or not bool(torch.isfinite(x).all()) for x in (gram, keys, values, shared)):
                raise ValueError("diagnostic inputs must be finite CPU tensors")
            count = min(n, max_samples)
            ids = torch.arange(count, device="cpu") * (n - 1) // max(1, count - 1)
            for head in range(hk):
                covariance = gram[head] / (n * data["gqa_ratio"])
                covariance = (covariance + covariance.T) / 2
                all_keys = keys[:, head].double()
                mean = all_keys.mean(0)
                mean_ratio = float(mean.square().sum() / all_keys.square().sum(-1).mean().clamp_min(1e-30))
                sample = keys[ids, head].bfloat16()
                basis = _basis(torch, covariance)
                variants = {}
                for name, rotation, centered in (("shared_v1", shared, False), ("perhead", basis, False),
                                                  ("perhead_center", basis, True)):
                    shift = mean.float().bfloat16() if centered else torch.zeros(dim, dtype=torch.bfloat16, device="cpu")
                    source = (sample - shift).bfloat16() if centered else sample
                    rotated = rotate_bf16(source, rotation)
                    rotation_only = rotate_bf16(rotated, rotation.T).float() + shift.float()
                    codes, scale, zero = quantize_rows(rotated, clip)
                    variants[name] = {}
                    for mode in (0, 1):
                        decoded = dequantize_rows(codes, scale, zero, mode).bfloat16()
                        restored = rotate_bf16(decoded, rotation.T).float() + shift.float()
                        variants[name]["prefill" if mode == 0 else "decode"] = _metrics(
                            torch, sample, restored, rotation_only, covariance)
                results.append({"layer_id": layer_id, "kv_head": r * hk + head, "rank": r,
                                "calibration_tokens": n, "sample_count": count,
                                "mean_energy_ratio": mean_ratio, "variants": variants})
        del rank
    return {"status": "complete", "scope": "offline_calibration_K_proxies_only", "model_fingerprint": fingerprint,
            "sample_limit_per_head": max_samples, "k_clip_ratio": clip,
            "configured_k_clip_ratio": (config or {}).get("k_clip_ratio", .96),
            "basis": "unit_mean_eig(sym(M_q)); FP64 eigh; U H Pbr; FP32 PT then BF16 operands",
            "score_proxy": "mean(e_centered.T M_q e_centered)/D; common key-error mean removed; not attention loss",
            "query_rotation_rounding": "not_observed_from_qgram; score proxies use query second moments only",
            "reconstruction": "BF16 center subtraction/rotation/QDQ/inverse; add BF16 mean back in FP32 for diagnostics",
            "quality_claim": "No model, attention-KL or LongBench recovery is established by these calibration-sample proxies.",
            "model_or_npu_loaded": False, "production_changed": False, "heads": results}
