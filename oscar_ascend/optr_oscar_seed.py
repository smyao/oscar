"""Bounded OSCAR-base initialization for OptR; never a serving operation.

Archive #86/#97/#170/#173/#175/#176. The user explicitly selected the local
oscar_base reference at 9cd2429f. Its FP32 qqt/sst statistics, rank-weighted
shared fit and CPU FP64 U H P_br composition are reproduced here. Reference
trees are not imported. The current native GPQA calibration source is kept;
the first 512 aligned rows replace that reference's short Chinese prompt.
This sampling adaptation is part of the identity, not a quality guarantee.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping

RECIPE_VERSION = 6
SEED_RECIPE = "oscar_base_shared_v1"
SAMPLE_CAP = 512
SOURCE_COMMIT = "9cd2429fb4fb934889fa6f51da5d54396012b5df"
OBJECTIVE = "oscar_base_shared_qqt_sst_r_h_pbr"
SOURCE_FILES = {
    "oscar_ascend/calib.py": "a4db413470e123102b1423bfb933e83a789b417590ebde2ddfd9eb2fc1b5b9b7",
    "tools/gen_rotations.py": "0dfbd105e9ff63018bbaacd93a84e6a8dea1efe3c99a07fa7e2d410e38ab405e",
}


def source_provenance():
    return {"source": "user_selected_local_oscar_base", "commit": SOURCE_COMMIT,
            "files": dict(SOURCE_FILES), "recipe": SEED_RECIPE,
            "statistics_dtype": "float32", "eigensolver_dtype": "float64",
            "fitted_grouping": "layer", "key_centering": "none",
            "query_trace_normalization": False, "diagonal_epsilon": 1e-6,
            "sample_cap": SAMPLE_CAP,
            "sampling": "first_512_aligned_rows_complete_gpqa_prompts",
            "sampling_adaptation": "current_GPQA_source_not_reference_short_Chinese_prompt"}


def seed_profile(base_profile):
    """Separate seed identity; perhead_center below describes storage only."""
    if (not isinstance(base_profile, Mapping) or base_profile.get("recipe_version") != 3 or
            base_profile.get("rotation_variant") != "perhead_center" or
            not base_profile.get("model_fingerprint") or
            type(base_profile.get("max_tokens")) is not int or base_profile["max_tokens"] <= 0):
        raise ValueError("OSCAR-base seed requires an identified recipe3 sampling profile")
    result = copy.deepcopy(dict(base_profile))
    result.pop("official_commit", None)
    result.update(recipe_version=RECIPE_VERSION, rotation_seed=SEED_RECIPE,
                  objective=OBJECTIVE, accumulator_dtype="float32",
                  fitted_grouping="layer", key_centering="none",
                  source_provenance=source_provenance())
    return result


def legacy_base_profile(profile):
    """Recover only the old sampling identity; never reuse old fitted means."""
    if not isinstance(profile, Mapping) or profile.get("rotation_seed") != SEED_RECIPE:
        return copy.deepcopy(profile)
    from .rotation_calibration import OFFICIAL_COMMIT
    result = _profile(profile)
    for key in ("rotation_seed", "fitted_grouping", "key_centering", "source_provenance"):
        result.pop(key)
    result.update(recipe_version=3, objective="qqt_sst_r_h_pbr", accumulator_dtype="float64",
                  official_commit=OFFICIAL_COMMIT)
    return result


def _profile(profile):
    if (not isinstance(profile, Mapping) or profile.get("recipe_version") != RECIPE_VERSION or
            profile.get("rotation_seed") != SEED_RECIPE or profile.get("objective") != OBJECTIVE or
            profile.get("accumulator_dtype") != "float32" or
            profile.get("fitted_grouping") != "layer" or profile.get("key_centering") != "none" or
            profile.get("rotation_variant") != "perhead_center" or
            profile.get("source_provenance") != source_provenance() or
            "official_commit" in profile or not profile.get("model_fingerprint") or
            type(profile.get("max_tokens")) is not int or profile["max_tokens"] <= 0):
        raise ValueError("OSCAR-base seed profile/source differs")
    return json.loads(json.dumps(profile, sort_keys=True, allow_nan=False))


def validate_seed_profile(profile, model_fingerprint):
    result = _profile(profile)
    if result["model_fingerprint"] != model_fingerprint:
        raise ValueError("OSCAR-base seed profile model fingerprint differs")
    return result


def shared_covariance(q, k, v):
    """Literal FP32 finalize_cov order, evaluated on the input's device.

    No FP64 QGram reconstruction, trace normalization or key centering.
    The caller owns the bounded, aligned native sample selection.
    """
    import torch
    if (any(not torch.is_tensor(x) or x.ndim != 3 for x in (q, k, v)) or
            not (q.device == k.device == v.device) or k.shape != v.shape or
            q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or
            q.shape[0] < 1 or q.shape[0] > SAMPLE_CAP or k.shape[1] < 1 or
            q.shape[1] < k.shape[1] or q.shape[1] % k.shape[1] or
            any(x.dtype not in (torch.float32, torch.float16, torch.bfloat16) for x in (q, k, v))):
        raise ValueError("OSCAR-base aligned bounded Q/K/V geometry/dtype differs")
    if any(not bool(torch.isfinite(x).all()) for x in (q, k, v)):
        raise ValueError("OSCAR-base nonfinite native samples")
    n, hk, dim = k.shape
    group = q.shape[1] // hk
    cq = torch.zeros(dim, dim, dtype=torch.float32, device=q.device)
    cv = torch.zeros_like(cq)
    # This is intentionally sqrt-weighted rather than an algebraically
    # equivalent FP64 einsum: reproduce the selected reference's operations.
    for head in range(hk):
        qg = q[:, head * group:(head + 1) * group].float().reshape(-1, dim)
        kh, vh = k[:, head].float(), v[:, head].float()
        qtq = qg.t() @ qg / qg.shape[0]
        qtq = (qtq + qtq.T) / 2
        cq += qtq
        weights = (kh @ qtq * kh).sum(1)
        weights = weights / weights.sum().clamp(min=1e-12) * n
        vw = vh * weights.unsqueeze(1).sqrt()
        sst = vw.t() @ vw / n
        cv += (sst + sst.T) / 2
    cq, cv = cq / hk, cv / hk
    if not bool(torch.isfinite(cq).all() & torch.isfinite(cv).all()):
        raise ValueError("OSCAR-base nonfinite FP32 covariance")
    return {"q": {"cov": cq.detach().cpu(), "count": n},
            "v": {"cov": cv.detach().cpu(), "count": n}}


class BaseSeedAccumulator:
    """Capture only the bounded prefix; count all native calibration rows."""
    def __init__(self, *, head_dim, local_kv_heads, local_q_heads, token_budget, testing=False):
        if (any(type(x) is not int or x <= 0 for x in
                (head_dim, local_kv_heads, local_q_heads, token_budget)) or
                head_dim & (head_dim - 1) or local_q_heads % local_kv_heads):
            raise ValueError("OSCAR-base positive power-of-two D and valid GQA required")
        self.head_dim, self.local_kv_heads = head_dim, local_kv_heads
        self.local_q_heads, self.token_budget = local_q_heads, token_budget
        self.testing, self.count = testing, 0
        self.statistics_count = min(SAMPLE_CAP, token_budget)
        self._rows = {key: [] for key in ("q", "k", "v")}
        self._captured = 0
        self._statistics = None
        self._device = None

    def observe(self, q, k, v):
        import torch
        if self.count == self.token_budget:
            return
        if (any(not torch.is_tensor(x) for x in (q, k, v)) or
                not (q.device == k.device == v.device) or
                q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]):
            raise ValueError("OSCAR-base aligned Q/K/V tensors required")
        if q.device.type != "npu" and not (self.testing and q.device.type == "cpu"):
            raise RuntimeError("OSCAR-base production sampling requires NPU")
        if self._device is not None and self._device != q.device:
            raise ValueError("OSCAR-base native sample device changed")
        if (any(x.dtype not in (torch.float32, torch.bfloat16, torch.float16) for x in (q, k, v)) or
                q.numel() != q.shape[0] * self.local_q_heads * self.head_dim or
                any(x.numel() != x.shape[0] * self.local_kv_heads * self.head_dim for x in (k, v))):
            raise ValueError("OSCAR-base native sample geometry/dtype changed")
        n = min(q.shape[0], self.token_budget - self.count)
        take = min(n, self.statistics_count - self._captured)
        if take:
            self._device = q.device
            for label, value, heads in (("q", q, self.local_q_heads),
                                       ("k", k, self.local_kv_heads), ("v", v, self.local_kv_heads)):
                # clone owns exactly `take` rows: a view must not retain the
                # potentially much larger native forward allocation.
                row = value[:take].detach().reshape(take, heads, self.head_dim).float().clone()
                if not bool(torch.isfinite(row).all()):
                    raise ValueError("OSCAR-base nonfinite native sample")
                self._rows[label].append(row)
            self._captured += take
            if self._captured == self.statistics_count:
                self._statistics = shared_covariance(*(torch.cat(self._rows[key], dim=0)
                                                       for key in ("q", "k", "v")))
                self._rows = None  # Release all owned NPU activation buffers.
        self.count += n

    def export_payload(self):
        if self.count != self.token_budget or self._statistics is None:
            raise ValueError("OSCAR-base calibration token budget not reached")
        return {"count": self.count, "token_budget": self.token_budget,
                "statistics_count": self.statistics_count, "head_dim": self.head_dim,
                "local_kv_heads": self.local_kv_heads, "local_q_heads": self.local_q_heads,
                "gqa_ratio": self.local_q_heads // self.local_kv_heads,
                "source_device": "cpu_test" if self.testing else "npu",
                "test_only": self.testing, "source_provenance": source_provenance(),
                "oscar_base_seed": self._statistics}


def _rotation(stats, dim):
    import torch
    covariance = stats["cov_sum"] / max(1, int(stats["count"]))
    # Reference creates epsilon in FP32, then promotes it for CPU eigh.
    epsilon = 1e-6 * torch.eye(dim, dtype=torch.float32, device="cpu")
    cov64 = (covariance.double() + covariance.double().T) / 2
    values, vectors = torch.linalg.eigh(cov64 + epsilon.double())
    h = torch.ones(1, 1, dtype=torch.float64, device="cpu")
    while h.shape[0] < dim:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0) / math.sqrt(2)
    br = torch.tensor([int(f"{i:0{dim.bit_length()-1}b}"[::-1], 2) for i in range(dim)],
                      dtype=torch.long, device="cpu")
    permutation = torch.zeros(dim, dtype=torch.long, device="cpu")
    permutation[br] = torch.argsort(values, descending=True)
    rotation = vectors @ h @ torch.eye(dim, dtype=torch.float64, device="cpu")[:, permutation]
    return rotation.float().contiguous(), values.float().contiguous()


def _geometry(head_dim, global_q_heads, global_kv_heads, tp_size):
    if (any(type(x) is not int or x <= 0 for x in
            (head_dim, global_q_heads, global_kv_heads, tp_size)) or
            head_dim & (head_dim - 1) or global_q_heads % global_kv_heads or
            global_q_heads % tp_size or global_kv_heads % tp_size):
        raise ValueError("OSCAR-base requires power-of-two D and disjoint TP GQA shards")


def _matrix_digest(layers):
    import torch
    digest = hashlib.sha256()
    for name, layer in sorted(layers.items()):
        digest.update(name.encode())
        for key in ("Rk", "Rv", "k_mean"):
            tensor = layer[key].detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fit_seed_artifact(rank_payloads, profile, *, layer_names, head_dim, global_q_heads,
                      global_kv_heads, tp_size, prompt_sha256, testing=False):
    import torch
    from .rotations import _names
    profile = _profile(profile)
    names = _names(layer_names)
    ids = {int(name.split(".")[2]) for name in names}
    _geometry(head_dim, global_q_heads, global_kv_heads, tp_size)
    if (len(rank_payloads) != tp_size or
            {r.get("rank") for r in rank_payloads} != set(range(tp_size))):
        raise ValueError("OSCAR-base complete unique rank payloads required")
    sums = {layer: {kind: {"cov_sum": torch.zeros(head_dim, head_dim, dtype=torch.float32,
                                                  device="cpu"), "count": 0}
                    for kind in ("q", "v")} for layer in ids}
    counts = set()
    statistics_digest = hashlib.sha256()
    for rank in sorted(rank_payloads, key=lambda row: row["rank"]):
        for key, expected in (("model_fingerprint", profile["model_fingerprint"]),
                              ("prompt_sha256", prompt_sha256), ("tp_size", tp_size),
                              ("global_q_heads", global_q_heads), ("global_kv_heads", global_kv_heads)):
            if rank.get(key) != expected:
                raise ValueError("OSCAR-base rank identity differs: " + key)
        if set(rank.get("layers", {})) != ids:
            raise ValueError("OSCAR-base rank FULL layer coverage differs")
        for layer in sorted(ids):
            data = rank["layers"][layer]
            count = data.get("count")
            if (type(count) is not int or not 0 < count <= profile["max_tokens"] or
                    data.get("token_budget") != count or data.get("statistics_count") != min(SAMPLE_CAP, count) or
                    data.get("head_dim") != head_dim or data.get("local_q_heads") != global_q_heads // tp_size or
                    data.get("local_kv_heads") != global_kv_heads // tp_size or
                    data.get("gqa_ratio") != global_q_heads // global_kv_heads or
                    data.get("source_provenance") != source_provenance()):
                raise ValueError("OSCAR-base sample geometry/source identity differs")
            if data.get("test_only") is not testing or data.get("source_device") != ("cpu_test" if testing else "npu"):
                raise ValueError("OSCAR-base native production/test sampling identity differs")
            counts.add(count)
            rows = data.get("oscar_base_seed", {})
            for kind in ("q", "v"):
                entry = rows.get(kind, {})
                cov, n = entry.get("cov"), entry.get("count")
                if (not torch.is_tensor(cov) or cov.dtype != torch.float32 or
                        cov.device.type != "cpu" or cov.shape != (head_dim, head_dim) or
                        not bool(torch.isfinite(cov).all()) or not torch.equal(cov, cov.T) or
                        type(n) is not int or n != min(SAMPLE_CAP, count)):
                    raise ValueError("OSCAR-base FP32 source covariance missing or invalid")
                aggregate = sums[layer][kind]
                aggregate["cov_sum"] = aggregate["cov_sum"] + cov * n
                aggregate["count"] += n
                statistics_digest.update(f"{rank['rank']}:{layer}:{kind}:{n}".encode())
                statistics_digest.update(cov.contiguous().view(torch.uint8).numpy().tobytes())
    if len(counts) != 1:
        raise ValueError("OSCAR-base aligned rank/layer token counts differ")
    layers = {}
    for name in names:
        layer_id = int(name.split(".")[2])
        rk, _ = _rotation(sums[layer_id]["q"], head_dim)
        rv, _ = _rotation(sums[layer_id]["v"], head_dim)
        layers[name] = {"layer_id": layer_id,
                        "Rk": rk.expand(global_kv_heads, -1, -1).clone().contiguous(),
                        "Rv": rv.expand(global_kv_heads, -1, -1).clone().contiguous(),
                        "k_mean": torch.zeros(global_kv_heads, head_dim, dtype=torch.float32, device="cpu")}
    tokens = next(iter(counts))
    artifact = {"format_version": 2, "source_grouping": "layer_kv_head",
                "rotation_variant": "perhead_center", "global_kv_heads": global_kv_heads,
                "head_dim": head_dim, "model_fingerprint": profile["model_fingerprint"],
                "pr_fingerprint": SOURCE_COMMIT, "objective": OBJECTIVE,
                "calibrated": True, "accumulation_dtype": "float32", "test_only": testing,
                "calibration_profile": profile, "source_provenance": source_provenance(),
                "calibration": {"tokens": tokens, "statistics_tokens_per_rank": min(SAMPLE_CAP, tokens),
                                "prompt_sha256": prompt_sha256, "global_q_heads": global_q_heads,
                                "global_kv_heads": global_kv_heads, "tp_size": tp_size,
                                "statistics_device": "cpu_test" if testing else "npu",
                                "statistics_sha256": statistics_digest.hexdigest()},
                "layers": layers, "matrices_sha256": _matrix_digest(layers)}
    return validate_seed_artifact(artifact, expected_profile=profile, layer_names=names,
                                 head_dim=head_dim, global_q_heads=global_q_heads,
                                 global_kv_heads=global_kv_heads, tp_size=tp_size, testing=testing)


def validate_seed_artifact(artifact, *, expected_profile, layer_names=None, head_dim=None,
                           global_q_heads=None, global_kv_heads=None, tp_size=None,
                           model_fingerprint=None, testing=False):
    import torch
    from .rotations import _names, ORTHOGONALITY_ATOL
    profile = _profile(expected_profile)
    if not isinstance(artifact, Mapping) or artifact.get("calibration_profile") != profile:
        raise ValueError("OSCAR-base artifact profile differs")
    if (artifact.get("format_version") != 2 or artifact.get("source_grouping") != "layer_kv_head" or
            artifact.get("rotation_variant") != "perhead_center" or artifact.get("objective") != OBJECTIVE or
            artifact.get("calibrated") is not True or artifact.get("accumulation_dtype") != "float32" or
            artifact.get("pr_fingerprint") != SOURCE_COMMIT or
            artifact.get("source_provenance") != source_provenance() or
            artifact.get("model_fingerprint") != profile["model_fingerprint"] or
            model_fingerprint not in (None, profile["model_fingerprint"])):
        raise ValueError("OSCAR-base artifact source/model identity differs")
    if type(artifact.get("test_only")) is not bool or (artifact["test_only"] and not testing):
        raise ValueError("OSCAR-base test-only artifact cannot enter production")
    meta = artifact.get("calibration", {})
    dim, heads = artifact.get("head_dim"), artifact.get("global_kv_heads")
    _geometry(dim, meta.get("global_q_heads"), heads, meta.get("tp_size"))
    if (head_dim not in (None, dim) or global_kv_heads not in (None, heads) or
            global_q_heads not in (None, meta["global_q_heads"]) or tp_size not in (None, meta["tp_size"]) or
            meta.get("global_kv_heads") != heads):
        raise ValueError("OSCAR-base artifact geometry differs")
    tokens = meta.get("tokens")
    if (type(tokens) is not int or not 0 < tokens <= profile["max_tokens"] or
            meta.get("statistics_tokens_per_rank") != min(SAMPLE_CAP, tokens) or
            not isinstance(meta.get("prompt_sha256"), str) or not meta["prompt_sha256"] or
            meta.get("statistics_device") != ("cpu_test" if artifact["test_only"] else "npu")):
        raise ValueError("OSCAR-base artifact sampling identity differs")
    for key in ("statistics_sha256",):
        value = meta.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("OSCAR-base source statistics hash missing")
    layers = artifact.get("layers", {})
    names = _names(layers if layer_names is None else layer_names)
    if set(layers) != set(names):
        raise ValueError("OSCAR-base artifact FULL layer coverage differs")
    for name in names:
        layer = layers[name]
        if layer.get("layer_id") != int(name.split(".")[2]):
            raise ValueError("OSCAR-base artifact layer identity differs")
        for key in ("Rk", "Rv"):
            rotation = layer.get(key)
            if (not torch.is_tensor(rotation) or rotation.dtype != torch.float32 or
                    rotation.device.type != "cpu" or rotation.shape != (heads, dim, dim) or
                    not bool(torch.isfinite(rotation).all()) or
                    not torch.equal(rotation, rotation[:1].expand_as(rotation))):
                raise ValueError("OSCAR-base requires finite identical shared rotations for all heads")
            if float((rotation[0].T @ rotation[0] - torch.eye(dim, device="cpu")).abs().max()) > ORTHOGONALITY_ATOL:
                raise ValueError("OSCAR-base rotation is not orthogonal")
        mean = layer.get("k_mean")
        if (not torch.is_tensor(mean) or mean.dtype != torch.float32 or mean.device.type != "cpu" or
                mean.shape != (heads, dim) or bool(torch.count_nonzero(mean))):
            raise ValueError("OSCAR-base requires exactly zero key mean")
    if artifact.get("matrices_sha256") != _matrix_digest(layers):
        raise ValueError("OSCAR-base seed matrix digest differs")
    return dict(artifact)
