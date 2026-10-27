"""Explicit, fingerprinted OSCAR rotation artifacts and sample statistics.

Archive G19/G20/G23, #23-#25: sample identity, finite covariance and solver
validation; #86: target and draft layers are enumerated separately. #96/#97:
lazy torch import and explicit, traceable data-free construction.
#165/#166: bind the effective native sampling configuration to cached pairs;
#86/#97: missing import provenance must never be inferred from the target.
Paper reference: rotation/compute_kv_rotation.py:23-46,93-136,234-312.
Serving rotation computation/validation requires NPU. CPU is an explicit test
option; requested official FP64 offline fitting lives in rotation_calibration.
This module alone does not establish target-NPU correctness or model quality.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


FORMAT_VERSION = 1
# Version the local sampling recipe independently of the pinned upstream math.
# v2 binds actual native BF16 GDN state and the authenticated capture/pair
# execution profile; it does not change the serving or sampling state dtype.
CALIBRATION_RECIPE_VERSION = 2
CENTER_CALIBRATION_RECIPE_VERSION = 3
HEAD_CLIP_CALIBRATION_RECIPE_VERSION = 4
PR_FINGERPRINT = "57286d5d2cb08c3dcd8c17bb59e132d6985e6796"
ORTHOGONALITY_ATOL = 5e-4
EIGEN_RESIDUAL_RTOL = 5e-4
_LAYER = re.compile(r"^model\.layers\.(\d+)\.self_attn\.attn$")


@dataclass(frozen=True)
class DeviceRotation:
    """Persistent, validated FP32 matrices for the out-only AscendC operators.

    Kernels consume transposed matrices for contiguous vector dot products.
    These allocations belong to initialization, never an inference step.
    The exact Hadamard pattern is checked before enabling its butterfly path.
    """

    key_transposed: Any
    value_transposed: Any
    inverse_value_transposed: Any
    hadamard: bool
    key_mean: Any = None
    key_clip_ratio: float | None = None


def prepare_device_rotations(artifact: Mapping[str, Any], *, device: str = "npu",
                             testing: bool = False, tp_rank: int = 0,
                             tp_size: int = 1) -> dict[str, DeviceRotation]:
    """Prepare already validated artifact matrices once, outside graph capture.

    Archive #111: keep rotation inputs/outputs FP32; do not narrow rotated K/V
    to FP16. Archive G27/#38: this transforms neither exact raw window data
    nor prefix state. A forged ``objective=hadamard`` cannot select a wrong
    numerical implementation: both matrices must equal the Sylvester pattern.
    """
    torch, selected = _runtime(device, testing)
    if artifact.get("test_only") and not testing:
        raise ValueError("CPU test rotation artifacts cannot enter production")
    dim = artifact.get("head_dim")
    if dim not in (64, 128, 256):
        raise ValueError("AscendC rotations require head_dim 64/128/256")
    layers = artifact.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError("rotation artifact must contain layer matrices")
    pattern = None
    if artifact.get("objective") == "hadamard":
        # Integer signs are dimension-only metadata. Numerical equality is a
        # one-time device check and cannot introduce a per-token host sync.
        signs = [[-1.0 if (i & j).bit_count() & 1 else 1.0
                  for j in range(dim)] for i in range(dim)]
        pattern = torch.tensor(signs, dtype=torch.float32, device=selected)
    result = {}
    per_head = artifact.get("source_grouping") == "layer_kv_head"
    if per_head:
        heads = artifact.get("global_kv_heads")
        # The deployed Qwen27B TP4 route has exactly one KV head per rank.
        # Never broadcast a global/per-head matrix as a shared rotation.
        if (type(tp_size) is not int or type(tp_rank) is not int or
                not 0 <= tp_rank < tp_size or heads != tp_size):
            raise ValueError("perhead_center currently requires one KV head per TP rank (Qwen27B TP4)")
    for name, entry in layers.items():
        k, v = entry.get("Rk"), entry.get("Rv")
        mean, clip = None, None
        if per_head:
            if any(not torch.is_tensor(x) or tuple(x.shape) != (heads, dim, dim) for x in (k, v)):
                raise ValueError(f"{name}: per-head matrix geometry differs")
            mean = entry.get("k_mean")
            if (not torch.is_tensor(mean) or mean.dtype != torch.float32 or
                    tuple(mean.shape) != (heads, dim) or mean.device != selected or
                    not bool(torch.isfinite(mean).all())):
                raise ValueError(f"{name}: finite per-head key mean required")
            k, v, mean = k[tp_rank].contiguous(), v[tp_rank].contiguous(), mean[tp_rank].contiguous()
            if artifact.get('calibration_profile', {}).get('rotation_k_clip') == 'per_head_calibrated':
                ratios = entry.get('k_clip_ratio')
                if (not torch.is_tensor(ratios) or ratios.dtype != torch.float64 or
                        ratios.device.type != 'cpu' or tuple(ratios.shape) != (heads,) or
                        not bool(torch.isfinite(ratios).all()) or bool(((ratios <= 0) | (ratios > 1)).any())):
                    raise ValueError(f'{name}: finite per-head host clipping constants required')
                clip = float(ratios[tp_rank])
        for label, matrix in (("Rk", k), ("Rv", v)):
            if (not torch.is_tensor(matrix) or matrix.dtype != torch.float32
                    or tuple(matrix.shape) != (dim, dim) or matrix.device != selected
                    or not bool(torch.isfinite(matrix).all())):
                raise ValueError(f"{name}/{label}: expected finite FP32 rotation on {selected}")
        fast = pattern is not None
        if fast:
            for label, matrix in (("Rk", k), ("Rv", v)):
                if not bool(matrix[0, 0] > 0) or not torch.equal(matrix, pattern * matrix[0, 0]):
                    raise ValueError(f"{name}/{label}: hadamard objective does not match exact Sylvester pattern")
        result[name] = DeviceRotation(k.T.contiguous(), v.T.contiguous(), v.contiguous(), fast, mean, clip)
    return result


def model_fingerprint(config: Mapping[str, Any], weight_manifest_sha256: str) -> str:
    """Bind model configuration to a caller-verified weight manifest digest."""
    if not re.fullmatch(r"[a-fA-F0-9]{64}", weight_manifest_sha256):
        raise ValueError("weight_manifest_sha256 must be a full SHA-256 digest")
    data = {"config": config, "weight_manifest_sha256": weight_manifest_sha256.lower()}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _runtime(device: str | Any, testing: bool):
    torch = importlib.import_module("torch")
    device_text = str(device)
    if device_text.startswith("npu"):
        importlib.import_module("torch_npu")
    selected = torch.device(device)
    if selected.type != "npu" and not (testing and selected.type == "cpu"):
        raise RuntimeError("rotation computation requires NPU; CPU is permitted only with testing=True")
    if selected.type == "npu" and selected.index is None:
        selected = torch.device("npu", torch.npu.current_device())
    return torch, selected


def _names(layer_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(layer_names)
    if not names or len(set(names)) != len(names) or any(not _LAYER.fullmatch(n) for n in names):
        raise ValueError("layer names must be unique model.layers.N.self_attn.attn entries")
    return names


def _base_artifact(layer_names: Sequence[str], head_dim: int, model_id: str, pr_id: str,
                   objective: str, testing: bool) -> dict[str, Any]:
    _names(layer_names)
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if not model_id or not pr_id:
        raise ValueError("model and PR fingerprints are mandatory")
    return {"format_version": FORMAT_VERSION, "source_grouping": "layer",
            "objective": objective, "head_dim": head_dim,
            "model_fingerprint": model_id, "pr_fingerprint": pr_id,
            "test_only": testing, "layers": {}}


def validate_artifact(artifact: Mapping[str, Any], *, layer_names: Sequence[str],
                      head_dim: int, model_fingerprint: str,
                      pr_fingerprint: str = PR_FINGERPRINT, device: str = "npu",
                      testing: bool = False, allow_identity: bool = False,
                      require_calibrated: bool = False) -> dict[str, Any]:
    """Validate exact coverage, provenance, fp32 geometry and orthogonality.

    Validation returns matrices on the requested device. A test artifact
    cannot be used in production even if it happens to have matching hashes.
    """
    torch, selected = _runtime(device, testing)
    names = _names(layer_names)
    per_head = (artifact.get("format_version") == 2 and artifact.get("source_grouping") == "layer_kv_head")
    if not per_head and (artifact.get("format_version") != FORMAT_VERSION or artifact.get("source_grouping") != "layer"):
        raise ValueError("unsupported rotation artifact schema")
    heads = artifact.get("global_kv_heads") if per_head else None
    clipped = artifact.get('calibration_profile', {}).get('rotation_k_clip') == 'per_head_calibrated'
    if clipped and (not per_head or artifact.get('calibration_profile', {}).get('recipe_version') != HEAD_CLIP_CALIBRATION_RECIPE_VERSION):
        raise ValueError('per-head clipping requires the recipe-4 center artifact')
    if per_head and (type(heads) is not int or heads <= 0 or artifact.get("rotation_variant") != "perhead_center"):
        raise ValueError("perhead_center artifact requires its global KV-head identity")
    if artifact.get("model_fingerprint") != model_fingerprint:
        raise ValueError("rotation model fingerprint mismatch")
    if artifact.get("pr_fingerprint") != pr_fingerprint:
        raise ValueError("rotation PR fingerprint mismatch")
    if artifact.get("head_dim") != head_dim:
        raise ValueError("rotation head_dim mismatch")
    if artifact.get("test_only") and not testing:
        raise ValueError("CPU test rotation artifacts cannot enter production")
    if not artifact.get("objective"):
        raise ValueError("rotation objective is required")
    if artifact.get("objective") == "optr_post_gate_output_v1":
        # Archive #86/#97: independently authenticated refinement; never
        # relabel fitted constants as the original FP64 qqt/sst checkpoint.
        from .optr_artifacts import validate_refinement_metadata
        validate_refinement_metadata(artifact)
    elif require_calibrated and (artifact.get("calibrated") is not True or
            artifact.get("objective") != "qqt_sst_r_h_pbr" or
            artifact.get("accumulation_dtype") != "float64"):
        raise ValueError("official FP64 calibrated qqt/sst rotation required; data-free cache is not accepted")
    layers = artifact.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != set(names):
        raise ValueError(f"rotation layer coverage mismatch: expected {names}, got {tuple(layers or ())}")
    identity = torch.eye(head_dim, dtype=torch.float32, device=selected)
    result = dict(artifact)
    result["layers"] = {}
    for name in names:
        source = layers[name]
        layer_id = int(_LAYER.fullmatch(name).group(1))
        if not isinstance(source, Mapping) or source.get("layer_id") != layer_id:
            raise ValueError(f"rotation layer id mismatch at {name}")
        entry = {"layer_id": layer_id}
        for key in ("Rk", "Rv"):
            matrix = source.get(key)
            shape = (heads, head_dim, head_dim) if per_head else (head_dim, head_dim)
            if not torch.is_tensor(matrix) or matrix.dtype != torch.float32 or tuple(matrix.shape) != shape:
                raise ValueError(f"{name}/{key}: expected fp32 {shape}")
            matrix = matrix.to(device=selected).contiguous()
            if not bool(torch.isfinite(matrix).all()):
                raise ValueError(f"{name}/{key}: non-finite rotation")
            error = (matrix.transpose(-1, -2) @ matrix - identity).abs().amax()
            if float(error) > ORTHOGONALITY_ATOL:
                raise ValueError(f"{name}/{key}: non-orthogonal rotation, max_error={float(error)}")
            if bool(((matrix - identity) == 0).all(dim=-1).all(dim=-1).any()) and not allow_identity:
                raise ValueError(f"{name}/{key}: identity requires explicit allow_identity=True")
            entry[key] = matrix
        if per_head:
            mean = source.get("k_mean")
            if (not torch.is_tensor(mean) or mean.dtype != torch.float32 or
                    tuple(mean.shape) != (heads, head_dim) or not bool(torch.isfinite(mean).all())):
                raise ValueError(f"{name}/k_mean: finite fp32 [{heads},{head_dim}] required")
            entry["k_mean"] = mean.to(device=selected).contiguous()
        if clipped:
            ratios = source.get('k_clip_ratio')
            if (not torch.is_tensor(ratios) or ratios.dtype != torch.float64 or tuple(ratios.shape) != (heads,)
                    or not bool(torch.isfinite(ratios).all()) or bool(((ratios <= 0) | (ratios > 1)).any())):
                raise ValueError(f'{name}/k_clip_ratio: finite FP64 per-head ratios in (0,1] required')
            if not isinstance(source.get('clip_selection'), Mapping):
                raise ValueError(f'{name}: calibrated clip selection provenance required')
            entry['k_clip_ratio'] = ratios.detach().cpu().contiguous()
            entry['clip_selection'] = source['clip_selection']
        elif 'k_clip_ratio' in source or 'clip_selection' in source:
            raise ValueError('per-head clip fields cannot enter an older recipe')
        result["layers"][name] = entry
    return result


def build_hadamard_artifact(*, layer_names: Sequence[str], head_dim: int,
                            model_fingerprint: str, pr_fingerprint: str = PR_FINGERPRINT,
                            device: str = "npu", testing: bool = False) -> dict[str, Any]:
    """Paper's explicit data-free method, not a data-calibrated checkpoint."""
    torch, selected = _runtime(device, testing)
    result = _base_artifact(layer_names, head_dim, model_fingerprint, pr_fingerprint, "hadamard", testing)
    if head_dim < 2 or head_dim & (head_dim - 1):
        raise ValueError("Hadamard head_dim must be a power of two greater than one")
    matrix = torch.ones((1, 1), device=selected, dtype=torch.float32)
    while matrix.shape[0] < head_dim:
        matrix = torch.cat((torch.cat((matrix, matrix), 1),
                            torch.cat((matrix, -matrix), 1)), 0) / math.sqrt(2)
    for name in _names(layer_names):
        result["layers"][name] = {"layer_id": int(_LAYER.fullmatch(name).group(1)),
                                   "Rk": matrix.clone(), "Rv": matrix.clone()}
    result["calibrated"] = False
    return validate_artifact(result, layer_names=layer_names, head_dim=head_dim,
                             model_fingerprint=model_fingerprint, pr_fingerprint=pr_fingerprint,
                             device=device, testing=testing)


def build_identity_artifact(*, layer_names: Sequence[str], head_dim: int,
                            model_fingerprint: str, allow_identity: bool = False,
                            pr_fingerprint: str = PR_FINGERPRINT,
                            device: str = "npu", testing: bool = False) -> dict[str, Any]:
    if not allow_identity:
        raise ValueError("identity must be explicitly authorized with allow_identity=True")
    torch, selected = _runtime(device, testing)
    result = _base_artifact(layer_names, head_dim, model_fingerprint, pr_fingerprint, "identity", testing)
    for name in _names(layer_names):
        result["layers"][name] = {"layer_id": int(_LAYER.fullmatch(name).group(1)),
                                   "Rk": torch.eye(head_dim, device=selected),
                                   "Rv": torch.eye(head_dim, device=selected)}
    result["calibrated"] = False
    result["identity_authorized"] = True
    return validate_artifact(result, layer_names=layer_names, head_dim=head_dim,
                             model_fingerprint=model_fingerprint, pr_fingerprint=pr_fingerprint,
                             device=device, testing=testing, allow_identity=True)


def sample_covariances(query, key, value, *, sample_id: str,
                       device: str = "npu", testing: bool = False) -> tuple[Any, Any]:
    """Compute qqt/sst from ONE aligned sample, never two engine replays.

    Inputs are [tokens, heads, dimension]. NPU fp32 accumulation differs
    from the paper's CPU fp64 implementation and needs numerical validation.
    This function neither captures samples nor supplies an eigensolver.
    """
    torch, selected = _runtime(device, testing)
    if not sample_id:
        raise ValueError("sample_id is required to trace Q/K/V identity")
    if query.ndim != 3 or key.ndim != 3 or key.shape != value.shape:
        raise ValueError("sample tensors must be Q[T,Hq,D] and K/V[T,Hkv,D]")
    count, hq, dim = query.shape
    if count < 1 or dim < 1 or key.shape[0] != count or key.shape[2] != dim:
        raise ValueError("sample Q/K/V token or dimension mismatch")
    hk = key.shape[1]
    if hk < 1 or hq < 1 or hq % hk:
        raise ValueError("invalid sample GQA ratio")
    if any(t.device != selected for t in (query, key, value)):
        raise ValueError("samples must already reside on the selected device")
    if any(not bool(torch.isfinite(t).all()) for t in (query, key, value)):
        raise ValueError("non-finite calibration sample")
    grouped = query.float().reshape(count, hk, hq // hk, dim).permute(1, 0, 2, 3).reshape(hk, -1, dim)
    qtq = grouped.transpose(1, 2) @ grouped / grouped.shape[1]
    kh = key.float().transpose(0, 1)
    vh = value.float().transpose(0, 1)
    weights = ((kh @ qtq) * kh).sum(-1)
    totals = weights.sum(-1, keepdim=True)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or bool((totals <= 0).any()):
        raise ValueError("invalid sst weights: require finite nonnegative weights and positive sums")
    weights = weights / totals * count
    weighted = vh * weights.sqrt().unsqueeze(-1)
    covariance_k = qtq.mean(0)
    covariance_v = (weighted.transpose(1, 2) @ weighted / count).mean(0)
    covariance_k = (covariance_k + covariance_k.T) / 2
    covariance_v = (covariance_v + covariance_v.T) / 2
    if not bool(torch.isfinite(covariance_k).all() & torch.isfinite(covariance_v).all()):
        raise ValueError("non-finite calibration covariance")
    return covariance_k, covariance_v


def _validated_eigh(covariance, solver: Callable, torch):
    values, vectors = solver(covariance)
    dim = covariance.shape[0]
    if (values.device != covariance.device or vectors.device != covariance.device
            or values.shape != (dim,) or vectors.shape != (dim, dim)):
        raise ValueError("eigensolver changed device or returned invalid geometry")
    if not bool(torch.isfinite(values).all() & torch.isfinite(vectors).all()):
        raise ValueError("eigensolver returned non-finite values")
    values, vectors = values.float(), vectors.float()
    residual = torch.linalg.vector_norm(covariance @ vectors - vectors * values.unsqueeze(0))
    norm = torch.linalg.vector_norm(covariance)
    if float(norm) == 0 or float(residual / norm) > EIGEN_RESIDUAL_RTOL:
        raise ValueError("eigensolver residual validation failed")
    orthogonal = (vectors.T @ vectors - torch.eye(dim, device=vectors.device)).abs().amax()
    if float(orthogonal) > ORTHOGONALITY_ATOL:
        raise ValueError("eigensolver orthogonality validation failed")
    return values, vectors


def build_sample_artifact(samples: Mapping[str, tuple[Any, Any, Any]], *,
                          sample_ids: Mapping[str, str], model_fingerprint: str,
                          eigensolver: Callable, device: str = "npu", testing: bool = False,
                          pr_fingerprint: str = PR_FINGERPRINT) -> dict[str, Any]:
    """Legacy FP32 research helper, not the official automatic FP64 pipeline.

    qqt/sst R @ H @ P construction with an explicitly supplied NPU solver.

    The caller must prove the solver uses NPU numerical kernels. Returning
    NPU tensors alone is insufficient evidence against hidden CPU execution.
    """
    torch, selected = _runtime(device, testing)
    names = _names(tuple(samples))
    if set(sample_ids) != set(names):
        raise ValueError("sample identifiers must cover exactly the sampled layers")
    dim = samples[names[0]][0].shape[-1]
    hadamard_artifact = build_hadamard_artifact(layer_names=names, head_dim=dim,
        model_fingerprint=model_fingerprint, pr_fingerprint=pr_fingerprint, device=device, testing=testing)
    hadamard = hadamard_artifact["layers"][names[0]]["Rk"]
    result = _base_artifact(names, dim, model_fingerprint, pr_fingerprint, "qqt_sst_r_h_pbr", testing)
    bits = dim.bit_length() - 1
    # Index construction is dimension-only host metadata, not sample processing.
    reversal = [int(f"{i:0{bits}b}"[::-1], 2) for i in range(dim)]
    br = torch.tensor(reversal, dtype=torch.int64, device=selected)
    for name, tensors in samples.items():
        ck, cv = sample_covariances(*tensors, sample_id=sample_ids[name], device=device, testing=testing)
        if ck.shape != (dim, dim):
            raise ValueError("sample head dimensions differ across layers")
        entry = {"layer_id": int(_LAYER.fullmatch(name).group(1))}
        for key, covariance in (("Rk", ck), ("Rv", cv)):
            eigenvalues, eigenvectors = _validated_eigh(covariance, eigensolver, torch)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            permutation = torch.empty(dim, dtype=torch.int64, device=selected)
            permutation[br] = sorted_indices
            entry[key] = (eigenvectors @ hadamard).index_select(1, permutation).contiguous()
        result["layers"][name] = entry
    result["sample_ids"] = dict(sample_ids)
    result["calibrated"] = True
    result["accumulation_dtype"] = "fp32"
    return validate_artifact(result, layer_names=names, head_dim=dim,
        model_fingerprint=model_fingerprint, pr_fingerprint=pr_fingerprint, device=device, testing=testing)


def save_artifact(artifact: Mapping[str, Any], path: str | Path) -> None:
    """Serialize already-validated constants; no inference samples are copied."""
    torch = importlib.import_module("torch")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    portable = dict(artifact)
    portable["layers"] = {name: {key: value.detach().cpu() if torch.is_tensor(value) else value
                                 for key, value in layer.items()}
                           for name, layer in artifact["layers"].items()}
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(portable, temporary)
    temporary.replace(destination)


def load_artifact(path: str | Path, **validation) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping):
        raise ValueError("rotation artifact must be a mapping")
    return validate_artifact(artifact, **validation)


def calibration_cache_identity(config: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    """Pure host identity; no model load, tensor operation, or online download."""
    from .rotation_calibration import OFFICIAL_COMMIT
    from tools.calibration_data import GPQA_SHA256
    from tools.model_config import resolve_mamba_cache_dtypes
    prompts = config.get("rotation_calibration_prompts")
    source = ({"path": str(Path(prompts).expanduser().resolve()),
               "sha256": hashlib.sha256(Path(prompts).expanduser().read_bytes()).hexdigest()}
              if prompts else {"source": "openai-simple-evals/gpqa_diamond-seed0-v1",
                               "sha256": GPQA_SHA256})
    budget = config.get("rotation_calibration_tokens", 30000)
    if type(budget) is not int or budget <= 0:
        raise ValueError("rotation_calibration_tokens must be a positive integer")
    variant = config.get("rotation_variant", "shared_v1")
    if variant not in ("shared_v1", "perhead_center"):
        raise ValueError("rotation_variant must be shared_v1 or perhead_center")
    center = variant == "perhead_center"
    clip_method = config.get('rotation_k_clip')
    if clip_method not in (None, 'per_head_calibrated') or (clip_method and not center):
        raise ValueError('per_head_calibrated clipping requires perhead_center')
    clip_profile = {}
    if clip_method:
        baseline = float(config.get('k_clip_ratio', .96))
        if not math.isfinite(baseline) or not 0 < baseline <= 1:
            raise ValueError('per-head clipping baseline must be in (0,1]')
        clip_profile = {'rotation_k_clip': clip_method, 'clip_calibration': {
            'policy': 'centered_rotated_score_proxy_v1', 'baseline_ratio': baseline,
            'candidate_ratios': [.80, .85, .90, .92, .94, .96, .98, 1.0],
            'max_samples': 1024, 'dequant_modes': [0, 1],
            'criterion': 'both_modes_raw_and_centered_nonworse_minimax_relative'}}
    profile = {"official_commit": OFFICIAL_COMMIT,
            "recipe_version": (HEAD_CLIP_CALIBRATION_RECIPE_VERSION if clip_method else
                               CENTER_CALIBRATION_RECIPE_VERSION if center else CALIBRATION_RECIPE_VERSION),
            **({"rotation_variant": "perhead_center"} if center else {}),
            **clip_profile,
            "model_fingerprint": fingerprint,
            "objective": "qqt_sst_r_h_pbr", "accumulator_dtype": "float64",
            "prompt_source": source, "max_tokens": budget, "GPQA_option_seed": 0,
            "model_execution": {"dtype": "bfloat16", "quantization": config.get("quantization"),
                                "revision": config.get("revision"), "hf_overrides": config.get("hf_overrides"),
                                **resolve_mamba_cache_dtypes(config)}}
    from .optr_policy import validate_refinement_config, refinement_profile
    validate_refinement_config(config)
    return refinement_profile(profile, config) if config.get("rotation_refinement") == "optr" else profile


def model_artifact_fingerprint(model: str | Path) -> str:
    """Bind model location, config/index bytes and weight-file identity.

    This is a cache invalidation identity, not a content hash of all weights.
    Weight path/size/mtime/inode changes invalidate a same-architecture cache.
    """
    model = Path(model).expanduser().resolve()
    config = (model / "config.json").read_bytes()
    index = model / "model.safetensors.index.json"
    index_bytes = index.read_bytes() if index.is_file() else b""
    if index_bytes:
        names = sorted(set(json.loads(index_bytes)["weight_map"].values()))
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("weight index must name relative model files")
        weights = [model / name for name in names]
    else:
        weights = sorted(set(model.glob("*.safetensors")) | set(model.glob("pytorch_model*.bin")))
    records = []
    for path in weights:
        stat = path.stat()
        records.append({"name": str(path.relative_to(model)), "size": stat.st_size,
                        "resolved_path": str(path.resolve()),
                        "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino})
    # These small descriptors affect native quantized execution and prompt
    # rendering even when the weight shards themselves are unchanged.
    descriptors = {}
    for name in ("quant_model_description.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                 "tokenizer.json", "vocab.json", "merges.txt", "generation_config.json"):
        path = model / name
        if path.is_file():
            descriptors[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hasher = hashlib.sha256(config + b"\0" + index_bytes + b"\0")
    hasher.update(json.dumps({"model_path": str(model), "weight_files": records, "descriptors": descriptors},
                            sort_keys=True, separators=(",", ":")).encode())
    return hasher.hexdigest()


def checkpoint_identity(model: str | Path) -> tuple[list[str], int, str]:
    """Shared model geometry/cache identity for preparation and runtime."""
    model = Path(model)
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    count = text["num_hidden_layers"]
    kinds = text.get("layer_types")
    if kinds is None:
        interval = text.get("full_attention_interval")
        if type(interval) is not int or interval <= 0:
            raise ValueError("model must declare layer_types or full_attention_interval")
        kinds = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)]
    if len(kinds) != count or any(kind not in {"full_attention", "linear_attention"} for kind in kinds):
        raise ValueError("unexpected hybrid layer_types")
    layers = [f"model.layers.{i}.self_attn.attn" for i, kind in enumerate(kinds) if kind == "full_attention"]
    dim = text.get("head_dim")
    if dim is None:
        hidden, heads = text["hidden_size"], text["num_attention_heads"]
        if hidden % heads:
            raise ValueError("hidden_size is not divisible by query heads")
        dim = hidden // heads
    if not layers or type(dim) is not int or dim <= 0:
        raise ValueError("model has no FULL layers or invalid head_dim")
    return layers, dim, model_artifact_fingerprint(model)


def load_active_artifact(config: Mapping[str, Any], *, device="npu", testing=False,
                         layer_names=None, head_dim=None, model_fingerprint=None,
                         path=None):
    """Require a complete hashed official pair, not a hand-written boolean.

    The combined constants must match both pair files bit-for-bit. Runtime
    matrix validation runs on the requested device, outside inference/capture.
    """
    from .rotation_calibration import OFFICIAL_COMMIT,verified_pair_files,checkpoint_pair_artifact
    names, dim, fingerprint = checkpoint_identity(config["model"])
    if ((layer_names is not None and list(layer_names) != names) or
            (head_dim is not None and head_dim != dim) or
            (model_fingerprint is not None and model_fingerprint != fingerprint)):
        raise ValueError("rotation caller/model identity differs")
    destination = Path(path) if path is not None else rotation_artifact_path(config, fingerprint)
    options = dict(layer_names=names, head_dim=dim, model_fingerprint=fingerprint,
                   device=device, testing=testing)
    from .optr_policy import validate_refinement_config
    validate_refinement_config(config)
    if config.get("rotation_refinement") == "optr":
        from .optr_artifacts import load_refined_artifact
        return load_refined_artifact(config, destination, **options)
    if config.get("rotation_method", "qqt_sst") == "hadamard":
        return load_artifact(destination, **options)
    if config.get("rotation_method", "qqt_sst") != "qqt_sst":
        raise ValueError("unsupported runtime rotation method")
    k_state, v_state, artifact = verified_pair_files(destination.parent, artifact_name=destination.name)
    profile = calibration_cache_identity(config, fingerprint)
    expected = checkpoint_pair_artifact(k_state, v_state, layer_names=names, head_dim=dim,
        model_fingerprint=fingerprint, calibration_profile=profile,
        testing=testing, validate_math=False)
    if artifact.get("calibration") != expected.get("calibration"):
        raise ValueError("combined artifact calibration metadata differs from pair")
    if artifact.get("calibration_profile") != profile:
        raise ValueError("calibration cache profile differs")
    for key in ("format_version", "source_grouping", "global_kv_heads", "rotation_variant"):
        if artifact.get(key) != expected.get(key):
            raise ValueError("combined artifact recipe/schema differs from pair: " + key)
    if profile.get("rotation_variant") == "perhead_center":
        model_config = json.loads((Path(config['model']) / 'config.json').read_text())
        text = model_config.get('text_config', model_config)
        if artifact.get('global_kv_heads') != text.get('num_key_value_heads'):
            raise ValueError('per-head artifact KV-head count differs from selected model')
    torch = importlib.import_module("torch")
    for name in names:
        keys = ('Rk', 'Rv', 'k_mean') if profile.get('rotation_variant') == 'perhead_center' else ('Rk', 'Rv')
        if profile.get('rotation_k_clip') == 'per_head_calibrated':
            keys += ('k_clip_ratio',)
            if artifact.get('layers', {}).get(name, {}).get('clip_selection') != expected['layers'][name].get('clip_selection'):
                raise ValueError('combined clip selection provenance differs from official pair')
        for key in keys:
            value = artifact.get("layers", {}).get(name, {}).get(key)
            if not torch.is_tensor(value) or not torch.equal(value, expected["layers"][name][key]):
                raise ValueError("combined rotation constants differ from official pair")
    return validate_artifact(artifact, **options, pr_fingerprint=OFFICIAL_COMMIT, require_calibrated=True)


def rotation_artifact_path(config: Mapping[str, Any], fingerprint: str, *, root=None) -> Path:
    """One path contract shared by preparation and runtime; old H stays distinct."""
    import os
    explicit = config.get("rotation_artifact") or os.environ.get("OSCAR_ROTATION_ARTIFACT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    if config.get("rotation_method", "qqt_sst") == "hadamard":
        return base / "artifacts/rotations" / fingerprint / "hadamard.pt"
    identity = calibration_cache_identity(config, fingerprint)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    if config.get("rotation_refinement") == "optr":
        return base / "artifacts/rotations" / fingerprint / ("optr_" + digest) / "refined.pt"
    return base / "artifacts/rotations" / fingerprint / ("qqt_sst_" + digest) / "calibrated.pt"
