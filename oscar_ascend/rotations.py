"""Explicit, fingerprinted OSCAR rotation artifacts and sample statistics.

Archive G19/G20/G23, #23-#25: sample identity, finite covariance and solver
validation; #86: target and draft layers are enumerated separately. #96/#97:
lazy torch import and explicit, traceable data-free construction.
Paper reference: rotation/compute_kv_rotation.py:23-46,93-136,234-312.
All production numerical work requires NPU. CPU is an explicit test option.
This module alone does not establish target-NPU correctness or calibration.
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


def prepare_device_rotations(artifact: Mapping[str, Any], *, device: str = "npu",
                             testing: bool = False) -> dict[str, DeviceRotation]:
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
    for name, entry in layers.items():
        k, v = entry.get("Rk"), entry.get("Rv")
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
        result[name] = DeviceRotation(k.T.contiguous(), v.T.contiguous(), v.contiguous(), fast)
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
                      testing: bool = False, allow_identity: bool = False) -> dict[str, Any]:
    """Validate exact coverage, provenance, fp32 geometry and orthogonality.

    Validation returns matrices on the requested device. A test artifact
    cannot be used in production even if it happens to have matching hashes.
    """
    torch, selected = _runtime(device, testing)
    names = _names(layer_names)
    if artifact.get("format_version") != FORMAT_VERSION or artifact.get("source_grouping") != "layer":
        raise ValueError("unsupported rotation artifact schema")
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
            if not torch.is_tensor(matrix) or matrix.dtype != torch.float32 or matrix.shape != (head_dim, head_dim):
                raise ValueError(f"{name}/{key}: expected fp32 [{head_dim},{head_dim}]")
            matrix = matrix.to(device=selected).contiguous()
            if not bool(torch.isfinite(matrix).all()):
                raise ValueError(f"{name}/{key}: non-finite rotation")
            error = (matrix.T @ matrix - identity).abs().amax()
            if float(error) > ORTHOGONALITY_ATOL:
                raise ValueError(f"{name}/{key}: non-orthogonal rotation, max_error={float(error)}")
            if torch.equal(matrix, identity) and not allow_identity:
                raise ValueError(f"{name}/{key}: identity requires explicit allow_identity=True")
            entry[key] = matrix
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
    """qqt/sst R @ H @ P construction with an explicitly supplied NPU solver.

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
