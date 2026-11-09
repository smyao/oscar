"""Content-bound remote cache namespace, separate from physical page identity.

Archive #86/#97/#127/#128/#157/#163/#171: PT/model/math/physical geometry and
MTP/GDN protocols cannot share remote objects by model path or token hash alone.
The existing rotation pair publication lock and SHA manifest are reused. This
module reads host metadata once at registration, never on an attention step.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .layout import LayerCacheExport

NAMESPACE_SCHEMA = "oscar-physical-cache-v1"


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def rotation_artifact_identity(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint the validated artifact actually loaded by the runtime.

    Call immediately after load_active_artifact, before preparing device views.
    This binds cache keys to the loaded constants even if the published PT files
    are replaced before connector registration. Copies, if needed, occur only
    at initialization; this function must never enter an inference/capture path.
    """
    import torch

    def encode(value):
        if isinstance(value, torch.Tensor):
            data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            return {"tensor_dtype": str(value.dtype).removeprefix("torch."),
                    "tensor_shape": list(value.shape),
                    "tensor_sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest()}
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("rotation artifact mapping keys must be strings")
            return {key: encode(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if value is None or type(value) in (bool, str, int, float):
            return value
        raise ValueError(f"unsupported rotation artifact identity value: {type(value).__name__}")

    fingerprint = artifact.get("model_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
        raise ValueError("loaded rotation artifact lacks a model fingerprint")
    return {"identity_schema": "loaded-rotation-artifact-v1", "model_fingerprint": fingerprint,
            "artifact_content_sha256": hashlib.sha256(_canonical(encode(artifact)).encode()).hexdigest()}


def rotation_file_identity(path: str | Path, *, require_pair: bool = True,
                           expected_model_fingerprint: str | None = None) -> dict[str, Any]:
    """Hash actual PT bytes under the publisher's lock, without unpickling."""
    from ..rotation_calibration import K_FILENAME, V_FILENAME

    path = Path(path).expanduser().resolve()
    if not require_pair:
        return {"artifact_sha256": _digest_file(path), "pair": False}
    directory = path.parent
    with (directory / ".oscar.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        if (directory / ".oscar.pending.json").exists():
            raise ValueError("incomplete rotation pair publication")
        manifest = json.loads((directory / ".oscar.complete.json").read_text())
        expected = {K_FILENAME, V_FILENAME, path.name}
        if set(manifest.get("files", {})) != expected:
            raise ValueError("rotation pair manifest file set differs")
        if (expected_model_fingerprint is not None and
                manifest.get("model_fingerprint") != expected_model_fingerprint):
            raise ValueError("rotation pair model fingerprint differs")
        hashes = {name: _digest_file(directory / name) for name in sorted(expected)}
        if hashes != manifest["files"]:
            raise ValueError("rotation pair file hash mismatch")
        return {"pair": True, "artifact_sha256": hashes[path.name],
                "key_sha256": hashes[K_FILENAME], "value_sha256": hashes[V_FILENAME],
                "generation_id": manifest.get("generation_id"),
                "model_fingerprint": manifest.get("model_fingerprint")}


@dataclass(frozen=True)
class NamespaceIdentity:
    digest: str
    tp_rank: int
    tp_size: int
    manifest_json: str

    @property
    def common_digest(self) -> str:
        return self.digest

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_json)

    def shard_key(self, suffix: str, *, rank: int | None = None) -> str:
        rank = self.tp_rank if rank is None else rank
        if type(rank) is not int or not 0 <= rank < self.tp_size:
            raise ValueError("object shard rank is outside TP world")
        if not isinstance(suffix, str) or not suffix:
            raise ValueError("object suffix is required")
        return f"{NAMESPACE_SCHEMA}:{self.digest}:tp{rank}:{suffix}"


def build_namespace_identity(config: Mapping[str, Any], layouts: Mapping[str, LayerCacheExport], *,
                             tp_rank: int, tp_size: int, rotation_path: str | Path | None = None,
                             model_fingerprint: str | None = None,
                             loaded_rotation_identity: Mapping[str, Any] | None = None) -> NamespaceIdentity:
    """Build a common all-rank namespace and a separate object shard rank.

    Model evidence is the existing rotation cache fingerprint (config/index,
    descriptor contents and weight-file identity), not merely a path and not
    a claim to rehash all model weight bytes. PT files are content-hashed.
    """
    from ..rotations import model_artifact_fingerprint, rotation_artifact_path

    if type(tp_size) is not int or tp_size <= 0 or type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
        raise ValueError("invalid tensor parallel rank/world")
    if config.get("tensor_parallel_size", tp_size) != tp_size:
        raise ValueError("configured tensor parallel world differs from native worker")
    if not layouts:
        raise ValueError("namespace requires registered cache geometry")
    model = Path(config["model"]).expanduser().resolve()
    actual_fingerprint = model_artifact_fingerprint(model)
    if model_fingerprint is not None and model_fingerprint != actual_fingerprint:
        raise ValueError("model fingerprint differs from current model evidence")
    fingerprint = actual_fingerprint
    if not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
        raise ValueError("model fingerprint must be a full SHA-256 digest")
    method = config.get("rotation_method", "qqt_sst")
    if method not in {"qqt_sst", "hadamard"}:
        raise ValueError("unsupported rotation method")
    if loaded_rotation_identity is not None:
        rotation = json.loads(_canonical(dict(loaded_rotation_identity)))
        content_hash = rotation.get("artifact_content_sha256")
        if (rotation.get("identity_schema") != "loaded-rotation-artifact-v1"
                or rotation.get("model_fingerprint") != fingerprint
                or not isinstance(content_hash, str)
                or not re.fullmatch(r"[a-f0-9]{64}", content_hash)):
            raise ValueError("loaded rotation identity differs from the selected model/schema")
    else:
        if rotation_path is None:
            rotation_path = rotation_artifact_path(config, fingerprint)
        rotation = rotation_file_identity(rotation_path, require_pair=method == "qqt_sst",
                                          expected_model_fingerprint=fingerprint)
    # Keep every numerical experiment switch conservative. Transport addresses,
    # port/device IDs, scheduler capacities and tp_rank cannot split all-rank keys.
    keys = {"dtype", "quantization", "quantization_mode", "revision", "hf_overrides",
            "mamba_cache_dtype", "mamba_ssm_cache_dtype", "sink_tokens", "recent_tokens",
            "speculative_config", "k_clip_ratio", "v_clip_ratio", "rotation_method",
            "rotation_variant", "rotation_k_clip", "oscar_dequant_policy",
            "oscar_current_only_policy", "cache_format"}
    keys.update(key for key in config if key.startswith("experimental_"))
    semantics = {key: config[key] for key in sorted(keys) if key in config}
    manifest = {"schema": NAMESPACE_SCHEMA, "model_fingerprint": fingerprint,
                "model_config": json.loads((model / "config.json").read_text()),
                "rotation": rotation, "semantics": semantics, "tp_size": tp_size,
                "codec": "INT2-packed-codes-with-exact-physical-window",
                "draft_rotation_policy": "identity-with-zero-mean-v1",
                "layers": [layouts[name].geometry() for name in sorted(layouts)]}
    serialized = _canonical(manifest)
    return NamespaceIdentity(hashlib.sha256(serialized.encode()).hexdigest(), tp_rank, tp_size, serialized)
