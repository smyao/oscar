"""Archive #86/#97/#127/#128/#157/#163/#171: remote identity mismatch gates.

Hash actual published PT bytes, and keep common namespace separate from TP
shards. Host identity validation does not establish device offload acceptance.
"""
import hashlib
import json

import pytest
import torch

from test_runtime_native import native_specs  # noqa: F401
from test_offload_layout import make_exports
from oscar_ascend.offload.identity import (
    build_namespace_identity, rotation_file_identity, rotation_artifact_identity,
)
from oscar_ascend.rotation_calibration import K_FILENAME, V_FILENAME
from oscar_ascend.rotations import model_artifact_fingerprint


def publish_pair(directory, fingerprint, value=1):
    directory.mkdir(exist_ok=True)
    (directory / ".oscar.lock").touch()
    files = {}
    for name in (K_FILENAME, V_FILENAME, "calibrated.pt"):
        torch.save({"constant": torch.tensor([value]), "model_fingerprint": fingerprint}, directory / name)
        files[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    (directory / ".oscar.complete.json").write_text(json.dumps({
        "files": files, "model_fingerprint": fingerprint, "generation_id": str(value)}))
    return directory / "calibrated.pt"


@pytest.fixture
def namespace_case(tmp_path, native_specs):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"rope_theta": 10000, "num_attention_heads": 24}))
    (model / "model.safetensors").write_bytes(b"test weight identity")
    fingerprint = model_artifact_fingerprint(model)
    path = publish_pair(tmp_path / "rotations", fingerprint)
    config = {"model": str(model), "tensor_parallel_size": 4, "rotation_method": "qqt_sst",
              "dtype": "bfloat16", "k_clip_ratio": .96, "v_clip_ratio": .92,
              "sink_tokens": 64, "recent_tokens": 256,
              "oscar_dequant_policy": "fp32_metadata",
              "experimental_official_math": True,
              "speculative_config": {"method": "qwen3_5_mtp", "num_speculative_tokens": 3}}
    exports, *_ = make_exports(native_specs)
    return config, exports, path, fingerprint


def build(case, rank=0, **changes):
    config, exports, path, _ = case
    config = dict(config, **changes)
    return build_namespace_identity(config, exports, tp_rank=rank,
                                    tp_size=config["tensor_parallel_size"], rotation_path=path)


def test_namespace_is_common_and_rank_is_object_shard(namespace_case):
    identities = [build(namespace_case, rank=rank) for rank in range(4)]
    assert len({identity.digest for identity in identities}) == 1
    identity = identities[0]
    assert identity.common_digest == identity.digest
    assert len({identity.shard_key("checkpoint", rank=rank) for rank in range(4)}) == 4
    assert identity.shard_key("checkpoint", rank=3) == identities[3].shard_key("checkpoint")
    with pytest.raises(ValueError, match="rank"):
        identity.shard_key("checkpoint", rank=4)
    assert build(namespace_case, port=123, devices=[4, 5, 6, 7], tp_rank=3).digest == identity.digest


@pytest.mark.parametrize("changes", [
    {"k_clip_ratio": .94}, {"v_clip_ratio": .90}, {"dtype": "float16"},
    {"oscar_dequant_policy": "native_phase"}, {"experimental_official_math": False},
    {"speculative_config": {"method": "qwen3_5_mtp", "num_speculative_tokens": 1}},
    {"hf_overrides": {"rope_scaling": {"factor": 2}}}, {"tensor_parallel_size": 2},
])
def test_numerical_and_geometry_protocol_changes_split_namespace(namespace_case, changes):
    assert build(namespace_case, **changes).digest != build(namespace_case).digest


def test_same_pt_path_new_contents_never_reuse_namespace(namespace_case):
    _, _, path, fingerprint = namespace_case
    original = build(namespace_case).digest
    publish_pair(path.parent, fingerprint, value=2)
    assert build(namespace_case).digest != original
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        build(namespace_case)


def test_pair_must_be_complete_and_model_bound(namespace_case):
    _, _, path, _ = namespace_case
    (path.parent / ".oscar.pending.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        build(namespace_case)
    (path.parent / ".oscar.pending.json").unlink()
    with pytest.raises(ValueError, match="model fingerprint"):
        rotation_file_identity(path, expected_model_fingerprint="0" * 64)


def test_model_file_evidence_changes_invalidate_old_pt_pair(namespace_case):
    config, _, _, _ = namespace_case
    from pathlib import Path
    (Path(config["model"]) / "model.safetensors").write_bytes(b"different weight file")
    with pytest.raises(ValueError, match="model fingerprint"):
        build(namespace_case)


def test_loaded_identity_tracks_actual_tensors_across_disk_replacement(namespace_case):
    config, exports, path, fingerprint = namespace_case
    artifact = {"model_fingerprint": fingerprint, "format_version": 2,
                "layers": {"full": {"Rk": torch.eye(4), "k_mean": torch.zeros(4),
                                     "clip": torch.tensor(.96)}}}
    loaded = rotation_artifact_identity(artifact)
    before = build_namespace_identity(config, exports, tp_rank=0, tp_size=4,
                                      loaded_rotation_identity=loaded)
    publish_pair(path.parent, fingerprint, value=99)
    after = build_namespace_identity(config, exports, tp_rank=1, tp_size=4,
                                     loaded_rotation_identity=loaded)
    assert before.digest == after.digest
    artifact["layers"]["full"]["k_mean"][1] = 1
    changed = rotation_artifact_identity(artifact)
    assert changed != loaded
    newer = build_namespace_identity(config, exports, tp_rank=0, tp_size=4,
                                     loaded_rotation_identity=changed)
    assert newer.digest != before.digest
    clone = {"model_fingerprint": fingerprint, "format_version": 2,
             "layers": {"full": {key: tensor.clone() for key, tensor in artifact["layers"]["full"].items()}}}
    assert rotation_artifact_identity(clone) == changed


def test_loaded_identity_rejects_unbound_constants(namespace_case):
    config, exports, _, _ = namespace_case
    with pytest.raises(ValueError, match="model fingerprint"):
        rotation_artifact_identity({"layers": {"Rk": torch.eye(4)}})
    with pytest.raises(ValueError, match="loaded rotation identity"):
        build_namespace_identity(config, exports, tp_rank=0, tp_size=4,
            loaded_rotation_identity={"identity_schema": "loaded-rotation-artifact-v1",
                                      "model_fingerprint": "0" * 64,
                                      "artifact_content_sha256": "f" * 64})
