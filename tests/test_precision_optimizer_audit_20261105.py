"""Archive #170/#172/#174-177: recipe3 calibration limitations, CPU evidence.

The user is testing OSCAR without OPTR. These tests call the existing recipe3
fitter and offline BF16 replay only; they do not activate an optimizer, alter
clip/windows, or certify a model-quality cause or a replacement default.
"""
import hashlib
import json

import pytest
import torch

from oscar_ascend.rotation_calibration import (
    OFFICIAL_COMMIT, OfflineLayerAccumulator, checkpoint_pair_artifact,
    fit_checkpoint_pair, publish_checkpoint_pair,
)
from oscar_ascend.rotations import (
    calibration_cache_identity, checkpoint_identity, load_active_artifact,
    prepare_device_rotations, rotation_artifact_path,
)


NAME = "model.layers.3.self_attn.attn"


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _sha(tensor):
    return hashlib.sha256(tensor.float().contiguous().numpy().tobytes()).hexdigest()


def _profile(tokens=64):
    return {"official_commit": OFFICIAL_COMMIT, "model_fingerprint": "synthetic",
            "recipe_version": 3, "rotation_variant": "perhead_center", "max_tokens": tokens,
            "model_execution": {"dtype": "bfloat16", "mamba_cache_dtype": "bfloat16",
                                "mamba_ssm_cache_dtype": "bfloat16"}}


def _fit(q, k, v):
    tokens, heads, dim = k.shape
    accumulator = OfflineLayerAccumulator(head_dim=dim, local_kv_heads=heads,
        local_q_heads=q.shape[1], token_budget=tokens, testing=True)
    accumulator.observe(q, k, v)
    payload = {"rank": 0, "tp_size": 1, "global_q_heads": q.shape[1], "global_kv_heads": heads,
               "prompt_sha256": "synthetic", "model_fingerprint": "synthetic",
               "layers": {3: accumulator.export_payload()}}
    key, value = fit_checkpoint_pair([payload], layer_names=[NAME], head_dim=dim,
        model_path="/synthetic-not-a-model", model_fingerprint="synthetic", global_q_heads=q.shape[1],
        global_kv_heads=heads, tp_size=1, prompt_sha256="synthetic", max_tokens=tokens,
        calibration_profile=_profile(tokens), testing=True)
    return key["layers"][3]["rotation"][0], value["layers"][3]["rotation"][0], key["layers"][3]["k_mean"][0]


def recipe3_key_shift_example(*, norm_preserving=False):
    """An irrelevant exact key translation changes the current V proxy/basis."""
    from oscar_ascend.optr_calibration import replay_sample  # numerical replay, no OptR fitting

    dim, tokens = 64, 64
    generator = torch.Generator().manual_seed(20261105)
    if norm_preserving:
        query = (torch.randint(0, 2, (tokens, 1, dim), generator=generator) * 2 - 1).bfloat16()
        half = (torch.randint(0, 2, (tokens // 2, 1, dim), generator=generator) * 2 - 1).bfloat16()
    else:
        query = torch.randint(-3, 4, (tokens, 1, dim), generator=generator).bfloat16()
        half = torch.randint(-4, 5, (tokens // 2, 1, dim), generator=generator).bfloat16()
    keys = torch.cat((half, -half))  # exact zero mean; no BF16 centering ambiguity
    if norm_preserving:
        keys[:, :, 0] = -4
    values = torch.randn(tokens, 1, dim, generator=generator).bfloat16()
    shift = torch.zeros(dim, dtype=torch.bfloat16)
    shift[0] = 8 if norm_preserving else 64  # keys and mean remain BF16-exact
    records = []
    for current_keys in (keys, keys + shift):
        rk, rv, mean = _fit(query, current_keys, values)
        positions = torch.arange(60, 64)
        logits = query[-4:, 0].float() @ current_keys[:, 0].float().T / 8
        logits.masked_fill_(torch.arange(tokens)[None, :] > positions[:, None], -torch.inf)
        teacher = (logits.softmax(-1) @ values[:, 0].float()).bfloat16()
        sample = {"request_id": "synthetic-held-forward", "q": query[-4:], "k": current_keys[:, 0],
            "v": values[:, 0], "gate": torch.ones(4, 1, dim, dtype=torch.bfloat16),
            "teacher": teacher[:, None], "wo": torch.eye(dim)[None], "positions": positions,
            "contexts": torch.full((4,), 60, dtype=torch.int64),
            "current_ends": torch.full((4,), 64, dtype=torch.int64)}
        actual = replay_sample(rk, rv, mean, sample, {"sink_tokens": 2, "recent_tokens": 4})[:, 0]
        centered = (current_keys[:, 0].float() - mean.bfloat16().float()).bfloat16()
        records.append({"Rk": rk, "Rv": rv, "mean": mean, "centered": centered, "teacher": teacher,
                        "squared_error": float((actual - teacher).double().square().sum())})
    first, second = records
    return {"synthetic": True, "head_dim": dim, "tokens": tokens, "gqa": 1,
            "seed": 20261105, "key_shift_coordinate": 0, "key_shift_value": float(shift[0]),
            "norm_preserving": norm_preserving,
            "key_squared_norms_equal": torch.equal(keys.float().square().sum(-1),
                                                    (keys + shift).float().square().sum(-1)),
            "Rk_equal": torch.equal(first["Rk"], second["Rk"]),
            "Rv_equal": torch.equal(first["Rv"], second["Rv"]),
            "mean_delta_equals_shift": torch.equal(second["mean"] - first["mean"], shift.float()),
            "centered_K_equal": torch.equal(first["centered"], second["centered"]),
            "native_attention_equal": torch.equal(first["teacher"], second["teacher"]),
            "centered_K_sha256": _sha(first["centered"]), "teacher_sha256": _sha(first["teacher"]),
            "Rk_sha256": _sha(first["Rk"]), "Rv_sha256": [_sha(row["Rv"]) for row in records],
            "squared_error": [row["squared_error"] for row in records],
            "relative_error_increase": second["squared_error"] / first["squared_error"] - 1,
            "interpretation": "Raw-key energy used for S_v is not invariant to an exact constant key translation; "
                              "this is a limitation of the proxy, not evidence of a mismatched official formula or model cause."}


def test_recipe3_value_proxy_changes_under_attention_invariant_centered_key_translation():
    result = recipe3_key_shift_example()
    assert result["Rk_equal"] and result["mean_delta_equals_shift"]
    assert result["centered_K_equal"] and result["native_attention_equal"]
    assert not result["Rv_equal"]
    assert result["squared_error"][1] > result["squared_error"][0]
    assert result["squared_error"][0] == pytest.approx(95.1706514817, rel=2e-6)
    assert result["squared_error"][1] == pytest.approx(107.5031493472, rel=2e-6)


def test_translation_limitation_also_exists_with_every_key_norm_preserved():
    result = recipe3_key_shift_example(norm_preserving=True)
    assert result["key_squared_norms_equal"]
    assert result["Rk_equal"] and result["centered_K_equal"] and result["native_attention_equal"]
    assert not result["Rv_equal"]
    assert result["squared_error"][0] == pytest.approx(2.66502878095, rel=2e-6)
    assert result["squared_error"][1] == pytest.approx(2.35849616048, rel=2e-6)


def test_recipe3_query_statistics_discard_query_time_order():
    generator = torch.Generator().manual_seed(52)
    q = torch.randn(32, 2, 64, generator=generator).bfloat16()
    k = torch.randn(32, 1, 64, generator=generator).bfloat16()
    v = torch.randn(32, 1, 64, generator=generator).bfloat16()
    original = _fit(q, k, v)
    reordered = _fit(q.flip(0), k, v)
    # Integer-valued BF16 products and these small sums are exactly represented
    # in FP64, so this is exact sufficient-statistic equality, not loose fitting.
    assert all(torch.equal(left, right) for left, right in zip(original, reordered))
    original_output = (q[-1, 0].float() @ k[:, 0].float().T / 8).softmax(-1) @ v[:, 0].float()
    reordered_output = (q[0, 0].float() @ k[:, 0].float().T / 8).softmax(-1) @ v[:, 0].float()
    assert not torch.equal(original_output, reordered_output)


def test_current_recipe3_cache_identity_does_not_bind_tp_or_additional_config(tmp_path):
    """Record an execution-provenance gap, without asserting a current TP4 cause."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"], "head_dim": 64,
        "num_attention_heads": 8, "num_key_value_heads": 4}))
    (model / "model.safetensors").write_bytes(b"synthetic-file-identity-only")
    config = {"model": str(model), "rotation_variant": "perhead_center", "rotation_method": "qqt_sst",
              "tensor_parallel_size": 2, "rotation_calibration_tokens": 16,
              "mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "bfloat16"}
    names, dim, fingerprint = checkpoint_identity(model)
    other_tp = {**config, "tensor_parallel_size": 4}
    other_execution = {**config, "additional_config": {"enable_flashcomm1": True}}
    profile = calibration_cache_identity(config, fingerprint)
    assert calibration_cache_identity(other_tp, fingerprint) == profile
    assert calibration_cache_identity(other_execution, fingerprint) == profile
    assert rotation_artifact_path(config, fingerprint) == rotation_artifact_path(other_tp, fingerprint)
    ranks = []
    for rank in range(2):
        generator = torch.Generator().manual_seed(72 + rank)
        accumulator = OfflineLayerAccumulator(head_dim=64, local_q_heads=4,
            local_kv_heads=2, token_budget=16, testing=True)
        accumulator.observe(*[torch.randn(16, count, 64, generator=generator).bfloat16() for count in (4, 2, 2)])
        ranks.append({"rank": rank, "tp_size": 2, "global_q_heads": 8, "global_kv_heads": 4,
                      "prompt_sha256": "synthetic", "model_fingerprint": fingerprint,
                      "layers": {3: accumulator.export_payload()}})
    key, value = fit_checkpoint_pair(ranks, layer_names=names, head_dim=dim, model_path=str(model),
        model_fingerprint=fingerprint, global_q_heads=8, global_kv_heads=4, tp_size=2,
        prompt_sha256="synthetic", max_tokens=16, calibration_profile=profile, testing=True)
    artifact = checkpoint_pair_artifact(key, value, layer_names=names, head_dim=dim,
        model_fingerprint=fingerprint, calibration_profile=profile, testing=True)
    output = tmp_path / "pair" / "calibrated.pt"
    publish_checkpoint_pair(key, value, output.parent, artifact=artifact)
    loaded = load_active_artifact(other_tp, path=output, device="cpu", testing=True)
    assert loaded["calibration"]["tp_size"] == 2
    for rank in range(4):
        matrices = prepare_device_rotations(loaded, device="cpu", testing=True, tp_rank=rank, tp_size=4)
        # Head identity remains correct despite reuse of a different TP capture.
        assert torch.equal(matrices[NAME].key_transposed, artifact["layers"][NAME]["Rk"][rank].T)
