"""CPU-only rotation mathematics; does not validate NPU eigensolver support.

Archive G19/G20/G23, #23-#25, #86/#97 motivate aligned samples, coverage,
finite matrices and traceable data-free artifacts; no silent identity.
"""

import hashlib

import pytest
import torch

from oscar_ascend.rotations import (
    PR_FINGERPRINT, build_hadamard_artifact, build_identity_artifact,
    build_sample_artifact, load_artifact, model_fingerprint, sample_covariances,
    save_artifact, validate_artifact,
)


NAMES = ["model.layers.3.self_attn.attn", "model.layers.7.self_attn.attn"]
MODEL = hashlib.sha256(b"explicit-test-weight-manifest").hexdigest()


def options():
    return {"layer_names": NAMES, "head_dim": 8, "model_fingerprint": MODEL,
            "device": "cpu", "testing": True}


def test_hadamard_is_exact_paper_sign_pattern_and_data_free():
    artifact = build_hadamard_artifact(**options())
    expected = torch.tensor([[(-1.0) ** ((i & j).bit_count()) for j in range(8)] for i in range(8)]) / 8 ** .5
    for layer in artifact["layers"].values():
        torch.testing.assert_close(layer["Rk"], expected, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(layer["Rk"].T @ layer["Rk"], torch.eye(8), atol=5e-7, rtol=0)
    assert artifact["objective"] == "hadamard"
    assert artifact["calibrated"] is False
    assert artifact["test_only"] is True


def test_cpu_is_not_an_implicit_production_execution_path():
    kwargs = options()
    kwargs["testing"] = False
    with pytest.raises(RuntimeError, match="requires NPU"):
        build_hadamard_artifact(**kwargs)


def test_identity_requires_explicit_authorization_at_construction_and_load():
    with pytest.raises(ValueError, match="explicitly"):
        build_identity_artifact(**options())
    artifact = build_identity_artifact(**options(), allow_identity=True)
    with pytest.raises(ValueError, match="identity requires"):
        validate_artifact(artifact, **options())
    valid = validate_artifact(artifact, **options(), allow_identity=True)
    assert valid["identity_authorized"] is True


@pytest.mark.parametrize("mutation, message", [
    (lambda a: a.update(model_fingerprint="wrong"), "model fingerprint"),
    (lambda a: a.update(pr_fingerprint="wrong"), "PR fingerprint"),
    (lambda a: a.update(head_dim=16), "head_dim"),
    (lambda a: a["layers"].pop(NAMES[0]), "coverage"),
    (lambda a: a["layers"][NAMES[0]].update(layer_id=4), "layer id"),
    (lambda a: a["layers"][NAMES[0]].update(Rk=torch.ones(8, 8)), "non-orthogonal"),
    (lambda a: a["layers"][NAMES[0]].update(Rk=torch.full((8, 8), float("nan"))), "non-finite"),
    (lambda a: a["layers"][NAMES[0]].update(Rk=torch.eye(8).half()), "expected fp32"),
])
def test_bad_artifacts_fail_closed(mutation, message):
    artifact = build_hadamard_artifact(**options())
    mutation(artifact)
    with pytest.raises(ValueError, match=message):
        validate_artifact(artifact, **options())


def test_artifact_roundtrip_is_weights_only_and_preserves_coverage(tmp_path):
    artifact = build_hadamard_artifact(**options())
    path = tmp_path / "nested" / "rotations.pt"
    save_artifact(artifact, path)
    actual = load_artifact(path, **options())
    assert set(actual["layers"]) == set(NAMES)
    assert actual["pr_fingerprint"] == PR_FINGERPRINT
    torch.testing.assert_close(actual["layers"][NAMES[0]]["Rk"], artifact["layers"][NAMES[0]]["Rk"])


def test_model_fingerprint_binds_configuration_and_weights():
    first = model_fingerprint({"head_dim": 8, "layers": [3, 7]}, MODEL)
    assert first == model_fingerprint({"layers": [3, 7], "head_dim": 8}, MODEL)
    assert first != model_fingerprint({"head_dim": 16, "layers": [3, 7]}, MODEL)
    assert first != model_fingerprint({"head_dim": 8, "layers": [3, 7]}, "a" * 64)


def _samples(seed=12):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(37, 4, 8, generator=generator),
            torch.randn(37, 2, 8, generator=generator),
            torch.randn(37, 2, 8, generator=generator))


def test_sample_statistics_match_independent_paper_fp64_equations():
    q, k, v = _samples()
    got_k, got_v = sample_covariances(q, k, v, sample_id="immutable-sample-12", device="cpu", testing=True)
    ref_k, ref_v = torch.zeros(8, 8, dtype=torch.float64), torch.zeros(8, 8, dtype=torch.float64)
    for h in range(2):
        qg = q[:, 2*h:2*h+2].reshape(-1, 8).double()
        kh, vh = k[:, h].double(), v[:, h].double()
        qtq = qg.T @ qg / len(qg)
        weights = (kh @ qtq * kh).sum(1)
        weights = weights / weights.sum() * len(kh)
        weighted = vh * weights.sqrt()[:, None]
        ref_k += qtq / 2
        ref_v += weighted.T @ weighted / len(kh) / 2
    torch.testing.assert_close(got_k.double(), ref_k, atol=5e-7, rtol=5e-7)
    torch.testing.assert_close(got_v.double(), ref_v, atol=5e-7, rtol=5e-7)


def test_empty_or_invalid_sample_is_never_repaired_silently():
    q, k, v = _samples()
    with pytest.raises(ValueError, match="sample_id"):
        sample_covariances(q, k, v, sample_id="", device="cpu", testing=True)
    with pytest.raises(ValueError, match="sst weights"):
        sample_covariances(q, torch.zeros_like(k), v, sample_id="bad", device="cpu", testing=True)
    with pytest.raises(ValueError, match="token"):
        sample_covariances(q[:-1], k, v, sample_id="misaligned", device="cpu", testing=True)


def test_sample_artifact_uses_explicit_solver_and_records_sample_ids():
    samples = {name: _samples(i) for i, name in enumerate(NAMES)}
    identifiers = {name: f"sample-{i}" for i, name in enumerate(NAMES)}
    artifact = build_sample_artifact(samples, sample_ids=identifiers, model_fingerprint=MODEL,
                                     eigensolver=torch.linalg.eigh, device="cpu", testing=True)
    assert artifact["objective"] == "qqt_sst_r_h_pbr"
    assert artifact["sample_ids"] == identifiers
    assert artifact["calibrated"] is True
    assert artifact["test_only"] is True
    validate_artifact(artifact, **options())


def test_eigensolver_that_returns_wrong_values_fails_validation():
    def broken_solver(covariance):
        return torch.zeros(8), torch.eye(8)
    with pytest.raises(ValueError, match="residual"):
        build_sample_artifact({NAMES[0]: _samples()}, sample_ids={NAMES[0]: "sample"},
                               model_fingerprint=MODEL, eigensolver=broken_solver,
                               device="cpu", testing=True)
