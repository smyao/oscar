"""Offline/loader Gram checks fail closed on a nonfinite backend result.

The NaN result is deliberately injected: finite large matrices on this Mac's
CPU backend produced Inf (already rejected), not a reproduced NPU failure.
No optimizer, tolerance, source seed equation or serving forward is changed.
"""
import pytest
import torch


def validator(kind, tmp_path):
    if kind == "fit":
        from oscar_ascend.optr_calibration import _matrices
        rotation = torch.linalg.qr(torch.randn(4, 4, generator=torch.Generator().manual_seed(91))).Q
        return lambda: _matrices(rotation, rotation, torch.zeros(4))
    if kind == "seed":
        from test_optr_oscar_seed import fit
        from oscar_ascend.optr_oscar_seed import validate_seed_artifact
        value = fit(dim=4)
        return lambda: validate_seed_artifact(value, expected_profile=value["calibration_profile"], testing=True)
    from test_optr_integration import NAME, artifact, configuration
    from oscar_ascend.rotations import validate_artifact
    value = artifact(configuration(tmp_path))
    return lambda: validate_artifact(value, layer_names=[NAME], head_dim=value["head_dim"],
        model_fingerprint=value["model_fingerprint"], pr_fingerprint=value["pr_fingerprint"],
        device="cpu", testing=True, require_calibrated=True)


@pytest.mark.parametrize("kind", ["fit", "seed", "artifact"])
@pytest.mark.parametrize("result", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_nonfinite_backend_gram_is_rejected(kind, result, tmp_path, monkeypatch):
    validate = validator(kind, tmp_path)
    original = torch.Tensor.__matmul__
    calls = []

    def injected(left, right):
        gram = original(left, right)
        calls.append(tuple(gram.shape))
        return torch.full_like(gram, result)

    monkeypatch.setattr(torch.Tensor, "__matmul__", injected)
    with pytest.raises(ValueError, match="orthogonal"):
        validate()
    # Reuse the existing Gram product; reject before computing the next one.
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["fit", "seed", "artifact"])
def test_valid_orthogonal_input_uses_only_existing_gram_products(kind, tmp_path, monkeypatch):
    validate = validator(kind, tmp_path)
    original = torch.Tensor.__matmul__
    calls = []

    def counted(left, right):
        result = original(left, right)
        calls.append(tuple(result.shape))
        return result

    monkeypatch.setattr(torch.Tensor, "__matmul__", counted)
    validate()
    assert len(calls) == 2
