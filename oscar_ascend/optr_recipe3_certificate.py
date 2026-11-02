"""Archive #170/#172/#174: strict, recipe3-anchored offline publication.

These checks certify local replay only; no claim of model-quality acceptance
is inferred from a certificate. They run during preparation/loading only.
"""
from __future__ import annotations

POLICY_VERSION = "optr_recipe3_centered_candidates_ascend_v10"
PUBLICATION_POLICY = "recipe3_validation_and_independent_test_required_v1"


class RefitRequired(ValueError):
    """Authentic old fitting code requires fresh fitting, not fresh sampling."""


def is_recipe3(policy):
    return policy.get("policy_version") == POLICY_VERSION


def strict_certificate(certificate):
    from .optr_artifacts import independent_test_summary
    return {**certificate, "schema_version": 3,
            "publication_policy": PUBLICATION_POLICY, "experimental": True,
            "independent_test": independent_test_summary(certificate["layers"])}


def require_publishable(artifact):
    """Completed-layer checkpoints may be unchanged; a published fit may not."""
    certificate = artifact["refinement_certificate"]
    if not is_recipe3(certificate["policy"]):
        return
    names = certificate.get("full_layer_names")
    if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or set(names) != set(artifact["layers"])):
        raise ValueError("recipe3 OptR publication requires every model FULL layer")
    if not certificate.get("accepted_layers"):
        raise ValueError("recipe3 OptR found no strict validation improvement; no refined PT is published")
    if certificate.get("independent_test", {}).get("status") != "passed":
        raise ValueError("recipe3 OptR independent test must pass before publication or reuse")
    from .optr_checkpoint import fitting_implementation_digest
    if certificate.get("fitting_implementation_sha256") != fitting_implementation_digest():
        raise RefitRequired("recipe3 OptR fitting implementation changed; refit from authenticated captures")


def validate_baseline_binding(artifact, config, *, testing):
    """Do not accept a valid-looking certificate against a substituted seed."""
    import torch
    from .optr_artifacts import file_sha256, matrix_digest
    from tools.prepare_optr_recipe3_base import base_config, load_base
    certificate = artifact["refinement_certificate"]
    if not is_recipe3(certificate["policy"]):
        return
    selected = base_config(config, artifact["model_fingerprint"])
    if file_sha256(selected["rotation_artifact"]) != certificate["base_artifact_sha256"]:
        raise ValueError("recipe3 OptR baseline file changed; its old certificate cannot be reused")
    baseline = load_base(selected, testing=testing)
    if set(baseline["layers"]) != set(artifact["layers"]):
        raise ValueError("recipe3 OptR baseline FULL layer coverage differs")
    for name, constants in artifact["layers"].items():
        seed = baseline["layers"][name]
        if (certificate["layers"][name]["base_constants_sha256"] != matrix_digest({name: seed})
                or not torch.equal(constants["k_mean"], seed["k_mean"])):
            raise ValueError("recipe3 OptR baseline constants or preserved key mean differ")
