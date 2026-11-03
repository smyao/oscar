"""Archive #170/#172/#174: strict, recipe3-anchored offline publication.

These checks certify local replay only; no claim of model-quality acceptance
is inferred from a certificate. They run during preparation/loading only.
"""
from __future__ import annotations

POLICY_VERSION = "optr_recipe3_centered_candidates_ascend_v10"
REPAIR_POLICY_VERSION = "optr_recipe3_trust_region_ascend_v11"
PUBLICATION_POLICY = "recipe3_validation_and_independent_test_required_v1"


class RefitRequired(ValueError):
    """Authentic old fitting code requires fresh fitting, not fresh sampling."""


def is_recipe3(policy):
    return policy.get("policy_version") in (POLICY_VERSION, REPAIR_POLICY_VERSION)


def validate_repair_record(record, policy, heads):
    """Archive #176: a repair certificate cannot bypass the original gates."""
    from .optr_artifacts import nonworse
    from .optr_trust_region import REPAIR_POLICY
    if policy.get("policy_version") != REPAIR_POLICY_VERSION:
        return
    if policy.get("recipe3_repair") != REPAIR_POLICY:
        raise ValueError("recipe3 OptR repair policy differs")
    original = record.get("shared_layer_selection", {})
    if "recipe3_repair_selection" not in record:
        raise ValueError("recipe3 OptR repair selection evidence missing")
    repair = record["recipe3_repair_selection"]
    if original.get("accepted") is True:
        if repair is not None:
            raise ValueError("recipe3 repair cannot replace a passing original candidate")
        return
    if (not isinstance(repair, dict) or repair.get("policy") != REPAIR_POLICY or
            repair.get("method") != REPAIR_POLICY["version"] or
            repair.get("test_used_for_selection") is not False or
            repair.get("accepted") is not record.get("accepted") or
            repair.get("base_constants_sha256") != record.get("base_constants_sha256") or
            repair.get("selected_constants_sha256") != record.get("selected_constants_sha256")):
        raise ValueError("recipe3 OptR repair decision/identity differs")
    rows = repair.get("evaluated_candidates")
    if (type(heads) is not int or not 1 <= heads <= REPAIR_POLICY["max_kv_heads"] or
            not isinstance(rows, list) or len(rows) > heads * 16 or
            repair.get("max_candidates") != heads * 16 or repair.get("candidate_count") != len(rows)):
        raise ValueError("recipe3 OptR repair candidate budget differs")
    before = record["validation"]["baseline"]
    best_id, best_metric, best_digest = "seed", before, record["base_constants_sha256"]
    seen = set()
    for row in rows:
        head, side, radius = (row.get(key) for key in ("head", "side", "radius"))
        if (type(head) is not int or not 0 <= head < heads or side not in ("Rk", "Rv") or
                type(radius) not in (int, float) or radius not in REPAIR_POLICY["radii"]):
            raise ValueError("recipe3 OptR repair head/side/radius differs")
        identity = f"repair_h{head}_{side}_r{REPAIR_POLICY['radii'].index(radius)}"
        if row.get("candidate") != identity or identity in seen:
            raise ValueError("recipe3 OptR repair candidate identity differs")
        seen.add(identity)
        measured = row.get("validation")
        passes = nonworse(before, measured, require_frames=True, require_terminal=True,
                          require_queries=True, require_projection=True)
        passes = passes and measured["squared_error"] < before["squared_error"]
        if row.get("accepted") is not passes:
            raise ValueError("recipe3 OptR repair gate differs")
        if passes and measured["squared_error"] < best_metric["squared_error"]:
            best_id, best_metric, best_digest = identity, measured, row.get("constants_sha256")
    if (repair.get("selected_candidate") != best_id or
            record["selected_constants_sha256"] != best_digest or
            record["validation"]["selected"] != best_metric):
        raise ValueError("recipe3 OptR repair selection differs from validation-only minimum")


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
