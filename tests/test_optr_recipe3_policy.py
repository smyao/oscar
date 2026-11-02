"""Recipe3 is the comparison baseline; all online launch behavior stays fixed."""
import copy
import json
from pathlib import Path
import subprocess
from types import ModuleType

import pytest

from oscar_ascend.optr_policy import (
    RECIPE3_DEPLOYMENT_CANDIDATE_SELECTION, RECIPE3_POLICY_VERSION,
    RECIPE3_PUBLICATION_POLICY, compatible_capture_profiles,
    legacy_capture_policy, optr_policy, validate_refinement_config,
)
from oscar_ascend.rotations import calibration_cache_identity, rotation_artifact_path
from tools.serving_variants import runtime_feature_summary, variant_config, variant_features
from tools.target_cli import serve_argv


ROOT = Path(__file__).resolve().parents[1]
PRE_RECIPE3 = "7c4a621"


@pytest.fixture
def candidate():
    return variant_config(json.loads((ROOT / "configs/target.json").read_text()), "candidate")


def old_policy_module():
    source = subprocess.check_output(
        ["git", "show", f"{PRE_RECIPE3}:oscar_ascend/optr_policy.py"], cwd=ROOT)
    module = ModuleType("oscar_ascend._previous_optr_policy")
    exec(compile(source, "committed_optr_policy.py", "exec"), module.__dict__)
    return module


def test_legacy_without_marker_keeps_exact_v9_policy_and_older_identities(candidate):
    old = old_policy_module()
    legacy = {key: value for key, value in candidate.items() if key != "optr_baseline"}
    assert optr_policy(legacy) == old.optr_policy(legacy)
    for version in ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"):
        assert legacy_capture_policy(legacy, version) == old.legacy_capture_policy(legacy, version)
        assert legacy_capture_policy(candidate, version) == old.legacy_capture_policy(legacy, version)


def test_new_policy_only_changes_the_baseline_and_offline_search_contract(candidate):
    original = copy.deepcopy(candidate)
    legacy = optr_policy({key: value for key, value in candidate.items() if key != "optr_baseline"})
    policy = optr_policy(candidate)
    changes = {
        "policy_version": RECIPE3_POLICY_VERSION,
        "fitted_grouping": "layer_kv_head",
        "selection": "recipe3_layer_each_query_and_projection_nonworse",
        "deployment_candidate_selection": RECIPE3_DEPLOYMENT_CANDIDATE_SELECTION,
        "max_deployment_candidates": 12,
        "source_plan": "recipe3_excluded_records_independent_buckets_v1",
    }
    assert policy == {**legacy, **changes}
    assert candidate == original
    assert compatible_capture_profiles({}, candidate) == []


def test_recipe3_profile_keeps_original_centered_perhead_identity_and_isolates_cache(candidate, tmp_path, monkeypatch):
    monkeypatch.delenv("OSCAR_ROTATION_ARTIFACT", raising=False)
    base_config = {key: value for key, value in candidate.items()
                   if key not in ("optr_baseline", "rotation_refinement")}
    legacy_config = {key: value for key, value in candidate.items() if key != "optr_baseline"}
    base = calibration_cache_identity(base_config, "model")
    profile = calibration_cache_identity(candidate, "model")
    assert base["recipe_version"] == 3
    assert profile["base_calibration_profile"] == base
    assert profile["base_calibration_profile"] != calibration_cache_identity(legacy_config, "model")["base_calibration_profile"]
    paths = {rotation_artifact_path(config, "model", root=tmp_path)
             for config in (base_config, legacy_config, candidate)}
    assert len(paths) == 3
    # The preparation process authenticates this dynamic source proof through
    # the certificate/trace chain, without changing the predetermined PT path.
    prepared = {**candidate, "optr_seed_capture": {"baseline_artifact_sha256": "a" * 64},
                "baseline_artifact_sha256": "a" * 64}
    assert calibration_cache_identity(prepared, "model") == profile
    assert rotation_artifact_path(prepared, "model", root=tmp_path) == rotation_artifact_path(candidate, "model", root=tmp_path)


@pytest.mark.parametrize("override", [
    {"policy_version": "old"}, {"fitted_grouping": "layer"},
    {"selection": "average_only"}, {"max_deployment_candidates": 100},
    {"source_plan": "disjoint_records_packed_calibration_only_v1"},
    {"validation_projection_views": ["fp32"]},
    {"deployment_candidate_selection": "test_reselection"},
])
def test_new_policy_cannot_relax_frozen_acceptance_or_source_plan(candidate, override):
    with pytest.raises(ValueError, match="frozen"):
        optr_policy({**candidate, "optr_calibration": override})


@pytest.mark.parametrize("marker", ["shared_zero", "recipe4", "", False, 3])
def test_unknown_baseline_is_rejected(candidate, marker):
    with pytest.raises(ValueError, match="optr_baseline"):
        validate_refinement_config({**candidate, "optr_baseline": marker})
    with pytest.raises(ValueError, match="optr_baseline"):
        variant_config({**candidate, "optr_baseline": marker}, "candidate")


def test_baseline_marker_requires_refinement(candidate):
    with pytest.raises(ValueError, match="requires OptR"):
        validate_refinement_config({**candidate, "rotation_refinement": None})


@pytest.mark.parametrize("variant", ["native", "baseline"])
def test_controls_remove_recipe3_capture_metadata_without_changing_online_bundle(candidate, variant):
    old = {key: value for key, value in candidate.items() if key != "optr_baseline"}
    generated = {**candidate, "optr_seed_capture": {"baseline_artifact_sha256": "a" * 64},
                 "optr_calibration": {"steps": 1}}
    assert variant_config(generated, variant) == variant_config(old, variant)
    assert not {"optr_baseline", "optr_seed_capture", "optr_calibration"} & variant_config(generated, variant).keys()


def test_summary_reports_real_baseline_and_strict_publication_without_online_changes(candidate):
    legacy = {key: value for key, value in candidate.items() if key != "optr_baseline"}
    summary = runtime_feature_summary(candidate, "candidate")
    old_summary = runtime_feature_summary(legacy, None)
    assert summary["optr_publication_policy"] == RECIPE3_PUBLICATION_POLICY
    assert summary["independent_test_blocks_publication"] is True
    assert summary["optr_seed"] == "recipe3_perhead_center"
    assert summary["optr_fitted_grouping"] == "layer_kv_head"
    assert summary["key_centering"] is True
    assert old_summary["optr_seed"] == "oscar_base_shared_zero_mean"
    assert old_summary["key_centering"] is False
    assert old_summary["independent_test_blocks_publication"] is False
    assert serve_argv(candidate) == serve_argv(legacy)
    assert variant_features(candidate) == variant_features(legacy)
    assert variant_config(legacy, None) == legacy
    assert variant_config(legacy, "candidate") == candidate
