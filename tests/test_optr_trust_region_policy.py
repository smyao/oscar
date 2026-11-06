"""Archive #176/#177: explicit v11 retains its policy and online bundle."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import types

import pytest

from oscar_ascend.optr_checkpoint import fitting_implementation_digest
from oscar_ascend.optr_policy import (compatible_capture_profiles, optr_policy,
                                     validate_refinement_config)
from oscar_ascend.rotations import calibration_cache_identity, rotation_artifact_path
from tools.serving_variants import runtime_feature_summary, variant_config, variant_features
from tools.target_cli import serve_argv, target_env


from test_optr_cpu_scope_20261105 import OFFLINE, _without_profile, _front_card_target


ROOT = Path(__file__).resolve().parents[1]
BASE = "e62d6d2"
REPAIR = "single_head_trust_region_v1"
OFFLINE_SEAMS = {
    "oscar_ascend/optr_artifacts.py", "oscar_ascend/optr_checkpoint.py",
    "oscar_ascend/optr_policy.py", "oscar_ascend/optr_recipe3_certificate.py",
    "tools/optr_fit_worker.py", "tools/optr_rejection_report.py",
    "tools/prepare_optr_recipe3.py", "tools/prepare_optr_recipe3_base.py",
    "tools/prepare_optr_rotations.py", "tools/serving_variants.py",
}


def _historical(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def _target():
    return json.loads(_historical("configs/target.json"))


def _legacy_candidate():
    selected = variant_config(_target(), "candidate")
    selected.pop("optr_recipe3_optimizer")
    selected["optr_recipe3_repair"] = REPAIR
    return variant_config(selected, None)


def test_explicit_v11_changes_only_the_offline_marker_and_policy_receipt():
    old = types.ModuleType("historical_serving_variants")
    exec(compile(_historical("tools/serving_variants.py"), "historical_serving_variants.py", "exec"),
         old.__dict__)
    config = _target()
    previous = old.variant_config(config, "candidate")
    selected = _legacy_candidate()
    assert selected == {**previous, "optr_recipe3_repair": REPAIR}
    assert variant_features(selected) == old.variant_features(previous)
    assert serve_argv(selected) == serve_argv(previous)
    assert target_env(selected, base={}) == target_env(previous, base={})
    before = old.runtime_feature_summary(previous, "candidate")
    after = runtime_feature_summary(selected, "candidate")
    assert before["optr_fitting_policy"] == "optr_recipe3_centered_candidates_ascend_v10"
    assert after["optr_fitting_policy"] == "optr_recipe3_trust_region_ascend_v11"
    assert after.pop("optr_recipe3_repair") == optr_policy(selected)["recipe3_repair"]
    after["optr_fitting_policy"] = before["optr_fitting_policy"]
    assert after == before
    assert config == _target()


def test_only_exact_v10_sampling_is_compatible_and_all_artifact_profiles_are_isolated(tmp_path):
    selected = _legacy_candidate()
    old = {key: value for key, value in selected.items() if key != "optr_recipe3_repair"}
    shared = {key: value for key, value in old.items() if key != "optr_baseline"}
    fingerprint = "policy-test-model"
    profile, old_profile, shared_profile = [calibration_cache_identity(row, fingerprint)
                                            for row in (selected, old, shared)]
    assert compatible_capture_profiles(profile, selected) == [old_profile]
    assert compatible_capture_profiles(old_profile, old) == []
    assert shared_profile not in compatible_capture_profiles(profile, selected)
    assert profile != old_profile != shared_profile
    assert profile["base_calibration_profile"] == old_profile["base_calibration_profile"]
    assert profile["refinement_policy"]["source_plan"] == old_profile["refinement_policy"]["source_plan"]
    paths = {rotation_artifact_path(row, fingerprint, root=tmp_path) for row in (selected, old, shared)}
    assert len(paths) == 3
    changed = copy.deepcopy(selected)
    changed["optr_calibration"] = {"capture_lengths": {split: [4096, 32768]
                                                      for split in ("train", "validation", "test")}}
    changed_profile = calibration_cache_identity(changed, fingerprint)
    assert old_profile not in compatible_capture_profiles(changed_profile, changed)
    assert compatible_capture_profiles(changed_profile, changed)[0]["refinement_policy"]["capture_lengths"] == changed["optr_calibration"]["capture_lengths"]


@pytest.mark.parametrize("variant", ["native", "baseline"])
def test_other_serving_variants_clear_repair_marker(variant):
    selected = _legacy_candidate()
    actual = variant_config(selected, variant)
    assert "optr_recipe3_repair" not in actual
    assert actual.get("rotation_refinement") is None
    assert actual.get("optr_baseline") is None
    assert not any(variant_features(actual).values())


@pytest.mark.parametrize("changes", [
    {"optr_recipe3_repair": "unknown"},
    {"optr_recipe3_repair": True},
    {"optr_baseline": None},
    {"rotation_refinement": None},
])
def test_invalid_explicit_repair_configuration_is_rejected(changes):
    config = {**_legacy_candidate(), **changes}
    with pytest.raises(ValueError):
        validate_refinement_config(config)
    if changes.get("rotation_refinement", "optr") == "optr":
        with pytest.raises(ValueError):
            optr_policy(config)


def test_candidate_does_not_silently_correct_an_unknown_explicit_repair():
    config = {**_target(), "optr_recipe3_repair": "unknown"}
    with pytest.raises(ValueError, match="repair"):
        variant_config(config, "candidate")


def test_current_implementation_digest_covers_both_new_algorithm_and_source_reader():
    source = ROOT / "oscar_ascend/optr_checkpoint.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "fitting_implementation_digest")
    names = next(ast.literal_eval(node.value) for node in function.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "names" for target in node.targets))
    assert {"oscar_ascend/optr_trust_region.py", "tools/optr_recipe3_source.py"} <= set(names)
    expected = hashlib.sha256()
    for name in names:
        expected.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    assert fitting_implementation_digest() == expected.hexdigest()
    from tools.optr_recipe3_source import LEGACY_FITTING_SHA256
    assert fitting_implementation_digest() != LEGACY_FITTING_SHA256


def test_every_existing_production_file_outside_named_offline_seams_is_byte_identical():
    records = subprocess.check_output(["git", "ls-tree", "-r", "-z", BASE,
        "csrc", "cmake", "oscar_ascend", "configs", "scripts", "tools"], cwd=ROOT).split(b"\0")
    changed = []
    checked = 0
    for record in filter(None, records):
        header, encoded = record.split(b"\t", 1)
        _, kind, expected = header.split()
        name = encoded.decode()
        if kind != b"blob" or name in (OFFLINE_SEAMS | OFFLINE):
            continue
        path = ROOT / name
        if not path.is_file():
            changed.append(name)
            continue
        data = path.read_bytes()
        if name in ("configs/target.json", "tools/probe_history_reuse.py"):
            # User 2026-11-06 authorized only the new target placement.
            previous = subprocess.check_output(["git", "show", f"7720e3d:{name}"], cwd=ROOT)
            assert data == (_front_card_target(previous) if name == "configs/target.json" else previous)
            checked += 1
            continue
        if name == "oscar_ascend/rotations.py":
            previous = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=ROOT)
            assert _without_profile(data) == _without_profile(previous)
            checked += 1
            continue
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected.decode():
            changed.append(name)
        checked += 1
    assert checked == 880  # Four more pre-existing files are named offline CPU seams.
    assert changed == []
