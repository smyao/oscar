"""Archive #148/#163/#170/#172/#174: isolate the requested OptR migration.

The independently committed oscar-new-6 base is the performance/source oracle.
OptR changes offline constants and preparation only. These CPU/source checks
do not assert NPU execution time, graph replay, or model-quality acceptance.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import ModuleType

import pytest

from tools import install_serve, observe_serve, serving_variants
from tools.target_cli import serve_argv

ROOT = Path(__file__).resolve().parents[1]
BASE = "ca46f57e0267f15aadba08d6913cef81b46e21b1"
SOURCE = "e7dc8a92000e0d9b49800eadda6d1d16f978436f"
EXISTING_CALIBRATION_SEAMS = {
    "oscar_ascend/rotations.py", "oscar_ascend/integration/rotation_capture.py",
    "tools/calibrate_model_rotations.py", "tools/calibration_data.py",
    "tools/prepare_rotations.py", "tools/serving_variants.py",
    "tools/install_serve.py", "tools/deploy.py", "tools/observe_serve.py",
    "tools/phase.py", "tools/audit_precision_identity.py",
}


def committed(path, revision=BASE):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


@pytest.fixture(scope="module")
def previous_variants():
    module = ModuleType("tools._optr_migration_base_variants")
    module.__file__ = str(ROOT / "tools/serving_variants.py")
    exec(compile(committed("tools/serving_variants.py"), module.__file__, "exec"), module.__dict__)
    return module


def test_existing_production_tree_changes_only_at_named_calibration_seams():
    # Git's committed blob IDs are independent of the current implementation.
    # Include generator/probes/timers so unchanged kernels cannot be replaced
    # during build or selected by an altered runtime after this test passes.
    records = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", BASE, "csrc", "cmake", "oscar_ascend",
         "configs", "scripts", "tools"], cwd=ROOT).split(b"\0")
    changed = []
    for record in filter(None, records):
        header, raw_path = record.split(b"\t", 1)
        _, kind, expected = header.split()
        path = raw_path.decode()
        if kind != b"blob" or path in EXISTING_CALIBRATION_SEAMS:
            continue
        current = ROOT / path
        if not current.is_file():
            changed.append(path + " (missing)")
            continue
        data = current.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected.decode():
            changed.append(path)
    assert changed == []


@pytest.mark.parametrize("path", [
    "csrc/kernels/attention_cv_official_cluster16.cpp",
    "csrc/kernels/attention_official_finite.h",
    "oscar_ascend/integration/prefill_state_contract.py",
    "oscar_ascend/plugin.py",
])
def test_source_branch_older_online_implementations_are_not_imported(path):
    source_entry = subprocess.check_output(["git", "ls-tree", SOURCE, "--", path], cwd=ROOT)
    # The source branch predates the finite helper altogether.
    assert not source_entry or committed(path, SOURCE) != committed(path)
    assert (ROOT / path).read_bytes() == committed(path)


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("mode", ["default", "native_budget", "hadamard", "w8a8"])
@pytest.mark.parametrize("variant", [None, "candidate", "baseline", "native"])
def test_all_existing_flags_and_native_arguments_match_base(previous_variants, prefix, mode, variant):
    original = json.loads(committed("configs/target.json"))
    original.update(enable_prefix_caching=prefix, oscar_current_only_policy="native_exact")
    if mode == "native_budget":
        original["use_native_token_budget"] = True
    elif mode == "hadamard":
        original["rotation_method"] = "hadamard"
    elif mode == "w8a8":
        original["quantization"] = "ascend"
    before = json.loads(json.dumps(original))
    expected = previous_variants.variant_config(original, variant)
    actual = serving_variants.variant_config(original, variant)
    enabled = variant == "candidate" and mode == "default"
    assert actual == {**expected, **({"rotation_refinement": "optr"} if enabled else {})}
    assert serving_variants.variant_features(actual) == previous_variants.variant_features(expected)
    assert serve_argv(actual) == serve_argv(expected)
    assert "oscar_current_only_policy" not in actual
    assert original == before


@pytest.mark.parametrize("variant", ["baseline", "native"])
def test_reused_optr_candidate_does_not_contaminate_controls(previous_variants, variant):
    raw = json.loads(committed("configs/target.json"))
    old_candidate = previous_variants.variant_config(raw, "candidate")
    candidate = serving_variants.variant_config(raw, "candidate")
    candidate.update(optr_calibration={"steps": 5}, oscar_current_only_policy="native_exact")
    cleaned = serving_variants.variant_config(candidate, variant)
    assert cleaned == previous_variants.variant_config(old_candidate, variant)
    assert not {"rotation_refinement", "optr_calibration", "oscar_current_only_policy"} & cleaned.keys()


@pytest.mark.parametrize("entry", [install_serve, observe_serve])
@pytest.mark.parametrize("prefix", [False, True])
def test_fixed_entrypoints_keep_base_launch_and_online_selection(
        previous_variants, entry, prefix, capsys, monkeypatch, tmp_path):
    args = ["--variant", "candidate", "--plan", "--log-dir", str(tmp_path / "logs"),
            "--enable-prefix-caching" if prefix else "--no-enable-prefix-caching"]
    assert entry.main(args) == 0
    actual = json.loads(capsys.readouterr().out)
    with monkeypatch.context() as patch:
        patch.setattr(entry, "variant_config", previous_variants.variant_config)
        patch.setattr(entry, "runtime_feature_summary", previous_variants.runtime_feature_summary)
        assert entry.main(args) == 0
        expected = json.loads(capsys.readouterr().out)
    selection = actual.pop("runtime_selection")
    old_selection = expected.pop("runtime_selection")
    assert actual == expected
    offline_summary_keys = {key for key in selection if key.startswith("optr_")} | {
        "rotation_refinement", "independent_test_blocks_publication", "key_centering",
        "calibration_recipe_version"}
    assert {k: v for k, v in selection.items() if k not in offline_summary_keys} == {
        k: v for k, v in old_selection.items() if k not in offline_summary_keys}
    assert selection["rotation_refinement"] == "optr"
    assert selection["calibration_recipe_version"] == 5
    assert selection["port"] == 7878 and selection["devices"] == [0, 1, 2, 3]


@pytest.mark.parametrize("path", ["tools/install_serve.py", "tools/deploy.py", "tools/observe_serve.py"])
def test_launchers_only_change_offline_preparation_deadline(path):
    # Structurally erase exactly the approved timeout seam, then compare the
    # complete executable AST to the committed launcher. Process ownership,
    # service exec, cleanup and original failure handling must remain intact.
    tree = ast.parse((ROOT / path).read_text())
    baseline = ast.parse(committed(path))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "rotation_preparation_limits"
        and [(item.name, item.asname) for item in node.names] == [("phase_timeout", None)])]
    expected_timeout = ast.parse(
        'float(config["phase_timeout_seconds"])' if path == "tools/observe_serve.py"
        else 'config["phase_timeout_seconds"]', mode="eval").body

    class RestoreDeadline(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "phase_timeout":
                assert ast.dump(node) == ast.dump(ast.parse("phase_timeout(config, name)", mode="eval").body)
                return expected_timeout
            return self.generic_visit(node)

    tree = RestoreDeadline().visit(tree)
    assert ast.dump(tree) == ast.dump(baseline)
