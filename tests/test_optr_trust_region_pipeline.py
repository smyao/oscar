"""Archive #175/#176: authenticated old proposals, new bounded CPU repair.

These synthetic CPU fixtures exercise real pair/source/capture certificates,
BF16 replay and strict publication. They do not measure model accuracy/NPU.
"""
import hashlib
import json

import pytest
import torch

from oscar_ascend import optr_checkpoint, optr_sequential, optr_trust_region
from oscar_ascend.optr_policy import optr_policy
from oscar_ascend.rotations import calibration_cache_identity, load_active_artifact
from tools import prepare_optr_recipe3, prepare_optr_rotations, prepare_rotations
from test_optr_recipe3_pipeline import NAME, case, capture, forbidden, proposal


LEGACY_DIGEST = "a30488344d57137d7bfc0623e074716744369eef999ac1827014fd907e02a917"
REPAIR = "single_head_trust_region_v1"
VERSION = "optr_recipe3_trust_region_ascend_v11"


def _microstep_teacher(seed, source_result):
    direction = optr_trust_region._source_directions(seed, source_result)[0]
    head, side = direction["head"], direction["side"]
    teacher = {**seed, side: seed[side].clone()}
    teacher[side][head] = seed[side][head] @ torch.matrix_exp(
        direction["unit_tangent"] * optr_trust_region.REPAIR_POLICY["radii"][0])
    return teacher


def _make_rejected_source(case, monkeypatch, *, fail_test=False):
    """Generate real old checkpoints; only their historic code identity is pinned.

    The teacher equals one fixed small single-head proposal. The original
    full-head large proposal therefore loses the strict gate. That prescribed
    synthetic teacher provides an accepted repair without mocking metrics.
    """
    source_case = {**case, "candidate": case["candidate"]}
    seed = case["baseline"]["layers"][NAME]
    old_result = proposal(source_case)(seed["Rk"], seed["Rv"], seed["k_mean"],
        [[{"split": "train"}]], [[{"split": "validation"}]], policy=optr_policy(case["config"]))
    case["calls"].clear()
    case["candidate"] = _microstep_teacher(seed, old_result)

    profile = calibration_cache_identity(case["config"], case["fingerprint"])
    profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()[:20]
    pool = case["output"].parent / "generations"
    old_output = pool / ("optr_" + profile_hash) / "refined.pt"
    # Do not replace source validation/hash code. The source adapter requires
    # the historic, immutable v10 implementation identity even in CPU tests.
    with monkeypatch.context() as historical:
        historical.setattr(optr_checkpoint, "fitting_implementation_digest", lambda: LEGACY_DIGEST)
        historical.setattr(prepare_optr_recipe3, "fitting_implementation_digest", lambda: LEGACY_DIGEST)
        with pytest.raises(ValueError, match="no strict validation improvement"):
            prepare_optr_rotations.prepare(case["config"], output=old_output, testing=True,
                capture_runner=capture(case, fail_test=fail_test), fitter=proposal(source_case),
                base_runner=forbidden)
    assert case["calls"] == ["capture", "fit"]
    assert not old_output.exists()
    source_files = {path: path.read_bytes() for path in old_output.parent.rglob("*") if path.is_file()}
    config = {**case["config"], "optr_recipe3_repair": REPAIR}
    output = pool / "repair" / "refined.pt"
    return config, output, old_output, source_files


def _forbid_new_fit_or_capture(monkeypatch):
    monkeypatch.setattr(prepare_optr_recipe3, "fit_perhead", forbidden)
    monkeypatch.setattr(optr_sequential, "fitting_device", forbidden)
    monkeypatch.setattr(prepare_optr_rotations.subprocess, "run", forbidden)


def _prepare(config, output):
    return prepare_rotations.prepare(config, output=output, testing=True,
                                     capture_runner=forbidden)


def test_v11_fixed_preparer_reuses_real_v10_proposals_and_resume_keeps_selected_layer(
        case, monkeypatch, capsys):
    config, output, _, source_files = _make_rejected_source(case, monkeypatch)
    _forbid_new_fit_or_capture(monkeypatch)
    selected_calls = []
    select = optr_trust_region.select_repair

    def record_selection(*args, **kwargs):
        selected_calls.append(1)
        return select(*args, **kwargs)

    monkeypatch.setattr(optr_trust_region, "select_repair", record_selection)
    result = _prepare(config, output)
    assert result["accepted_layers"] == [NAME]
    assert result["refinement_policy_version"] == VERSION
    assert selected_calls == [1] and case["calls"] == ["capture", "fit"]
    artifact = load_active_artifact(config, path=output, device="cpu", testing=True)
    certificate = artifact["refinement_certificate"]
    record = certificate["layers"][NAME]
    assert certificate["policy"]["policy_version"] == VERSION
    assert certificate["independent_test"]["status"] == "passed"
    assert record["source_fit_provenance"]["fitting_implementation_sha256"] == LEGACY_DIGEST
    assert record["recipe3_repair_selection"]["test_used_for_selection"] is False
    assert record["recipe3_repair_selection"]["accepted"] is True
    assert record["validation"]["selected"]["squared_error"] < record["validation"]["baseline"]["squared_error"]
    assert torch.equal(artifact["layers"][NAME]["k_mean"], case["baseline"]["layers"][NAME]["k_mean"])
    assert all(path.read_bytes() == content for path, content in source_files.items())

    # Recreate publication from the completed new layer, rather than merely
    # taking the final-PT cache early return. Neither new selection nor replay
    # may run when the authenticated completed layer already exists.
    output.unlink()
    (output.parent / ".optr.complete.json").unlink()
    monkeypatch.setattr(optr_trust_region, "select_repair", forbidden)
    monkeypatch.setattr(prepare_optr_rotations, "refine_layer", forbidden)
    capsys.readouterr()
    resumed = _prepare(config, output)
    assert resumed["accepted_layers"] == [NAME] and output.is_file()
    assert "layer_checked_resumed" in capsys.readouterr().out
    assert record == load_active_artifact(config, path=output, device="cpu", testing=True)["refinement_certificate"]["layers"][NAME]
    assert all(path.read_bytes() == content for path, content in source_files.items())


def test_v11_independent_test_failure_never_reselects_or_publishes(case, monkeypatch):
    config, output, _, source_files = _make_rejected_source(case, monkeypatch, fail_test=True)
    _forbid_new_fit_or_capture(monkeypatch)
    selected_calls = []
    select = optr_trust_region.select_repair

    def record_selection(*args, **kwargs):
        selected_calls.append(1)
        return select(*args, **kwargs)

    monkeypatch.setattr(optr_trust_region, "select_repair", record_selection)
    with pytest.raises(prepare_optr_rotations.IndependentTestRegression, match="not published"):
        _prepare(config, output)
    assert selected_calls == [1]
    assert not output.exists() and not (output.parent / ".optr.complete.json").exists()
    assert list(output.parent.rglob("*independent-test-failed.json"))
    monkeypatch.setattr(optr_trust_region, "select_repair", forbidden)
    with pytest.raises(prepare_optr_rotations.IndependentTestRegression, match="not published"):
        _prepare(config, output)
    assert not output.exists() and case["calls"] == ["capture", "fit"]
    assert all(path.read_bytes() == content for path, content in source_files.items())


def test_v11_source_hash_corruption_stops_before_repair(case, monkeypatch):
    config, output, old_output, _ = _make_rejected_source(case, monkeypatch)
    pointer = next((old_output.parent / "fit-checkpoints").glob("*/layer-3-shared.json"))
    manifest = json.loads(pointer.read_text())
    payload = pointer.parent / manifest["file"]
    payload.write_bytes(payload.read_bytes() + b"corrupt-source")
    _forbid_new_fit_or_capture(monkeypatch)
    monkeypatch.setattr(optr_trust_region, "select_repair", forbidden)
    with pytest.raises(ValueError, match="hash|source|checkpoint"):
        _prepare(config, output)
    assert not output.exists() and case["calls"] == ["capture", "fit"]


def test_fresh_v11_capture_repairs_twelve_rejected_proposals_without_historical_source(case):
    config = {**case["config"], "optr_recipe3_repair": REPAIR,
              "optr_calibration": {**case["config"]["optr_calibration"], "steps": 80, "eval_every": 20}}
    policy = optr_policy(config)
    seed = case["baseline"]["layers"][NAME]
    far = case["candidate"]
    result = proposal(case)(seed["Rk"], seed["Rv"], seed["k_mean"],
        [[{"split": "train"}]], [[{"split": "validation"}]], policy=policy)
    case["calls"].clear()
    candidates = list(result["candidates"])
    heads, dim = seed["Rk"].shape[:2]

    def displaced(side, radius):
        relative = seed[side].transpose(-1, -2) @ far[side]
        tangent = (relative - relative.transpose(-1, -2)) * .5
        rms = torch.linalg.vector_norm(tangent, dim=(-2, -1)) / dim ** .5
        return seed[side] @ torch.matrix_exp(tangent / rms[:, None, None] * radius)

    for index, step in enumerate((1, 21, 41, 61, 80)):
        candidates.append({"id": f"k_step_{step}", "phase": "k", "step": step,
            "Rk": displaced("Rk", .35 + index * .04), "Rv": seed["Rv"], "proxy_losses": [1.] * heads})
    for index, step in enumerate((0, 1, 21, 41, 61, 80)):
        candidates.append({"id": f"v_step_{step}", "phase": "v", "step": step,
            "Rk": far["Rk"], "Rv": seed["Rv"] if step == 0 else displaced("Rv", .35 + index * .04),
            "proxy_losses": [1.] * heads})
    result["candidates"] = candidates
    result["decision"]["candidate_count"] = len(candidates)
    assert len(candidates) == 12
    case["candidate"] = _microstep_teacher(seed, result)

    def fitted(_rk, _rv, _mean, train, validation, *, policy):
        assert all(sample["split"] == "train" for head in train for sample in head)
        assert all(sample["split"] == "validation" for head in validation for sample in head)
        assert policy == result["decision"]["policy"]
        case["calls"].append("fit")
        return result

    prepared = prepare_optr_rotations.prepare(config, output=case["output"], testing=True,
        capture_runner=capture(case), fitter=fitted, base_runner=forbidden)
    assert prepared["accepted_layers"] == [NAME] and case["calls"] == ["capture", "fit"]
    artifact = load_active_artifact(config, path=case["output"], device="cpu", testing=True)
    certificate = artifact["refinement_certificate"]
    record = certificate["layers"][NAME]
    assert certificate["policy"]["policy_version"] == VERSION
    assert "source_fit_provenance" not in record
    assert record["shared_layer_selection"]["accepted"] is False
    assert len(record["shared_layer_selection"]["evaluated_candidates"]) == 12
    assert all(not row["accepted"] for row in record["shared_layer_selection"]["evaluated_candidates"])
    assert record["recipe3_repair_selection"]["accepted"] is True
    assert certificate["independent_test"]["status"] == "passed"
    trace = json.loads((case["output"].parent / "native-traces/capture-manifest.json").read_text())
    assert trace["optr_policy"]["policy_version"] == VERSION


def test_new_authenticated_source_generation_cannot_reuse_old_completed_repair(case, monkeypatch):
    config, output, old_output, _ = _make_rejected_source(case, monkeypatch)
    _forbid_new_fit_or_capture(monkeypatch)
    assert _prepare(config, output)["accepted_layers"] == [NAME]
    completed = next((output.parent / "fit-checkpoints").glob("*/layer-3.json"))
    old_completed_bytes = completed.read_bytes()
    old_identity = json.loads(old_completed_bytes)["identity"]
    assert old_identity["source_fit_snapshot_sha256"]
    output.unlink()
    (output.parent / ".optr.complete.json").unlink()

    # Publish a fully valid new v10 proposal generation with identical tensor
    # contents. Authentication must succeed: only its atomic pointer/file
    # snapshot changes. A corrupted source would test a different early gate.
    pointer = next((old_output.parent / "fit-checkpoints").glob("*/layer-3-shared.json"))
    before = json.loads(pointer.read_text())
    payload = optr_checkpoint.FitCheckpoints._load_from(pointer, "layer-3-shared", before["identity"])
    source_cache = optr_checkpoint.FitCheckpoints(pointer.parent.parent, before["identity"])
    source_cache.save("layer-3-shared", payload)
    after = json.loads(pointer.read_text())
    assert after["identity"] == before["identity"] and after["file"] != before["file"]

    searched = []

    def require_new_selection(*_args, **_kwargs):
        searched.append(1)
        raise RuntimeError("new authenticated source snapshot requires new repair selection")

    monkeypatch.setattr(optr_trust_region, "select_repair", require_new_selection)
    with pytest.raises(RuntimeError, match="new authenticated source snapshot"):
        _prepare(config, output)
    assert searched == [1]  # Reusing the old completed layer would never call it.
    assert not output.exists() and not (output.parent / ".optr.complete.json").exists()
    assert completed.read_bytes() == old_completed_bytes
    identities = [json.loads(path.read_text())["identity"]
                  for path in (output.parent / "fit-checkpoints").glob("*/layer-3-shared.json")]
    snapshots = {identity["source_fit_snapshot_sha256"] for identity in identities}
    assert len(identities) == len(snapshots) == 2
    assert old_identity["source_fit_snapshot_sha256"] in snapshots
