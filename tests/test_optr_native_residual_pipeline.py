"""Archive #170/#175/#176/#177: authenticated v12 preparation and strict gates.

Real CPU recipe3 pairs, trace/source hashes, replay, gates and publication are
used. Capture teachers are deliberately synthetic replay outputs. Positive
publication cases inject one disclosed tangent, never acceptance or metrics;
the zero-improvement case also runs the actual streamed gradient. No example
certifies NPU equivalence, model quality, or performance.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

from oscar_ascend import optr_native_residual as native
from oscar_ascend import optr_sequential, optr_traces
from oscar_ascend.optr_artifacts import nonworse
from oscar_ascend.optr_checkpoint import FitCheckpoints
from oscar_ascend.optr_native_residual_certificate import POLICY_VERSION
from oscar_ascend.rotations import calibration_cache_identity, load_active_artifact
from tools import optr_parallel, prepare_optr_rotations as prepare
from tools import prepare_optr_native_residual as native_prepare
from test_optr_recipe3_pipeline import NAME, case, capture, forbidden, proposal


OPTIMIZER = "joint_native_residual_v1"
NO_IMPROVEMENT = "no strict validation improvement"


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(case):
    return {**case["config"], "optr_recipe3_optimizer": OPTIMIZER}


def _tangent(dim):
    value = torch.zeros(dim, dim)
    value[0, 1], value[1, 0] = (dim / 2.) ** .5, -(dim / 2.) ** .5
    return value


def _prescribe_teacher_and_direction(case, monkeypatch):
    """Disclosed synthetic optimizer-direction injection; real strict scoring.

    Only head0/K moves. Its larger fixed radius exactly generates the captured
    synthetic teacher, so acceptance remains a real BF16 replay comparison.
    """
    baseline = case["baseline"]["layers"][NAME]
    tangent = _tangent(baseline["Rk"].shape[-1])
    direction = {"head": 0, "side": "Rk", "unit_tangent": tangent}
    case["candidate"] = native._proposal(baseline, direction, native.NATIVE_RESIDUAL_POLICY["radii"][0])
    calls = []

    def prescribed(seed, head, side, samples, policy, cache, *, progress=None):
        assert all(sample["split"] == "train" for rows in samples for sample in rows)
        assert all(torch.equal(seed[key], baseline[key]) for key in ("Rk", "Rv", "k_mean"))
        calls.append((head, side))
        live = head == 0 and side == "Rk"
        unit = tangent.clone() if live else torch.zeros_like(tangent)
        metrics = [cache.before, *cache.before["projection_checks"].values()]
        normalized = [query["normalized_error"] for metric in metrics
                      for request in metric["requests"] for frame in request["frames"]
                      for query in frame["queries"]]
        return {"head": head, "side": side, "gradient_rms": 1. if live else 0.,
            "training_objective": sum(normalized) / len(normalized), "zero_gradient": not live,
            "unit_tangent_sha256": native._tensor_digest(unit), "unit_tangent": unit}

    monkeypatch.setattr(native, "_gradient_direction", prescribed)
    return calls


def _read_checkpoint(output, owner):
    pointer = next((output.parent / "fit-checkpoints").glob("*/" + owner + ".json"))
    manifest = json.loads(pointer.read_text())
    cache = FitCheckpoints(pointer.parent.parent, manifest["identity"])
    return pointer, manifest, cache, cache.load(owner)


def _prepare(case, *, capture_runner=None, parallel_testing=False):
    return prepare.prepare(_config(case), output=case["output"], testing=True,
        capture_runner=capture_runner or forbidden, base_runner=forbidden,
        parallel_testing=parallel_testing)


def _observed_real_worker_entry(target, args):
    """Observe calls only; execute the unchanged production worker entry.

    Spawn has a fresh interpreter, so a parent monkeypatch of fitting_device
    would prove nothing. This call profiler fails if that function is entered;
    it changes no operands, directions, proposals, losses or acceptance gates.
    """
    # The shared ``case`` fixture relocates the authenticated canonical cache
    # with a parent monkeypatch. Spawn does not inherit that module constant;
    # explicitly restore the *same test fixture root* before real load_base.
    # Identity, pair, file, trace and selection authentication still run intact.
    from tools import prepare_optr_recipe3_base
    spec = args[0]
    assert spec["testing"] is True
    prepare_optr_recipe3_base.ROOT = Path(spec["base_config"]["model"]).parent

    def observe(frame, event, _arg):
        if (event == "call" and frame.f_code.co_name == "fitting_device" and
                frame.f_globals.get("__name__") == "oscar_ascend.optr_sequential"):
            raise AssertionError("native-residual spawn worker entered proxy fitting_device")

    args[-1].send({"kind": "progress", "layer": None,
                  "fields": {"phase": "test_real_worker_call_observer_installed"}})
    sys.setprofile(observe)
    try:
        # This process imports the original module; the parent-only override
        # selected this top-level, picklable observation wrapper as spawn target.
        optr_parallel._worker_entry(target, args)
    finally:
        sys.setprofile(None)


def test_real_spawn_worker_performs_first_native_fit_on_cpu_and_matches_serial_metrics(case, monkeypatch, capsys):
    # Use the actual nonzero-gradient fixture, with no injected direction.
    # The small case's optimization outcome is honestly allowed to reject.
    def run(current, runner, parallel):
        try:
            return _prepare(current, capture_runner=runner, parallel_testing=parallel)
        except ValueError as error:
            assert NO_IMPROVEMENT in str(error)
            return None

    serial_result = run(case, capture(case), False)
    _, _, _, serial = _read_checkpoint(case["output"], "layer-3")
    serial_record = serial["artifact"]["refinement_certificate"]["layers"][NAME]
    assert any(row["gradient_rms"] > 0 for row in serial_record["native_residual_selection"]["directions"])
    parallel = {**case, "output": case["output"].parent.parent / "parallel-native" / "refined.pt"}
    assert not parallel["output"].parent.exists()
    # The deterministic fixture emits the same raw tensors into its own
    # authenticated capture directory. There is no completed layer/selection
    # lock in the new directory; a real child must perform first fitting.
    monkeypatch.setattr(optr_parallel, "_worker_entry", _observed_real_worker_entry)
    capsys.readouterr()
    parallel_result = run(parallel, capture(parallel), True)
    events = [json.loads(line.split("OPTR_PROGRESS ", 1)[1])
              for line in capsys.readouterr().out.splitlines() if line.startswith("[oscar] OPTR_PROGRESS ")]
    assert next(row for row in events if row.get("phase") == "parallel_fit_plan")["workers"] == 1
    assert any(row.get("phase") == "test_real_worker_call_observer_installed" for row in events)
    assert next(row for row in events if row.get("phase") == "fit_worker_ready")["fitting_device"] == "cpu"
    assert any(row.get("event") == "native_residual_gradient_request" for row in events)
    assert any(row.get("phase") == "layer_checked_saved" for row in events)
    assert any(row.get("phase") == "layer_checked_collected" for row in events)
    assert next(row for row in events if row.get("phase") == "fit_worker_exited")["returncode"] == 0
    assert next(row for row in events if row.get("phase") == "fit_workers_cleanup")["cleanup_complete"] is True
    _, _, _, completed = _read_checkpoint(parallel["output"], "layer-3")
    certificate = completed["artifact"]["refinement_certificate"]
    record = certificate["layers"][NAME]
    assert certificate["policy"]["policy_version"] == POLICY_VERSION
    assert record["validation"] == serial_record["validation"]
    assert record["test"] == serial_record["test"]
    assert record["accepted"] is serial_record["accepted"]
    observed = record["native_residual_selection"]
    expected = serial_record["native_residual_selection"]
    assert observed["train_baseline"] == expected["train_baseline"]
    assert [row["train"] for row in observed["evaluated_train_candidates"]] == [
        row["train"] for row in expected["evaluated_train_candidates"]]
    assert (parallel_result is None) is (serial_result is None)
    if not record["accepted"]:
        assert not parallel["output"].exists()
        assert not (parallel["output"].parent / ".optr.complete.json").exists()
    else:
        assert load_active_artifact(_config(parallel), path=parallel["output"], device="cpu", testing=True)


def test_native_pipeline_publishes_only_actual_strict_train_validation_and_test_improvement(case, monkeypatch):
    before = Path(case["base_config"]["rotation_artifact"]).read_bytes()
    means = case["baseline"]["layers"][NAME]["k_mean"].clone()
    calls = _prescribe_teacher_and_direction(case, monkeypatch)
    monkeypatch.setattr(optr_sequential, "fitting_device", forbidden)
    result = _prepare(case, capture_runner=capture(case))
    assert result["accepted_layers"] == [NAME]
    assert result["refinement_policy_version"] == POLICY_VERSION
    assert result["optimization_device"] == "cpu_native_residual"
    assert calls == [(head, side) for head in range(2) for side in ("Rk", "Rv")]
    assert case["calls"] == ["capture"]
    artifact = load_active_artifact(_config(case), path=case["output"], device="cpu", testing=True)
    certificate = artifact["refinement_certificate"]
    record = certificate["layers"][NAME]
    decision = record["native_residual_selection"]
    assert certificate["independent_test"]["status"] == "passed"
    assert decision["accepted"] and not decision["test_used_for_selection"]
    assert len(decision["evaluated_train_candidates"]) == 10
    assert len(decision["evaluated_validation_candidates"]) == 2
    assert all(key not in record for key in ("source_fit_provenance", "shared_layer_selection", "recipe3_repair_selection"))
    for split in ("validation", "test"):
        metric = record[split]
        assert nonworse(metric["baseline"], metric["selected"], **native.STRICT)
        assert metric["selected"]["squared_error"] < metric["baseline"]["squared_error"]
    train_winner = next(row for row in decision["evaluated_train_candidates"]
                        if row["candidate"] == decision["selected_candidate"])
    assert nonworse(decision["train_baseline"], train_winner["train"], **native.STRICT)
    assert train_winner["train"]["squared_error"] < decision["train_baseline"]["squared_error"]
    assert Path(case["base_config"]["rotation_artifact"]).read_bytes() == before
    assert torch.equal(artifact["layers"][NAME]["k_mean"], means)
    assert torch.equal(case["baseline"]["layers"][NAME]["k_mean"], means)
    monkeypatch.setattr(native, "select_native_residual", forbidden)
    assert _prepare(case)["reused"]


def test_actual_native_gradient_rejects_all_unchanged_and_completed_resume_spawns_zero_workers(case, monkeypatch, capsys):
    # A teacher equal to baseline has zero error; no nonzero candidate can
    # improve it. This executes the actual gradient, selector and strict gate.
    case["candidate"] = copy.deepcopy(case["baseline"]["layers"][NAME])
    with pytest.raises(ValueError, match=NO_IMPROVEMENT):
        _prepare(case, capture_runner=capture(case))
    assert not case["output"].exists()
    _, _, _, lock = _read_checkpoint(case["output"], "layer-3-shared")
    assert not lock["decision"]["accepted"]
    assert all(row["zero_gradient"] for row in lock["decision"]["directions"])
    assert lock["decision"]["validation_candidate_count"] == 0
    assert case["calls"] == ["capture"]
    capsys.readouterr()
    before = {path: path.read_bytes() for path in case["output"].parent.rglob("*") if path.is_file()}
    monkeypatch.setattr(native, "select_native_residual", forbidden)
    monkeypatch.setattr(native_prepare, "refine_layer", forbidden)
    monkeypatch.setattr(prepare, "refine_layer", forbidden)
    monkeypatch.setattr(prepare, "_metrics", forbidden)
    monkeypatch.setattr(optr_traces, "load_layer_samples", forbidden)
    monkeypatch.setattr(optr_sequential, "fitting_device", forbidden)
    monkeypatch.setattr(optr_parallel.mp, "get_context", forbidden)
    with pytest.raises(ValueError, match=NO_IMPROVEMENT):
        _prepare(case, parallel_testing=True)
    events = [json.loads(line.split("OPTR_PROGRESS ", 1)[1])
              for line in capsys.readouterr().out.splitlines() if line.startswith("[oscar] OPTR_PROGRESS ")]
    assert any(row.get("phase") == "layer_checked_resumed" for row in events)
    assert next(row for row in events if row.get("phase") == "parallel_fit_plan")["workers"] == 0
    assert all(path.read_bytes() == value for path, value in before.items())
    assert not case["output"].exists()


def test_actual_nonzero_gradient_is_admitted_only_by_full_training_and_validation_gates(case):
    # The original fixture's unrelated teacher matrices give nonzero native
    # residuals. No gradient, proposal, metric or acceptance function is mocked.
    # A valid optimization outcome may be rejection; forcing a pass would hide
    # exactly the observed all-candidates-rejected failure family.
    try:
        _prepare(case, capture_runner=capture(case))
    except ValueError as error:
        assert NO_IMPROVEMENT in str(error)
    _, _, _, saved = _read_checkpoint(case["output"], "layer-3-shared")
    decision = saved["decision"]
    assert any(row["gradient_rms"] > 0 for row in decision["directions"])
    assert any(not row["same_bf16_operand_as_seed"] for row in decision["evaluated_train_candidates"])
    for row in decision["evaluated_train_candidates"]:
        expected = nonworse(decision["train_baseline"], row["train"], **native.STRICT)
        expected = expected and row["train"]["squared_error"] < decision["train_baseline"]["squared_error"]
        assert row["accepted"] is expected
    for row in decision["evaluated_validation_candidates"]:
        expected = nonworse(saved["before"], row["validation"], **native.STRICT)
        expected = expected and row["validation"]["squared_error"] < saved["before"]["squared_error"]
        assert row["accepted"] is expected
    if case["output"].exists():
        artifact = load_active_artifact(_config(case), path=case["output"], device="cpu", testing=True)
        record = artifact["refinement_certificate"]["layers"][NAME]
        assert record["accepted"]
        assert nonworse(record["test"]["baseline"], record["test"]["selected"], **native.STRICT)
    else:
        assert decision["accepted"] is False


def test_independent_test_failure_preserves_locked_selection_and_never_reselects(case, monkeypatch):
    calls = _prescribe_teacher_and_direction(case, monkeypatch)
    before = Path(case["base_config"]["rotation_artifact"]).read_bytes()
    with pytest.raises(prepare.IndependentTestRegression, match="not published"):
        _prepare(case, capture_runner=capture(case, fail_test=True))
    assert len(calls) == 4
    assert not case["output"].exists()
    pointer, _, _, lock = _read_checkpoint(case["output"], "layer-3-shared")
    assert lock["decision"]["accepted"]
    locked = pointer.read_bytes()
    failures = list(case["output"].parent.rglob("*independent-test-failed.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text())
    assert failure["artifact_publish_allowed"] is False and failure["test_used_for_selection"] is False
    monkeypatch.setattr(native, "select_native_residual", forbidden)
    monkeypatch.setattr(native, "_gradient_direction", forbidden)
    with pytest.raises(prepare.IndependentTestRegression, match="not published"):
        _prepare(case)
    assert pointer.read_bytes() == locked
    assert Path(case["base_config"]["rotation_artifact"]).read_bytes() == before
    assert not case["output"].exists() and case["calls"] == ["capture"]


@pytest.mark.parametrize("corrupt", ["file_hash", "binding", "selected_hash", "tangent", "accepted"])
def test_locked_candidate_corruption_is_rejected_without_selection(case, monkeypatch, corrupt):
    _prescribe_teacher_and_direction(case, monkeypatch)
    with pytest.raises(prepare.IndependentTestRegression):
        _prepare(case, capture_runner=capture(case, fail_test=True))
    pointer, manifest, cache, saved = _read_checkpoint(case["output"], "layer-3-shared")
    if corrupt == "file_hash":
        path = pointer.parent / manifest["file"]
        path.write_bytes(path.read_bytes() + b"explicit tampering fixture")
    else:
        if corrupt == "binding":
            saved["native_residual_binding"]["base_constants_sha256"] = "f" * 64
        elif corrupt == "selected_hash":
            saved["decision"]["selected_constants_sha256"] = "f" * 64
        elif corrupt == "tangent":
            saved["proposal"]["directions"][0]["unit_tangent"] *= -1
        else:
            saved["decision"]["accepted"] = False
        cache.save("layer-3-shared", saved)  # Valid outer file hash; semantic corruption remains.
    monkeypatch.setattr(native, "select_native_residual", forbidden)
    with pytest.raises(ValueError, match="hash|checkpoint|native residual"):
        _prepare(case)
    assert not case["output"].exists()


def test_published_pt_matrix_hash_and_recipe3_baseline_file_are_authenticated(case, monkeypatch):
    _prescribe_teacher_and_direction(case, monkeypatch)
    _prepare(case, capture_runner=capture(case))
    original = case["output"].read_bytes()
    artifact = torch.load(case["output"], map_location="cpu", weights_only=True)
    artifact["layers"][NAME]["Rv"][0] *= -1  # Still orthogonal; digest must catch it.
    torch.save(artifact, case["output"])
    with pytest.raises(ValueError, match="hash|digest|constants|matrix|matrices"):
        load_active_artifact(_config(case), path=case["output"], device="cpu", testing=True)
    case["output"].write_bytes(original)
    baseline_path = Path(case["base_config"]["rotation_artifact"])
    baseline_path.write_bytes(baseline_path.read_bytes() + b"explicit baseline tampering")
    with pytest.raises(ValueError, match="baseline file changed"):
        _prepare(case)


@pytest.mark.parametrize("generation", ["v10", "v11"])
def test_old_native_capture_reused_but_old_fitting_payload_never_becomes_v12_evidence(case, monkeypatch, generation):
    old_config = dict(case["config"])
    if generation == "v11":
        old_config["optr_recipe3_repair"] = "single_head_trust_region_v1"
    profile = calibration_cache_identity(old_config, case["fingerprint"])
    profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    pool = case["output"].parent / "generations"
    old_output = pool / ("optr_" + profile_hash) / "refined.pt"
    case["candidate"] = copy.deepcopy(case["baseline"]["layers"][NAME])
    with pytest.raises(ValueError, match=NO_IMPROVEMENT):
        prepare.prepare(old_config, output=old_output, testing=True,
            capture_runner=capture(case), fitter=proposal(case, unchanged=True), base_runner=forbidden)
    pointer, manifest, _, _ = _read_checkpoint(old_output, "layer-3-shared")
    old_payload = pointer.parent / manifest["file"]
    old_payload.write_bytes(old_payload.read_bytes() + b"v12 must not read rejected proxy fit")
    old_bytes = {path: path.read_bytes() for path in old_output.parent.rglob("*") if path.is_file()}
    case["output"] = pool / "native" / "refined.pt"
    monkeypatch.setattr(optr_sequential, "fitting_device", forbidden)
    monkeypatch.setattr(optr_sequential, "fit_head", forbidden)
    with pytest.raises(ValueError, match=NO_IMPROVEMENT):
        _prepare(case)
    assert case["calls"] == ["capture", "fit"]
    assert all(path.read_bytes() == value for path, value in old_bytes.items())
    _, new_manifest, _, completed = _read_checkpoint(case["output"], "layer-3")
    record = completed["artifact"]["refinement_certificate"]["layers"][NAME]
    assert "native_residual_selection" in record
    assert all(key not in record for key in ("source_fit_provenance", "shared_layer_selection", "recipe3_repair_selection"))
    assert "source_fit_snapshot_sha256" not in new_manifest["identity"]
    assert new_manifest["identity"]["profile"]["refinement_policy"]["policy_version"] == POLICY_VERSION
    old_trace = old_output.parent / "native-traces" / "capture-manifest.json"
    assert new_manifest["identity"]["trace_manifest_sha256"] == hashlib.sha256(old_trace.read_bytes()).hexdigest()


def test_v12_authenticated_raw_profile_is_reused_while_previous_fit_payload_is_ignored(case, monkeypatch):
    from oscar_ascend.optr_policy import compatible_capture_profiles
    config = _config(case)
    current = calibration_cache_identity(config, case["fingerprint"])
    historical = compatible_capture_profiles(current, config)[-1]
    assert historical["refinement_policy"]["policy_version"] == "optr_recipe3_native_residual_ascend_v12"
    profile_hash = hashlib.sha256(json.dumps(historical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    pool = case["output"].parent / "generations"
    old_output = pool / ("optr_" + profile_hash) / "refined.pt"
    old_case = {**case, "output": old_output,
                "candidate": copy.deepcopy(case["baseline"]["layers"][NAME])}
    # Create authentic synthetic raw payload/source hashes, then record the
    # exact historical policy: v12/v13 capture geometry and teacher are identical.
    with pytest.raises(ValueError, match=NO_IMPROVEMENT):
        _prepare(old_case, capture_runner=capture(old_case))
    manifest = old_output.parent / "native-traces" / "capture-manifest.json"
    value = json.loads(manifest.read_text())
    value.update(optr_policy=historical["refinement_policy"], calibration_profile=historical)
    manifest.write_text(json.dumps(value))
    pointer, saved_manifest, _, _ = _read_checkpoint(old_output, "layer-3-shared")
    fitted_payload = pointer.parent / saved_manifest["file"]
    fitted_payload.write_bytes(fitted_payload.read_bytes() + b"old fit must never be read")
    before = {path: path.read_bytes() for path in old_output.parent.rglob("*") if path.is_file()}
    case["output"] = pool / "native-v13" / "refined.pt"
    monkeypatch.setattr(optr_sequential, "fit_head", forbidden)
    with pytest.raises(ValueError, match=NO_IMPROVEMENT):
        _prepare(case)
    assert case["calls"] == ["capture"]
    assert all(path.read_bytes() == data for path, data in before.items())
    _, new_manifest, _, completed = _read_checkpoint(case["output"], "layer-3")
    assert new_manifest["identity"]["trace_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert completed["artifact"]["refinement_certificate"]["policy"]["policy_version"] == POLICY_VERSION
    assert completed["artifact"]["refinement_certificate"]["layers"][NAME]["native_residual_selection"]["train_candidate_count"] == 10
