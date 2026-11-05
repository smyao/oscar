"""Archive #97/#166/#175-177: portable CPU work retains original identities.

The tiny recipe3 pair and trace teachers are explicitly synthetic. Real file
authentication, optimizer, spawn worker, replay and strict gates execute; no
case is evidence of NPU arithmetic, model accuracy or preparation throughput.
"""
from __future__ import annotations

import copy
import importlib.abc
import json
from pathlib import Path
import sys

import pytest
import torch

from oscar_ascend.optr_artifacts import file_sha256
from tools import optr_offline as offline, optr_parallel
from tools import prepare_optr_rotations as prepare
from tools.prepare_optr_recipe3_base import capture_source_metadata
from test_optr_recipe3_pipeline import NAME, case, capture, forbidden, json_write
from test_optr_native_residual_pipeline import _config, _prepare, _read_checkpoint, NO_IMPROVEMENT


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _raw_capture(case):
    config = _config(case)
    selected = {**config, "optr_seed_capture": {
        **capture_source_metadata(case["baseline"]),
        "baseline_artifact_sha256": file_sha256(case["base_config"]["rotation_artifact"])}}
    root = case["output"].parent / "raw"
    root.mkdir(parents=True)
    config_path = root / "config.json"
    json_write(config_path, selected)
    manifest = root / "capture-manifest.json"
    json_write(manifest, capture(case)(config_path, root))
    return config, manifest


@pytest.fixture
def exported(case):
    config, manifest = _raw_capture(case)
    bundle = case["output"].parent.parent / "bundle"
    result = offline.export_bundle(config, bundle, trace_manifest=manifest, testing=True)
    assert result["model_forward"] == "not_run" and result["npu"] == "not_used"
    return case, config, manifest, bundle


class _NoAcceleratorOrModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"torch_npu", "vllm", "vllm_ascend", "transformers"}:
            raise AssertionError("offline CPU worker imported model/accelerator module: " + fullname)
        return None


def _observed_offline_worker_entry(target, args):
    """Observe only: invoke the unchanged real worker in a fresh interpreter."""
    spec = args[0]
    assert spec["testing"] is True and spec["offline_bundle"]
    assert not Path(spec["config"]["model"]).exists()
    forbidden_modules = {"torch_npu", "vllm", "vllm_ascend", "transformers"}
    assert not any(name.split(".", 1)[0] in forbidden_modules for name in sys.modules)
    guard = _NoAcceleratorOrModelImports()
    sys.meta_path.insert(0, guard)

    def observe(frame, event, _arg):
        if event != "call":
            return
        module, name = frame.f_globals.get("__name__", ""), frame.f_code.co_name
        denied = {
            ("oscar_ascend.optr_sequential", "fitting_device"),
            ("oscar_ascend.rotations", "checkpoint_identity"),
            ("oscar_ascend.rotations", "model_artifact_fingerprint"),
            ("tools.prepare_optr_rotations", "_capture"),
            ("tools.calibrate_optr_traces", "collect"),
            ("tools.calibrate_model_rotations", "collect"),
        }
        if (module, name) in denied:
            raise AssertionError("offline worker entered forbidden source-host function: " + module + "." + name)

    args[-1].send({"kind": "progress", "layer": None,
                  "fields": {"phase": "test_offline_worker_guards_installed"}})
    sys.setprofile(observe)
    try:
        optr_parallel._worker_entry(target, args)
    finally:
        sys.setprofile(None)
        sys.meta_path.remove(guard)
        assert not any(name.split(".", 1)[0] in forbidden_modules for name in sys.modules)


def _attempt(run):
    try:
        return run(), None
    except ValueError as error:
        assert NO_IMPROVEMENT in str(error)
        return None, str(error)


def test_real_export_hidden_sources_spawn_cpu_fit_matches_serial_and_resumes(case, monkeypatch, capsys):
    serial, serial_error = _attempt(lambda: _prepare(case, capture_runner=capture(case)))
    _, _, _, serial_saved = _read_checkpoint(case["output"], "layer-3")
    serial_record = serial_saved["artifact"]["refinement_certificate"]["layers"][NAME]
    assert any(row["gradient_rms"] > 0
               for row in serial_record["native_residual_selection"]["directions"])
    manifest = case["output"].parent / "native-traces" / "capture-manifest.json"
    bundle = case["output"].parent.parent / "bundle"
    offline.export_bundle(_config(case), bundle, trace_manifest=manifest, testing=True)
    header = json.loads((bundle / "bundle.json").read_text())
    # The bundle is byte transport: no rewrite of original JSON, PT or source.
    assert all((bundle / row["path"]).read_bytes() == Path(logical).read_bytes()
               for logical, row in header["files"].items())
    assert not any(Path(logical).suffix == ".safetensors" for logical in header["files"])
    paths = [Path(case["config"]["model"]), Path(case["base_config"]["rotation_artifact"]).parents[3],
             case["source"], case["output"].parent]
    moved = []
    output = bundle.parent / "offline-result"
    try:
        for path in paths:
            hidden = path.with_name("hidden-" + path.name)
            path.rename(hidden)
            moved.append((path, hidden))
        assert all(not Path(logical).exists() for logical in header["files"])
        monkeypatch.setattr(optr_parallel, "_worker_entry", _observed_offline_worker_entry)
        capsys.readouterr()
        portable, portable_error = _attempt(lambda: offline.fit_bundle(
            bundle, output, workers=2, threads=1, testing=True))
        events = [json.loads(line.split("OPTR_PROGRESS ", 1)[1])
                  for line in capsys.readouterr().out.splitlines() if line.startswith("[oscar] OPTR_PROGRESS ")]
        assert any(row.get("phase") == "test_offline_worker_guards_installed" for row in events)
        assert next(row for row in events if row.get("phase") == "fit_worker_ready")["fitting_device"] == "cpu"
        assert next(row for row in events if row.get("phase") == "fit_worker_exited")["returncode"] == 0
        assert next(row for row in events if row.get("phase") == "fit_workers_cleanup")["cleanup_complete"]
        _, _, _, portable_saved = _read_checkpoint(output / "refined.pt", "layer-3")
        portable_record = portable_saved["artifact"]["refinement_certificate"]["layers"][NAME]
        for key in ("validation", "test", "accepted"):
            assert portable_record[key] == serial_record[key]
        for key in ("train_baseline", "evaluated_train_candidates", "evaluated_validation_candidates"):
            assert portable_record["native_residual_selection"][key] == serial_record["native_residual_selection"][key]
        assert portable_error == serial_error
        assert (portable is None) is (serial is None)
        if portable_error:
            assert not (output / "refined.pt").exists()
            assert not (output / ".optr.complete.json").exists()
        else:
            assert portable["independent_test"] == "passed"
        monkeypatch.setattr(optr_parallel.mp, "get_context", forbidden)
        again, again_error = _attempt(lambda: offline.fit_bundle(bundle, output, testing=True))
        assert again_error == portable_error
        if again is not None:
            assert again["reused"]
        assert all(file_sha256(bundle / row["path"]) == row["sha256"]
                   for row in header["files"].values())
        # Authentication may create this lock; sample/PT bytes remain fixed.
        assert (bundle / "base" / ".oscar.lock").is_file()
    finally:
        for path, hidden in reversed(moved):
            hidden.rename(path)


@pytest.mark.parametrize("fault", ["hash", "size", "traversal", "absolute", "symlink", "duplicate", "missing", "row_type"])
def test_portable_inventory_rejects_corruption_and_path_escape(exported, fault):
    _case, _config_value, _manifest, bundle = exported
    header_path = bundle / "bundle.json"
    header = json.loads(header_path.read_text())
    logical = header["trace_manifest"]
    row = header["files"][logical]
    physical = bundle / row["path"]
    if fault == "hash":
        physical.write_bytes(physical.read_bytes() + b" ")
    elif fault == "size":
        row["size"] += 1
    elif fault in ("traversal", "absolute", "symlink"):
        outside = bundle.parent / "outside.json"
        outside.write_bytes(physical.read_bytes())
        if fault == "symlink":
            physical.unlink()
            physical.symlink_to(outside)
        else:
            row["path"] = "../outside.json" if fault == "traversal" else str(outside)
    elif fault == "duplicate":
        header["files"][str(bundle.parent / "different-logical.json")] = copy.deepcopy(row)
    elif fault == "missing":
        del header["files"][logical]
    else:
        header["files"][logical] = []
    json_write(header_path, header)
    with pytest.raises(ValueError):
        offline.PortableInputs(bundle, testing=True).load_trace()


@pytest.mark.parametrize("fault", ["source_bytes", "excluded_records", "seed_binding", "identity", "combined_tensor"])
def test_portable_semantic_authentication_survives_rehashed_inventory(exported, fault):
    case, _config_value, _manifest, bundle = exported
    header_path = bundle / "bundle.json"
    header = json.loads(header_path.read_text())
    if fault == "seed_binding":
        header["config"]["optr_seed_capture"]["baseline_artifact_sha256"] = "a" * 64
    elif fault == "identity":
        header["identity"][2] = "another-model"
    else:
        logical = (str(case["source"]) if fault == "source_bytes" else
                   header["base_config"]["rotation_artifact"] if fault == "combined_tensor" else
                   header["trace_manifest"])
        row = header["files"][logical]
        path = bundle / row["path"]
        if fault == "source_bytes":
            path.write_text(path.read_text() + "\n")
        elif fault == "excluded_records":
            value = json.loads(path.read_text())
            value["recipe3_sampling"]["excluded_record_sha256"].pop()
            json_write(path, value)
        else:
            value = torch.load(path, weights_only=True)
            del value["layers"][NAME]["Rk"]
            torch.save(value, path)
        row.update(sha256=file_sha256(path), size=path.stat().st_size)
    json_write(header_path, header)
    with pytest.raises(ValueError):
        files = offline.PortableInputs(bundle, testing=True)
        files.load_base()
        files.load_trace()


def test_test_only_bundle_cannot_be_promoted_and_cli_has_no_testing_switch(exported, monkeypatch):
    _case, _config_value, _manifest, bundle = exported
    with pytest.raises(ValueError, match="test-only"):
        offline.PortableInputs(bundle)
    header_path = bundle / "bundle.json"
    header = json.loads(header_path.read_text())
    header["test_only"] = False
    json_write(header_path, header)
    with pytest.raises(ValueError):
        offline.PortableInputs(bundle).load_base()
    monkeypatch.setattr(sys, "argv", ["optr_offline", "fit", "--bundle", str(bundle),
                                     "--output", str(bundle.parent / "result"), "--testing"])
    with pytest.raises(SystemExit) as error:
        offline.main()
    assert error.value.code == 2


def test_fit_rejects_mutable_bundle_output_and_invalid_cpu_budget(exported):
    _case, _config_value, _manifest, bundle = exported
    with pytest.raises(ValueError, match="outside"):
        offline.fit_bundle(bundle, bundle / "results", testing=True)
    for budget in (0, True, 65):
        with pytest.raises(ValueError, match="workers/threads"):
            offline.fit_bundle(bundle, bundle.parent / "result", workers=budget, testing=True)


def test_resolve_rechecks_hash_and_symlink_after_open(exported):
    _case, _config_value, _manifest, bundle = exported
    files = offline.PortableInputs(bundle, testing=True)
    logical = files.header["trace_manifest"]
    physical = files.resolve(logical)
    content = physical.read_bytes()
    outside = bundle.parent / "same-bytes-outside.json"
    outside.write_bytes(content)
    physical.unlink()
    physical.symlink_to(outside)
    with pytest.raises(ValueError, match="escaped"):
        files.resolve(logical)
    physical.unlink()
    physical.write_bytes(content)
    changed = bytearray(content)
    changed[-1] ^= 1
    physical.write_bytes(changed)
    with pytest.raises(ValueError, match="changed"):
        files.resolve(logical)


def test_export_canonicalizes_relative_model_and_source_before_changing_cwd(case, monkeypatch):
    config, manifest = _raw_capture(case)
    origin = case["output"].parent.parent
    monkeypatch.chdir(origin)
    config = {**config, "model": "model", "rotation_calibration_prompts": "source.jsonl"}
    bundle = origin / "bundle"
    offline.export_bundle(config, bundle, trace_manifest=manifest, testing=True)
    elsewhere = origin / "another-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    files = offline.PortableInputs(bundle, testing=True)
    assert Path(files.config["model"]).is_absolute()
    assert Path(files.config["rotation_calibration_prompts"]).is_absolute()
    files.load_base()
    files.load_trace()


def test_original_logical_paths_do_not_follow_fitting_host_symlinks(exported):
    from oscar_ascend.optr_traces import load_layer_samples
    case, _config_value, _manifest, bundle = exported
    original_header = (bundle / "bundle.json").read_bytes()
    alien = bundle.parent / "unrelated-fitting-host-path"
    alien.mkdir()
    alien_source = alien / "source.jsonl"
    alien_source.write_text("different local file; never a capture input")
    # Model, base, raw capture and custom source logical names now resolve on
    # this host to unrelated files. All original identities remain in bundle.
    paths = [Path(case["config"]["model"]), Path(case["base_config"]["rotation_artifact"]).parents[3],
             case["output"].parent, case["source"]]
    moved = []
    try:
        for path in paths:
            hidden = path.with_name("hidden-" + path.name)
            path.rename(hidden)
            path.symlink_to(alien_source if path == case["source"] else alien,
                            target_is_directory=path != case["source"])
            moved.append((path, hidden))
        files = offline.PortableInputs(bundle, testing=True)
        baseline = files.load_base()
        manifest = files.load_trace()
        samples = load_layer_samples(manifest, baseline["layers"][NAME]["layer_id"])
        assert set(samples) == {"train", "validation", "test"}
        assert manifest["model_fingerprint"] == case["fingerprint"]
        assert (bundle / "bundle.json").read_bytes() == original_header
        source = str(case["source"])
        assert str(files.logical_path(source)) == source
        assert files.logical_path("child/../leaf", root="/tmp/source") == Path("/tmp/source/leaf")
    finally:
        for path, hidden in reversed(moved):
            path.unlink()
            hidden.rename(path)


def test_finished_fit_reuses_authentication_and_only_refits_implementation_change(case, monkeypatch):
    from oscar_ascend import optr_checkpoint
    from test_optr_native_residual_pipeline import _prescribe_teacher_and_direction
    _prescribe_teacher_and_direction(case, monkeypatch)
    # Explicit prescribed tangent only for this cache/publication seam;
    # captured teachers, replay, metrics, independent test and hashes are real.
    _prepare(case, capture_runner=capture(case))
    manifest = case["output"].parent / "native-traces" / "capture-manifest.json"
    bundle = case["output"].parent.parent / "bundle"
    offline.export_bundle(_config(case), bundle, trace_manifest=manifest, testing=True)
    assert offline.fit_bundle(bundle, case["output"].parent, testing=True)["reused"]
    with pytest.raises(ValueError, match="test-only"):
        offline.import_result(_config(case), case["output"].parent)

    class RefitReached(Exception):
        pass

    def observe_refit(*_args, **_kwargs):
        raise RefitReached

    monkeypatch.setattr(prepare, "_parallel_layers", observe_refit)
    with monkeypatch.context() as changed:
        changed.setattr(optr_checkpoint, "fitting_implementation_digest", lambda: "f" * 64)
        with pytest.raises(RefitReached):
            offline.fit_bundle(bundle, case["output"].parent, testing=True)
    case["output"].write_bytes(case["output"].read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash or identity"):
        offline.fit_bundle(bundle, case["output"].parent, testing=True)


def test_export_authenticates_compatible_v12_raw_without_importing_old_fit(case):
    from oscar_ascend.optr_policy import compatible_capture_profiles
    config, manifest = _raw_capture(case)
    top = json.loads(manifest.read_text())
    v12 = next(profile for profile in compatible_capture_profiles(top["calibration_profile"], config)
               if profile["refinement_policy"]["policy_version"] == "optr_recipe3_native_residual_ascend_v12")
    top.update(calibration_profile=v12, optr_policy=v12["refinement_policy"])
    json_write(manifest, top)
    old_fit = case["output"].parent / "fit-checkpoints"
    old_fit.mkdir()
    (old_fit / "do-not-load.pt").write_bytes(b"old incompatible fitted work")
    bundle = case["output"].parent.parent / "bundle"
    offline.export_bundle(config, bundle, trace_manifest=manifest, testing=True)
    files = offline.PortableInputs(bundle, testing=True)
    assert files.header["profile"]["refinement_policy"]["policy_version"].endswith("v13")
    assert files.load_trace()["optr_policy"]["policy_version"].endswith("v12")
    assert all("fit-checkpoints" not in path for path in files.header["files"])


@pytest.mark.parametrize("command", ["export", "import"])
@pytest.mark.parametrize("explicit_trace", [False, True])
def test_cli_applies_fixed_candidate_variant_without_changing_fit_bundle(command, explicit_trace, tmp_path, monkeypatch):
    from tools.serving_variants import variant_config
    source = Path(__file__).resolve().parents[1] / "configs/target.json"
    config = json.loads(source.read_text())
    seen = []
    if command == "export":
        monkeypatch.setattr(offline, "export_bundle", lambda selected, output, **kw: seen.append((selected, kw)) or {})
        args = ["--output", str(tmp_path / "bundle")]
    else:
        monkeypatch.setattr(offline, "import_result", lambda selected, result, **kw: seen.append((selected, kw)) or {})
        args = ["--result", str(tmp_path / "result")]
    trace = str(tmp_path / "external-source" / "capture-manifest.json") if explicit_trace else None
    if trace is not None:
        args.extend(("--trace-manifest", trace))
    monkeypatch.setattr(sys, "argv", ["optr_offline", command, "--config", str(source), *args])
    offline.main()
    assert seen[0][0] == variant_config(config, "candidate")
    assert seen[0][1]["trace_manifest"] == trace
    assert "testing" not in seen[0][1]
