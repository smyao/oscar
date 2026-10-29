"""Real spawned host workers; synthetic CPU traces are never NPU evidence."""
import copy
import json
import os
from pathlib import Path
import signal
import time

import pytest

from tools.optr_parallel import run_layer_workers, worker_plan


def fake_worker(spec, index, threads, pipe):
    root = Path(spec["root"])
    if spec.get("mode") == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pipe.send({"kind": "ready"})
    while True:
        name = pipe.recv()
        if name is None:
            return
        (root / f"started-{name}").write_text(str(os.getpid()))
        pipe.send({"kind": "progress", "layer": name,
                   "fields": {"phase": "assigned", "layer": name, "threads": threads}})
        # A concurrency barrier, not a speed assertion: all four independent
        # tasks must start before any of them can finish.
        until = time.monotonic() + 10
        while len(list(root.glob("started-*"))) < 4:
            if time.monotonic() > until:
                raise RuntimeError("layer workers did not overlap")
            time.sleep(.01)
        if spec.get("mode") in ("crash", "stubborn") and index == 0:
            os._exit(7)
        if spec.get("mode") in ("crash", "stall") and index == 0:
            time.sleep(30)
        if spec.get("mode") in ("crash", "stubborn"):
            time.sleep(30)
        time.sleep(.1)
        pipe.send({"kind": "done", "layer": name})


def pool_spec(tmp_path, mode="ok", count=8):
    return {"config": {"devices": [0, 1, 2, 3]}, "testing": True,
            "root": str(tmp_path), "mode": mode, "names": [f"layer-{n}" for n in range(count)]}


def assert_workers_gone(events):
    pids = [e["pid"] for e in events if e.get("phase") == "fit_worker_started"]
    assert len(set(pids)) == 4
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_four_live_processes_dynamically_complete_all_layers(tmp_path):
    spec, events, done = pool_spec(tmp_path), [], []
    run_layer_workers(spec, spec["names"], progress=lambda **x: events.append(x),
                      completed=done.append, timeout=15, worker_target=fake_worker)
    assert sorted(done) == sorted(spec["names"])
    assert {e["worker"] for e in events if e.get("phase") == "assigned"} == {0, 1, 2, 3}
    assert sum(e.get("phase") == "fit_worker_exited" and e["returncode"] == 0 for e in events) == 4
    assert_workers_gone(events)


@pytest.mark.parametrize("mode", ["crash", "stubborn"])
def test_worker_crash_aborts_pool_with_real_exit_code(tmp_path, mode):
    spec, events = pool_spec(tmp_path, mode), []
    with pytest.raises(RuntimeError, match="exitcode=7"):
        run_layer_workers(spec, spec["names"], progress=lambda **x: events.append(x),
                          completed=lambda _: None, timeout=15, worker_target=fake_worker)
    assert_workers_gone(events)


def test_one_stalled_worker_is_not_kept_alive_by_peer_progress(tmp_path):
    spec, events = pool_spec(tmp_path, "stall", 80), []
    with pytest.raises(TimeoutError, match="worker 0 stalled"):
        run_layer_workers(spec, spec["names"], progress=lambda **x: events.append(x),
                          completed=lambda _: None, timeout=2, worker_target=fake_worker)
    assert sum(e.get("phase") == "assigned" for e in events) > 4
    assert_workers_gone(events)


def test_signal_and_completion_validation_failure_cleanup_owned_workers(tmp_path):
    for mode in ("signal", "invalid_checkpoint"):
        root = tmp_path / mode
        root.mkdir()
        spec, events = pool_spec(root), []
        def progress(**row):
            events.append(row)
            if mode == "signal" and row.get("phase") == "assigned":
                os.kill(os.getpid(), signal.SIGTERM)
        def completed(_):
            raise ValueError("completed checkpoint failed authentication")
        with pytest.raises(SystemExit if mode == "signal" else ValueError) as error:
            run_layer_workers(spec, spec["names"], progress=progress, completed=completed,
                              timeout=15, worker_target=fake_worker)
        if mode == "signal":
            assert error.value.code == 128 + signal.SIGTERM
        assert_workers_gone(events)


def test_plan_bounds_cpu_pools_preserves_card_order_and_parent_environment(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1, 2, 3, 4, 5}, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    assert worker_plan({"devices": [4, 5, 6, 7]}, 16) == {
        "workers": 4, "devices": [4, 5, 6, 7], "cpu_threads_per_worker": 1,
        "start_method": "spawn", "unit": "complete_layer"}
    assert worker_plan({"devices": [0, 1, 2, 3]}, 1)["cpu_threads_per_worker"] == 4
    assert worker_plan({"devices": [0, 1, 2, 3]}, 0)["workers"] == 0
    assert os.environ["OMP_NUM_THREADS"] == "64"
    with pytest.raises(ValueError, match="four physical"):
        worker_plan({"devices": [0, 0, 2, 3]}, 4)


def test_spawn_hooks_are_not_a_production_cpu_bypass(tmp_path):
    spec = pool_spec(tmp_path)
    spec["testing"] = False
    with pytest.raises(ValueError, match="test-only"):
        run_layer_workers(spec, spec["names"], progress=lambda **_: None,
                          completed=lambda _: None, timeout=1, worker_target=fake_worker)


@pytest.mark.parametrize("mode", ["start_failure", "cancel_during_start"])
def test_partial_pool_start_is_owned_and_cleanup_cannot_lose_child(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace
    from tools import optr_parallel as parallel
    real = parallel.mp.get_context("spawn")
    made = []
    def process(**kwargs):
        child = real.Process(**kwargs)
        made.append(child)
        class OwnedHandle:
            def __getattr__(self, key):
                return getattr(child, key)
            def start(self):
                if mode == "start_failure" and len(made) == 2:
                    raise OSError("synthetic spawn failure")
                child.start()
                if mode == "cancel_during_start":
                    # Arrives after creation, before start returns to caller.
                    os.kill(os.getpid(), signal.SIGTERM)
        return OwnedHandle()
    monkeypatch.setattr(parallel.mp, "get_context", lambda _: SimpleNamespace(Pipe=real.Pipe, Process=process))
    spec = pool_spec(tmp_path)
    with pytest.raises(OSError if mode == "start_failure" else SystemExit):
        run_layer_workers(spec, spec["names"], progress=lambda **_: None,
                          completed=lambda _: None, timeout=15, worker_target=fake_worker)
    for child in made:
        if child.pid is not None:
            assert not child.is_alive()
            with pytest.raises(ProcessLookupError):
                os.kill(child.pid, 0)


@pytest.mark.parametrize("implementation", [
    "db3b72ec7d53ee57649fff212186d5c8c9a04030fe8e30fb472e17c6c33ce2db",
    "5ed45fb740f865a23ac1d2af6116f2c51300f01822d80265e2b5b301e6fe9490"])
def test_v3_fits_cannot_resume_into_joint_gqa_objective(tmp_path, implementation):
    import torch
    from oscar_ascend.optr_checkpoint import FitCheckpoints, fitting_implementation_digest
    from oscar_ascend.optr_policy import POLICY_VERSION
    identity = {"profile": {"refinement_policy": {"policy_version": "optr_frame_bf16_ascend_v3"}},
                "fitting_implementation_sha256": implementation,
                "base_artifact_sha256": "seed", "trace_manifest_sha256": "trace", "test_only": True}
    old = FitCheckpoints(tmp_path, identity)
    old.save("layer-3-head-0", {"Rk": torch.eye(4)})
    current = {**identity, "fitting_implementation_sha256": fitting_implementation_digest(),
               "profile": {"refinement_policy": {"policy_version": POLICY_VERSION}}}
    assert FitCheckpoints(tmp_path, current).load("layer-3-head-0") is None
    # Even relabelling the old profile cannot grant cross-implementation reuse.
    assert FitCheckpoints(tmp_path, {**current, "profile": identity["profile"]}).load("layer-3-head-0") is None
    active = FitCheckpoints(tmp_path, current)
    active.save("layer-3-head-0", {"Rk": torch.eye(4)})
    assert torch.equal(active.load("layer-3-head-0")["Rk"], torch.eye(4))
    for key in ("base_artifact_sha256", "trace_manifest_sha256", "test_only"):
        changed = {**current, key: "different"}
        assert FitCheckpoints(tmp_path, changed).load("layer-3-head-0") is None
    bad = copy.deepcopy(current)
    bad["profile"]["refinement_policy"]["policy_version"] = "optr_sequential_ascend_v2"
    assert FitCheckpoints(tmp_path, bad).load("layer-3-head-0") is None
    pointer = json.loads((active.root / "layer-3-head-0.json").read_text())
    (active.root / pointer["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        FitCheckpoints(tmp_path, current).load("layer-3-head-0")


def two_layer_bundle(tmp_path):
    import torch
    from test_optr_traces import build_bundle, json_save
    from oscar_ascend.optr_artifacts import file_sha256
    from oscar_ascend.rotations import checkpoint_identity, calibration_cache_identity
    path, config, top = build_bundle(tmp_path, dim=64, steps=2)
    config["devices"] = [0, 1, 2, 3]
    model = Path(config["model"]) / "config.json"
    raw = json.loads(model.read_text())
    raw["num_hidden_layers"] = 8
    json_save(model, raw)
    names, dim, fingerprint = checkpoint_identity(config["model"])
    top.update(layer_ids=[3, 7], model_fingerprint=fingerprint,
               calibration_profile=calibration_cache_identity(config, fingerprint))
    for rank_path in top["rank_paths"]:
        rank = json.loads(Path(rank_path).read_text())
        origin = copy.deepcopy(rank["layers"]["3"])
        for layer_id in (3, 7):
            layer = copy.deepcopy(origin)
            wo = torch.load(origin["wo_path"], weights_only=True)
            wo["layer_id"] = layer_id
            layer["wo_path"] = origin["wo_path"].replace("layer3", f"layer{layer_id}")
            torch.save(wo, layer["wo_path"])
            layer["wo_sha256"] = file_sha256(layer["wo_path"])
            for seq, original in zip(layer["sequences"], origin["sequences"]):
                payload = torch.load(original["path"], weights_only=True)
                payload.update(layer_id=layer_id, model_fingerprint=fingerprint)
                seq["path"] = original["path"].replace("layer3", f"layer{layer_id}")
                torch.save(payload, seq["path"])
                seq["sha256"] = file_sha256(seq["path"])
            rank["layers"][str(layer_id)] = layer
        rank["model_fingerprint"] = fingerprint
        json_save(Path(rank_path), rank)
        top["rank_sha256"][rank_path] = file_sha256(rank_path)
    json_save(path, top)
    return path, config, names, dim, fingerprint


def test_real_parallel_prepare_matches_serial_and_reuses_validated_layers(tmp_path, monkeypatch):
    import torch
    from tools import prepare_optr_rotations as prep
    from oscar_ascend.rotations import calibration_cache_identity
    from oscar_ascend.rotation_calibration import OFFICIAL_COMMIT
    from oscar_ascend.optr_artifacts import file_sha256
    path, config, names, dim, fingerprint = two_layer_bundle(tmp_path)
    seed_path = tmp_path / "seed.pt"
    base_config = {k: v for k, v in config.items() if k not in ("rotation_refinement", "optr_calibration")}
    base_config["rotation_artifact"] = str(seed_path)
    generator = torch.Generator().manual_seed(7)
    baseline = {"format_version": 2, "source_grouping": "layer_kv_head", "rotation_variant": "perhead_center",
        "global_kv_heads": 2, "head_dim": dim, "model_fingerprint": fingerprint, "pr_fingerprint": OFFICIAL_COMMIT,
        "objective": "qqt_sst_r_h_pbr", "calibrated": True, "accumulation_dtype": "float64", "test_only": True,
        "calibration_profile": calibration_cache_identity(base_config, fingerprint), "layers": {}}
    for name, layer_id in zip(names, (3, 7)):
        rotation = torch.linalg.qr(torch.randn(2, dim, dim, generator=generator)).Q
        baseline["layers"][name] = {"layer_id": layer_id, "Rk": rotation, "Rv": rotation.clone(),
                                     "k_mean": torch.zeros(2, dim)}
    from oscar_ascend.rotation_calibration import checkpoint_pair_artifact, publish_checkpoint_pair
    metadata = {"model_fingerprint": fingerprint, "calibration_profile": baseline["calibration_profile"],
                "global_kv_heads": 2, "accumulator_dtype": "float64", "post_rope": True, "test_only": True,
                "generation_id": "synthetic-parallel-test"}
    states = []
    for key, objective in (("Rk", "qqt_r_h_pbr"), ("Rv", "sst_r_h_pbr")):
        state = {"format_version": 2, "source_grouping": "layer_kv_head", "global_kv_heads": 2,
                 "objective": objective, "calibration": metadata, "layers": {}}
        for entry in baseline["layers"].values():
            state["layers"][entry["layer_id"]] = {"layer_id": entry["layer_id"], "rotation": entry[key]}
            if key == "Rk":
                state["layers"][entry["layer_id"]]["k_mean"] = entry["k_mean"]
        states.append(state)
    baseline = checkpoint_pair_artifact(*states, layer_names=names, head_dim=dim,
        model_fingerprint=fingerprint, calibration_profile=baseline["calibration_profile"], testing=True)
    publish_checkpoint_pair(*states, seed_path.parent, artifact=baseline, artifact_name=seed_path.name)
    monkeypatch.setattr(prep, "_base_config", lambda *_: base_config)
    monkeypatch.setattr(prep, "_prepare_base", lambda *a, **k: baseline)
    monkeypatch.setattr(prep, "_existing_trace_manifest", lambda *a: path)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    events = []
    monkeypatch.setattr(prep, "_progress", lambda **x: events.append(x))
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        serial = tmp_path / "serial" / "refined.pt"
        parallel = tmp_path / "parallel" / "refined.pt"
        prep.prepare(config, output=serial, testing=True)
        prep.prepare(config, output=parallel, testing=True, parallel_testing=True)
    finally:
        torch.set_num_threads(prior)
    left = torch.load(serial, weights_only=True)
    right = torch.load(parallel, weights_only=True)
    assert list(right["layers"]) == names
    for name in names:
        for key in ("Rk", "Rv", "k_mean"):
            assert torch.equal(left["layers"][name][key], right["layers"][name][key])
    assert left["refinement_certificate"] == right["refinement_certificate"]
    assert len({e["pid"] for e in events if e.get("phase") == "fit_worker_started"}) == 2
    assert all(e["returncode"] == 0 for e in events if e.get("phase") == "fit_worker_exited")
    assert not any(e.get("fitting_device", "").startswith("npu") for e in events)
    # Removing only final publication simulates interruption after saved layers.
    parallel.unlink()
    events.clear()
    prep.prepare(config, output=parallel, testing=True, parallel_testing=True)
    assert not any(e.get("phase") == "fit_worker_started" for e in events)
    assert torch.load(parallel, weights_only=True)["refinement_certificate"] == right["refinement_certificate"]
    assert file_sha256(path) == right["refinement_certificate"]["trace_manifest_sha256"]
