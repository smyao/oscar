"""Archive #97/#120/#172: interruption must not lose completed native/fit work."""
import copy
import json
from pathlib import Path
import signal
import sys
import threading

import pytest
import torch

from oscar_ascend.optr_checkpoint import FitCheckpoints
from tools.optr_progress import ProgressWatchdog


def test_checkpoint_pointer_is_atomic_and_rejects_modified_work(tmp_path, monkeypatch):
    from tools import phase
    store = FitCheckpoints(tmp_path, {"base_hash": "seed", "trace_hash": "trace", "policy": {}, "test_only": True})
    first = {"Rk": torch.eye(2), "decision": {"accepted": False}}
    store.save("layer-3-head-0", first)
    def interrupted(*_):
        raise RuntimeError("simulated termination before publishing pointer")
    monkeypatch.setattr(phase, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="simulated termination"):
        store.save("layer-3-head-0", {"Rk": -torch.eye(2)})
    assert torch.equal(store.load("layer-3-head-0")["Rk"], first["Rk"])
    pointer = json.loads((store.root / "layer-3-head-0.json").read_text())
    (store.root / pointer["file"]).write_bytes(b"not a completed tensor checkpoint")
    with pytest.raises(ValueError, match="hash"):
        store.load("layer-3-head-0")
    with pytest.raises(ValueError, match="owner"):
        store.load("../../other")


@pytest.mark.parametrize("field", ["base_hash", "trace_hash", "profile", "implementation_hash", "test_only"])
def test_changed_source_or_implementation_does_not_resume_another_fit(tmp_path, field):
    identity = {"base_hash": "a", "trace_hash": "b", "profile": {}, "implementation_hash": "c", "test_only": True}
    store = FitCheckpoints(tmp_path, identity)
    store.save("layer-3-head-0", {"Rk": torch.eye(2)})
    changed = copy.deepcopy(identity)
    changed[field] = False if field == "test_only" else "changed"
    other = FitCheckpoints(tmp_path, changed)
    assert other.root != store.root and other.load("layer-3-head-0") is None


def test_only_real_progress_advances_watchdog_deadline():
    now = [0.]
    log = []
    watcher = ProgressWatchdog(10, clock=lambda: now[0], output=lambda **x: log.append(x))
    now[0] = 8
    watcher.progress(event="step_done", head=0, step=1)
    now[0] = 17
    assert not watcher.expired()
    # A parent's wall-time heartbeat does not call progress or reset time.
    now[0] = 18
    assert watcher.expired()
    assert log == [{"event": "step_done", "head": 0, "step": 1}]


def test_stall_reports_owner_and_invokes_owned_termination():
    stopped = threading.Event()
    log = []
    with ProgressWatchdog(.03, output=lambda **x: log.append(x), on_timeout=stopped.set) as watcher:
        watcher.progress(layer="layer3", head=2, event="validation_start")
        assert stopped.wait(2)
    assert log[-1]["reason"] == "offline_progress_timeout"
    assert log[-1]["last_completed_or_started_work"]["head"] == 2


def test_success_closes_watchdog_without_terminating_owner():
    stopped = threading.Event()
    with ProgressWatchdog(10, output=lambda **_: None, on_timeout=stopped.set) as watcher:
        watcher.progress(event="layer_checked_saved")
    assert not watcher.thread.is_alive() and not stopped.is_set()


def test_real_stalled_preparer_terminates_and_launcher_cleans_owned_children(tmp_path):
    from tools.phase import run_phase
    program = ("import subprocess, sys, time\n"
               "from tools.optr_progress import ProgressWatchdog\n"
               "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
               "with ProgressWatchdog(.1) as progress:\n"
               "    progress.progress(layer='layer3', head=2, event='validation_start')\n"
               "    time.sleep(30)\n")
    result = run_phase("prepare-rotations", [sys.executable, "-c", program],
                       cwd=Path(__file__).resolve().parents[1], log_dir=tmp_path, timeout=5, grace=.2)
    assert result.returncode == -signal.SIGTERM and not result.timed_out
    assert result.cleanup_complete
    log = Path(result.log).read_text()
    assert "offline_progress_timeout" in log and '"head": 2' in log


def test_completed_head_survives_interruption_but_joint_test_still_runs(tmp_path, monkeypatch):
    from tools import prepare_optr_rotations as prep
    baseline = {"layer_id": 3, "Rk": -torch.eye(4).repeat(2, 1, 1),
                "Rv": -torch.eye(4).repeat(2, 1, 1), "k_mean": torch.zeros(2, 4)}
    samples = {split: [[{"request_id": split}], [{"request_id": split}]] for split in ("train", "validation", "test")}
    calls = []
    def first(rk, rv, mean, *_args, **_kw):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("interrupted during second head")
        return rk, rv, {"accepted": False}
    store = FitCheckpoints(tmp_path, {"source": "same-authenticated-traces", "test_only": True})
    with pytest.raises(RuntimeError, match="second head"):
        prep.refine_layer("model.layers.3.self_attn.attn", baseline, samples, {}, fitter=first, checkpoints=store)
    assert store.load("layer-3-head-0") is not None and store.load("layer-3-head-1") is None
    checks = []
    def measure(constants, rows, policy):
        split = rows[0][0]["request_id"]
        checks.append(split)
        row = {"request_id": split, "sample_sha256": ("a" if split == "validation" else "b") * 64,
               "squared_error": 1., "teacher_energy": 2., "normalized_error": .5}
        return {"squared_error": 1., "teacher_energy": 2., "normalized_error": .5, "requests": [row]}
    monkeypatch.setattr(prep, "_metrics", measure)
    def resumed(rk, rv, *_args, **_kw):
        calls.append(True)
        return rk, rv, {"accepted": False}
    selected, record = prep.refine_layer("model.layers.3.self_attn.attn", baseline, samples, {}, fitter=resumed, checkpoints=store)
    assert len(calls) == 3  # The completed head was not fitted again.
    assert checks == ["validation", "validation", "test"]
    assert not record["accepted"] and torch.equal(selected["Rk"], baseline["Rk"])
