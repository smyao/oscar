"""Archive G15/G16/G17/#84/#166: fresh calibration process/thread-pool ownership.

The two-level test runs real host processes and Torch autograd, using the native
context selector extracted from the read-only reference. It does not load an
NPU model or claim engine/hardware acceptance.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap

import pytest

from tools import calibrate_model_rotations, prepare_rotations


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method", [None, "fork", "spawn"])
def test_calibration_process_environment_is_a_copy_and_overrides_inherited_fork(method):
    base = {"OSCAR_ENABLED": "1", "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
            "OMP_NUM_THREADS": "2", "UNRELATED_SETTING": "preserved",
            "OSCAR_TRACE_DIR": "/tmp/old-trace", "OSCAR_PASSIVE_TIMING_CONTROL": "/tmp/old-control"}
    if method is not None:
        base["VLLM_WORKER_MULTIPROC_METHOD"] = method
    before = dict(base)
    selected = calibrate_model_rotations.calibration_process_env(base)
    assert base == before
    assert selected is not base
    assert selected["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert selected["OSCAR_ENABLED"] == "0"
    assert "OSCAR_TRACE_DIR" not in selected and "OSCAR_PASSIVE_TIMING_CONTROL" not in selected
    assert selected["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3"
    assert selected["OMP_NUM_THREADS"] == "2" and selected["UNRELATED_SETTING"] == "preserved"


def test_capture_passes_isolated_environment_without_mutating_launcher(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    monkeypatch.setenv("OSCAR_ENABLED", "1")
    monkeypatch.setenv("OSCAR_TRACE_DIR", "/tmp/launcher-trace")
    monkeypatch.setenv("OSCAR_PASSIVE_TIMING_CONTROL", "/tmp/launcher-control")
    before = dict(os.environ)
    config = tmp_path / "config.json"
    directory = tmp_path / "capture"
    directory.mkdir()
    called = []

    def run(command, **kwargs):
        called.append(command)
        assert command == [sys.executable, "-m", "tools.calibrate_model_rotations",
                           "--config", str(config), "--output-dir", str(directory)]
        assert kwargs["check"] is True
        assert kwargs["env"]["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        assert kwargs["env"]["OSCAR_ENABLED"] == "0"
        assert "OSCAR_TRACE_DIR" not in kwargs["env"]
        assert "OSCAR_PASSIVE_TIMING_CONTROL" not in kwargs["env"]
        # Output remains inherited, so exceptions and progress are live.
        assert not kwargs.get("capture_output")
        assert kwargs.get("stdout") is None and kwargs.get("stderr") is None
        (directory / "capture-manifest.json").write_text(json.dumps({"status": "host_double"}))

    monkeypatch.setattr(prepare_rotations.subprocess, "run", run)
    assert prepare_rotations._capture(config, directory) == {"status": "host_double"}
    assert len(called) == 1 and dict(os.environ) == before


TWO_LEVEL_SCRIPT = r'''
import ast
import json
import multiprocessing
import os
from pathlib import Path
import sys

# Keep the native environment -> get_context call intact. Platform auto-spawn
# probes are not relevant to this deliberately CPU-only test process.
class NativeEnvironment:
    @property
    def VLLM_WORKER_MULTIPROC_METHOD(self):
        return os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")

native_path = Path(sys.argv[1])
tree = ast.parse(native_path.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_mp_context")
namespace = {"multiprocessing": multiprocessing, "envs": NativeEnvironment(),
             "_maybe_force_spawn": lambda: None, "_sync_visible_devices_env_vars": lambda: None}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(native_path), "exec"), namespace)
get_mp_context = namespace["get_mp_context"]
INITIALIZED_BY = None

def initialize_own_pool(role):
    global INITIALIZED_BY
    inherited = INITIALIZED_BY
    assert inherited is None, (role, "inherited live parent memory", inherited)
    import torch
    torch.set_num_threads(2)
    # Materialize real intra-op and autograd work before creating descendants.
    value = torch.ones((128, 128), requires_grad=True)
    (value @ value.T).sum().backward()
    assert torch.equal(value.grad, torch.full_like(value, 256.0))
    INITIALIZED_BY = os.getpid()
    return {"role": role, "pid": os.getpid(), "ppid": os.getppid(),
            "method": get_mp_context().get_start_method(),
            "actual_method": multiprocessing.get_start_method(),
            "inherited_pool_owner": inherited, "threads": torch.get_num_threads(),
            "gradient": float(value.grad[0, 0])}

def worker(result_queue, expected_parent):
    assert os.getppid() == expected_parent
    result_queue.put(initialize_own_pool("worker"))

def engine(result_queue, expected_parent):
    assert os.getppid() == expected_parent
    row = initialize_own_pool("engine")
    context = get_mp_context()
    assert context.get_start_method() == "spawn"
    children_queue = context.Queue()
    children = [context.Process(target=worker, args=(children_queue, os.getpid())) for _ in range(2)]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(12)
            assert not child.is_alive(), "worker exceeded bounded functional check"
            assert child.exitcode == 0, child.exitcode
        workers = [children_queue.get(timeout=2) for _ in children]
        result_queue.put({"engine": row, "workers": workers,
                          "worker_exitcodes": [child.exitcode for child in children]})
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(2)
            if child.is_alive():
                child.kill()
                child.join(2)
        children_queue.close()
        children_queue.join_thread()

def main():
    parent = initialize_own_pool("parent")
    context = get_mp_context()
    assert context.get_start_method() == "spawn"
    result_queue = context.Queue()
    child = context.Process(target=engine, args=(result_queue, os.getpid()))
    try:
        child.start()
        child.join(35)
        assert not child.is_alive(), "engine exceeded bounded functional check"
        assert child.exitcode == 0, child.exitcode
        result = result_queue.get(timeout=2)
        print(json.dumps({"parent": parent, **result, "engine_exitcode": child.exitcode}), flush=True)
    finally:
        if child.is_alive():
            child.terminate()
            child.join(2)
        if child.is_alive():
            child.kill()
            child.join(2)
        result_queue.close()
        result_queue.join_thread()

if __name__ == "__main__":
    main()
'''


def test_native_context_spawns_two_fresh_levels_after_parent_torch_pool_is_warm(tmp_path):
    script = tmp_path / "two_level_pool.py"
    script.write_text(textwrap.dedent(TWO_LEVEL_SCRIPT))
    native = ROOT / "references/vllm/vllm/utils/system_utils.py"
    environment = calibrate_model_rotations.calibration_process_env()
    environment["OMP_NUM_THREADS"] = "2"
    environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    process = subprocess.Popen([sys.executable, str(script), str(native)], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f"two-level host process check timed out\n{stdout}\n{stderr}")
    assert process.returncode == 0, stdout + stderr
    result = json.loads(stdout.strip().splitlines()[-1])
    parent, engine, workers = result["parent"], result["engine"], result["workers"]
    assert len(workers) == 2
    assert len({row["pid"] for row in [parent, engine, *workers]}) == 4
    assert engine["ppid"] == parent["pid"]
    assert all(worker["ppid"] == engine["pid"] for worker in workers)
    for row in [engine, *workers]:
        assert row["method"] == row["actual_method"] == "spawn"
        assert row["inherited_pool_owner"] is None
        assert row["threads"] == 2 and row["gradient"] == 256.0
    assert parent["threads"] == 2 and parent["gradient"] == 256.0
    assert result["engine_exitcode"] == 0 and result["worker_exitcodes"] == [0, 0]
