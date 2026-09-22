# 档案 #75/#94/#95/#101/#116/#117/#120：注入真实失败/超时，验证退出码、日志与清理。
import json
from pathlib import Path
import sys
import signal
import subprocess
import time
import pytest
from tools.phase import run_phase
from tools import phase
from tools.environment import compare_integrity, file_fingerprint
from tools.build_ops import normalize_soc, cann_root, clear_owned_build
from tools.target_cli import serve_argv, target_env
from tools.prepare_rotations import model_geometry

ROOT = Path(__file__).resolve().parents[1]


def test_failure_keeps_original_code_and_diagnostic(tmp_path):
    result = run_phase("failure", [sys.executable, "-c", "import sys;print('original failure');sys.exit(7)"],
                       cwd=ROOT, log_dir=tmp_path, timeout=3)
    assert result.returncode == 7 and result.cleanup_complete
    assert "original failure" in Path(result.log).read_text()
    assert json.loads((tmp_path / "failure.json").read_text())["returncode"] == 7


def test_silent_failure_still_has_log(tmp_path):
    result = run_phase("silent", [sys.executable, "-c", "raise SystemExit(3)"], cwd=ROOT, log_dir=tmp_path, timeout=3)
    assert result.returncode == 3
    assert "START phase=silent" in Path(result.log).read_text()
    assert "RESULT" in Path(result.log).read_text()


def test_timeout_is_bounded_and_reaped(tmp_path):
    result = run_phase("timeout", [sys.executable, "-c", "import time;time.sleep(10)"], cwd=ROOT,
                       log_dir=tmp_path, timeout=0.15, grace=0.2)
    assert result.returncode == 124 and result.timed_out and result.cleanup_complete
    assert result.elapsed_seconds < 3


def test_exec_failure_recorded(tmp_path):
    result = run_phase("missing", [str(tmp_path / "no-executable")], cwd=ROOT, log_dir=tmp_path, timeout=1)
    assert result.returncode == 127 and "EXEC_ERROR" in Path(result.log).read_text()


class ExitingChild:
    """A deterministic owned child in Darwin's exit-before-waitpid window."""
    pid = 42900

    def __init__(self, *, live=False):
        self.returncode = None
        self.waits = []
        self.live = live

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.live:
            raise subprocess.TimeoutExpired(["owned-fixture"], timeout)
        self.returncode = -15
        return self.returncode


@pytest.mark.parametrize("failing_signal", [0, signal.SIGTERM, signal.SIGKILL])
def test_cleanup_reaps_exiting_owned_child_before_confirming_group_disappearance(monkeypatch, failing_signal):
    # Local macOS reproduction, not target NPU evidence: killpg(0) EPERM for
    # a Z/<defunct> child disappears after waitpid reaps that owned child.
    child = ExitingChild()
    calls = []
    injected = False

    def killpg(pid, signum):
        nonlocal injected
        assert pid == child.pid
        calls.append(signum)
        if child.returncode is not None:
            raise ProcessLookupError("group is now absent")
        if signum == failing_signal and not injected:
            injected = True
            raise PermissionError("owned child exiting")

    monkeypatch.setattr(phase.os, "killpg", killpg)
    assert phase.cleanup_group(child, grace=0)
    assert injected and child.returncode == -15
    assert child.waits and max(child.waits) <= 1
    assert calls[-1] == 0  # only a fresh ESRCH probe establishes disappearance


def test_cleanup_never_treats_persistent_group_permission_error_as_absence(monkeypatch):
    child = ExitingChild()
    calls = []
    denied = PermissionError("persistent group denial")

    def killpg(pid, signum):
        calls.append((pid, signum))
        raise denied

    monkeypatch.setattr(phase.os, "killpg", killpg)
    with pytest.raises(PermissionError) as failure:
        phase.cleanup_group(child, grace=0)
    assert failure.value is denied and child.returncode == -15
    assert calls == [(child.pid, 0), (child.pid, 0)]


def test_cleanup_live_child_permission_error_has_one_bounded_wait_and_still_fails(monkeypatch):
    child = ExitingChild(live=True)
    denied = PermissionError("live child denied")
    calls = []

    def killpg(pid, signum):
        calls.append((pid, signum))
        raise denied

    monkeypatch.setattr(phase.os, "killpg", killpg)
    with pytest.raises(PermissionError) as failure:
        phase.cleanup_group(child, grace=0)
    assert failure.value is denied and child.returncode is None
    assert child.waits == [.2] and calls == [(child.pid, 0)]


@pytest.mark.parametrize("failing_signal", [signal.SIGTERM, signal.SIGKILL])
def test_signal_denial_is_not_hidden_when_owned_group_still_exists(monkeypatch, failing_signal):
    child = ExitingChild()
    denied = PermissionError("signal denied for remaining group")

    def killpg(pid, signum):
        assert pid == child.pid
        if signum == failing_signal:
            raise denied

    monkeypatch.setattr(phase.os, "killpg", killpg)
    with pytest.raises(PermissionError) as failure:
        phase.cleanup_group(child, grace=0)
    assert failure.value is denied and child.returncode == -15


def test_native_mutation_detects_new_deleted_and_changed(tmp_path):
    (tmp_path / "source.py").write_text("original")
    before = file_fingerprint(tmp_path)
    (tmp_path / "source.py").write_text("changed")
    (tmp_path / "new.py").write_text("new")
    assert compare_integrity(before, file_fingerprint(tmp_path))["changed"] == ["new.py", "source.py"]


def test_soc_mapping_does_not_select_310p_headers():
    assert normalize_soc("Ascend910B4") == "ascend910b4"
    with pytest.raises(ValueError, match="wrong header"):
        normalize_soc("ascend910b")


def test_build_cleanup_is_project_owned(tmp_path):
    with pytest.raises(ValueError, match="non-project"):
        clear_owned_build(tmp_path)
    assert tmp_path.exists()


def test_target_command_preserves_graph_mtp_and_dtype():
    config = json.loads((ROOT / "configs/target.json").read_text())
    argv = serve_argv(config)
    assert json.loads(argv[argv.index("--compilation-config")+1])["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert json.loads(argv[argv.index("--speculative_config")+1])["num_speculative_tokens"] == 3
    assert argv[argv.index("--tensor-parallel-size")+1] == "4"
    assert argv[argv.index("--max-model-len")+1] == "262144"
    assert "--enforce-eager" not in argv
    assert argv[argv.index("--mamba-ssm-cache-dtype")+1] == "bfloat16"


def test_profiler_config_env_arms_native_window_without_config_edit(tmp_path):
    from tools.target_cli import profiler_config_env
    assert profiler_config_env({}, {}) is None
    configured = {"profiler": "torch", "torch_profiler_dir": "/explicit"}
    assert profiler_config_env({"profiler_config": configured}, {"OSCAR_PROFILE_DIR": "ignored"}) is configured
    armed = profiler_config_env({}, {"OSCAR_PROFILE_DIR": str(tmp_path / "trace")})
    assert armed == {"profiler": "torch", "torch_profiler_dir": str((tmp_path / "trace").resolve())}
    with pytest.raises(ValueError):
        profiler_config_env({}, {"OSCAR_PROFILE_DIR": "../outside/"})


def test_device_selection_does_not_inherit_somebody_elses_devices():
    with pytest.raises(ValueError, match="physical NPU"):
        target_env({"devices": None}, {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"})
    selected = target_env({"devices": [4, 5, 6, 7]}, {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3", "VLLM_PLUGINS": "ascend"})
    assert selected["ASCEND_RT_VISIBLE_DEVICES"] == "4,5,6,7"
    assert selected["VLLM_PLUGINS"] == "ascend,oscar_ascend"


def test_model_geometry_reads_real_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 8,
        "full_attention_interval": 4, "head_dim": 256}}))
    layers, dim, fingerprint = model_geometry(tmp_path)
    assert layers == ["model.layers.3.self_attn.attn", "model.layers.7.self_attn.attn"]
    assert dim == 256 and len(fingerprint) == 64
