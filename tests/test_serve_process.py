# Archive G21/G22/#94/#95/#125/#135: local process fault injection proves
# owned cleanup/logging only; these tests do not simulate NPU memory release.
"""Exercise the direct service supervisor with real, bounded host children."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from tools import install_serve, phase, serve_process


ROOT = Path(__file__).resolve().parents[1]
MEMORY_ERROR = (
    "ValueError: Free memory on device (1.19/29.49 GiB) on startup is less than "
    "desired GPU memory utilization (0.9, 26.54 GiB). Decrease GPU memory "
    "utilization or reduce GPU memory used by other processes."
)


def config():
    value = json.loads((ROOT / "configs/target.json").read_text())
    value.update(phase_timeout_seconds=.01, shutdown_timeout_seconds=.15)
    return value


def substitute_service(monkeypatch, command):
    original = subprocess.Popen
    seen = []

    def spawn(actual, **kwargs):
        assert actual[1:3] == ["-m", "tools.target_cli"]
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4,5,6,7"
        assert kwargs["env"]["OSCAR_TERMINAL_LOG_MODE"] == "full"
        # It must be a regular log file, not PIPE: an orphan inheriting the
        # descriptor must never keep the supervisor blocked on pipe EOF.
        assert kwargs["stdout"].fileno() >= 0
        child = original(command, **kwargs)
        seen.append(child)
        return child

    monkeypatch.setattr(phase.subprocess, "Popen", spawn)
    return seen


def test_startup_failure_keeps_log_exit_code_and_unrelated_service(monkeypatch, tmp_path, capsys):
    outsider = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    try:
        seen = substitute_service(monkeypatch, [sys.executable, "-c",
            f"import time;time.sleep(.15);print({MEMORY_ERROR!r});raise SystemExit(7)"])
        settings = config()
        result = serve_process.run_service(settings, ROOT / "configs/target.json", log_dir=tmp_path)
        assert result.returncode == 7 and result.cleanup_complete and not result.timed_out
        assert result.elapsed_seconds > settings["phase_timeout_seconds"]
        assert outsider.poll() is None
        assert not phase.group_exists(seen[0].pid)
        assert MEMORY_ERROR in Path(result.log).read_text()
        output = capsys.readouterr()
        assert MEMORY_ERROR in output.out
        assert "STARTUP_MEMORY_INSUFFICIENT" in output.err
        assert "npu-smi info" in output.err and "ps -eo pid,ppid,pgid,etime,args" in output.err
        report = json.loads((tmp_path / "serve-diagnostic.json").read_text())
        assert report["reported_gib"] == {"free": "1.19", "total": "29.49", "required": "26.54"}
        assert report["memory_owner"] == "not_established"
        assert report["npu_resource_release"] == "not_measured"
        assert settings["gpu_memory_utilization"] == .9
    finally:
        assert phase.cleanup_group(outsider, .2)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
@pytest.mark.parametrize("repeated", [False, True])
def test_signal_reaps_term_ignoring_service_and_keeps_signal_code(monkeypatch, tmp_path, signum, repeated):
    # Send the signal only after the owned child installed its ignore handler;
    # the supervisor must reach its bounded SIGKILL escalation, not hang.
    repeat = f"time.sleep(.03);os.kill(os.getppid(),{int(signum)});" if repeated else ""
    command = [sys.executable, "-c",
        "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"os.kill(os.getppid(),{int(signum)});{repeat}time.sleep(30)"]
    seen = substitute_service(monkeypatch, command)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    result = serve_process.run_service(config(), ROOT / "configs/target.json", log_dir=tmp_path)
    assert result.interrupted and result.returncode == 128 + signum
    assert result.cleanup_complete and result.elapsed_seconds < 3
    assert seen[0].returncode == -signal.SIGKILL
    assert not phase.group_exists(seen[0].pid)
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


def test_service_child_signal_retains_raw_code_and_correct_shell_code(monkeypatch, tmp_path):
    substitute_service(monkeypatch, [sys.executable, "-c",
        "import os,signal;os.kill(os.getpid(),signal.SIGTERM)"])
    result = serve_process.run_service(config(), ROOT / "configs/target.json", log_dir=tmp_path)
    assert result.returncode == -signal.SIGTERM and result.cleanup_complete
    assert serve_process.shell_returncode(result.returncode) == 143


def test_killed_orphan_transient_eperm_needs_fresh_esrch(monkeypatch):
    # Independent host fault model for the locally observed Darwin sequence:
    # killpg(SIGKILL)=0; killpg(0)=EPERM twice; then the orphan is reaped/ESRCH.
    class ExitedLeader:
        pid = 424242

        def poll(self):
            return 9

        def wait(self, timeout):
            return 9

    seen = []
    killed = False
    denial_count = 0

    def killpg(pid, signum):
        nonlocal killed, denial_count
        assert pid == ExitedLeader.pid
        seen.append(signum)
        if signum == signal.SIGKILL:
            killed = True
        elif killed and signum == 0:
            denial_count += 1
            if denial_count <= 2:
                raise PermissionError("orphan has exited but has not been reaped")
            raise ProcessLookupError("group has disappeared")

    monkeypatch.setattr(phase.os, "killpg", killpg)
    assert phase.cleanup_group(ExitedLeader(), 0)
    assert denial_count == 3 and seen[-1] == 0


def test_group_exit_wait_never_accepts_persistent_permission_denial(monkeypatch):
    class ExitedLeader:
        pid = 424242

        def poll(self):
            return 9

    denied = PermissionError("group visibility is unavailable")
    calls = []

    def killpg(pid, signum):
        calls.append((pid, signum))
        raise denied

    monkeypatch.setattr(phase.os, "killpg", killpg)
    with pytest.raises(PermissionError) as error:
        phase._wait_owned_group_exit(ExitedLeader(), 0)
    assert error.value is denied
    assert calls == [(ExitedLeader.pid, 0), (ExitedLeader.pid, 0)]


def test_leader_exit_cleans_descendant_holding_log_fd_without_waiting_for_eof(monkeypatch, tmp_path):
    ready = tmp_path / "descendant.pid"
    # A failed API server can leave workers behind. The descendant inherits
    # stdout/stderr, ignores TERM and remains in the owned session/group.
    program = tmp_path / "leader.py"
    program.write_text(
        "import os,signal,time\nfrom pathlib import Path\n"
        "pid=os.fork()\n"
        "if pid == 0:\n"
        " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        f" Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        " print('worker-still-holds-log-fd',flush=True)\n"
        " time.sleep(30)\n"
        " os._exit(0)\n"
        f"while not Path({str(ready)!r}).exists(): time.sleep(.01)\n"
        "os._exit(9)\n"
    )
    seen = substitute_service(monkeypatch, [sys.executable, str(program)])
    result = serve_process.run_service(config(), ROOT / "configs/target.json", log_dir=tmp_path)
    assert result.returncode == 9 and result.cleanup_complete
    assert result.elapsed_seconds < 3 and ready.exists()
    assert seen[0].returncode == 9 and not phase.group_exists(seen[0].pid)
    assert "worker-still-holds-log-fd" in Path(result.log).read_text()


@pytest.mark.parametrize("code,interrupted,status,expected", [
    (0, False, "stopped", 0), (7, False, "failed", 7),
    (-signal.SIGTERM, False, "failed", 143), (130, True, "interrupted", 130),
])
def test_install_launcher_records_formal_service_outcome(monkeypatch, tmp_path, code, interrupted, status, expected):
    logs = tmp_path / "logs"
    monkeypatch.setattr(install_serve, "run_phase", lambda name, command, **kwargs:
        phase.PhaseResult(name, command, 0, .01, False, False, str(logs / name), True))

    def launch(selected, selected_path, *, log_dir, env):
        assert selected["devices"] == [4, 5, 6, 7] and selected["port"] == 7878
        assert selected["gpu_memory_utilization"] == .9
        assert selected["experimental_decode_bundle"] is True
        assert json.loads(Path(selected_path).read_text()) == selected
        assert env["OSCAR_TARGET_CONFIG"] == str(selected_path)
        return phase.PhaseResult("serve", ["formal-service"], code, .2, False, interrupted,
                                 str(logs / "serve.log"), True)

    monkeypatch.setattr(install_serve, "run_service", launch)
    assert install_serve.main(["--variant", "candidate", "--log-dir", str(logs)]) == expected
    state = json.loads((logs / "status.json").read_text())
    assert state["status"] == status and state["service_returncode"] == code
    assert state["service"]["cleanup_complete"] and state["probes"] == "none"
    assert state["npu_resource_release"] == "not_measured"


def test_unknown_failure_has_no_memory_diagnosis_and_diagnostic_io_cannot_change_rc(monkeypatch, tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("RuntimeError: an unrelated startup error\n")
    assert serve_process.startup_diagnostic(log, config()) is None
    log.write_text(MEMORY_ERROR)
    result = phase.PhaseResult("serve", ["service"], 7, .1, False, False, str(log), False)
    monkeypatch.setattr(serve_process, "run_phase", lambda *args, **kwargs: result)
    monkeypatch.setattr(serve_process, "atomic_json", lambda *args: (_ for _ in ()).throw(OSError("read-only")))
    assert serve_process.run_service(config(), ROOT / "configs/target.json", log_dir=tmp_path) is result


def test_direct_script_command_inspection_stays_read_only(tmp_path):
    result = subprocess.run(["bash", str(ROOT / "scripts/serve_direct.sh"), "--print-command",
        "--log-dir", str(tmp_path / "must-not-exist")], cwd=ROOT,
        env={**os.environ, "OSCAR_PYTHON": sys.executable}, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--port 7878" in result.stdout and "--gpu-memory-utilization 0.9" in result.stdout
    assert not (tmp_path / "must-not-exist").exists()
