"""Archive #118/#122: actual temporary log fixtures and owned subprocesses.

These files are deliberately synthetic test inputs. No NPU/CANN plog is
fabricated as acceptance evidence, and tests do not inspect real user logs.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from tools import plog, probe_ops, service_probe


def log(path, text, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (mtime, mtime))
    return path


def test_only_recent_owned_files_or_explicit_owned_lines_are_read(tmp_path, monkeypatch):
    started = time.time()
    root = tmp_path / "cann"
    home = tmp_path / "home"
    own = log(root / "debug/plog/plog-321_1.log",
              "ERROR generic shell\noscar_attention_cv_kernel launch failed\n[ERROR] RUNTIME(999,python): foreign error\n",
              started + 1)
    aggregate = log(root / "runtime.log",
                    "[ERROR] RUNTIME(321,python): kernel owned error\n"
                    "[ERROR] RUNTIME(1321,python): kernel foreign error\n"
                    "[ERROR] RUNTIME(999,python): foreign IPC peer pid=321 oscar error\n"
                    "ERROR missing pid\n", started + 1)
    foreign = log(root / "plog-1321_1.log", "pid=321 oscar foreign file must not be opened\n", started + 1)
    old = log(root / "plog-321_old.log", "oscar old error\n", started - 1)
    outside = log(tmp_path / "other-project/plog-321_1.log", "oscar outside error\n", started + 1)
    (root / "linked.log").symlink_to(outside)
    (root / "linked-directory").symlink_to(outside.parent, target_is_directory=True)
    opened = []
    native_open = os.open

    def guarded_open(path, flags, *args, **kwargs):
        assert Path(path) not in {foreign, old, outside}
        opened.append(Path(path))
        return native_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    report = plog.collect_plog(started_at=started, owned_pids={321},
                               environ={"ASCEND_PROCESS_LOG_PATH": str(root)}, home=home)
    assert set(opened) == {own, aggregate}
    assert report["status"] == "evidence_collected"
    assert report["excerpts"][0]["text"] == "oscar_attention_cv_kernel launch failed"
    assert all(entry["pid"] == 321 for entry in report["excerpts"])
    text = "\n".join(entry["text"] for entry in report["excerpts"])
    assert "foreign" not in text and "missing pid" not in text and "old error" not in text


def test_default_user_root_and_tilde_are_explicit_and_broad_roots_are_rejected(tmp_path):
    started = time.time()
    home = tmp_path / "home"
    path = log(home / "ascend/log/debug/plog-pid_432_1.log", "pid=432 OscarStore kernel failed\n", started + 1)
    report = plog.collect_plog(started_at=started, owned_pids={432},
                               environ={"ASCEND_PROCESS_LOG_PATH": "/"}, home=home)
    assert report["roots"] == [str(home / "ascend/log")]
    assert report["excerpts"][0]["path"] == str(path)
    report = plog.collect_plog(started_at=started, owned_pids={432},
                               environ={"ASCEND_PROCESS_LOG_PATH": "~/ascend/log"}, home=home)
    assert report["roots"] == [str(home / "ascend/log")]


def test_log_file_byte_excerpt_and_scan_bounds(tmp_path):
    started = time.time()
    root = tmp_path / "cann"
    for i in range(10):
        log(root / f"plog-321_{i}.log", "pid=321 oscar kernel error data\n" * 100, started + i + 1)
    limits = plog.PlogLimits(files=2, entries=100, depth=1, bytes_per_file=128,
                             total_bytes=180, lines=2, excerpt_chars=50, line_chars=30)
    report = plog.collect_plog(started_at=started, owned_pids={321},
                               environ={"ASCEND_PROCESS_LOG_PATH": str(root)}, home=tmp_path / "home", limits=limits)
    assert report["files_read"] <= 2 and report["bytes_read"] <= 180
    assert len(report["excerpts"]) <= 2 and sum(len(x["text"]) for x in report["excerpts"]) <= 50
    assert report["truncated"]


def test_root_cause_error_is_not_crowded_out_by_oscar_info_lines(tmp_path):
    started = time.time()
    root = tmp_path / "cann"
    log(root / "plog-321_fixture.log",
        "INFO oscar artifact discovered\n" * 100 + "ERROR kernel resolution failed at first cause\n", started + 1)
    report = plog.collect_plog(started_at=started, owned_pids={321},
                               environ={"ASCEND_PROCESS_LOG_PATH": str(root)}, home=tmp_path / "home",
                               limits=plog.PlogLimits(lines=1))
    assert report["excerpts"][0]["text"] == "ERROR kernel resolution failed at first cause"


def test_diagnostic_error_never_overwrites_the_original_failure(monkeypatch):
    def unavailable(**kwargs):
        raise PermissionError("fixture denied diagnostic read")
    monkeypatch.setattr(plog, "collect_plog", unavailable)
    report = {"status": "failed", "returncode": 7, "error": "primary failure"}
    plog.attach_plog(report, started_at=time.time(), owned_pids={os.getpid()})
    assert report["status"] == "failed" and report["returncode"] == 7 and report["error"] == "primary failure"
    assert report["cann_plog"]["status"] == "diagnostic_failed"


def test_operator_probe_failure_attaches_current_pid_fixture_without_npu(tmp_path, monkeypatch):
    root, output = tmp_path / "cann", tmp_path / "probe.json"
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def fail_probe():
        log(root / f"plog-{os.getpid()}_fixture.log", "oscar fixture operator failure\n", time.time())
        raise RuntimeError("primary operator fixture failure")

    monkeypatch.setattr(probe_ops, "probe", fail_probe)
    monkeypatch.setattr(sys, "argv", ["probe_ops", "--output", str(output)])
    assert probe_ops.main() == 1
    report = json.loads(output.read_text())
    assert report["error"] == "primary operator fixture failure"
    assert report["device_completion"] == "not_established"
    assert report["cann_plog"]["excerpts"][0]["pid"] == os.getpid()


def test_process_group_ledger_excludes_a_real_unrelated_group():
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
    outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
    try:
        ledger = plog.OwnedProcessGroup(owner.pid)
        ledger.refresh(force=True)
        assert owner.pid in ledger.pids
        assert outsider.pid not in ledger.pids and os.getpid() not in ledger.pids
        assert outsider.poll() is None
    finally:
        owner.terminate()
        outsider.terminate()
        owner.wait(timeout=3)
        outsider.wait(timeout=3)


def test_failed_owned_server_fixture_is_attributed_after_cleanup(tmp_path, monkeypatch):
    root = tmp_path / "cann"
    root.mkdir()
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 4, "head_dim": 64, "full_attention_interval": 4}))
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/target.json").read_text())
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    config.update(model=str(model), devices=[40, 41, 42, 43], host="127.0.0.1", port=port,
                  phase_timeout_seconds=5, service_startup_timeout_seconds=2,
                  shutdown_timeout_seconds=.1, resource_release_timeout_seconds=.1)
    config_path = tmp_path / "target.json"
    config_path.write_text(json.dumps(config))
    command = [sys.executable, "-c",
               "import os,pathlib; p=os.getpid(); "
               "root=pathlib.Path(os.environ['ASCEND_PROCESS_LOG_PATH']); "
               "(root/f'plog-{p}_fixture.log').write_text(f'[ERROR] RUNTIME({p},fixture): oscar kernel fixture failure\\n'); "
               "raise SystemExit(7)"]

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [17]

    def resources(target, **kwargs):
        return {"devices": target["devices"], "memory": [
            {"physical_device": device, "logical_device": index, "free_bytes": 2_000_000_000,
             "total_bytes": 4_000_000_000} for index, device in enumerate(target["devices"])]}

    report = service_probe.run_service(config_path, output=tmp_path / "report.json", log_dir=tmp_path / "logs",
                                       command=command, tokenizer_factory=lambda _model: Tokenizer(), resource_reader=resources)
    assert report["status"] == "failed" and "rc=7" in report["error"]
    assert report["server"]["exit_code"] == 7 and report["server"]["cleanup_complete"]
    assert report["cann_plog"]["excerpts"][0]["pid"] == report["server"]["pid"]
    assert report["device_completion"] == "not_run"
