"""Archive #74/#75/#94/#125/#175/#176/#177: explicit host actions behind the lifecycle guard.

Host changes are intercepted, and shell entrypoints run only against SDK/Python
stubs. These tests never write /proc, mount filesystems, or initialize an SDK.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import node93_host

ROOT = Path(__file__).resolve().parents[1]


def test_host_preparation_uses_only_the_three_supplied_operations(monkeypatch, capsys):
    operations = []
    monkeypatch.setattr(Path, "write_text", lambda path, value:
                        operations.append(("write", str(path), value)))
    monkeypatch.setattr(node93_host.subprocess, "run", lambda command, **kwargs:
                        operations.append(("run", command, kwargs)))
    assert node93_host.main([]) == 0
    assert operations == [
        ("write", "/proc/sys/vm/nr_hugepages", "200000\n"),
        ("run", ("mount", "-o", "remount,size=8G", "/dev/shm"), {"check": True}),
        ("run", ("mkdir", "-p", "/home/kvcache"), {"check": True}),
    ]
    assert "nr_hugepages=200000" in capsys.readouterr().out


def test_host_plan_performs_no_system_operation(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("plan must not change the host")
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(node93_host.subprocess, "run", forbidden)
    assert node93_host.main(["--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["executed"] is False
    assert plan["write"] == {"path": "/proc/sys/vm/nr_hugepages", "value": "200000\n"}
    assert plan["commands"] == [["mount", "-o", "remount,size=8G", "/dev/shm"],
                                ["mkdir", "-p", "/home/kvcache"]]


def test_hugepages_permission_failure_is_visible_and_stops_before_mount(monkeypatch, capsys):
    def denied(path, value):
        raise PermissionError(13, "Permission denied", str(path))
    monkeypatch.setattr(Path, "write_text", denied)
    monkeypatch.setattr(node93_host.subprocess, "run", lambda *a, **k:
                        pytest.fail("write failure must stop host preparation"))
    assert node93_host.main([]) == 1
    error = capsys.readouterr().err
    assert "PermissionError" in error and "/proc/sys/vm/nr_hugepages" in error


@pytest.mark.parametrize("failed_index,returncode", [(0, 7), (1, 29), (0, -9)])
def test_host_command_failure_preserves_status_and_stops(monkeypatch, capsys, failed_index, returncode):
    commands = []
    monkeypatch.setattr(Path, "write_text", lambda *args: None)
    def run(command, **kwargs):
        commands.append(command)
        if len(commands) == failed_index + 1:
            raise subprocess.CalledProcessError(returncode, command)
    monkeypatch.setattr(node93_host.subprocess, "run", run)
    assert node93_host.main([]) == (returncode if returncode >= 0 else 128 - returncode)
    assert commands == list(node93_host.HOST_COMMANDS[:failed_index + 1])
    assert f"rc={returncode}" in capsys.readouterr().err


@pytest.fixture
def launch_stubs(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("start_node93_oscar.sh", "start_node93_memcache.sh", "install_serve.sh"):
        (scripts / name).write_text((ROOT / "scripts" / name).read_text())
    (scripts / "env_node93_memcache.sh").write_text(
        'printf \'{"kind":"source"}\\n\' >> "$STUB_TRACE"\n'
        'export SDK_MARKER=installed MMC_META_CONFIG_PATH=/sdk/config/mmc-meta.conf\n'
        'return "${STUB_SOURCE_RC:-0}"\n')
    python = tmp_path / "fake-python"
    python.write_text(f"#!{sys.executable}\n" +
        "import json, os, sys\n"
        "with open(os.environ['STUB_TRACE'], 'a') as output:\n"
        "    output.write(json.dumps({'kind':'python', 'args':sys.argv[1:], "
        "'sdk':os.environ.get('SDK_MARKER')}) + '\\n')\n"
        "stage = os.environ.get('STUB_FAILURE_STAGE')\n"
        "sys.exit(int(os.environ.get('STUB_FAILURE_RC', '7')) "
        "if stage and stage in sys.argv else 0)\n")
    python.chmod(0o755)
    trace = tmp_path / "trace.jsonl"
    env = dict(os.environ, OSCAR_PYTHON=str(python), STUB_TRACE=str(trace))
    env.pop("SDK_MARKER", None)
    def launch(name, args=(), overrides=None):
        result = subprocess.run(["bash", str(scripts / name), *args],
                                env=dict(env, **(overrides or {})), text=True, capture_output=True)
        events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        return result, events
    return launch


def test_oscar_entry_sources_then_passes_node93_candidate_and_user_args(launch_stubs):
    extra = ["--plan", "--no-kv-offload", "--log-dir", "path with spaces"]
    result, events = launch_stubs("start_node93_oscar.sh", extra)
    assert result.returncode == 0
    assert events == [{"kind": "source"}, {"kind": "python", "sdk": "installed", "args": [
        "-m", "tools.install_serve", "--config", "configs/benchmark_node93.json",
        "--variant", "candidate", *extra]}]


def test_oscar_entry_retains_formal_service_exit_code(launch_stubs):
    result, events = launch_stubs("start_node93_oscar.sh", overrides={
        "STUB_FAILURE_STAGE": "tools.install_serve", "STUB_FAILURE_RC": "31"})
    assert result.returncode == 31 and len(events) == 2


@pytest.mark.parametrize("prepare", [False, True])
def test_meta_entry_delegates_preparation_to_guarded_lifecycle(launch_stubs, prepare):
    result, events = launch_stubs("start_node93_memcache.sh", ["--prepare-host"] if prepare else [])
    expected = [{"kind": "source"},
        {"kind": "python", "sdk": "installed", "args": ["-m", "tools.memcache_service", "node93",
            "--config-dir", "/sdk/config", *(["--prepare-host"] if prepare else [])]}]
    assert result.returncode == 0 and events == expected


@pytest.mark.parametrize("script", ["start_node93_oscar.sh", "start_node93_memcache.sh"])
def test_source_failure_preserves_status_and_stops_launch(launch_stubs, script):
    result, events = launch_stubs(script, overrides={"STUB_SOURCE_RC": "17"})
    assert result.returncode == 17 and events == [{"kind": "source"}]


def test_meta_lifecycle_failure_retains_real_code(launch_stubs):
    result, events = launch_stubs("start_node93_memcache.sh", ["--prepare-host"],
        {"STUB_FAILURE_STAGE": "tools.memcache_service", "STUB_FAILURE_RC": "23"})
    assert result.returncode == 23 and len(events) == 2
    assert "tools.memcache_service" in events[-1]["args"]


@pytest.mark.parametrize("script", ["start_node93_oscar.sh", "start_node93_memcache.sh"])
@pytest.mark.parametrize("args", [["--help"], ["--prepare-host", "--help"]])
def test_help_does_not_source_sdk_or_prepare_host(launch_stubs, script, args):
    result, events = launch_stubs(script, args)
    assert result.returncode == 0 and not events
    assert "Usage:" in result.stdout


def test_unknown_meta_option_does_not_touch_host(launch_stubs):
    result, events = launch_stubs("start_node93_memcache.sh", ["--unexpected"])
    assert result.returncode == 2 and not events
    assert "unknown argument" in result.stderr
