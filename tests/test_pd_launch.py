"""#94/#125/#148/#172: PD roles reach the same candidate service/probe argv.

Read-only plans; no SDK, target model, build, NPU or performance experiment.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def plan(module, *args):
    completed = subprocess.run([sys.executable, "-m", module, "--variant", "candidate", "--plan", *args],
                               cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


@pytest.mark.parametrize("role,wire", [("prefill", "kv_producer"), ("decode", "kv_consumer")])
def test_direct_plan_reaches_service_and_keeps_math_budget_mtp(role, wire):
    baseline = plan("tools.install_serve")
    pd = plan("tools.install_serve", "--pd-role", role)
    actual, previous = list(pd["vllm_argv"]), list(baseline["vllm_argv"])
    offset = actual.index("--kv-transfer-config")
    transfer = json.loads(actual[offset + 1])
    assert transfer["kv_role"] == wire
    assert transfer["kv_connector_extra_config"]["pd_role"] == role
    assert transfer["kv_connector_extra_config"]["load_async"] is True
    del actual[offset:offset + 2]
    if "--kv-transfer-config" in previous:
        offset = previous.index("--kv-transfer-config")
        del previous[offset:offset + 2]
    assert actual == previous
    assert pd["runtime_selection"]["pd_role"] == role
    assert pd["runtime_selection"]["optimizations"] == baseline["runtime_selection"]["optimizations"]


def test_rear_cards_explicit_port_and_probe_entry_accept_role():
    d = plan("tools.install_serve", "--pd-role", "decode", "--rear-cards", "--port", "7880")
    assert d["port"] == 7880 and d["target_devices"] == [4, 5, 6, 7]
    probe = plan("tools.deploy", "--pd-role", "prefill")
    assert probe  # Parses and validates the same effective config without NPU.


@pytest.mark.parametrize("args", [("--pd-role", "decode", "--no-kv-offload"),
                                  ("--pd-role", "decode", "--variant", "baseline"),
                                  ("--port", "0")])
def test_conflicting_launch_cannot_silently_drop_pd(args):
    completed = subprocess.run([sys.executable, "-m", "tools.install_serve", "--plan", *args],
                               cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode != 0
