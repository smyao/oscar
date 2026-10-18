"""Archive #94/#125/#136: BF16 deployment must fail closed without timing runs.

Only host orchestration receipts are injected; these tests are not NPU proof.
"""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from tools import deploy, install_serve, observe_serve, service_probe
from tools.phase import PhaseResult


@pytest.mark.parametrize("entry", [install_serve.main, observe_serve.main, deploy.main])
def test_every_entrypoint_uses_same_model_controls(tmp_path, capsys, entry):
    model = tmp_path / "original-looking-name"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    (model / "quant_model_description.json").write_text('{"layer.weight":"W8A8"}')
    args = ["--model", str(model), "--served-name", "chosen", "--variant", "candidate",
            "--plan", "--log-dir", str(tmp_path / "unused")]
    assert entry(args) == 0
    plan = json.loads(capsys.readouterr().out)
    argv = plan["vllm_argv"]
    assert argv[:2] == ["serve", str(model)]
    assert argv[argv.index("--served-model-name") + 1] == "chosen"
    assert argv[argv.index("--quantization") + 1] == "ascend"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "16384"
    assert "--no-enable-prefix-caching" in argv and "--scheduler-cls" not in argv
    assert not (tmp_path / "unused").exists()


def test_default_bf16_deploy_plan_has_no_synthetic_performance(tmp_path, capsys):
    logs = tmp_path / "unused"
    assert deploy.main(["--plan", "--log-dir", str(logs)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [name for name, _ in plan["stages"]] == ["bf16-correctness-gates", "service-probe"]
    assert "--probe-only" in plan["stages"][0][1]
    assert "--correctness-only" in plan["stages"][1][1]
    assert plan["performance"] == "not_run_user_requested"
    assert "--quantization" not in plan["vllm_argv"]
    assert not logs.exists()


def _setup(tmp_path, monkeypatch, failure=None):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"architectures":["Qwen3_5ForConditionalGeneration"]}')
    logs = tmp_path / "logs"
    calls = []
    def phase(name, command, *, env, **kwargs):
        calls.append(name)
        if failure == "phase_exception":
            raise OSError("host phase launch failed")
        path = Path(env["OSCAR_TARGET_CONFIG"])
        config = json.loads(path.read_text())
        assert config["experimental_bf16_attention"] is True
        assert config["model"] == str(model) and config["quantization"] is None
        assert command[command.index("--config") + 1] == str(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name == "bf16-correctness-gates":
            out = logs / "bf16-correctness" / "status.json"
            report = {"status": "operator_probes_passed", "service_started": False,
                "performance_acceptance": "not_run_user_requested",
                "target_config": {"original_sha256": digest, "effective_sha256": digest},
                "aligned_attention_gate": {"status": "passed"}}
            if failure == "stale_config":
                report["target_config"]["effective_sha256"] = "stale"
            if failure == "fake_gate":
                report["aligned_attention_gate"]["status"] = "not_run"
            if failure == "malformed_receipt":
                report["target_config"] = "invalid"
        else:
            assert name == "service-probe" and "--correctness-only" in command
            out = logs / "service-probe-report.json"
            report = {"status": "passed", "resource_release": "passed", "server": {"cleanup_complete": True},
                "device_completion": "real_service_requests_completed", "mtp": {"status": "passed"},
                "telemetry": {"status": "passed"}, "validation_mode": "correctness_only_user_requested",
                "performance": "not_run_user_requested", "target_config_sha256": digest}
            if failure == "resource":
                report["resource_release"] = "failed"
            if failure == "telemetry":
                report["telemetry"]["status"] = "failed"
        out.parent.mkdir(parents=True, exist_ok=True)
        if failure != "missing_receipt":
            out.write_text(json.dumps(report))
        return PhaseResult(name, command, 29 if failure == "child" else 0, 0.0,
                           False, False, str(logs / f"{name}.log"), True)
    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(deploy, "_compare_performance", lambda *a, **k: pytest.fail("BF16 must not compare speeds"))
    return model, logs, calls


def test_correctness_success_reaches_formal_service_without_performance(monkeypatch, tmp_path):
    model, logs, calls = _setup(tmp_path, monkeypatch)
    served = []
    def serve():
        served.append(list(sys.argv))
        assert "--serve" in sys.argv and "--correctness-only" in sys.argv
        return 0
    monkeypatch.setattr(service_probe, "main", serve)
    assert deploy.main(["--model", str(model), "--log-dir", str(logs)]) == 0
    assert calls == ["bf16-correctness-gates", "service-probe"]
    assert len(served) == 1
    state = json.loads((logs / "status.json").read_text())
    assert state["status"] == "stopped" and state["performance"] == "not_run_user_requested"
    assert "paired_performance" not in state


@pytest.mark.parametrize("failure", ["child", "missing_receipt", "stale_config", "fake_gate", "resource", "telemetry",
                                     "malformed_receipt", "phase_exception"])
def test_bad_correctness_receipt_blocks_service(monkeypatch, tmp_path, failure):
    model, logs, _ = _setup(tmp_path, monkeypatch, failure)
    monkeypatch.setattr(service_probe, "main", lambda: pytest.fail("failed correctness gate must block formal serve"))
    rc = deploy.main(["--model", str(model), "--log-dir", str(logs)])
    assert rc == (29 if failure == "child" else 1)
    assert json.loads((logs / "status.json").read_text())["status"] == "failed"
