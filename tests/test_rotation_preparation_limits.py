"""Archive #86/#94/#95/#97/#166: calibration bounds and nested launch wiring.

Host-only orchestration. No model execution, installation or NPU connections.
"""
import json
from pathlib import Path
import sys

import pytest

from tools import deploy, observe_serve
from tools.phase import PhaseResult, _compact_should_mirror, run_phase
from tools.rotation_preparation_limits import (phase_timeout,
    preparation_phase_timeout, preparation_progress_timeout)


def _config(tmp_path, *, layers=64, heads=4, steps=16, nested=True):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    metadata = {"num_hidden_layers": layers, "num_key_value_heads": heads,
                "full_attention_interval": 4,
                "architectures": ["Qwen3_5ForConditionalGeneration"]}
    if nested:
        metadata = {"architectures": metadata["architectures"], "text_config": metadata}
    (model / "config.json").write_text(json.dumps(metadata))
    config = json.loads((deploy.ROOT / "configs/target.json").read_text())
    return {**config, "model": str(model), "rotation_refinement": "optr",
            "optr_calibration": {"steps": steps}}


def test_default_budget_covers_all_heads_but_progress_deadline_does_not_grow(tmp_path):
    config = _config(tmp_path)
    before = json.dumps(config, sort_keys=True)
    assert preparation_phase_timeout(config) == 86400
    assert preparation_progress_timeout(config) == 1800
    assert phase_timeout(config, "bf16-correctness-gates") == 88200
    assert json.dumps(config, sort_keys=True) == before


def test_budget_scales_full_layers_heads_steps_and_stays_bounded(tmp_path):
    full = preparation_phase_timeout(_config(tmp_path, nested=False))
    fewer_layers = preparation_phase_timeout(_config(tmp_path, layers=32))
    fewer_heads = preparation_phase_timeout(_config(tmp_path, heads=2))
    fewer_steps = preparation_phase_timeout(_config(tmp_path, steps=1))
    assert 1800 < fewer_layers == fewer_heads < full
    assert 1800 < fewer_steps < full
    assert preparation_phase_timeout(_config(tmp_path, steps=160)) == 86400
    config = _config(tmp_path)
    path = Path(config["model"]) / "config.json"
    metadata = json.loads(path.read_text())
    text = metadata["text_config"]
    text.pop("full_attention_interval")
    text["layer_types"] = ["full_attention"] * 4 + ["linear_attention"] * 60
    path.write_text(json.dumps(metadata))
    assert preparation_phase_timeout(config) < fewer_layers


@pytest.mark.parametrize("value,expected", [(1, 1), (.05, .05), (1800, 1800), (7200, 1800)])
def test_progress_deadline_preserves_stricter_limit(value, expected):
    assert preparation_progress_timeout({"phase_timeout_seconds": value}) == expected


def test_progress_deadline_preserves_original_capture_default():
    assert preparation_progress_timeout({}) == 1800


@pytest.mark.parametrize("value", [0, -1, True, "1800", float("inf"), float("nan")])
def test_invalid_progress_deadline_fails(value):
    with pytest.raises(ValueError, match="finite and positive"):
        preparation_progress_timeout({"phase_timeout_seconds": value})


def test_unknown_plan_geometry_uses_bound_without_device_or_model_work(tmp_path):
    config = {"phase_timeout_seconds": 1800, "rotation_refinement": "optr",
              "model": str(tmp_path / "absent")}
    assert preparation_phase_timeout(config) == 86400
    config["optr_calibration"] = {"steps": 0}
    with pytest.raises(ValueError, match="steps"):
        preparation_phase_timeout(config)


@pytest.mark.parametrize("name", ["build-ops", "install-plugin", "probe-ops", "service-probe",
                                  "native-synthetic", "official-attention-npu", "q4-hotpath"])
def test_other_phases_keep_exact_existing_limit(name):
    config = {"phase_timeout_seconds": 123.5, "rotation_refinement": "optr"}
    assert phase_timeout(config, name) == 123.5


@pytest.mark.parametrize("name", ["prepare-rotations", "bf16-correctness-gates"])
def test_non_optr_recipes_keep_exact_existing_limit(name):
    assert phase_timeout({"phase_timeout_seconds": 123.5}, name) == 123.5


@pytest.mark.parametrize("name", ["prepare-rotations", "probe-ops", "operator-build"])
def test_observe_phase_selects_only_calibration_budget(tmp_path, monkeypatch, name):
    config = _config(tmp_path)
    seen = []
    def phase(phase_name, command, **kwargs):
        seen.append((phase_name, kwargs["timeout"]))
        return PhaseResult(phase_name, command, 0, 0., False, False, "log", True)
    monkeypatch.setattr(observe_serve, "run_phase", phase)
    status = {"phases": []}
    observe_serve._phase(name, ["injected-host-only"], config=config, env={},
                         log_dir=tmp_path, status=status)
    assert seen == [(name, 86400 if name == "prepare-rotations" else 1800)]


def test_deploy_only_prepare_uses_effective_candidate_and_retains_failure(tmp_path, monkeypatch):
    config = _config(tmp_path)
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    seen = []
    def phase(name, command, **kwargs):
        selected = json.loads(Path(kwargs["env"]["OSCAR_TARGET_CONFIG"]).read_text())
        seen.append((name, kwargs["timeout"], selected.get("rotation_refinement")))
        return PhaseResult(name, command, 124, 1., True, False, "log", True)
    monkeypatch.setattr(deploy, "run_phase", phase)
    assert deploy.main(["--config", str(path), "--variant", "candidate", "--only", "prepare-rotations",
                        "--log-dir", str(tmp_path / "logs")]) == 124
    assert seen == [("prepare-rotations", 86400, "optr")]
    status = json.loads((tmp_path / "logs/status.json").read_text())
    assert status["failed_phase"] == "prepare-rotations"


def test_compact_filter_keeps_exact_calibration_events_only():
    assert _compact_should_mirror('[oscar] OPTR_PROGRESS {"step": 1}', {})
    assert _compact_should_mirror('{"optr_capture": "native_request_saved"}', {})
    for line in ('{"native": "ordinary-info"}', 'INFO native model startup',
                 '[oscar] OPTR_PROGRESS_EXTRA {}', 'prefix {"optr_capture": "other"}'):
        assert not _compact_should_mirror(line, {})


def test_real_compact_child_shows_progress_and_preserves_complete_log(tmp_path, capfd):
    progress = '[oscar] OPTR_PROGRESS {"phase": "step_complete", "step": 1}'
    native = '{"optr_capture": "native_request_saved"}'
    noise = 'INFO native model startup'
    script = "import sys\nfor line in sys.argv[1:]: print(line, flush=True)\n"
    result = run_phase("prepare-rotations", [sys.executable, "-c", script, progress, native, noise],
                       cwd=tmp_path, log_dir=tmp_path, timeout=5,
                       env={"OSCAR_TERMINAL_LOG_MODE": "compact"}, grace=1)
    assert result.returncode == 0 and result.cleanup_complete
    terminal = capfd.readouterr().out
    assert progress in terminal and native in terminal and noise not in terminal
    assert noise in Path(result.log).read_text()
