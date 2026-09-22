# 档案 #75/#94/#95/#101/#116/#117/#120：注入真实失败/超时，验证退出码、日志与清理。
import json
from pathlib import Path
import sys
import time
import pytest
from tools.phase import run_phase
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
