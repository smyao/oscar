"""Archive #27/#74/#77/#94/#125/#148/#175/#176: explicit deployment and SDK sources."""
import json
from pathlib import Path
import subprocess

import pytest

from tools import deploy, install_serve, memcache_config, observe_serve
from tools.offload_dependencies import offload_install_dependencies
from tools.target_cli import target_env


def test_memcache_generation_uses_supplied_addresses_and_per_rank_capacity(tmp_path):
    result = memcache_config.generate(tmp_path / "config with spaces", meta="tcp://meta.example:5000",
        config_store="tcp://config.example:6000", dram_gib=8, world_size=16,
        local_ip="192.0.2.5", interface="eth0")
    assert result["tp4_dram_gib"] == 32 and result["service_started"] is False
    local = Path(result["local_config"]).read_text()
    assert "ock.mmc.local_service.dram.size = 8GB" in local
    assert "ock.mmc.local_service.protocol = device_rdma" in local
    assert "tcp://config.example:6000" in local
    assert "tcp://meta.example:5000" in Path(result["meta_config"]).read_text()
    sourced = subprocess.run(["bash", "-c", 'source "$1"; printf "%s" "$MMC_LOCAL_CONFIG_PATH"',
        "test", result["environment"]], check=True, text=True, capture_output=True)
    assert sourced.stdout == result["local_config"]
    with pytest.raises(FileExistsError):
        memcache_config.generate(Path(result["environment"]).parent,
            meta="tcp://meta.example:5000", config_store="tcp://config.example:6000", dram_gib=8, world_size=16)


@pytest.mark.parametrize("address", ["http://example:1", "tcp://example", "tcp://example:65536",
    "tcp://user:password@example:10", "tcp://example:10/path", "tcp://example:10\nother = bad"])
def test_invalid_memcache_endpoints_do_not_create_configuration(tmp_path, address):
    with pytest.raises(ValueError):
        memcache_config.generate(tmp_path / "unused", meta=address,
                                config_store="tcp://example:6000", dram_gib=8, world_size=4)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("source", [None, "pip", "installed"])
def test_install_dependencies_follow_effective_override(tmp_path, capsys, enabled, source):
    config = json.loads((deploy.ROOT / "configs/target.json").read_text())
    config["kv_offload"]["enabled"] = not enabled
    if source is not None:
        config["offload_sdk_source"] = source
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    assert install_serve.main(["--config", str(path), "--plan", "--variant", "candidate",
        "--kv-offload" if enabled else "--no-kv-offload", "--log-dir", str(tmp_path / "unused")]) == 0
    plan = json.loads(capsys.readouterr().out)
    install = dict(plan["stages"])["build-dependencies"]
    for dependency in ("memfabric-hybrid", "memcache-hybrid"):
        assert (dependency in install) is (enabled and source != "installed")
    assert ("pyzmq" in install) is enabled
    assert ("--kv-transfer-config" in plan["vllm_argv"]) is enabled
    assert not (tmp_path / "unused").exists()


def test_disabled_probe_plan_keeps_existing_correctness_stages(tmp_path):
    plan = deploy._bf16_plan(tmp_path / "not-created.json", tmp_path, {"kv_offload": {"enabled": False}})
    assert [name for name, _ in plan] == ["bf16-correctness-gates", "service-probe"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("source", [None, "pip", "installed"])
def test_probe_entry_uses_selected_sdk_source_before_device_gates(tmp_path, monkeypatch, enabled, source):
    calls = []
    class Installed(Exception):
        pass
    def phase(name, command, **kwargs):
        calls.append((name, command))
        if name == "install-plugin":
            raise Installed
    monkeypatch.setattr(observe_serve, "_phase", phase)
    config = {"kv_offload": {"enabled": enabled}}
    if source is not None:
        config["offload_sdk_source"] = source
    with pytest.raises(Installed):
        observe_serve._preflight(tmp_path / "unused.json", config, {}, tmp_path, {}, "candidate")
    for dependency in ("memfabric-hybrid", "memcache-hybrid"):
        assert (dependency in calls[0][1]) is (enabled and source != "installed")
    assert ("pyzmq" in calls[0][1]) is enabled
    assert [name for name, _ in calls] == ["install-dependencies", "install-plugin"]


@pytest.mark.parametrize("source", [None, "auto", "", True, [], {}])
def test_invalid_sdk_source_is_not_silently_replaced(source):
    with pytest.raises(ValueError, match="offload_sdk_source"):
        offload_install_dependencies({"offload_sdk_source": source, "kv_offload": {"enabled": True}})


def test_node93_default_plan_preserves_installed_sdk_and_transfer_contract(tmp_path, capsys):
    path = deploy.ROOT / "configs/benchmark_node93.json"
    config = json.loads(path.read_text())
    assert config["offload_sdk_source"] == "installed"
    assert install_serve.main(["--config", str(path), "--plan", "--log-dir", str(tmp_path / "unused")]) == 0
    plan = json.loads(capsys.readouterr().out)
    install = dict(plan["stages"])["build-dependencies"]
    assert "pyzmq" in install
    assert "memfabric-hybrid" not in install and "memcache-hybrid" not in install
    assert plan["target_devices"] == [0, 1, 2, 3] and plan["port"] == 7878
    generic = dict(config, offload_sdk_source="pip")
    generic_path = tmp_path / "pip-target.json"
    generic_path.write_text(json.dumps(generic))
    assert install_serve.main(["--config", str(generic_path), "--plan", "--log-dir", str(tmp_path / "unused")]) == 0
    pip_plan = json.loads(capsys.readouterr().out)
    assert plan["vllm_argv"] == pip_plan["vllm_argv"]
    assert not (tmp_path / "unused").exists()


def test_node93_service_env_preserves_sourced_sdk_bindings():
    config = json.loads((deploy.ROOT / "configs/benchmark_node93.json").read_text())
    sourced = {
        "PATH": "/sdk/bin:/usr/bin", "LD_LIBRARY_PATH": "/sdk/lib",
        "PYTHONPATH": "/sdk/python", "PYTHONHASHSEED": "0",
        "MMC_LOCAL_SERVICE_BACKEND_ID": "1", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
        "MMC_META_CONFIG_PATH": "/sdk/mmc-meta.conf",
        "MMC_LOCAL_CONFIG_PATH": "/sdk/mmc-local.conf",
        "MMC_DIR_STORE_CONFIG_PATH": "/sdk/dir_store.conf",
    }
    env = target_env(config, base=sourced)
    assert {name: env[name] for name in sourced} == sourced
