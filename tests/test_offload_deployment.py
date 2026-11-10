"""Archive #27/#77/#94/#125/#148: explicit deploy configuration and fail-closed gates."""
import json
from pathlib import Path
import subprocess

import pytest

from tools import deploy, install_serve, memcache_config, observe_serve


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
def test_install_dependencies_follow_effective_override(tmp_path, capsys, enabled):
    config = json.loads((deploy.ROOT / "configs/target.json").read_text())
    config["kv_offload"]["enabled"] = not enabled
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    assert install_serve.main(["--config", str(path), "--plan", "--variant", "candidate",
        "--kv-offload" if enabled else "--no-kv-offload", "--log-dir", str(tmp_path / "unused")]) == 0
    plan = json.loads(capsys.readouterr().out)
    install = dict(plan["stages"])["build-dependencies"]
    for dependency in ("memfabric-hybrid", "memcache-hybrid", "pyzmq"):
        assert (dependency in install) is enabled
    assert ("--kv-transfer-config" in plan["vllm_argv"]) is enabled
    assert not (tmp_path / "unused").exists()


def test_disabled_probe_plan_keeps_existing_correctness_stages(tmp_path):
    plan = deploy._bf16_plan(tmp_path / "not-created.json", tmp_path, {"kv_offload": {"enabled": False}})
    assert [name for name, _ in plan] == ["bf16-correctness-gates", "service-probe"]


def test_sourced_sdk_launch_preserves_full_service_argv(tmp_path, monkeypatch, capsys):
    # User's installed SDKs must not be replaced; this selection must not
    # change token budget, prefix cache, RoPE, MTP, graph, placement or port.
    plans = []
    for source in ("pip", "installed"):
        monkeypatch.setenv("OSCAR_OFFLOAD_SDK_SOURCE", source)
        assert install_serve.main(["--plan", "--variant", "candidate", "--kv-offload",
                                   "--log-dir", str(tmp_path / "unused")]) == 0
        plans.append(json.loads(capsys.readouterr().out))
    assert plans[0]["vllm_argv"] == plans[1]["vllm_argv"]
    for index, source in enumerate(("pip", "installed")):
        command = dict(plans[index]["stages"])["build-dependencies"]
        assert "pyzmq" in command
        for sdk in ("memfabric-hybrid", "memcache-hybrid"):
            assert (sdk in command) is (source == "pip")
    assert not (tmp_path / "unused").exists()


def test_offload_sdk_selection_rejects_unknown_source():
    from tools.offload_dependencies import offload_install_dependencies
    with pytest.raises(ValueError, match="OSCAR_OFFLOAD_SDK_SOURCE"):
        offload_install_dependencies({"kv_offload": {"enabled": True}},
                                     {"OSCAR_OFFLOAD_SDK_SOURCE": "typo"})
    assert offload_install_dependencies({"kv_offload": {"enabled": False}},
                                        {"OSCAR_OFFLOAD_SDK_SOURCE": "installed"}) == []


def test_offload_keeps_both_device_masks_equal_for_explicit_placement():
    from tools.target_cli import target_env
    for devices in ([0, 1, 2, 3], [4, 5, 6, 7]):
        env = target_env({"devices": devices, "kv_offload": {"enabled": True}},
                         {"CUDA_VISIBLE_DEVICES": "stale"})
        assert env["CUDA_VISIBLE_DEVICES"] == env["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, devices))


@pytest.mark.parametrize("enabled", [False, True])
def test_probe_entry_installs_sdk_before_device_gates(tmp_path, monkeypatch, enabled):
    calls = []
    class Installed(Exception):
        pass
    def phase(name, command, **kwargs):
        calls.append((name, command))
        if name == "install-plugin":
            raise Installed
    monkeypatch.setattr(observe_serve, "_phase", phase)
    with pytest.raises(Installed):
        observe_serve._preflight(tmp_path / "unused.json", {"kv_offload": {"enabled": enabled}},
                                 {}, tmp_path, {}, "candidate")
    for dependency in ("memfabric-hybrid", "memcache-hybrid", "pyzmq"):
        assert (dependency in calls[0][1]) is enabled
    assert [name for name, _ in calls] == ["install-dependencies", "install-plugin"]
