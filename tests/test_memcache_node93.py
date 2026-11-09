"""Archive #27/#29/#94/#125: exact supplied configuration, visible failure, no SDK/device initialization."""
import pytest

from tools.memcache_config import prepare_node93, _settings


def configs(tmp_path):
    local = tmp_path / "mmc-local.conf"
    meta = tmp_path / "mmc-meta.conf"
    local.write_text("""# ock.mmc.local_service.dram.size example remains a comment
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000
ock.mmc.local_service.dram.size = 32GB
ock.mmc.local_service.protocol = device_rdma
ock.mmc.local_service.storage.enabled = true
dir_store.base_path = /home/kvcache
dir_store.io_workers = 32
""")
    meta.write_text("""ock.mmc.meta_service_url = tcp://127.0.0.1:5000
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:6000
ock.mmc.log_level = info
ock.mmc.evict_threshold_high = 90
ock.mmc.evict_threshold_low = 80
ock.mmc.storage.prefetch.enabled = true
""")
    return local, meta


def test_user_values_are_applied_without_changing_comments_unknown_keys_or_cache(tmp_path):
    local, meta = configs(tmp_path)
    report = prepare_node93(tmp_path)
    assert report["cache_cleared"] is report["service_started"] is False
    assert report["ssd_backend_validation"].startswith("requires_")
    assert local.read_text().startswith("# ock.mmc.local_service.dram.size example remains a comment\n")
    assert _settings(local.read_text())["ock.mmc.local_service.dram.size"] == "40GB"
    actual = _settings(meta.read_text())
    assert actual["ock.mmc.log_level"] == "error"
    assert actual["ock.mmc.evict_threshold_high"] == "70"
    assert actual["ock.mmc.evict_threshold_low"] == "60"
    assert actual["ock.mmc.storage.prefetch.enabled"] == "true"
    assert report["dir_store_config_exists"] is report["dir_store_config_created"] is False
    assert not (tmp_path / "dir_store.conf").exists()
    assert _settings(local.read_text())["dir_store.base_path"] == "/home/kvcache"
    assert not (tmp_path / "kvcache").exists()


def test_plan_writes_nothing_and_rerun_preserves_existing_dir_store_bytes(tmp_path):
    local, meta = configs(tmp_path)
    before = local.read_bytes(), meta.read_bytes()
    result = prepare_node93(tmp_path, plan=True)
    assert not result["dir_store_config_exists"] and not result["dir_store_config_created"]
    assert (local.read_bytes(), meta.read_bytes()) == before
    assert not (tmp_path / "dir_store.conf").exists()
    # An installed version may use another schema. Preserve it byte-for-byte.
    directory = tmp_path / "dir_store.conf"
    directory.write_bytes(b"[SDK-owned section]\nactual_backend_option: preserve\n")
    before = directory.read_bytes()
    assert prepare_node93(tmp_path)["dir_store_config_created"] is False
    assert directory.read_bytes() == before
    assert prepare_node93(tmp_path)["dir_store_config_created"] is False
    assert directory.read_bytes() == before


@pytest.mark.parametrize("problem", ["endpoints", "duplicate", "malformed", "storage_disabled"])
def test_invalid_input_does_not_partially_rewrite_sdk_config(tmp_path, problem):
    local, meta = configs(tmp_path)
    if problem == "endpoints":
        meta.write_text(meta.read_text().replace(":5000", ":5001"))
    elif problem == "duplicate":
        meta.write_text(meta.read_text() + "ock.mmc.log_level = debug\n")
    elif problem == "malformed":
        local.write_text(local.read_text() + "malformed configuration line\n")
    else:
        local.write_text(local.read_text().replace("storage.enabled = true", "storage.enabled = false"))
    before = local.read_bytes(), meta.read_bytes()
    with pytest.raises(ValueError):
        prepare_node93(tmp_path)
    assert (local.read_bytes(), meta.read_bytes()) == before
    assert not (tmp_path / "dir_store.conf").exists()
