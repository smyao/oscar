# 档案 #27/#29/#77/#94/#125/#177：显式配置、当前环境、真实退出码；SDK启动失败不冒充复用成功。
"""Generate explicit MemCache deployment files, or run its foreground MetaService."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
from urllib.parse import urlsplit


def _settings(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            raise ValueError("MemCache configuration must use key = value lines")
        key = key.strip()
        if not key or key in values:
            raise ValueError(f"empty or duplicate active MemCache key: {key!r}")
        values[key] = value.strip()
    return values


def _updated_settings(text: str, updates: dict[str, str]) -> str:
    _settings(text)
    remaining = dict(updates)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        key = stripped.partition("=")[0].strip()
        if stripped and not stripped.startswith(("#", ";")) and key in remaining:
            lines.append(f"{key} = {remaining.pop(key)}")
        else:
            lines.append(line)
    lines.extend(f"{key} = {value}" for key, value in remaining.items())
    return "\n".join(lines) + "\n"


def prepare_node93(config_dir: Path, *, plan: bool = False) -> dict:
    """Apply the user's 40GB / device_rdma / 70:60 settings before startup.

    Only confirmed settings are changed. A DirStore path is configuration
    input, not proof that this installed SDK selected or exercised SSD storage.
    Existing cache objects are retained for separately controlled cold/warm runs.
    """
    directory = config_dir.resolve(strict=True)
    local_path, meta_path = directory / "mmc-local.conf", directory / "mmc-meta.conf"
    dir_path = directory / "dir_store.conf"
    local, meta = local_path.read_text(), meta_path.read_text()
    local_values, meta_values = _settings(local), _settings(meta)
    for key, expected in (("ock.mmc.meta_service_url", "tcp://127.0.0.1:5000"),
                          ("ock.mmc.local_service.config_store_url", "tcp://127.0.0.1:6000")):
        if local_values.get(key) != expected:
            raise ValueError(f"node93 LocalService endpoint differs from the confirmed configuration: {key}")
    if (meta_values.get("ock.mmc.meta_service_url") != "tcp://127.0.0.1:5000"
            or meta_values.get("ock.mmc.meta_service.config_store_url") != "tcp://127.0.0.1:6000"):
        raise ValueError("node93 MetaService endpoints differ from the confirmed LocalService configuration")
    if local_values.get("ock.mmc.local_service.storage.enabled") != "true":
        raise ValueError("node93 DRAM+SSD configuration must retain the supplied storage.enabled=true")
    # No version-matched evidence establishes that DistributedObjectStore
    # reads this path or that these local keys enable a DirStore backend.
    # Preserve existing SDK files and do not fabricate a missing config.
    dir_exists = dir_path.is_file()
    contents = {
        local_path: _updated_settings(local, {"ock.mmc.local_service.dram.size": "40GB",
                                             "ock.mmc.local_service.protocol": "device_rdma"}),
        meta_path: _updated_settings(meta, {"ock.mmc.log_level": "error",
                                          "ock.mmc.evict_threshold_high": "70",
                                          "ock.mmc.evict_threshold_low": "60"}),
    }
    if not plan:
        # Validate every input before changing anything. Writes are not a
        # multi-file transaction; I/O failure is surfaced to the caller.
        for path, content in contents.items():
            if path.read_text() != content:
                path.write_text(content, encoding="utf-8")
    return {"mode": "plan" if plan else "prepared", "local_config": str(local_path),
        "meta_config": str(meta_path), "dir_store_config": str(dir_path),
        "dir_store_config_exists": dir_exists, "dir_store_config_created": False,
        "dram_per_local_service": "40GB", "tp4_configured_dram": "160GB",
        "protocol": "device_rdma", "eviction_high": 70, "eviction_low": 60,
        "cache_cleared": False, "service_started": False,
        "ssd_backend_validation": "requires_installed_sdk_and_actual_ssd_io_evidence"}


def endpoint(value: str) -> str:
    """SDK configuration accepts tcp endpoints; reject INI/shell injection."""
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == "tcp" and parsed.hostname and parsed.port
                 and not parsed.username and not parsed.password
                 and parsed.path == "" and not parsed.query and not parsed.fragment
                 and not any(c.isspace() for c in value))
    except ValueError as error:
        raise ValueError("endpoint must be tcp://host:port") from error
    if not valid or not re.fullmatch(r"[A-Za-z0-9.:%_-]+", parsed.hostname):
        raise ValueError("endpoint must be tcp://host:port")
    return value


def generate(directory: Path, *, meta: str, config_store: str, dram_gib: int,
             world_size: int, local_ip: str | None = None, interface: str | None = None,
             force: bool = False) -> dict:
    meta, config_store = endpoint(meta), endpoint(config_store)
    if type(dram_gib) is not int or dram_gib <= 0:
        raise ValueError("dram_gib must be a positive per-LocalService capacity")
    if type(world_size) is not int or world_size < 4:
        raise ValueError("world_size must allow at least the four TP LocalServices")
    if (local_ip is None) != (interface is None):
        raise ValueError("local_ip and interface must be supplied together")
    if local_ip is not None:
        ipaddress.ip_address(local_ip)
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
            raise ValueError("invalid communication interface name")
    directory = directory.resolve()
    meta_path, local_path = directory / "mmc-meta.conf", directory / "mmc-local.conf"
    environment = directory / "environment.sh"
    files = (meta_path, local_path, environment)
    if not force and any(path.exists() for path in files):
        raise FileExistsError("MemCache files already exist; use --force to replace the supplied configuration")
    common = f"ock.mmc.meta_service_url = {meta}\nock.mmc.log_level = error\n"
    contents = {
        meta_path: common + f"ock.mmc.meta_service.config_store_url = {config_store}\n",
        local_path: common + f"ock.mmc.local_service.config_store_url = {config_store}\n"
            + f"ock.mmc.local_service.world_size = {world_size}\n"
            + "ock.mmc.local_service.protocol = device_rdma\n"
            + f"ock.mmc.local_service.dram.size = {dram_gib}GB\n",
        environment: "# Generated for Ascend 910B/A2; source before launching the service.\n"
            + f"export MMC_META_CONFIG_PATH={shlex.quote(str(meta_path))}\n"
            + f"export MMC_LOCAL_CONFIG_PATH={shlex.quote(str(local_path))}\n"
            + "export PYTHONHASHSEED=0\n",
    }
    if local_ip is not None:
        contents[environment] += f"export HCCL_IF_IP={shlex.quote(local_ip)}\n"
        for name in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME"):
            contents[environment] += f"export {name}={shlex.quote(interface)}\n"
    directory.mkdir(parents=True, exist_ok=True)
    for path, content in contents.items():
        path.write_text(content, encoding="utf-8")
    return {"meta_config": str(meta_path), "local_config": str(local_path),
            "environment": str(environment), "protocol": "device_rdma",
            "dram_gib_per_local_service": dram_gib, "maximum_local_services": world_size,
            "tp4_dram_gib": 4 * dram_gib, "service_started": False}


def serve_meta(config: Path) -> int:
    """Call the real foreground SDK entry; never turn a failure into reuse."""
    path = config.resolve(strict=True)
    os.environ["MMC_META_CONFIG_PATH"] = str(path)
    from memcache_hybrid import MetaService
    print(f"[oscar] MemCache MetaService foreground config={path}", flush=True)
    result = MetaService.main()
    if result is not None and result != 0:
        raise RuntimeError(f"MemCache MetaService failed: {result!r}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("generate", help="write deployment configuration without importing an SDK or using an NPU")
    config.add_argument("--output-dir", type=Path, required=True)
    config.add_argument("--meta-url", required=True)
    config.add_argument("--config-store-url", required=True)
    config.add_argument("--dram-gib", type=int, required=True, help="DRAM per LocalService / TP rank, not the TP total")
    config.add_argument("--world-size", type=int, default=256, help="maximum LocalServices supported by the pool")
    config.add_argument("--local-ip")
    config.add_argument("--interface")
    config.add_argument("--force", action="store_true")
    serve = commands.add_parser("serve-meta", help="run the SDK MetaService in the foreground; no daemon or SSH")
    serve.add_argument("--config", type=Path, required=True)
    node93 = commands.add_parser("prepare-node93", help="apply the user-supplied node93 configuration; no SDK initialization or cache deletion")
    node93.add_argument("--config-dir", type=Path, default=Path("/usr/local/memcache_hybrid/latest/config"))
    node93.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate":
        print(json.dumps(generate(args.output_dir, meta=args.meta_url,
            config_store=args.config_store_url, dram_gib=args.dram_gib,
            world_size=args.world_size, local_ip=args.local_ip,
            interface=args.interface, force=args.force), indent=2))
    elif args.command == "prepare-node93":
        print(json.dumps(prepare_node93(args.config_dir, plan=args.plan), indent=2))
    else:
        return serve_meta(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
