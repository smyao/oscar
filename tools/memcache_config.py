# 档案 #27/#29/#77/#94/#125：显式配置、当前环境、真实退出码；不继承旧设备、不隐式启动远程服务。
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
    args = parser.parse_args(argv)
    if args.command == "generate":
        print(json.dumps(generate(args.output_dir, meta=args.meta_url,
            config_store=args.config_store_url, dram_gib=args.dram_gib,
            world_size=args.world_size, local_ip=args.local_ip,
            interface=args.interface, force=args.force), indent=2))
    else:
        path = args.config.resolve(strict=True)
        os.environ["MMC_META_CONFIG_PATH"] = str(path)
        from memcache_hybrid import MetaService
        print(f"[oscar] MemCache MetaService foreground config={path}", flush=True)
        result = MetaService.main()
        if result is not None and result != 0:
            raise RuntimeError(f"MemCache MetaService failed: {result!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
