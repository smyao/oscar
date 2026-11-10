# 档案 #94/#125/#175/#177/#178：先只读观察监听，避免裸TCP探测污染SDK协议；失败不得冒充复用成功。
"""Idempotent node93 MetaService startup using a bounded SDK metadata query."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from urllib.parse import urlsplit
import uuid

from tools import node93_host
from tools.memcache_config import _settings, endpoint, prepare_node93, serve_meta
from tools.memcache_lock import sdk_lock_held

METADATA_TIMEOUT_SECONDS = 15
TCP4_TABLE = Path("/proc/net/tcp")
TCP6_TABLE = Path("/proc/net/tcp6")


def _read_tcp_listeners(path: Path, address_digits: int, *,
                        optional_missing: bool = False) -> list[tuple[int, int]]:
    """Read this process's network namespace without opening a connection."""
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except FileNotFoundError as exc:
        if optional_missing:
            return []
        raise RuntimeError(f"Cannot read MemCache listener table: {path}: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Cannot read MemCache listener table: {path}: {exc}") from exc
    header = lines[0].split() if lines else []
    if (len(header) < 4 or header[:2] != ["sl", "local_address"]
            or header[2] not in {"rem_address", "remote_address"} or header[3] != "st"):
        raise RuntimeError(f"Invalid TCP table header: {path}")
    address_pattern = rf"[0-9a-fA-F]{{{address_digits}}}:[0-9a-fA-F]{{4}}"
    listeners = []
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = line.split()
        if (len(fields) < 4 or not re.fullmatch(r"[0-9]+:", fields[0])
                or not re.fullmatch(address_pattern, fields[1])
                or not re.fullmatch(address_pattern, fields[2])
                or not re.fullmatch(r"[0-9a-fA-F]{2}", fields[3])):
            raise RuntimeError(f"Invalid TCP table row: {path}:{number}")
        if int(fields[3], 16) == 0x0A:  # TCP_LISTEN
            local, port = fields[1].split(":")
            listeners.append((int(local, 16), int(port, 16)))
    return listeners


def endpoint_is_listening(url: str) -> bool:
    """Observe node93's fixed IPv4 loopback endpoint; never send SDK traffic."""
    address = urlsplit(endpoint(url))
    if address.hostname != "127.0.0.1":
        raise RuntimeError("Read-only MemCache listener detection requires node93's IPv4 loopback endpoint")
    ipv4 = _read_tcp_listeners(TCP4_TABLE, 8)
    # /proc renders each IPv4 address as a native-order u32 hexadecimal value.
    loopback = int.from_bytes(bytes((127, 0, 0, 1)), sys.byteorder)
    if any(port == address.port and host in {0, loopback} for host, port in ipv4):
        return True
    # /proc does not expose IPV6_V6ONLY. A same-port IPv6 listener must not
    # become a false "absent" result that permits a second MetaService start.
    ipv6 = _read_tcp_listeners(TCP6_TABLE, 32, optional_missing=True)
    if any(port == address.port for _, port in ipv6):
        raise RuntimeError(f"Indeterminate IPv6/dual-stack listener for MemCache endpoint: {url}")
    return False


def metadata_query(local_config: Path) -> None:
    """Run only in an isolated child; create the native scheduler-style client.

    The native MemcacheBackend.create_scheduler_client uses init_bm=False.
    No LocalService, device buffer, put/get/remove, or worker initialization is
    requested here. A metadata response is not an SSD/NPU readiness certificate.
    """
    os.environ["MMC_LOCAL_CONFIG_PATH"] = str(local_config.resolve(strict=True))
    from memcache_hybrid import DistributedObjectStore

    store = DistributedObjectStore()
    try:
        result = store.init(0, init_bm=False)
        if type(result) is not int or result != 0:
            raise RuntimeError(f"MemCache metadata client initialization failed: {result!r}")
        result = store.batch_is_exist(["oscar-meta-check-" + uuid.uuid4().hex], 0)
        if (type(result) is not list or len(result) != 1
                or type(result[0]) is not int or result[0] not in (0, 1)):
            raise RuntimeError(f"MemCache metadata query returned an invalid status: {result!r}")
    finally:
        store.close()


def verify_metadata(local_config: Path) -> None:
    print("[oscar] checking existing MemCache with a metadata-only SDK client", flush=True)
    try:
        subprocess.run([sys.executable, "-m", "tools.memcache_service", "check-client",
                        "--local-config", str(local_config)],
                       check=True, timeout=METADATA_TIMEOUT_SECONDS)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Existing endpoints did not pass the MemCache metadata check; "
            "host/configuration unchanged, no second MetaService started. "
            f"{exc}"
        ) from exc


@contextmanager
def launcher_lock(meta_url: str, store_url: str, *, lock_root: Path = Path("/tmp")):
    """Serialize this launcher across check/prepare/foreground start.

    This is OSCAR's own lock, not the SDK lock. Keep the file after release to
    avoid unlink/open races. A second invocation may still verify and reuse the
    running service; it must not wait for the foreground service to exit.
    """
    identity = hashlib.sha256(f"{meta_url}\n{store_url}".encode()).hexdigest()[:24]
    path = lock_root / f"oscar-memcache-start-{os.getuid()}-{identity}.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(f"Invalid OSCAR startup lock: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            acquired = False
        else:
            acquired = True
        yield acquired
    finally:
        os.close(descriptor)


def start_node93(config_dir: Path, *, prepare_host: bool = False) -> int:
    # Read-only validation comes first. Do not silently rewrite a running
    # service's files, including when the user repeats --prepare-host.
    plan = prepare_node93(config_dir, plan=True)
    local_path, meta_path = Path(plan["local_config"]), Path(plan["meta_config"])
    values = _settings(meta_path.read_text())
    meta_url = values["ock.mmc.meta_service_url"]
    store_url = values["ock.mmc.meta_service.config_store_url"]
    os.environ["MMC_LOCAL_CONFIG_PATH"] = str(local_path)
    os.environ["MMC_META_CONFIG_PATH"] = str(meta_path)
    with launcher_lock(meta_url, store_url) as acquired:
        listening = (endpoint_is_listening(meta_url), endpoint_is_listening(store_url))
        if all(listening):
            verify_metadata(local_path)
            print("[oscar] reusing existing MemCache MetaService: metadata query passed; "
                  "host preparation, configuration writes and duplicate startup skipped", flush=True)
            print("[oscar] existing service runtime settings and SSD/NPU I/O are not verified here", flush=True)
            print("[oscar] existing MetaService remains owned by its original launcher; "
                  "keep that process running while OSCAR uses offload", flush=True)
            print("[oscar] next: bash scripts/start_node93_oscar.sh", flush=True)
            return 0
        if any(listening):
            raise RuntimeError(
                f"MemCache endpoints are only partially listening: {meta_url}={listening[0]}, "
                f"{store_url}={listening[1]}; host/configuration unchanged, no second service started"
            )
        if not acquired:
            raise RuntimeError("Another OSCAR MetaService launch is in progress; "
                               "host/configuration unchanged, no second service started")
        if sdk_lock_held():
            raise RuntimeError("MemCache SDK process lock is held but the configured endpoints are absent; "
                               "host/configuration unchanged, no second service started")
        if prepare_host:
            status = node93_host.main([])
            if status:
                return status
        print(json.dumps(prepare_node93(config_dir), indent=2), flush=True)
        return serve_meta(meta_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("node93", help="reuse a verified control service or start it in the foreground")
    start.add_argument("--config-dir", type=Path, required=True)
    start.add_argument("--prepare-host", action="store_true")
    client = commands.add_parser("check-client", help="internal isolated metadata check; no storage initialization")
    client.add_argument("--local-config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check-client":
        metadata_query(args.local_config)
        return 0
    return start_node93(args.config_dir, prepare_host=args.prepare_host)


if __name__ == "__main__":
    raise SystemExit(main())
