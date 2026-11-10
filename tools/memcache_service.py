# 档案 #94/#125/#175/#177：先识别已有控制服务，再准备主机/配置；失败不得冒充复用成功。
"""Idempotent node93 MetaService startup using a bounded SDK metadata query."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from urllib.parse import urlsplit
import uuid

from tools import node93_host
from tools.memcache_config import _settings, endpoint, prepare_node93, serve_meta
from tools.memcache_lock import sdk_lock_held

METADATA_TIMEOUT_SECONDS = 15


def endpoint_is_listening(url: str) -> bool:
    """A refusal means absent; a timeout/permission error is indeterminate."""
    address = urlsplit(endpoint(url))
    try:
        with socket.create_connection((address.hostname, address.port), timeout=1):
            return True
    except OSError as exc:
        if exc.errno == errno.ECONNREFUSED:
            return False
        raise RuntimeError(f"Cannot determine MemCache endpoint state: {url}: {exc}") from exc


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
