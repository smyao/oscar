"""Read-only SDK process-lock observation (archive #175/#176/#177).

Ascend's public MmcMetaServiceProcess::CheckIsRunning uses the path below and
fcntl(F_SETLK), with F_WRLCK / SEEK_SET / start=0 / len=0:
https://github.com/Ascend/memcache/blob/master/src/memcache/csrc/meta_service/mmc_meta_service_process.cpp
This public-source contract is not a binary identity claim about node93's
private SDK build. A held lock establishes contention, not service health.
Never create, remove, acquire, or release the SDK's lock here. The caller must
separately verify metadata service responses before reusing an existing service.
"""
from __future__ import annotations

import ctypes
import fcntl
import os
from pathlib import Path
import platform
import stat
import sys


SDK_LOCK_PATH = Path("/tmp/mmc_meta_service.lock")


class _LinuxFlock(ctypes.Structure):
    # Linux 64-bit aarch64/x86_64 off_t and pid_t ABI, including native padding.
    _fields_ = [("l_type", ctypes.c_short), ("l_whence", ctypes.c_short),
                ("l_start", ctypes.c_int64), ("l_len", ctypes.c_int64),
                ("l_pid", ctypes.c_int32)]


class _DarwinFlock(ctypes.Structure):
    # Darwin SDK usr/include/sys/fcntl.h: struct flock has a different order.
    _fields_ = [("l_start", ctypes.c_int64), ("l_len", ctypes.c_int64),
                ("l_pid", ctypes.c_int32), ("l_type", ctypes.c_short),
                ("l_whence", ctypes.c_short)]


def _flock_type():
    machine = platform.machine().lower()
    if sys.platform == "linux" and machine in {"aarch64", "x86_64"}:
        layout, size = _LinuxFlock, 32
    elif sys.platform == "darwin" and machine in {"arm64", "x86_64"}:
        layout, size = _DarwinFlock, 24
    else:
        raise RuntimeError(f"unsupported SDK lock-query ABI: {sys.platform}/{machine}")
    if (ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8
            or ctypes.sizeof(ctypes.c_short) != 2 or ctypes.sizeof(layout) != size):
        raise RuntimeError("unsupported native struct flock layout")
    return layout


def sdk_lock_held(path: Path = SDK_LOCK_PATH) -> bool:
    """Observe another process's conflicting POSIX lock without changing it.

    This is a momentary observation, not a startup reservation. A stale regular
    file is allowed; permission errors, symlinks, nonregular files, unsupported
    ABIs and kernel query errors propagate instead of being treated as unlocked.
    """
    layout = _flock_type()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"SDK process lock is not a regular file: {path}")
        query = layout(l_type=fcntl.F_WRLCK, l_whence=os.SEEK_SET, l_start=0, l_len=0)
        result = fcntl.fcntl(fd, fcntl.F_GETLK, bytes(query))
        if not isinstance(result, bytes) or len(result) != ctypes.sizeof(layout):
            raise RuntimeError("invalid SDK process-lock query response")
        observed = layout.from_buffer_copy(result)
        if observed.l_type == fcntl.F_UNLCK:
            return False
        if observed.l_type in {fcntl.F_RDLCK, fcntl.F_WRLCK}:
            return True
        raise RuntimeError(f"invalid SDK process-lock type: {observed.l_type}")
    finally:
        os.close(fd)
