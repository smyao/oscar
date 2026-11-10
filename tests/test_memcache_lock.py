"""Archive #175/#176/#177: real POSIX contention, never SDK/NPU acceptance."""
from contextlib import contextmanager
import errno
import fcntl
import os
import select
import subprocess
import sys

import pytest

from tools import memcache_lock


@contextmanager
def held_by_child(path, *, shared=False):
    # lockf uses POSIX record locking; flock(2) would test a different API.
    program = """import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.lockf(fd, (fcntl.LOCK_SH if sys.argv[2] == 'shared' else fcntl.LOCK_EX) | fcntl.LOCK_NB)
print('locked', flush=True)
sys.stdin.readline()
os.close(fd)
"""
    process = subprocess.Popen([sys.executable, "-c", program, str(path),
                                "shared" if shared else "exclusive"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 5)[0], "lock owner did not start"
        assert process.stdout.readline().strip() == "locked"
        yield process
    finally:
        try:
            _, error = process.communicate("\n", timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
            raise
        assert process.returncode == 0, error


def test_missing_lock_is_not_created(tmp_path):
    path = tmp_path / "absent.lock"
    assert memcache_lock.sdk_lock_held(path) is False
    assert not path.exists()


@pytest.mark.parametrize("content", [b"", b"SDK-owned bytes remain unchanged\n"])
def test_stale_lock_file_is_unlocked_and_unchanged(tmp_path, content):
    path = tmp_path / "meta.lock"
    path.write_bytes(content)
    before = path.stat()
    assert memcache_lock.sdk_lock_held(path) is False
    # A fresh process can still acquire the lock after the query.
    with held_by_child(path):
        assert memcache_lock.sdk_lock_held(path) is True
    assert memcache_lock.sdk_lock_held(path) is False
    after = path.stat()
    assert path.read_bytes() == content
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns)


@pytest.mark.parametrize("shared", [False, True])
def test_real_process_lock_survives_repeated_queries(tmp_path, shared):
    path = tmp_path / "meta.lock"
    path.touch()
    with held_by_child(path, shared=shared) as child:
        for _ in range(3):
            assert memcache_lock.sdk_lock_held(path) is True
            assert child.poll() is None
    assert memcache_lock.sdk_lock_held(path) is False


def test_query_uses_read_only_fd_and_only_f_getlk(tmp_path, monkeypatch):
    path = tmp_path / "meta.lock"
    path.touch()
    original = fcntl.fcntl
    calls = []
    def observe(fd, operation, payload):
        assert operation == fcntl.F_GETLK
        assert original(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        query = memcache_lock._flock_type().from_buffer_copy(payload)
        assert (query.l_type, query.l_whence, query.l_start, query.l_len) == (fcntl.F_WRLCK, os.SEEK_SET, 0, 0)
        calls.append(operation)
        return original(fd, operation, payload)
    monkeypatch.setattr(fcntl, "fcntl", observe)
    assert memcache_lock.sdk_lock_held(path) is False
    assert calls == [fcntl.F_GETLK]


def test_permission_failure_is_not_unlocked(tmp_path, monkeypatch):
    failure = PermissionError(errno.EACCES, "denied")
    def denied(*args):
        raise failure
    monkeypatch.setattr(os, "open", denied)
    with pytest.raises(PermissionError) as caught:
        memcache_lock.sdk_lock_held(tmp_path / "meta.lock")
    assert caught.value is failure


@pytest.mark.parametrize("kind", ["directory", "fifo", "symlink"])
def test_nonregular_or_symlink_lock_is_rejected(tmp_path, kind):
    path = tmp_path / "meta.lock"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        destination = tmp_path / "real.lock"
        destination.touch()
        path.symlink_to(destination)
    with pytest.raises((ValueError, OSError)):
        memcache_lock.sdk_lock_held(path)


@pytest.mark.parametrize("response", [b"bad", None, "invalid_type"])
def test_invalid_kernel_result_fails_and_closes_fd(tmp_path, monkeypatch, response):
    path = tmp_path / "meta.lock"
    path.touch()
    captured = []
    def invalid(fd, operation, payload):
        captured.append(fd)
        if response == "invalid_type":
            value = memcache_lock._flock_type()(l_type=127)
            return bytes(value)
        return response
    monkeypatch.setattr(fcntl, "fcntl", invalid)
    with pytest.raises(RuntimeError, match="invalid SDK process-lock"):
        memcache_lock.sdk_lock_held(path)
    with pytest.raises(OSError) as caught:
        os.fstat(captured[0])
    assert caught.value.errno == errno.EBADF


def test_kernel_query_error_propagates_and_closes_fd(tmp_path, monkeypatch):
    path = tmp_path / "meta.lock"
    path.touch()
    failure = OSError(errno.ENOTSUP, "unsupported lock query")
    captured = []
    def failed(fd, operation, payload):
        captured.append(fd)
        raise failure
    monkeypatch.setattr(fcntl, "fcntl", failed)
    with pytest.raises(OSError) as caught:
        memcache_lock.sdk_lock_held(path)
    assert caught.value is failure
    with pytest.raises(OSError) as caught:
        os.fstat(captured[0])
    assert caught.value.errno == errno.EBADF


@pytest.mark.parametrize("system,machine", [("linux", "armv7l"), ("darwin", "i386"), ("freebsd", "amd64")])
def test_unknown_abi_fails_before_file_access(tmp_path, monkeypatch, system, machine):
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(memcache_lock.platform, "machine", lambda: machine)
    with pytest.raises(RuntimeError, match="unsupported SDK lock-query ABI"):
        memcache_lock.sdk_lock_held(tmp_path / "absent.lock")
