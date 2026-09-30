# Archive #126/#148/#154 and startup D.4: local Lima CAModel experiment only.
# Existing compiled binaries and deterministic synthetic q4 fixture; not target NPU acceptance.
import ctypes
from ctypes import byref, c_bool, c_int, c_int64, c_size_t, c_uint32, c_void_p
import hashlib
import os
from pathlib import Path
import struct
import sys
import threading
import time

root = Path(sys.argv[1])
log_root = Path(sys.argv[2])
stop = threading.Event()


def quiet():
    sink = os.open("/dev/null", os.O_WRONLY)
    try:
        while not stop.is_set():
            for name in os.listdir("/proc/self/fd"):
                if not name.isdigit():
                    continue
                fd = int(name)
                if fd == sink:
                    continue
                try:
                    target = os.readlink(f"/proc/self/fd/{fd}")
                    if target.startswith(str(log_root) + "/") and target.endswith(".dump"):
                        os.dup2(sink, fd)
                except OSError:
                    pass
            time.sleep(0.001)
    finally:
        os.close(sink)


thread = threading.Thread(target=quiet, daemon=True)
thread.start()
acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)


def api(name, args):
    fn = getattr(acl, name)
    fn.argtypes = args
    fn.restype = c_int
    return fn


init = api("aclInit", [ctypes.c_char_p]); dev = api("aclrtSetDevice", [c_int])
reset = api("aclrtResetDevice", [c_int]); finalize = api("aclFinalize", [])
alloc = api("aclrtMalloc", [ctypes.POINTER(c_void_p), c_size_t, c_int])
free = api("aclrtFree", [c_void_p]); copy = api("aclrtMemcpy", [c_void_p, c_size_t, c_void_p, c_size_t, c_int])
create = api("aclrtCreateStream", [ctypes.POINTER(c_void_p)])
destroy = api("aclrtDestroyStream", [c_void_p]); sync = api("aclrtSynchronizeStream", [c_void_p])
assert init(None) == dev(0) == 0
lib = ctypes.CDLL(
    "/home/sunao2000.linux/batched4-experiment-20260929/build-binder-final/lib/liboscar_ascend_kernels_b634658ee1e3def6.so",
    mode=ctypes.RTLD_GLOBAL,
)
prep = lib.aclrtlaunch_oscar_prepare_attention_tasks_kernel
prep.argtypes = [c_uint32, c_void_p] + [c_void_p] * 5 + [c_int64] * 7 + [c_bool, c_void_p, c_int64, c_bool]
prep.restype = c_uint32
stream = c_void_p()
assert create(byref(stream)) == 0


def device(size, payload=None):
    ptr = c_void_p()
    assert alloc(byref(ptr), size, 0) == 0
    if payload is not None:
        host = ctypes.create_string_buffer(payload, size)
        assert copy(ptr, size, host, size, 1) == 0
    return ptr


for case_dir in sorted(root.glob("synthetic_q4_ctx*")):
    vals = (case_dir / "attrs.txt").read_text().split()
    n, h, hk, d, requests, columns, tasks_count, block_tokens, blocks, prefix, page_stride, \
        window_stride, tag_stride, sink, recent, spec, splits = map(int, vals[:-2])
    assert (n, h, hk, d, requests, tasks_count) == (4, 6, 1, 256, 1, 12)
    payloads = {name: (case_dir / f"{name}.bin").read_bytes() for name in ("starts", "lens", "slots")}
    ptrs = {name: device(len(data), data) for name, data in payloads.items()}
    ptrs["tasks"] = device(tasks_count * 16 * 8)
    ptrs["positions"] = device(n * 8)
    assert prep(1, stream, ptrs["starts"], ptrs["lens"], ptrs["slots"],
                ptrs["tasks"], ptrs["positions"], requests, n, h, hk,
                sink, recent, splits, True, None, 0, False) == 0
    assert sync(stream) == 0
    out = {}
    for name, size in (("tasks", tasks_count * 16 * 8), ("positions", n * 8)):
        buf = ctypes.create_string_buffer(size)
        assert copy(buf, size, ptrs[name], size, 2) == 0
        out[name] = buf.raw
        (case_dir / f"{name}.bin").write_bytes(buf.raw)
    context = int(case_dir.name.rsplit("ctx", 1)[1])
    assert struct.unpack("<4q", out["positions"]) == tuple(range(context, context + 4))
    print("PREPARED", case_dir.name, "columns", columns,
          "tasks_sha256", hashlib.sha256(out["tasks"]).hexdigest(), flush=True)
    for ptr in ptrs.values():
        assert free(ptr) == 0
assert destroy(stream) == reset(0) == finalize() == 0
stop.set(); thread.join(timeout=2)
