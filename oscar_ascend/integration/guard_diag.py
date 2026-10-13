# 档案 #142/#157：生产 Trap 不带故障族信息，两次真机故障（捕图 warmup、APC
# 混合负载）都无法从转储判定哪个 status 族非零。本模块提供一个默认关闭的
# 归因开关：OSCAR_GUARD_HOST_SYNC=1 时把同一个冻结 status_guard 内核按族
# 单独启动并逐族 host 同步，使 trap 从具体族的调用行抛出。默认路径保持
# 原来的单次融合启动，逐字节不改任何算子；这不是回退——任何非零 status
# 仍然中止当前 step，只是把"哪个族"写进异常链。
import os

import torch

from .runtime_api import OscarReadinessError

_ENABLED = None


def guard_diag_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("OSCAR_GUARD_HOST_SYNC") == "1"
    return _ENABLED


def run_status_guard(ops, site: str, segments, *, capture: bool = False) -> None:
    """Launch the frozen status trap, optionally with per-family attribution.

    segments: exactly four (name, tensor) pairs in the native a,b,c,d launch
    order. Default (or graph capture) path: one fused launch, identical to the
    historical behavior. Diagnostic path launches that same frozen kernel for
    each family and synchronizes to attribute the trap. It never reads status
    contents back to the host, including through an imported helper.
    """
    if not guard_diag_enabled() or capture:
        ops.status_guard(*[tensor for _, tensor in segments])
        return
    for name, tensor in segments:
        if tensor.numel() == 0:
            continue
        flat = tensor.reshape(-1)
        empty = flat[:0]
        try:
            ops.status_guard(flat, empty, empty, empty)
            torch.npu.synchronize()
        except Exception as error:
            raise OscarReadinessError(
                f"OSCAR status guard trapped at site={site} family={name}") from error
