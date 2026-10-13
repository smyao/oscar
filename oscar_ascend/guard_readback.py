# 档案 #142/#157：生产 Trap 不带故障值；诊断模式(OSCAR_GUARD_HOST_SYNC=1,
# 默认关闭)需要一次性读回非零 status 的索引与错误码来定点修复。此模块只在
# 该显式诊断路径被调用;在线热路径与默认融合 trap 路径均不经过这里。
import torch


def find_nonzero(tensor):
    """Return (count, indices, values) for a status tensor, or None if clean."""
    if tensor.numel() == 0:
        return None
    flat = tensor.reshape(-1)
    mask = flat != 0
    count = int(mask.sum().item())
    if count == 0:
        return None
    indices = mask.nonzero().reshape(-1).cpu()
    values = flat[mask].cpu()
    return count, indices.tolist(), values.tolist()
