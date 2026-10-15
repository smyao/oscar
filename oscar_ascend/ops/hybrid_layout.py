"""Archive #126/#148: immutable FP32 Rv address layout for the hybrid q4.

Called during rotation preparation before native KV profiling; never in the
attention/graph hot path. The independent oracle continues using natural Rv.
"""


def pack_rotation_nz(rotation):
    import torch
    if rotation.dtype != torch.float32 or tuple(rotation.shape) != (256, 256):
        raise ValueError('hybrid L1 rotation must be FP32 [256,256]')
    # offset(row,col) = (col//8)*256*8 + row*8 + col%8 (FP32 C0=8).
    return rotation.reshape(256, 32, 8).permute(1, 0, 2).contiguous().view(256, 256)
