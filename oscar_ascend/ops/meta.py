# Archive #27/#34/#36/#50: abstract shape propagation is separate from actual NPU graph capture/replay.
"""Meta implementations for out-only operators; never a numerical execution path."""
_registered=set()


def register_meta(namespace="oscar_ascend_ops"):
    if namespace in _registered:
        return
    import torch
    from .contracts import PRODUCTION_CAPABILITIES

    def out_only(*args,**kwargs):
        # All shapes/strides are caller-owned outputs. FakeTensor tracks the
        # schema mutations; no numerical tensors or fake success are produced.
        return None

    for name in sorted(PRODUCTION_CAPABILITIES):
        torch.library.register_fake(f"{namespace}::{name}")(out_only)
    _registered.add(namespace)
