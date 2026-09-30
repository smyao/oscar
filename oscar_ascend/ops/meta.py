# Archive #27/#34/#36/#50: abstract shape propagation is separate from actual NPU graph capture/replay.
# #148: optional cluster schema mutates caller-owned buffers; never computes fake outputs.
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

    names = set(PRODUCTION_CAPABILITIES)
    # #148: this optional experimental schema is never a numerical fallback.
    for optional in ("attention_cv_cluster4_out", "attention_cv_q1_out", "attention_cv_profile_out",
                     "attention_cv_fast_out", "attention_cv_fast_q1_out", "attention_cv_fast_cluster4_out",
                     "attention_cv_fast_balanced_out", "attention_cv_fast_cluster16_out"):
        if hasattr(getattr(torch.ops, namespace), optional):
            names.add(optional)
    for name in sorted(names):
        torch.library.register_fake(f"{namespace}::{name}")(out_only)
    _registered.add(namespace)
