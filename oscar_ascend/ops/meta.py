# Archive #27/#34/#36/#50: abstract shape propagation is separate from actual NPU graph capture/replay.
# #148: optional cluster schema mutates caller-owned buffers; never computes fake outputs.
"""Meta implementations for out-only operators; never a numerical execution path."""
_registered=set()
_libraries=[]


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
    # Diagnostic schemas remain registered for signed offline comparisons;
    # production capability requires the unified schema below.
    for optional in ("attention_cv_cluster4_out", "attention_cv_q1_out", "attention_cv_profile_out",
                     "attention_cv_fast_out", "attention_cv_fast_weighted_out", "attention_cv_fast_q1_out", "attention_cv_unified_out"):
        if hasattr(getattr(torch.ops, namespace), optional):
            names.add(optional)
    register_fake = getattr(torch.library, "register_fake", None)
    meta_library = None if register_fake is not None else torch.library.Library(namespace, "IMPL", "Meta")
    for name in sorted(names):
        if register_fake is not None:
            register_fake(f"{namespace}::{name}")(out_only)
        else:
            meta_library.impl(name, out_only)
    if meta_library is not None:
        _libraries.append(meta_library)
    _registered.add(namespace)
