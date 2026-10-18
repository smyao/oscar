# Archive #27/#34/#36/#50: abstract shape propagation is separate from actual NPU graph capture/replay.
# #148: optional cluster schema mutates caller-owned buffers; never computes fake outputs.
# #153: optional request-window/VectorInit schemas match the source/build capability lists.
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
    for optional in ("attention_cv_aligned_decode_out","attention_cv_aligned_decode_q1_out","attention_cv_aligned_cluster4_out","attention_cv_aligned_cluster16_out","store_aligned_int2_out","copy_validate_bf16_query_out","attention_cv_bf16_decode_out", "attention_cv_bf16_decode_q1_out", "attention_cv_bf16_cluster4_out", "attention_cv_bf16_cluster16_out",
                     "rotate_vector_init_out", "attention_cv_cluster4_out", "attention_cv_q1_out", "attention_cv_profile_out",
                     "attention_cv_fast_out", "attention_cv_fast_q1_out", "attention_cv_fast_cluster4_out",
                     "attention_cv_fast_balanced_out", "attention_cv_fast_cluster16_out",
                     "attention_cv_striped_out", "attention_cv_striped_q1_out",
                     "attention_cv_striped_cluster4_out", "attention_cv_striped_balanced_out",
                     "attention_cv_striped_cluster16_out", "attention_cv_striped_decode_out",
                     "rotate_clip_store_request_window_out", "attention_request_window_decode_out",
                     "attention_request_window_decode_q1_out", "attention_request_window_out",
                     "attention_request_window_balanced_out", "attention_request_window_cluster4_out",
                     "attention_request_window_cluster16_out",
                     "attention_cv_member_pipeline_cluster4_out", "attention_request_window_member_pipeline_cluster4_out",
                     "attention_cv_member_pipeline_cluster16_out", "attention_cv_hybrid_l1_decode_out",
                     "attention_cv_l1_reuse_cluster16_out",
                     "attention_cv_bundle_decode_out", "attention_cv_mdl_decode_out", "attention_cv_mdl_decode_q1_out", "attention_cv_bundle_q1_out", "attention_cv_window_range_out", "attention_cv_window_range_balanced_out", "attention_cv_window_range_cluster4_out", "attention_cv_window_range_cluster16_out", "rotate_clip_store_striped_out", "copy_validate_current_out"):
        if hasattr(getattr(torch.ops, namespace), optional):
            names.add(optional)
    for name in sorted(names):
        torch.library.register_fake(f"{namespace}::{name}")(out_only)
    _registered.add(namespace)
