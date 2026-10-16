"""Archive #148/#150/#153: explicit, complete OSCAR INT2 kernel selection.

C4 requires four full query groups in ONE request. Native host metadata can
prove that impossible without reading device tensors or adding graph nodes.
Unknown bounds conservatively keep C4. This never selects native attention.
"""

ATTENTION_QUERY_ROWS = 128
FE0_CV_OP = "attention_cv_out"
CLUSTER4_CV_OP = "attention_cv_cluster4_out"
Q1_CV_OP = "attention_cv_q1_out"
FAST_CV_OP = "attention_cv_fast_out"
FAST_Q1_CV_OP = "attention_cv_fast_q1_out"
FAST_CLUSTER4_CV_OP = "attention_cv_fast_cluster4_out"
FAST_BALANCED_CV_OP = "attention_cv_fast_balanced_out"
FAST_CLUSTER16_CV_OP = "attention_cv_fast_cluster16_out"
STRIPED_CV_OP = "attention_cv_striped_out"
STRIPED_Q1_CV_OP = "attention_cv_striped_q1_out"
STRIPED_CLUSTER4_CV_OP = "attention_cv_striped_cluster4_out"
STRIPED_BALANCED_CV_OP = "attention_cv_striped_balanced_out"
STRIPED_CLUSTER16_CV_OP = "attention_cv_striped_cluster16_out"
STRIPED_DECODE_CV_OP = "attention_cv_striped_decode_out"
MDL_DECODE_CV_OP = "attention_cv_mdl_decode_out"
MDL_DECODE_Q1_CV_OP = "attention_cv_mdl_decode_q1_out"
MDL_DECODE_CV_OPS = frozenset({MDL_DECODE_CV_OP, MDL_DECODE_Q1_CV_OP})
MEMBER_PIPELINE_CV_OP = "attention_cv_member_pipeline_cluster4_out"
BF16_HISTORY_OP_MAP = {
    MDL_DECODE_CV_OP: "attention_bf16_history_mdl_out",
    MDL_DECODE_Q1_CV_OP: "attention_bf16_history_mdl_q1_out",
    MEMBER_PIPELINE_CV_OP: "attention_bf16_history_member_pipeline_cluster4_out",
    "attention_cv_window_range_cluster16_out": "attention_bf16_history_window_range_cluster16_out",
    "attention_cv_window_range_out": "attention_bf16_history_window_range_out",
    "attention_cv_window_range_balanced_out": "attention_bf16_history_window_range_balanced_out",
}
BF16_HISTORY_CV_OPS = frozenset(BF16_HISTORY_OP_MAP.values())
BF16_HISTORY_CLUSTER_OPS = frozenset({
    BF16_HISTORY_OP_MAP[MEMBER_PIPELINE_CV_OP],
    BF16_HISTORY_OP_MAP["attention_cv_window_range_cluster16_out"],
})
OPTIMIZED_STRIPED_OPS = {
    STRIPED_CV_OP: "attention_cv_window_range_out",
    STRIPED_BALANCED_CV_OP: "attention_cv_window_range_balanced_out",
    STRIPED_CLUSTER4_CV_OP: "attention_cv_window_range_cluster4_out",
    STRIPED_CLUSTER16_CV_OP: "attention_cv_window_range_cluster16_out",
    STRIPED_DECODE_CV_OP: "attention_cv_bundle_decode_out",
    STRIPED_Q1_CV_OP: "attention_cv_bundle_q1_out",
}
DECODE_BUNDLE_OPS = frozenset(OPTIMIZED_STRIPED_OPS.values())
CLUSTER_CV_OPS = frozenset({CLUSTER4_CV_OP, FAST_CLUSTER4_CV_OP,
                            FAST_CLUSTER16_CV_OP, STRIPED_CLUSTER4_CV_OP,
                            STRIPED_CLUSTER16_CV_OP,
                            "attention_cv_window_range_cluster4_out",
                            "attention_cv_window_range_cluster16_out",
                            MEMBER_PIPELINE_CV_OP}) | BF16_HISTORY_CLUSTER_OPS
FAST_CV_OPS = frozenset({FAST_CV_OP, FAST_Q1_CV_OP, FAST_CLUSTER4_CV_OP})
MIXED_CV_OPS = frozenset({FAST_BALANCED_CV_OP, FAST_CLUSTER16_CV_OP})
STRIPED_CV_OPS = frozenset({STRIPED_CV_OP, STRIPED_Q1_CV_OP,
                            STRIPED_CLUSTER4_CV_OP, STRIPED_BALANCED_CV_OP,
                            STRIPED_CLUSTER16_CV_OP, STRIPED_DECODE_CV_OP})


def select_cv_op(cluster_size, heads, kv_heads, tokens, max_query_len, *, q1_draft=False,
                 fast_unpack=False, mixed_cv=False, striped_cache=False, decode_bundle=False,
                 c4_member_pipeline=False, mdl_decode=False, bf16_history=False):
    if any(type(flag) is not bool for flag in (fast_unpack, mixed_cv, striped_cache, decode_bundle,
                                              c4_member_pipeline, mdl_decode, bf16_history)):
        raise ValueError("CV experiment flags must be explicit booleans")
    if bf16_history:
        if cluster_size != 16 or not all((fast_unpack, mixed_cv, striped_cache,
                                         decode_bundle, c4_member_pipeline, mdl_decode)):
            raise ValueError("BF16 attention requires the complete C16 candidate bundle")
        old = select_cv_op(cluster_size, heads, kv_heads, tokens, max_query_len,
            q1_draft=q1_draft, fast_unpack=fast_unpack, mixed_cv=mixed_cv,
            striped_cache=striped_cache, decode_bundle=decode_bundle,
            c4_member_pipeline=c4_member_pipeline, mdl_decode=mdl_decode)
        if old not in BF16_HISTORY_OP_MAP:
            raise ValueError(f"BF16 attention has no registered reader for {old}")
        return BF16_HISTORY_OP_MAP[old]
    if mdl_decode:
        if not decode_bundle:
            raise ValueError("MDL decode requires the explicit decode bundle")
        # #126/#145/#154: replace the separately qualified q4/q1 layouts.
        # C4/C16 and all eligibility/error checks retain their original route.
        old = select_cv_op(cluster_size, heads, kv_heads, tokens, max_query_len,
            q1_draft=q1_draft, fast_unpack=fast_unpack, mixed_cv=mixed_cv,
            striped_cache=striped_cache, decode_bundle=decode_bundle,
            c4_member_pipeline=c4_member_pipeline)
        return {"attention_cv_bundle_decode_out": MDL_DECODE_CV_OP,
                "attention_cv_bundle_q1_out": MDL_DECODE_Q1_CV_OP}.get(old, old)
    if c4_member_pipeline:
        if not decode_bundle:
            raise ValueError("C4 member pipeline requires the explicit decode bundle")
        # #150: retain every host eligibility decision, including q1/q4/C16.
        # Only the existing optimized striped C4 route selects the new ABI.
        old = select_cv_op(cluster_size, heads, kv_heads, tokens, max_query_len,
            q1_draft=q1_draft, fast_unpack=fast_unpack, mixed_cv=mixed_cv,
            striped_cache=striped_cache, decode_bundle=decode_bundle)
        return (MEMBER_PIPELINE_CV_OP if old == OPTIMIZED_STRIPED_OPS[STRIPED_CLUSTER4_CV_OP]
                else old)
    if decode_bundle:
        if not striped_cache:
            raise ValueError("decode bundle requires the explicit striped cache bundle")
        old = select_cv_op(cluster_size, heads, kv_heads, tokens, max_query_len,
            q1_draft=q1_draft, fast_unpack=fast_unpack, mixed_cv=mixed_cv,
            striped_cache=striped_cache, decode_bundle=False)
        return OPTIMIZED_STRIPED_OPS[old]
    if striped_cache and (not mixed_cv or not fast_unpack or cluster_size != 16):
        raise ValueError("striped cache requires mixed/fast C16 candidate geometry")
    if mixed_cv and (not fast_unpack or cluster_size != 16):
        raise ValueError("mixed CV requires fast unpack and C16 workspace")
    if cluster_size == 16 and not mixed_cv:
        raise ValueError("C16 workspace requires explicit mixed CV routing")
    if fast_unpack and cluster_size not in (4, 16):
        raise ValueError("fast unpack requires explicit candidate geometry")
    if cluster_size == 1:
        return FE0_CV_OP
    if cluster_size not in (4, 16) or heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("invalid CV dispatch geometry")
    ratio = heads // kv_heads
    if ratio > 16:
        raise ValueError("unsupported CV GQA ratio")
    # #150 follow-up: later MTP q1 rounds retain padded model buffers. Keep
    # their full shape/task table, but distribute independent leaders rather
    # than serializing 21 requests on a single Cube. Target capture routing
    # is deliberately excluded by the caller's explicit later-draft marker.
    if q1_draft and type(max_query_len) is int and max_query_len == 1:
        if striped_cache:
            # Tile-one owner distribution is critical for the N16K-padded,
            # 32-real-request later MTP draft. GQA>8 cannot use the M32
            # kernel; keep its reader striped while retaining complete M128.
            if ratio > 8:
                return STRIPED_BALANCED_CV_OP if tokens > 128 else STRIPED_CV_OP
            return STRIPED_Q1_CV_OP
        return FAST_Q1_CV_OP if fast_unpack else Q1_CV_OP
    minimum = 4 * (ATTENTION_QUERY_ROWS // ratio)
    if mixed_cv:
        if striped_cache and type(max_query_len) is int and 0 <= max_query_len <= 4 and ratio <= 8:
            # Target FULL_DECODE_ONLY captures uniform q4. The device kernel
            # still checks every real task's qcount*GQA<=32 on graph replay.
            return STRIPED_DECODE_CV_OP
        # An exact same-domain C16 cluster needs at least sixteen full query
        # groups. The high threshold is a conservative host-only route: small
        # prefill has too few clusters to keep the measured Cubes busy.
        if type(max_query_len) is int and max_query_len >= 8192:
            return STRIPED_CLUSTER16_CV_OP if striped_cache else FAST_CLUSTER16_CV_OP
        if tokens < minimum or (type(max_query_len) is int and
                                0 <= max_query_len < minimum):
            # N128/S3 q4 already uses most Cubes and has a validated graph.
            # Balance only the larger padded buffers whose wasted token scan
            # and long per-core leader chains were observed in mixed traffic.
            if striped_cache:
                return STRIPED_BALANCED_CV_OP if tokens > 128 else STRIPED_CV_OP
            return FAST_BALANCED_CV_OP if tokens > 128 else FAST_CV_OP
        # Unknown per-request length stays on the already validated C4 path.
        return STRIPED_CLUSTER4_CV_OP if striped_cache else FAST_CLUSTER4_CV_OP
    # Total padded tokens alone do not establish per-request eligibility:
    # 32 independent q4 requests still cannot share any history with each other.
    if tokens < minimum or (type(max_query_len) is int and 0 <= max_query_len < minimum):
        return FAST_CV_OP if fast_unpack else FE0_CV_OP
    return FAST_CLUSTER4_CV_OP if fast_unpack else CLUSTER4_CV_OP
