"""Archive #148/#150: select only between complete OSCAR INT2 kernels.

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
BATCHED4_CV_OP = "attention_cv_batched4_out"
CLUSTER_CV_OPS = frozenset({CLUSTER4_CV_OP, FAST_CLUSTER4_CV_OP,
                            FAST_CLUSTER16_CV_OP, BATCHED4_CV_OP})
FAST_CV_OPS = frozenset({FAST_CV_OP, FAST_Q1_CV_OP, FAST_CLUSTER4_CV_OP})
MIXED_CV_OPS = frozenset({FAST_BALANCED_CV_OP, FAST_CLUSTER16_CV_OP})


def select_cv_op(cluster_size, heads, kv_heads, tokens, max_query_len, *, q1_draft=False,
                 fast_unpack=False, mixed_cv=False, batched_history=False,
                 batched_history_ready=False):
    if any(type(flag) is not bool for flag in
           (fast_unpack, mixed_cv, batched_history, batched_history_ready)):
        raise ValueError("CV experiment flags must be explicit booleans")
    if batched_history and (not mixed_cv or not fast_unpack or cluster_size != 16):
        raise ValueError("batched history requires mixed/fast C16 candidate geometry")
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
        return FAST_Q1_CV_OP if fast_unpack else Q1_CV_OP
    minimum = 4 * (ATTENTION_QUERY_ROWS // ratio)
    if mixed_cv:
        # CPU metadata proves one long continuation has >3 real old tokens.
        # Never infer it from max_seq_len/max_query_len across different reqs.
        if batched_history and batched_history_ready and (
                type(max_query_len) is int and max_query_len >= minimum):
            return BATCHED4_CV_OP
        # An exact same-domain C16 cluster needs at least sixteen full query
        # groups. The high threshold is a conservative host-only route: small
        # prefill has too few clusters to keep the measured Cubes busy.
        if type(max_query_len) is int and max_query_len >= 8192:
            return FAST_CLUSTER16_CV_OP
        if tokens < minimum or (type(max_query_len) is int and
                                0 <= max_query_len < minimum):
            # N128/S3 q4 already uses most Cubes and has a validated graph.
            # Balance only the larger padded buffers whose wasted token scan
            # and long per-core leader chains were observed in mixed traffic.
            return FAST_BALANCED_CV_OP if tokens > 128 else FAST_CV_OP
        # Unknown per-request length stays on the already validated C4 path.
        return FAST_CLUSTER4_CV_OP
    # Total padded tokens alone do not establish per-request eligibility:
    # 32 independent q4 requests still cannot share any history with each other.
    if tokens < minimum or (type(max_query_len) is int and 0 <= max_query_len < minimum):
        return FAST_CV_OP if fast_unpack else FE0_CV_OP
    return FAST_CLUSTER4_CV_OP if fast_unpack else CLUSTER4_CV_OP
