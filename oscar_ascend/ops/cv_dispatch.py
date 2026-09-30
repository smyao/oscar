"""Archive #148/#150 and P0 target pass: select complete OSCAR INT2 kernels.

C4 requires four full query groups in ONE request. Native host metadata can
prove that impossible without reading device tensors or adding graph nodes.
Unknown bounds conservatively keep C4. This never selects native attention.
"""

ATTENTION_QUERY_ROWS = 128
FE0_CV_OP = "attention_cv_out"
CLUSTER4_CV_OP = "attention_cv_cluster4_out"
Q1_CV_OP = "attention_cv_q1_out"
FAST_CV_OP = "attention_cv_fast_out"
FAST_WEIGHTED_CV_OP = "attention_cv_fast_weighted_out"
FAST_Q1_CV_OP = "attention_cv_fast_q1_out"
FAST_CLUSTER4_CV_OP = "attention_cv_fast_cluster4_out"
CLUSTER_CV_OPS = frozenset({CLUSTER4_CV_OP, FAST_CLUSTER4_CV_OP})
FAST_CV_OPS = frozenset({FAST_CV_OP, FAST_WEIGHTED_CV_OP,
                         FAST_Q1_CV_OP, FAST_CLUSTER4_CV_OP})


def select_source_splits(default_splits, cv_op, *, tokens,
                         weighted_q4_split2=False):
    """Apply only the S2 promotion proven for the exact weighted-q4 route."""
    if type(default_splits) is not int or not 1 <= default_splits <= 32:
        raise ValueError("invalid default CV split count")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("invalid active token count")
    if type(weighted_q4_split2) is not bool:
        raise ValueError("weighted q4 S2 flag must be an explicit boolean")
    if weighted_q4_split2 and cv_op == FAST_WEIGHTED_CV_OP and tokens == 128:
        return 2
    return default_splits


def select_cv_op(cluster_size, heads, kv_heads, tokens, max_query_len, *, q1_draft=False,
                 fast_unpack=False, weighted_q4=False, head_dim=None):
    if type(fast_unpack) is not bool or (fast_unpack and cluster_size != 4):
        raise ValueError("fast unpack requires explicit candidate geometry")
    if type(weighted_q4) is not bool or (weighted_q4 and not fast_unpack):
        raise ValueError("weighted q4 requires the proven fast candidate")
    if cluster_size == 1:
        return FE0_CV_OP
    if cluster_size != 4 or heads <= 0 or kv_heads <= 0 or heads % kv_heads:
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
    # Total padded tokens alone do not establish per-request eligibility:
    # 32 independent q4 requests still cannot share any history with each other.
    if tokens < minimum or (type(max_query_len) is int and 0 <= max_query_len < minimum):
        # Target run observe-20260929T081412: Hq6/Hkv1/D256, 32*q4,
        # N128/S3 passed bitwise/oracle/changed-input graph and all five Event
        # pairs (10.7628ms vs 11.2804ms median). Do not broaden that proof.
        if (weighted_q4 and heads == 6 and kv_heads == 1 and head_dim == 256 and
                tokens <= 128 and max_query_len == 4):
            return FAST_WEIGHTED_CV_OP
        return FAST_CV_OP if fast_unpack else FE0_CV_OP
    return FAST_CLUSTER4_CV_OP if fast_unpack else CLUSTER4_CV_OP
