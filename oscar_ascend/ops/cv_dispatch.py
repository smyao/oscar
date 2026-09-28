"""Archive #148/#150: select only between complete OSCAR INT2 kernels.

C4 requires four full query groups in ONE request. Native host metadata can
prove that impossible without reading device tensors or adding graph nodes.
Unknown bounds conservatively keep C4. This never selects native attention.
"""

ATTENTION_QUERY_ROWS = 128
FE0_CV_OP = "attention_cv_out"
CLUSTER4_CV_OP = "attention_cv_cluster4_out"


def select_cv_op(cluster_size, heads, kv_heads, tokens, max_query_len):
    if cluster_size == 1:
        return FE0_CV_OP
    if cluster_size != 4 or heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("invalid CV dispatch geometry")
    ratio = heads // kv_heads
    if ratio > 16:
        raise ValueError("unsupported CV GQA ratio")
    minimum = 4 * (ATTENTION_QUERY_ROWS // ratio)
    # Total padded tokens alone do not establish per-request eligibility:
    # 32 independent q4 requests still cannot share any history with each other.
    if tokens < minimum or (type(max_query_len) is int and 0 <= max_query_len < minimum):
        return FE0_CV_OP
    return CLUSTER4_CV_OP
