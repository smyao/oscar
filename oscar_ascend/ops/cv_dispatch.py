"""One production CV route for every supported attention geometry.

Archive #129/#140-151 and startup D.4: shape-specific q1/q4/C4 dispatch made
the measured optimization depend on a local case. Production always uses the
same exact INT2 operator. Query reuse is an internal device decision and never
selects a different numerical implementation.
"""

ATTENTION_QUERY_ROWS = 128
UNIFIED_CV_OP = "attention_cv_unified_out"

# Compatibility names for offline diagnostic code only. They intentionally
# resolve to the same ABI and must never be used to choose a production path.
FE0_CV_OP = CLUSTER4_CV_OP = Q1_CV_OP = UNIFIED_CV_OP
FAST_CV_OP = FAST_WEIGHTED_CV_OP = FAST_Q1_CV_OP = UNIFIED_CV_OP
FAST_CLUSTER4_CV_OP = UNIFIED_CV_OP


def select_cv_op(*_args, **_kwargs):
    return UNIFIED_CV_OP


def select_source_splits(default_splits, *_args, **_kwargs):
    if type(default_splits) is not int or not 1 <= default_splits <= 32:
        raise ValueError("invalid default CV split count")
    return default_splits
