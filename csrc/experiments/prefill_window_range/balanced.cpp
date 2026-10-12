// Isolated actual striped candidate clone. Archive #126/#129/#140/#145/#148/#151/#154/#155.
// D.4: full restore cost6499.8-6655.1ms is not repeated. This changes bounded
// window reads and score-mask writes only, with original FP32 history math,
// source/group boundaries and statuses. No current q4 gain is assumed here.
#include "../../kernels/attention_fast_unpack.h"
#include "../../kernels/attention_striped_unpack.h"
#define oscar_ascend_fast oscar_ascend_striped
#define OSCAR_STRIPED_D256_ONLY 1
#define attention_cv_fast_balanced_launch attention_cv_window_range_balanced_launch
#define oscar_attention_cv_fast_balanced_kernel oscar_attention_window_range_balanced_kernel
#include "balanced_body.cpp"
#undef OSCAR_STRIPED_D256_ONLY
