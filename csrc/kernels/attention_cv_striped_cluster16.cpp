// SPDX-License-Identifier: Apache-2.0
// Archive #126/#129/#148-154 and startup D.4: same CV math and bounded
// KV256 schedule, reading only striped-v1 packed pages; old arithmetic is unchanged.
// D.4 questions: fused history FIA; old whole restore cost 6.5s/card;
// remove repeated gather/plane staging via lossless stored code permutation;
// compilation/CPU agreement do not establish target speed or model quality.
#include "attention_fast_unpack.h"
#include "attention_striped_unpack.h"
#define oscar_ascend_fast oscar_ascend_striped
#define OSCAR_STRIPED_D256_ONLY 1
#define attention_cv_fast_cluster16_launch attention_cv_striped_cluster16_launch
#define oscar_attention_cv_fast_cluster16_kernel oscar_attention_cv_striped_cluster16_kernel
#include "attention_cv_fast_cluster16.cpp"
#undef OSCAR_STRIPED_D256_ONLY
