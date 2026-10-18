// Archive #111/#126/#164; BF16 current-chunk input, FP32 normalized metadata.
// Packed striped_fp32_v2 offsets/strides are bytes; raw window strides are
// BF16 elements; tags are int64. Original inputs preserve exact raw windows.
#pragma once
#include <cstdint>
namespace oscar_ascend {
void store_aligned_int2_launch(void* stream,void* rotated_key,void* rotated_value,
    void* original_key,void* original_value,void* slots,void* positions,
    void* packed,void* raw_key,void* raw_value,void* tags,void* status,
    int64_t tokens,int64_t heads,int64_t dim,int64_t block_tokens,int64_t blocks,
    int64_t offset,int64_t page_stride,int64_t key_stride,int64_t value_stride,
    int64_t tag_stride,int64_t sink,int64_t recent,float k_clip,float v_clip,
    uint32_t cores);
// Input and output are both BF16; FP32 exists only in bounded guard UB.
void copy_validate_bf16_query_launch(void* stream,void* input,void* slots,
    void* output,void* status,int64_t tokens,int64_t heads,int64_t dim,uint32_t cores);
}
