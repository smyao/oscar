// Archive #37-49/#126/#145/#154/#155; D.4: same-stream ordered two-pass store.
// Experiment-only ABI. Binding name: rotate_clip_store_request_window_out.
#pragma once
#include <cstdint>
namespace oscar_ascend {
void request_packed_store_launch(void* stream,uint32_t grid,void* key,void* value,
    void* rk,void* rv,void* slots,void* positions,void* packed,void* packedStatus,
    int64_t tokens,int64_t heads,int64_t dim,int32_t dtype,int64_t blockTokens,
    int64_t blocks,int64_t offset,int64_t pageStride,float kClip,float vClip,bool hadamard);
void rotate_clip_store_request_window_launch(void* stream,uint32_t grid,
    void* key,void* value,void* rk,void* rv,void* slots,void* positions,
    void* qstarts,void* windowMap,void* packed,void* wk,void* wv,void* tags,
    void* packedStatus,void* windowStatus,int64_t tokens,int64_t requests,
    int64_t heads,int64_t dim,int32_t dtype,int64_t blockTokens,int64_t blocks,
    int64_t offset,int64_t pageStride,int64_t kStride,int64_t vStride,int64_t tagStride,
    int64_t sink,int64_t recentWithSpec,float kClip,float vClip,bool hadamard,
    int64_t windowCapacity,int64_t positionBits);
}
