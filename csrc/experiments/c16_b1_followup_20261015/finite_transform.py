"""Archive #126/#134/#140/#145/#148/#154; D.4 reread in this iteration.

Batch only scalar readback of the ORIGINAL per-row ReduceSum. No sentinel,
range-based reduction removal, new arithmetic, relaxed status or buffer.
"""
from __future__ import annotations
from csrc.experiments.c16_anchor_skip_20261015.generate import method,sha

# Filled from the accepted B1/Rv unchanged Softmax body by the source pin.
SOFTMAX_SHA='9a5a43930345bd2c48118c1134e962716dcaf79743bc483811af2daadf3a901d'
PREFIX='''    // Complete history tile: the earliest live query must see every
    // column. Later live rows have monotone query position/cut, so their
    // original any=true and mask-complement branch is empty as well.
    // jEnd==256 proves tileEnd is bounded by this task's kvend.
    const int64_t earliestPosition=context+qbegin+rowBase/group-requestBegin;
    const int64_t earliestCut=Max64(g.sink,earliestPosition+1-g.recent);
    const bool fullyVisibleHistory=kind==0 && jEnd==kKvRows && start>=g.sink &&
        tileEnd<=Min64(context,Min64(earliestCut,earliestPosition+1));
    if(fullyVisibleHistory) {
      // Natural unpack is dead until later alpha Brcb. Each scalar result
      // has its own32B-aligned slot; the original tmp[16] reduction work
      // area stays separate. PIPE_V keeps its reuse in original row order.
      auto finiteSums=naturalBuf.Get<float>(kHalfRows*8);
      for(int32_t r=0;r<activeRows;++r) {
        PipeBarrier<PIPE_V>();
        ReduceSum(finiteSums[r*8],scores[r*kKvRows],tmp[16],kKvRows);
      }
      Fence<HardEvent::V_S>();
      for(int32_t r=0;r<activeRows;++r)
        if(!Finite(finiteSums.GetValue(r*8)))error=2;
      // Protect scalar reads before any later Vector reuse of naturalBuf.
      Fence<HardEvent::S_V>();
    } else {
'''


def apply_finite(text):
    start,end=method(text,'Softmax');before=text[start:end]
    if sha(before)!=SOFTMAX_SHA:raise ValueError('original ReduceSum/mask/error contract changed')
    begin=before.index('    for(int32_t r=0;r<activeRows;++r) {')
    stop=before.index('    // The two 64-row max/sum arrays',begin)
    original_loop=before[begin:stop]
    if original_loop.count('ReduceSum(tmp,scores[r*kKvRows],tmp[16],kKvRows);Fence<HardEvent::V_S>();')!=1:
        raise ValueError('original exact finite check missing')
    replacement=PREFIX+original_loop+'    }\n'
    after=before[:begin]+replacement+before[stop:]
    if after.replace(replacement,original_loop)!=before:raise ValueError('full Softmax inverse identity')
    result=text[:start]+after+text[end:]
    return result,{'old_Softmax_sha256':sha(before),'new_Softmax_sha256':sha(after),
        'fallback_loop_bytes_preserved':True,'reduction_algorithm':'same FP32 ReduceSum256 and tmp[16], same row order',
        'only_scalar_readback_batched':True,'natural_buffer_result_bytes':1024,
        'result_alignment_bytes':32,'additional_workspace_UB_L1_bytes':0,
        'required_numerical_cases':['normal_full','sum_overflow','cancellation_overflow','NaN','positive_negative_Inf','masked_partial','empty_row'],
        'CPU':'not_run','CAModel':'not_run','NPU':'not_run'}
