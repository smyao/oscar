"""Archive #129/#140/#148/#152/#155: isolate short queries in eager mixed batches.

Global N/max_query_len currently selects C16 and S1 for short decode requests
that share a batch with long prefill. Partition only at request boundaries.
This experiment keeps the original S choice, preserves the native model/GDN
call, and changes only FULL attention's internal launches. Eligibility requires
native prefix caching to be explicitly disabled: kv_cache_manager.py:212 then
returns no shared computed blocks; request-owned allocation remains native.
"""
from dataclasses import dataclass,replace
import torch
from ..ops.cv_dispatch import ATTENTION_QUERY_ROWS


@dataclass(frozen=True)
class MixedDecodePlan:
    prefix_requests:int
    prefix_tokens:int
    actual_tokens:int
    original_tokens:int


def plan_mixed_decode(metadata,tokens,*,query_heads=6,kv_heads=1):
    if (not getattr(metadata,'mixed_decode_split',False) or metadata.capture_origin or
            metadata.dummy_origin or (metadata.is_draft and not (
                metadata.draft_index==0 and metadata.first_draft_current_fia))):
        return None
    starts=metadata.query_start_loc_cpu
    if (not isinstance(starts,torch.Tensor) or starts.device.type!='cpu' or
            starts.ndim!=1 or starts.dtype not in (torch.int32,torch.int64) or
            starts.numel()<metadata.num_reqs+1):
        return None
    values=starts[:metadata.num_reqs+1].tolist()
    if (not values or values[0]!=0 or values[-1]!=metadata.num_actual_tokens or
            any(b<=a for a,b in zip(values,values[1:])) or
            type(tokens) is not int or not 0<values[-1]<=tokens):
        return None
    lengths=[b-a for a,b in zip(values,values[1:])]
    short=0
    while short<len(lengths) and lengths[short]<=4:short+=1
    if (type(query_heads) is not int or type(kv_heads) is not int or kv_heads<=0
            or query_heads<kv_heads or query_heads%kv_heads or query_heads//kv_heads>16):
        return None
    # Use the actual current dispatcher threshold, not an arbitrary 1K
    # cutoff: even a 512-token continuation selects C4 for the whole batch.
    clustered_minimum=4*(ATTENTION_QUERY_ROWS//(query_heads//kv_heads))
    if not 0<short<len(lengths) or values[short]>128 or max(lengths[short:])<clustered_minimum:
        return None
    original=metadata.cv_shape_tokens if metadata.cv_shape_tokens is not None else tokens
    if type(original) is not int or original<tokens:return None
    return MixedDecodePlan(short,values[short],values[-1],original)


def _slice_metadata(metadata,request_begin,request_end,token_begin,token_end,original,segment):
    cpu=metadata.query_start_loc_cpu[request_begin:request_end+1]-token_begin
    ends=tuple(cpu[1:].tolist())
    lengths=cpu[1:]-cpu[:-1]
    return replace(metadata,
        query_start_loc=metadata.query_start_loc[request_begin:request_end+1]-token_begin,
        query_start_loc_cpu=cpu,seq_lens=metadata.seq_lens[request_begin:request_end],
        block_tables=metadata.block_tables[request_begin:request_end],
        slot_mapping=metadata.slot_mapping[token_begin:token_end],
        num_reqs=request_end-request_begin,num_actual_tokens=token_end-token_begin,
        num_input_tokens=token_end-token_begin,max_query_len=int(lengths.max()),
        current_cumulative=ends,current_only_plan=None,first_draft_current_only=False,
        mixed_decode_split=False,mixed_split_segment=segment,cv_shape_tokens=original)


def dispatch_mixed_decode(impl,layer,query,key,value,kv_cache,metadata,output,plan):
    state=impl.provider.layer_state(layer.layer_name);ops=impl.provider.ops
    state.workspace.validate(query.shape[0],impl.num_heads,impl.num_kv_heads,impl.head_size)
    empty=state.workspace.store_status[:0].view(-1)
    # All checks remain on the device and precede any store. Never move a
    # sequence boundary to the CPU or hide a live trailing padding slot.
    starts=metadata.query_start_loc
    boundaries=torch.stack((starts[0],starts[plan.prefix_requests]-plan.prefix_tokens,
                            starts[metadata.num_reqs]-plan.actual_tokens))
    ops.status_guard((boundaries!=0).to(torch.int32),empty,empty,empty)
    if plan.actual_tokens<query.shape[0]:
        bad=(metadata.slot_mapping[plan.actual_tokens:query.shape[0]]>=0).to(torch.int32)
        ops.status_guard(bad,empty,empty,empty)
    for rb,re,tb,te,segment in (
        (0,plan.prefix_requests,0,plan.prefix_tokens,'short_prefix'),
        (plan.prefix_requests,metadata.num_reqs,plan.prefix_tokens,plan.actual_tokens,'long_suffix')):
        child=_slice_metadata(metadata,rb,re,tb,te,plan.original_tokens,segment)
        impl.forward(layer,query[tb:te],key[tb:te],value[tb:te],kv_cache,child,output=output[tb:te])
    if plan.actual_tokens<query.shape[0]:output[plan.actual_tokens:].zero_()
    return output
