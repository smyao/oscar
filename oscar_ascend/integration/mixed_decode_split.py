"""Archive #34/#36/#129/#140/#148/#152/#155/#157: eager request partitions.

Global N/max_query_len currently selects C16 and S1 for short decode requests
that share a batch with long prefill. Partition only at request boundaries.
This experiment keeps the original S choice, preserves the native model/GDN
call, and changes only FULL attention's internal launches. Archive #26/#37-49:
native FullAttentionManager shares only complete immutable prefix blocks;
allocate_new_blocks gives each request its own writable suffix. Both launches
may read shared prefix pages, but neither writes them. Native MTP's last-block
drop is retained. No block-table, allocation, hash or GDN semantics are changed.
Optional prepare reuse partitions existing scratch by the same native token
boundaries and consumes the builder's immutable CPU tuple, never a cached
mutable CPU mirror. Every boundary/slot/status guard remains per layer.
"""
from dataclasses import dataclass,replace
import torch
from ..ops.cv_dispatch import ATTENTION_QUERY_ROWS
from .metadata import slice_window_map
from .prepare_reuse import PrepareRegion, immutable_query_starts


@dataclass(frozen=True)
class MixedDecodePlan:
    prefix_requests:int
    prefix_tokens:int
    actual_tokens:int
    original_tokens:int


def plan_mixed_decode(metadata,tokens,*,query_heads=6,kv_heads=1,immutable_starts=False):
    if (not getattr(metadata,'mixed_decode_split',False) or metadata.capture_origin or
            metadata.dummy_origin or (metadata.is_draft and not (
                metadata.draft_index==0 and metadata.first_draft_current_fia))):
        return None
    if immutable_starts:
        values=immutable_query_starts(metadata)
        if values is None:return None
    else:
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


def _slice_metadata(metadata,request_begin,request_end,token_begin,token_end,original,segment,
                    *,prepare_reuse=False):
    region=None
    if prepare_reuse:
        starts=immutable_query_starts(metadata)
        if starts is None:raise ValueError('mixed prepare reuse requires immutable native query boundaries')
        values=tuple(value-token_begin for value in starts[request_begin:request_end+1])
        cpu=torch.tensor(values,dtype=metadata.query_start_loc.dtype)
        ends=values[1:];longest=max(b-a for a,b in zip(values,values[1:]))
        region=PrepareRegion(metadata,request_begin,request_end,token_begin,token_end,original)
    else:
        cpu=metadata.query_start_loc_cpu[request_begin:request_end+1]-token_begin
        ends=tuple(cpu[1:].tolist())
        lengths=cpu[1:]-cpu[:-1];longest=int(lengths.max())
    return replace(metadata,
        query_start_loc=metadata.query_start_loc[request_begin:request_end+1]-token_begin,
        query_start_loc_cpu=cpu,seq_lens=metadata.seq_lens[request_begin:request_end],
        block_tables=metadata.block_tables[request_begin:request_end],
        slot_mapping=metadata.slot_mapping[token_begin:token_end],
        num_reqs=request_end-request_begin,num_actual_tokens=token_end-token_begin,
        num_input_tokens=token_end-token_begin,max_query_len=longest,
        current_cumulative=ends,current_only_plan=None,first_draft_current_only=False,
        mixed_decode_split=False,mixed_split_segment=segment,cv_shape_tokens=original,
        window_map=slice_window_map(metadata, request_begin, request_end),prepare_region=region)


def guard_mixed_metadata(ops,workspace,metadata,plan,padded_tokens):
    """The same per-layer device guards serve production and its short probe."""
    empty=workspace.store_status[:0].view(-1)
    # All checks remain on the device and precede any store. Never move a
    # sequence boundary to the CPU or hide a live trailing padding slot.
    starts=metadata.query_start_loc
    boundaries=torch.stack((starts[0],starts[plan.prefix_requests]-plan.prefix_tokens,
                            starts[metadata.num_reqs]-plan.actual_tokens))
    from .guard_diag import run_status_guard
    run_status_guard(ops,'mixed_decode.boundaries',[("boundaries",(boundaries!=0).to(torch.int32)),
                                                    ("rotate",empty),("merge",empty),("store",empty)])
    if plan.actual_tokens<padded_tokens:
        bad=(metadata.slot_mapping[plan.actual_tokens:padded_tokens]>=0).to(torch.int32)
        run_status_guard(ops,'mixed_decode.padding',[("padding",bad),("rotate",empty),
                                                     ("merge",empty),("store",empty)])


def dispatch_mixed_decode(impl,layer,query,key,value,kv_cache,metadata,output,plan,*,prepare_reuse=False):
    state=impl.provider.layer_state(layer.layer_name);ops=impl.provider.ops
    state.workspace.validate(query.shape[0],impl.num_heads,impl.num_kv_heads,impl.head_size)
    guard_mixed_metadata(ops,state.workspace,metadata,plan,query.shape[0])
    for rb,re,tb,te,segment in (
        (0,plan.prefix_requests,0,plan.prefix_tokens,'short_prefix'),
        (plan.prefix_requests,metadata.num_reqs,plan.prefix_tokens,plan.actual_tokens,'long_suffix')):
        child=_slice_metadata(metadata,rb,re,tb,te,plan.original_tokens,segment,prepare_reuse=prepare_reuse)
        impl.forward(layer,query[tb:te],key[tb:te],value[tb:te],kv_cache,child,output=output[tb:te])
    if plan.actual_tokens<query.shape[0]:output[plan.actual_tokens:].zero_()
    return output
