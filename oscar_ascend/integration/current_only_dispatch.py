"""Experimental fresh-suffix execution; not enabled by serving presets.

Archive #126/#129/#142/#148/#155 and D.4: a proven zero-history suffix has only
exact current BF16 attention. Existing-prefix requests retain the complete
OSCAR path; every current K/V still enters the INT2 writer after attention.
No stored history is materialized or treated as BF16. Device checks reject
an incorrect host zero-context proof before executing current attention.
"""
from dataclasses import replace

import torch

from .current_attention import guard_current_slots, native_current_partial
from .runtime_api import OscarReadinessError
from ..timing import phase


def dispatch_current_suffix(impl, layer, query, key, value, kv_cache, metadata,
                            output, plan):
    first_draft = (metadata.is_draft and metadata.draft_index == 0
                   and metadata.first_draft_current_only)
    if (metadata.capture_origin or metadata.dummy_origin or
            (metadata.is_draft and not first_draft)):
        raise OscarReadinessError('current-only suffix requires real eager main prefill or qualified first MTP')
    n=query.shape[0]
    h,hk,d=impl.num_heads,impl.num_kv_heads,impl.head_size
    cut=plan.prefix_tokens
    fresh=plan.fresh_tokens
    actual=cut+fresh
    if actual!=metadata.num_actual_tokens or n-actual!=plan.padded_tokens:
        raise OscarReadinessError('current-only plan does not match active/padded tokens')
    state=impl.provider.layer_state(layer.layer_name)
    workspace=state.workspace
    workspace.validate(n,h,hk,d)
    ops=impl.provider.ops
    if cut:
        r=plan.prefix_requests
        cpu_starts=metadata.query_start_loc_cpu[:r+1]
        lengths=cpu_starts[1:]-cpu_starts[:-1]
        prefix=replace(metadata,query_start_loc=metadata.query_start_loc[:r+1],
                       query_start_loc_cpu=cpu_starts,seq_lens=metadata.seq_lens[:r],
                       block_tables=metadata.block_tables[:r],
                       slot_mapping=metadata.slot_mapping[:cut],
                       num_reqs=r,num_actual_tokens=cut,num_input_tokens=cut,
                       max_query_len=int(lengths.max()),
                       current_cumulative=tuple(int(v) for v in cpu_starts[1:].tolist() if v>0),
                       current_only_plan=None, first_draft_current_only=False)
        impl.forward(layer,query[:cut],key[:cut],value[:cut],kv_cache,prefix,
                     output=output[:cut])
    r=plan.prefix_requests
    fields={'layer':layer.layer_name,'tokens':fresh,'requests':plan.fresh_requests,
            'current_only':True,'stage':'draft' if first_draft else 'prefill',
            'is_draft':metadata.is_draft,'draft_index':metadata.draft_index}
    with phase('prepare',**fields):
        q=workspace.projection(query[cut:actual],workspace.query_input,fresh,h,d)
        k=workspace.projection(key[cut:actual],workspace.key_input,fresh,hk,d)
        v=workspace.projection(value[cut:actual],workspace.value_input,fresh,hk,d)
        slots=metadata.slot_mapping[cut:actual]
        if slots.dtype==torch.int32:
            workspace.slots[:fresh].copy_(slots)
            slots=workspace.slots[:fresh]
        starts=metadata.query_start_loc[r:r+plan.fresh_requests+1]-cut
        task_count=fresh*hk*3
        tasks=workspace.tasks[:task_count]
        positions=workspace.positions[:fresh]
        ops.prepare_attention_tasks_out(starts,metadata.seq_lens[r:r+plan.fresh_requests],
            slots,tasks,positions,h,hk,state.snapshots.sink_tokens,
            state.snapshots.recent_tokens,1,None,False)
        # Preserve all metadata failures. No host tensor reads or ignored
        # historical rows: a wrong zero-context proof is a same-stream trap.
        bad=(tasks[:,10]!=0)|((tasks[:,1]>=0)&(tasks[:,8]!=0))
        status=workspace.attention_status[:task_count]
        status.copy_(bad.to(torch.int32)[:,None])
        empty=workspace.store_status[:0].view(-1)
        ops.status_guard(status.view(-1),empty,empty,empty)
        guard_current_slots(ops,workspace.merge_status,slots,fresh)
        if actual<n:
            bad_padding=(metadata.slot_mapping[actual:n]>=0).to(torch.int32)
            ops.status_guard(bad_padding,empty,empty,empty)
    with phase('current_native_fia',**fields):
        current,lse=native_current_partial(q,k,v,plan.cumulative,
                                           heads=h,kv_heads=hk,scale=impl.scale)
    with phase('current_only_copy',**fields):
        if lse.dtype not in (torch.float16,torch.bfloat16,torch.float32):
            raise OscarReadinessError('current-only kernel received an unsupported native LSE dtype')
        # Canonical ND buffers, matching the existing copy into partials.
        # Q rotation is unused in this zero-history suffix; its status arena
        # can hold the temporary FP32 LSE after the prefix guard has executed.
        current_nd=workspace.query_input[:fresh]
        current_nd.copy_(current)
        lse_nd=workspace.rotate_status[:fresh].view(torch.float32)
        # Match write_current_partial: native FP16/BF16 LSE is widened into
        # the preallocated FP32 merge arena before the FP32 operator ABI.
        lse_nd.copy_(lse.squeeze(-1))
        ops.copy_validate_current_out(current_nd,lse_nd.view(fresh,h,1),
            output[cut:actual].view(fresh,h,d),workspace.lse[:fresh],
            workspace.merge_status[:fresh])
        if actual<n:
            output[actual:n].zero_()
    with phase('phase1_stores',**fields):
        name=('rotate_clip_store_striped_out' if state.cache_format=='striped_v1'
              else 'rotate_clip_store_out')
        getattr(ops,name)(k,v,state.rotation_k_transpose,state.rotation_v_transpose,
            slots,positions,state.raw,state.window_key,state.window_value,state.window_tags,
            workspace.store_status[:fresh],state.spec.block_size,state.num_blocks,
            state.num_blocks*state.spec.conv_bytes,state.spec.ssm_bytes,
            state.snapshots.sink_tokens,state.snapshots.ring_tokens,
            float(impl.provider.config.get('k_clip_ratio',0.)),
            float(impl.provider.config.get('v_clip_ratio',0.)),state.hadamard)
    with phase('status_guard',**fields):
        ops.status_guard(status.view(-1),empty,workspace.merge_status[:fresh].view(-1),
                         workspace.store_status[:fresh].view(-1))
    return output
