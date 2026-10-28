"""Verified fresh-suffix execution under the existing native-context proof.

Archive #126/#129/#142/#148/#155 and D.4: a proven zero-history suffix has only
exact current BF16 attention. Existing-prefix requests retain the complete
OSCAR path; every current K/V still enters the INT2 writer after attention.
No stored history is materialized or treated as BF16. Device checks reject
an incorrect host zero-context proof before executing current attention.
#34/#36/#127-129/#140: full-input buffers may exceed the historical-CV row
# budget. Validate the historical prefix separately before any prefix writes;
# the existing #142 dummy/capture and device context checks remain mandatory.
#13-20/#148/#164: legacy V4 exact-only rotates Q/K/V and applies one inverse.
#173: explicit native_exact policy keeps the proven fresh suffix in native
# attention arithmetic; its cache writer still uses the official rotated frame.
"""
from dataclasses import replace

import torch

from .current_attention import guard_current_slots, native_current_partial
from .guard_diag import run_status_guard
from .metadata import slice_window_map
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
    official_math=impl.provider.config.get("experimental_official_math",False)
    native_exact=(official_math and
        impl.provider.config.get("oscar_current_only_policy", "rotated_v4")=="native_exact")
    workspace=state.workspace
    workspace.validate_current_only(n,h,hk,d)
    if cut:
        workspace.validate(cut,h,hk,d)
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
                       current_only_plan=None, first_draft_current_only=False,
                       window_map=slice_window_map(metadata, 0, r))
        impl.forward(layer,query[:cut],key[:cut],value[:cut],kv_cache,prefix,
                     output=output[:cut])
    r=plan.prefix_requests
    window_map = slice_window_map(metadata, r, r + plan.fresh_requests)
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
        prepare_op=(ops.prepare_official_attention_tasks_out if official_math
                    else ops.prepare_attention_tasks_out)
        prepare_op(starts,metadata.seq_lens[r:r+plan.fresh_requests],
            slots,tasks,positions,h,hk,state.snapshots.sink_tokens,
            state.snapshots.recent_tokens,1,None,False)
        # Preserve all metadata failures. No host tensor reads or ignored
        # historical rows: a wrong zero-context proof is a same-stream trap.
        bad=(tasks[:,10]!=0)|((tasks[:,1]>=0)&(tasks[:,8]!=0))
        status=workspace.attention_status[:task_count]
        status.copy_(bad.to(torch.int32)[:,None])
        empty=workspace.store_status[:0].view(-1)
        run_status_guard(ops,'current_only.prepare',
                         [("metadata",status.view(-1)),("rotate",empty),("merge",empty),("store",empty)])
        guard_current_slots(ops,workspace.merge_status,slots,fresh)
        if actual<n:
            bad_padding=(metadata.slot_mapping[actual:n]>=0).to(torch.int32)
            run_status_guard(ops,'current_only.padding',
                             [("padding",bad_padding),("rotate",empty),("merge",empty),("store",empty)])
    with phase('current_native_fia',**fields):
        if native_exact:
            from .native_exact_current import verified_query
            attention_q=verified_query(ops,state,q,slots)
            attention_k,attention_v=k,v
        elif official_math:
            from .official_math import rotate_inputs
            attention_q,attention_k,attention_v=rotate_inputs(ops,state,q,k,v,slots,current_only=True)
        else:
            attention_q,attention_k,attention_v=q,k,v
        current,lse=native_current_partial(attention_q,attention_k,attention_v,plan.cumulative,
                                           heads=h,kv_heads=hk,scale=impl.scale)
    with phase('current_only_copy',**fields):
        if lse.dtype not in (torch.float16,torch.bfloat16,torch.float32):
            raise OscarReadinessError('current-only kernel received an unsupported native LSE dtype')
        # Canonical ND buffers, matching the existing copy into partials.
        # Official paths retain Q-validation status and separate FP32 LSE.
        # The legacy nonofficial path may reuse its unused rotate-status arena.
        if native_exact:
            from .native_exact_current import finalize_output
            finalize_output(ops,state,current,lse,output[cut:actual])
        elif official_math:
            from .official_math import finalize_output
            workspace.lse[:fresh].copy_(lse.squeeze(-1))
            finalize_output(ops,state,current.view(fresh,h,d),workspace.lse[:fresh],
                            output[cut:actual])
        else:
            current_nd=workspace.query_input[:fresh]
            current_nd.copy_(current)
            lse_nd=workspace.rotate_status[:fresh].view(torch.float32)
            # Match write_current_partial: native FP16/BF16 LSE is widened
            # before the FP32 copy ABI; V4 retains the query status separately.
            lse_nd.copy_(lse.squeeze(-1))
            ops.copy_validate_current_out(current_nd,lse_nd.view(fresh,h,1),
                output[cut:actual].view(fresh,h,d),workspace.lse[:fresh],
                workspace.merge_status[:fresh])
        if actual<n:
            output[actual:n].zero_()
    with phase('phase1_stores',**fields):
        from .cache_storage_ops import store_current
        store_current(ops,state,k,v,slots,positions,starts,window_map,impl.provider.config,
                      rotated_kv=(attention_k,attention_v) if official_math and not native_exact else None)
    with phase('status_guard',**fields):
        run_status_guard(ops,'current_only.final',
                         [("metadata",status.view(-1)),("rotate",workspace.rotate_status[:fresh] if official_math else empty),
                          ("merge",workspace.merge_status[:fresh].view(-1)),
                          ("store",workspace.store_status[:fresh].view(-1))])
        if official_math:
            if native_exact:
                from .native_exact_current import guard_output
                guard_output(ops,state,fresh)
            else:
                from .official_math import guard_final_output
                guard_final_output(ops,state,fresh)
    return output
