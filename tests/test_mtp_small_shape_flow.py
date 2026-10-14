"""Archive #160-162: CPU numerical boundary test for small later-MTP shapes.

Real native metadata update, production impl and flat workspace views are used.
Fake device operations independently compute dense attention by physical slot;
they test Python ordering/shape contracts, not AscendC or compiled NPU behavior.
"""
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import torch

from oscar_ascend.integration.current_attention import use_native_current
from oscar_ascend.integration.draft_discard import native_draft_scope
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.ops.cv_dispatch import CLUSTER_CV_OPS, select_cv_op
from oscar_ascend.runtime import GraphWorkspace, WorkspaceGeometry
from test_first_draft_current_only import builder as make_builder
from test_first_draft_padded_native import Common
from test_native_discard_pipeline import _native
from test_runtime_native import definitions


def test_real_impl_native_later_batches_reuse_poisoned_flat_workspace(monkeypatch):
    native, states, modes = _native(monkeypatch)
    builder = make_builder(monkeypatch)
    forward = definitions("oscar_ascend/integration/impl.py", "small_mtp_flow", set(),
        {"torch": torch, "use_native_current": use_native_current,
         "OscarReadinessError": OscarReadinessError, "phase": lambda *a, **k: nullcontext(),
         "select_cv_op": select_cv_op, "CLUSTER_CV_OPS": CLUSTER_CV_OPS,
         "emit_once": lambda *a, **k: None, "emit_throttled": lambda *a, **k: None,
         "run_status_guard": lambda ops, site, parts, **k: ops.status_guard(*[x for _, x in parts])},
        monkeypatch, methods=[("OscarAttentionImpl", "forward")]).forward
    cap, h, d = 512, 6, 64
    w = object.__new__(GraphWorkspace)
    w.geometry = WorkspaceGeometry(cap, h, 1, d, cube_cores=20, history_cluster_size=16)
    for name, shape, dtype in (
        ("query_input", (cap,h,d), torch.bfloat16), ("key_input", (cap,1,d), torch.bfloat16),
        ("value_input", (cap,1,d), torch.bfloat16), ("query_rot", (cap,h,d), torch.float32),
        ("partial", (cap,h,3,d), torch.float32), ("partial_lse", (cap,h,3), torch.float32),
        ("output", (cap,h,d), torch.float32), ("lse", (cap,h), torch.float32),
        ("rotate_status", (cap,h), torch.int32), ("merge_status", (cap,h), torch.int32),
        ("store_status", (cap,1), torch.int32), ("tasks", (cap*3,16), torch.int64),
        ("attention_status", (cap*3,2), torch.int32), ("positions", (cap,), torch.int64),
        ("slots", (cap,), torch.int64), ("cv", (1,), torch.uint8)):
        setattr(w, name, torch.empty(shape, dtype=dtype))
    w.cluster_stats = None
    memory = {}
    current = {}

    class Ops:
        def prepare_attention_tasks_out(self, starts, lens, slots, tasks, positions,
                                        heads, kvheads, sink, recent, splits, table, slotctx):
            assert slotctx and heads == h and kvheads == 1
            current.update(starts=starts.clone(), lens=lens.clone(), slots=slots.clone(), table=table.clone())
            tasks.zero_(); positions.fill_(-1)
            for row in range(lens.numel()):
                slot = int(slots[row]); block, offset = divmod(slot,128)
                column = table[row].tolist().index(block)
                positions[row] = column * 128 + offset
                assert int(positions[row]) < int(lens[row])

        def rotate_out(self, q, rotation, out, status, hadamard, slots):
            out.zero_(); out[slots >= 0] = q[slots >= 0].float(); status.zero_()

        def attention_cv_mdl_decode_q1_out(self, q, qr, k, v, rotation, raw, table,
                wk, wv, tags, tasks, partial, lse, status, workspace, *attrs):
            partial.zero_(); lse.fill_(-torch.inf); status.zero_()
            for row in range(current["lens"].numel()):
                length = int(w.positions[row])
                addresses = [int(table[row,p//128])*128+p%128 for p in range(length)]
                keys = torch.stack([memory[a][0] for a in addresses] + [k[row,0].float()])
                values = torch.stack([memory[a][1] for a in addresses] + [v[row,0].float()])
                scores = q[row].float() @ keys.T / d**0.5
                # Independent partition, deliberately unlike kernel source
                # layout: the production merge/view must still combine it.
                for part, indices in enumerate(torch.tensor_split(torch.arange(length+1), lse.shape[-1])):
                    if indices.numel():
                        chunk = scores[:,indices]
                        partial[row,:,part] = chunk.softmax(-1) @ values[indices]
                        lse[row,:,part] = chunk.logsumexp(-1)

        def merge_lse_out(self, partial, lse, output, out_lse, status):
            valid = torch.isfinite(lse).any(-1)
            output.zero_(); out_lse.fill_(-torch.inf)
            output[valid] = (partial[valid] * lse[valid].softmax(-1)[...,None]).sum(-2)
            out_lse[valid] = lse[valid].logsumexp(-1); status.zero_()

        def rotate_clip_store_striped_out(self, key, value, rk, rv, slots, positions,
                                          raw, wk, wv, tags, status, *attrs):
            for row in range(current["lens"].numel()):
                memory[int(slots[row])] = (key[row,0].float().clone(), value[row,0].float().clone())
            status.zero_()

        def status_guard(self, *statuses):
            assert all(torch.count_nonzero(x) == 0 for x in statuses)

    raw = torch.empty(1,dtype=torch.uint8)
    state = SimpleNamespace(packed=raw, raw=raw, workspace=w, cache_format="striped_v1",
        rotation_k_transpose=torch.eye(d), rotation_v_transpose=torch.eye(d), rotation_v=torch.eye(d),
        hadamard=False, num_blocks=1000, window_key=raw, window_value=raw, window_tags=raw,
        spec=SimpleNamespace(block_size=2816, conv_bytes=15360, ssm_bytes=393216),
        snapshots=SimpleNamespace(sink_tokens=64,recent_tokens=256,ring_tokens=259,speculative_tokens=3))
    config = dict(experimental_striped_cache=True,experimental_mixed_cv=True,
        experimental_fast_unpack=True,experimental_decode_bundle=True,experimental_mdl_decode=True)
    impl = SimpleNamespace(provider=SimpleNamespace(config=config,ops=Ops(),layer_state=lambda _:state),
        num_heads=h,num_kv_heads=1,head_size=d,scale=d**-0.5)
    generator = torch.Generator().manual_seed(162)
    for batch,n in ((32,128),(31,128),(24,96),(12,48),(32,128)):
        starts=torch.arange(batch+1,dtype=torch.int32)*4
        lens=torch.full((batch,),130,dtype=torch.int32)
        table=torch.arange(batch*3,dtype=torch.int32).reshape(batch,3)+10
        proposer=SimpleNamespace(runner=SimpleNamespace(input_batch=SimpleNamespace(num_reqs=batch),
            discard_request_mask=SimpleNamespace(gpu=torch.zeros(batch,dtype=torch.bool)),pcp_manager=None),
            pcp_size=1,dcp_size=1,method="mtp",arange=torch.arange(n+1,dtype=torch.int32),
            token_arange_np=torch.arange(n+1,dtype=torch.int32),uses_mrope=False,has_gdn=True,
            kernel_block_size=128,max_model_len=262144,use_compress=False,
            slot_mapping_group=[torch.full((n,),-1,dtype=torch.int32) for _ in range(3)],
            seq_lens_group=[lens.clone() for _ in range(3)],
            query_start_loc_group=[torch.zeros(batch+1,dtype=torch.int32) for _ in range(3)])
        proposer.shallow_copy_metadata=MethodType(native.shallow_copy_metadata,proposer)
        original=Common(query_start_loc=starts,query_start_loc_cpu=starts.clone(),seq_lens=lens,
            seq_lens_cpu=None,_seq_lens_cpu=lens.clone(),seq_lens_cpu_upper_bound=lens.clone(),
            num_reqs=batch,num_actual_tokens=batch*4,num_input_tokens=n,max_query_len=4,
            num_computed_tokens_cpu=None,_num_computed_tokens_cpu=None,block_table_tensor=table,
            slot_mapping=proposer.slot_mapping_group[0],positions=torch.zeros(n,dtype=torch.int64),
            positions_cpu=None,attn_state=states.SpecDecoding,decode_token_per_req=4,
            is_prefilling=torch.zeros(batch,dtype=torch.bool),max_seq_len=130)
        rejected=torch.arange(batch,dtype=torch.int32)%4
        used=(lens-rejected-1).to(torch.int64)
        memory.clear()
        for row in range(batch):
            for pos in range(130):
                memory[int(table[row,pos//128])*128+pos%128] = tuple(torch.randn(d,generator=generator) for _ in range(2))
        with native_draft_scope(proposer,common=original,rejected_tokens=rejected):
            previous=original
            for index in (1,2):
                previous,metadata=native.attn_update_stack_num_spec_norm(proposer,index,None,previous,
                    batch,n,used,modes.NONE,attn_group=SimpleNamespace(get_metadata_builder=lambda:builder))
                q=torch.randn(n,h,d,generator=generator,dtype=torch.bfloat16)
                k=torch.randn(n,1,d,generator=generator,dtype=torch.bfloat16)
                v=torch.randn(n,1,d,generator=generator,dtype=torch.bfloat16)
                q[batch:]=torch.nan;k[batch:]=torch.nan;v[batch:]=torch.nan
                expected=[]
                for row in range(batch):
                    length=int(used[row])
                    history=[memory[int(table[row,pos//128])*128+pos%128] for pos in range(length)]
                    keys=torch.stack([entry[0] for entry in history]+[k[row,0].float()])
                    values=torch.stack([entry[1] for entry in history]+[v[row,0].float()])
                    expected.append((q[row].float()@keys.T/d**0.5).softmax(-1)@values)
                w.partial.fill_(torch.nan);w.partial_lse.fill_(torch.nan)
                w.attention_status.fill_(99);w.store_status.fill_(99)
                output=torch.full_like(q,torch.nan)
                forward(impl,SimpleNamespace(layer_name="mtp"),q,k,v,raw,metadata,output=output)
                torch.testing.assert_close(output[:batch].float(),torch.stack(expected),atol=0.004,rtol=0.004)
                assert torch.count_nonzero(output[batch:]) == 0
