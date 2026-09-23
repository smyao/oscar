"""Native AttentionImpl backed entirely by the external AscendC operators.

Archive #27/#34/#36/#37-49/#69/#111: complete forward dispatch, fixed buffers,
device metadata, exact physical-page snapshots, and no BF16 history restore.
PR oscar_attn.py:486-577: current-chunk K/V are exact; cached history is INT2.
Native acl_graph.py:270 and attention_v1.py:454: graph-update interface.
"""
import torch
from vllm.v1.attention.backend import AttentionImpl

from .runtime_api import OscarReadinessError, require_runtime
from ..telemetry import emit_once, emit_throttled
from ..timing import phase


class OscarAttentionImpl(AttentionImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                 alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                 logits_soft_cap=None, attn_type="decoder",
                 kv_sharing_target_layer_name=None, **kwargs):
        if attn_type != "decoder" or sliding_window is not None or alibi_slopes is not None:
            raise OscarReadinessError("OSCAR target requires causal FULL decoder attention without ALiBi")
        if logits_soft_cap not in (None, 0) or kv_sharing_target_layer_name is not None:
            raise OscarReadinessError("target OSCAR ABI does not declare softcap or cross-layer KV sharing")
        if self.total_cp_world_size != 1:
            raise OscarReadinessError("target OSCAR configuration requires context parallel world size 1")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        self.provider = require_runtime()
        self.provider.ensure_workspace(num_heads, self.num_kv_heads, head_size,
                                       torch.device("npu", torch.npu.current_device()))

    @staticmethod
    def update_graph_params(update_stream, forward_context, num_tokens, vllm_config,
                            speculative_config=None, num_dcp_pcp_tokens=None,
                            draft_attn_metadatas=None):
        """Direct kernels consume native GPU buffers updated before replay.

        FIA's update-stream task rebinding is inapplicable: this implementation
        has no host sequence-length attrs and records no external-event waits.
        The metadata builder checks captured buffer identities on every build.
        GDN's own metadata/kernel capture remains native and independent.
        """
        if draft_attn_metadatas is not None:
            raise OscarReadinessError("target MTP draft is eager; a merged draft graph is not configured")
        require_runtime()

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if output_scale is not None or output_block_scale is not None:
            raise OscarReadinessError("OSCAR output is BF16; fused output quantization is not declared")
        if output is None:
            raise OscarReadinessError("native OSCAR backend requires its preallocated output buffer")
        if attn_metadata is None:
            # Native memory profiling invokes attention without a cache or
            # real metadata. This is profiling, not a serving attention route.
            return output.zero_()
        if query.shape[0] == 0:
            return output
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise OscarReadinessError("target CV production ABI requires BF16 Q/K/V")
        state = self.provider.layer_state(layer.layer_name)
        if kv_cache.data_ptr() != state.packed.data_ptr() or kv_cache.stride() != state.packed.stride():
            raise OscarReadinessError("native FULL cache no longer matches its OSCAR physical allocation")
        n, h, hk, d = query.shape[0], self.num_heads, self.num_kv_heads, self.head_size
        if key.shape[0] != n or value.shape[0] != n:
            raise OscarReadinessError("Q/K/V padded token counts disagree")
        if attn_metadata.slot_mapping.shape[0] < n:
            raise OscarReadinessError("native slot_mapping must cover the complete padded query buffer")
        if attn_metadata.slot_mapping.dtype not in (torch.int32, torch.int64):
            raise OscarReadinessError("native slot mapping must be int32 or int64")
        workspace = state.workspace
        workspace.validate(n, h, hk, d)
        g = workspace.geometry
        source_splits = g.splits_for_tokens(n)
        splits = 3 * source_splits
        task_count = n * hk * splits
        tasks = workspace.tasks[:task_count]
        statuses = workspace.attention_status[:task_count]
        qr = workspace.query_rot[:n]
        partial, partial_lse = workspace.partial_views(n, source_splits)
        positions = workspace.positions[:n]
        ops = self.provider.ops
        with phase("prepare", layer=layer.layer_name, tokens=n, requests=attn_metadata.num_reqs,
                   draft_index=attn_metadata.draft_index, source_splits=source_splits):
            q = workspace.projection(query, workspace.query_input, n, h, d)
            k = workspace.projection(key, workspace.key_input, n, hk, d)
            v = workspace.projection(value, workspace.value_input, n, hk, d)
            slots = attn_metadata.slot_mapping[:n]
            if slots.dtype == torch.int32:
                # Native llm_base_proposer.py:1662 uses int32 draft slot groups.
                workspace.slots[:n].copy_(slots)
                slots = workspace.slots[:n]
            ops.prepare_attention_tasks_out(
                attn_metadata.query_start_loc[:attn_metadata.num_reqs + 1],
                attn_metadata.seq_lens[:attn_metadata.num_reqs],
                slots, tasks, positions, h, hk,
                state.snapshots.sink_tokens, state.snapshots.recent_tokens, source_splits,
                attn_metadata.block_tables[:attn_metadata.num_reqs] if attn_metadata.draft_index > 0 else None,
                attn_metadata.draft_index > 0)
        with phase("rotate", layer=layer.layer_name, tokens=n, hadamard=state.hadamard,
                   requests=attn_metadata.num_reqs):
            ops.rotate_out(q, state.rotation_k_transpose, qr, workspace.rotate_status[:n], state.hadamard, slots)
        with phase("fia", layer=layer.layer_name, tokens=n, max_query_len=attn_metadata.max_query_len,
                   max_seq_len=attn_metadata.max_seq_len, splits=source_splits,
                   cube_cores=g.cube_cores, tasks=task_count, requests=attn_metadata.num_reqs):
            ops.attention_cv_out(
                q, qr, k, v, state.rotation_v, state.raw, attn_metadata.block_tables,
                state.window_key, state.window_value, state.window_tags, tasks,
                partial, partial_lse, statuses, workspace.cv,
                state.spec.block_size, state.num_blocks,
                state.num_blocks * state.spec.conv_bytes, state.spec.ssm_bytes,
                state.snapshots.sink_tokens, state.snapshots.recent_tokens,
                state.snapshots.speculative_tokens, source_splits, self.scale, g.cube_cores)
        with phase("merge", layer=layer.layer_name, tokens=n, splits=splits,
                   requests=attn_metadata.num_reqs):
            ops.merge_lse_out(
                partial.view(n * h, splits, d), partial_lse.view(n * h, splits),
                workspace.output[:n].view(n * h, d), workspace.lse[:n].view(n * h),
                workspace.merge_status[:n].view(n * h))
            output.view(n, h, d).copy_(workspace.output[:n])
        # Read before write is essential for a chunk longer than the ring:
        # every query sees the previous exact window until attention completes.
        # Current K/V was read directly; only now publish the new physical data.
        with phase("phase1_stores", layer=layer.layer_name, tokens=n, hadamard=state.hadamard,
                   requests=attn_metadata.num_reqs):
            ops.rotate_clip_store_out(
                k, v, state.rotation_k_transpose, state.rotation_v_transpose,
                slots, positions, state.raw,
                state.window_key, state.window_value, state.window_tags,
                workspace.store_status[:n], state.spec.block_size, state.num_blocks,
                state.num_blocks * state.spec.conv_bytes, state.spec.ssm_bytes,
                state.snapshots.sink_tokens, state.snapshots.ring_tokens,
                float(self.provider.config.get("k_clip_ratio", 0.0)),
                float(self.provider.config.get("v_clip_ratio", 0.0)), state.hadamard)
        with phase("status_guard", layer=layer.layer_name, tokens=n,
                   requests=attn_metadata.num_reqs):
            ops.status_guard(statuses, workspace.rotate_status[:n],
                             workspace.merge_status[:n], workspace.store_status[:n])
        emit_once("attention_dispatched", key=(layer.layer_name, n, attn_metadata.max_query_len),
                  layer=layer.layer_name, tokens=n, requests=attn_metadata.num_reqs,
                  max_query_len=attn_metadata.max_query_len,
                  max_seq_len=attn_metadata.max_seq_len,
                  source_splits=source_splits,
                  capture_origin=attn_metadata.capture_origin,
                  route="ascendc_int2_cv", device_completion="not_observed_here")
        emit_throttled("attention_progress", key=layer.layer_name,
                       layer=layer.layer_name, tokens=n, requests=attn_metadata.num_reqs,
                       max_query_len=attn_metadata.max_query_len,
                       max_seq_len=attn_metadata.max_seq_len,
                       route="ascendc_int2_cv", device_completion="not_observed_here")
        return output
