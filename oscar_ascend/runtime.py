"""Concrete AscendC runtime, allocation and per-layer bindings.

Archive #27/#31-49: real operators, module-owned state, physical-page lifetime,
and fixed workspaces. #28/#77/#78: imported only after native initialization.
#86/#97: fingerprinted target rotations and explicit PR draft identity.
#69/#111: NPU-only production; exact BF16 projection and FP32 rotation.
#148: fe0 is the accepted precision baseline. Cluster4 is an explicit
# experimental operator with separate bounded state, never the default route.
#34/#36/#127-129/#140/#142/#145/#155: optional history-query capacity is
# independent of current-only input capacity; ordinary CV, dummy and capture
# retain strict capacity checks. Every buffer is reserved before KV profiling.
#148/#153: unqualified BF16 history requests fail before any device loading.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import warnings
from typing import Any

from .integration.runtime_api import OscarReadinessError
from .bf16_history_policy import validate_bf16_history, require_bf16_history_production
from .lifecycle import SnapshotLayout
from .request_window_storage import RequestWindowLayout, FixedWindowMap, INVALID_TAG

# Archive #144: mirrored by oscar_attention_launch.h and checked in contracts.
from .ops.cv_dispatch import ATTENTION_QUERY_ROWS
ATTENTION_KV_ROWS = 256

# Explicit experimental bundle only. Capability presence is not device/graph
# acceptance; the serving presets do not enable this unqualified layout.
REQUEST_WINDOW_CAPABILITIES = frozenset({
    "rotate_clip_store_request_window_out", "attention_request_window_decode_out",
    "attention_request_window_decode_q1_out", "attention_request_window_out",
    "attention_request_window_balanced_out", "attention_request_window_cluster4_out",
    "attention_request_window_cluster16_out",
})


def canonical_rotation_name(name: str) -> str | None:
    """Qwen3.5 multimodal wrapper prefixes do not alter layer identity."""
    target = re.search(r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.attn$", name)
    if target:
        return f"model.layers.{int(target[1])}.self_attn.attn"
    if re.search(r"(?:^|\.)mtp\.layers\.\d+\.self_attn\.attn$", name):
        return None
    raise OscarReadinessError(f"unrecognized target/draft FULL layer identity: {name}")


@dataclass(frozen=True)
class WorkspaceGeometry:
    tokens: int
    query_heads: int
    kv_heads: int
    head_dim: int
    # Capacity multiplier, not the per-call split count. Small captured token
    # shapes can use otherwise idle arena rows without enlarging this budget.
    splits: int = 1
    cube_cores: int = 1
    history_cluster_size: int = 1
    # None preserves every existing allocation. This limits current query rows
    # processed by historical CV, never the number of cached historical tokens.
    history_token_capacity: int | None = None
    # #148/#154: independent bounded pipeline arena, never the old C4 size.
    c4_member_pipeline: bool = False
    bf16_history: bool = False

    def __post_init__(self):
        dimensions = (self.tokens, self.query_heads, self.kv_heads, self.head_dim,
                      self.splits, self.cube_cores, self.history_cluster_size)
        if any(type(x) is not int or x <= 0 for x in dimensions):
            raise ValueError("workspace dimensions must be positive integers")
        if self.history_token_capacity is not None and (
                type(self.history_token_capacity) is not int or
                not 1 <= self.history_token_capacity <= self.tokens):
            raise ValueError("history token capacity must be an integer in [1,input capacity]")
        if self.query_heads % self.kv_heads or self.query_heads // self.kv_heads > 16:
            raise ValueError("CV query grouping requires integral GQA ratio <= 16")
        if self.head_dim not in (64, 128, 256) or self.splits > 32 or self.cube_cores > 32:
            raise ValueError("unsupported CV workspace geometry")
        if self.history_cluster_size not in (1, 4, 16):
            raise ValueError("history cluster size must be 1, 4 or 16")
        if type(self.c4_member_pipeline) is not bool:
            raise ValueError("C4 member pipeline must be an explicit boolean")
        if self.c4_member_pipeline and self.history_cluster_size not in (4, 16):
            raise ValueError("C4 member pipeline requires a cluster workspace")
        if type(self.bf16_history) is not bool:
            raise ValueError("BF16 history must be an explicit boolean")
        if self.bf16_history and self.head_dim != 256:
            raise ValueError("BF16 history workspace requires D256")

    @property
    def history_tokens(self):
        return self.tokens if self.history_token_capacity is None else self.history_token_capacity

    @property
    def task_count(self):
        # Current-only still prepares/guards three S1 sources for every input
        # row. CV can reinterpret the same arena using its bounded split count.
        return max(self.tokens, self.history_tokens * self.splits) * self.kv_heads * 3

    def splits_for_tokens(self, tokens: int) -> int:
        """Choose parallelism from captured shape and measured Cube count.

        Archive #34/#36/#70-73: no seq_len readback, length-based route switch
        or new decode-only allocation. The same shape always chooses the same
        S. A partition's cv_shape_tokens can retain the original larger input
        shape even though its live CV rows are smaller. partial_views validates
        those live rows against the bounded history arena before any CV call.
        """
        if type(tokens) is not int or not 1 <= tokens <= self.tokens:
            raise ValueError(f"active tokens must be in [1,{self.tokens}]")
        query_tile = ATTENTION_QUERY_ROWS // (self.query_heads // self.kv_heads)
        groups = ((tokens + query_tile - 1) // query_tile) * self.kv_heads
        cube_parallelism = (self.cube_cores + groups - 1) // groups
        arena_limit = self.history_tokens * self.splits // tokens
        return max(1, min(32, cube_parallelism, arena_limit))

    @property
    def cv_bytes(self):
        if not self.bf16_history:
            return self._fp32_cv_bytes
        # Archive #126/#145/#148/#154, D.4: separate compact typed arenas,
        # matching Bf16AttentionWorkspace in oscar_bf16_history_launch.h.
        # #173: BF16 main/residual operand planes preserve rounding information.
        # Every family keeps its own fixed tile protocol. MDL has two KV
        # buffers and can now exceed the generic single-buffer arena despite
        # its smaller query tile. Reserve the larger size before profiling.
        # Arena size remains independent of history length.
        m, n, d = ATTENTION_QUERY_ROWS, ATTENTION_KV_ROWS, self.head_dim
        c = self.history_cluster_size
        buffers, slots = (2, 2) if c == 4 else (1, 1)
        floats = 2*m*d + slots*m*n + (c*m*d + 2*c*m if c > 1 else 0)
        operands = c*m*d + buffers*2*n*d + slots*m*n
        selected_bytes = floats*4 + operands*4
        mdl_rows = 32
        mdl_floats = 2*mdl_rows*d + mdl_rows*n
        mdl_operands = mdl_rows*d + 4*n*d + mdl_rows*n
        mdl_bytes = mdl_floats*4 + mdl_operands*4
        return self.cube_cores * max(selected_bytes, mdl_bytes)

    @property
    def _fp32_cv_bytes(self):
        if self.history_cluster_size == 16:
            # Same bounded GM layout as the independent C16 kernel: sixteen
            # Q/FP32 state groups share one KV tile. This arena is allocated
            # once per device, then sequentially reused by all FULL layers.
            m, b, d = ATTENTION_QUERY_ROWS, ATTENTION_KV_ROWS, self.head_dim
            return self.cube_cores * (((16 * m + 2 * b + m + 16 * m) * d
                                       + m * b + 16 * 2 * m) * 4)
        if self.history_cluster_size == 4:
            # Original Q buffer holds group0; extra3 Q +4 FP32 accumulators
            # and4 independent max/sum arrays. Independent of history length.
            base = (1664 * self.head_dim + 33792) * 4
            if self.c4_member_pipeline:
                # Second KV256 pair and second Q128 x KV256 score/P slot.
                base += (2 * ATTENTION_KV_ROWS * self.head_dim
                         + ATTENTION_QUERY_ROWS * ATTENTION_KV_ROWS) * 4
            return self.cube_cores * base
        # Archive #71/#144, D.4: bounded Q128/KV256; acc stays in Vector UB.
        m, b = ATTENTION_QUERY_ROWS, ATTENTION_KV_ROWS
        return self.cube_cores * ((2 * m + 2 * b) * self.head_dim + m * b) * 4

    @property
    def buffer_bytes(self):
        """Exact allocation ledger; shared current-only buffers keep input size."""
        n, c = self.tokens, self.history_tokens
        h, k, d, p = self.query_heads, self.kv_heads, self.head_dim, 3 * self.splits
        return {
            "query_rot": c * h * d * 4,
            "query_input": n * h * d * 2,
            "key_input": n * k * d * 2,
            "value_input": n * k * d * 2,
            "partial": c * h * p * d * 4,
            "partial_lse": c * h * p * 4,
            "output": c * h * d * 4,
            "lse": n * h * 4,
            "merge_status": n * h * 4,
            "rotate_status": n * h * 4,
            "store_status": n * k * 4,
            "tasks": self.task_count * 16 * 8,
            "attention_status": self.task_count * 2 * 4,
            "positions": n * 8,
            "slots": n * 8,
            "cv": self.cv_bytes,
            "cluster_stats": self.cube_cores * 8 * 8 if self.history_cluster_size in (4, 16) else 0,
        }

    @property
    def total_bytes(self):
        return sum(self.buffer_bytes.values())


class GraphWorkspace:
    """One stream-ordered scratch arena reused across sequential FULL layers."""
    def __init__(self, geometry: WorkspaceGeometry, device):
        import torch
        if torch.device(device).type != "npu":
            raise OscarReadinessError("production graph workspace requires NPU")
        self.geometry = geometry
        n, h, d, p = geometry.tokens, geometry.query_heads, geometry.head_dim, 3 * geometry.splits
        c = geometry.history_tokens
        self.query_rot = torch.empty(
            (c, h, d),
            dtype=torch.float32,
            device=device)
        # qkv.split leaves a strided value projection in the native model.
        # These bounded current-step copies are reused; no historical KV is
        # ever copied here. Contiguous projections use their original storage.
        self.query_input = torch.empty((n, h, d), dtype=torch.bfloat16, device=device)
        self.key_input = torch.empty((n, geometry.kv_heads, d), dtype=torch.bfloat16, device=device)
        self.value_input = torch.empty_like(self.key_input)
        self.partial = torch.empty((c, h, p, d), dtype=torch.float32, device=device)
        self.partial_lse = torch.empty((c, h, p), dtype=torch.float32, device=device)
        self.output = torch.empty((c, h, d), dtype=torch.float32, device=device)
        # Current-only writes validated LSE and reuses rotate_status for its
        # widened native LSE; these status/LSE tensors require all input rows.
        self.lse = torch.empty((n, h), dtype=torch.float32, device=device)
        self.merge_status = torch.empty((n, h), dtype=torch.int32, device=device)
        self.rotate_status = torch.empty((n, h), dtype=torch.int32, device=device)
        self.store_status = torch.empty((n, geometry.kv_heads), dtype=torch.int32, device=device)
        self.tasks = torch.empty((geometry.task_count, 16), dtype=torch.int64, device=device)
        self.attention_status = torch.empty((geometry.task_count, 2), dtype=torch.int32, device=device)
        self.positions = torch.empty((n,), dtype=torch.int64, device=device)
        self.slots = torch.empty((n,), dtype=torch.int64, device=device)
        self.cv = torch.empty((geometry.cv_bytes,), dtype=torch.uint8, device=device)
        self.cluster_stats = (torch.empty((geometry.cube_cores, 8), dtype=torch.int64, device=device)
                              if geometry.history_cluster_size in (4, 16) else None)

    def validate(self, tokens, heads, kv_heads, dim):
        """Validate live ordinary CV rows, including dummy/capture rows."""
        self._validate_capacity(tokens, heads, kv_heads, dim, self.geometry.history_tokens)

    def validate_current_only(self, tokens, heads, kv_heads, dim):
        """Capacity check for a caller that separately proves zero history.

        This does not establish routing eligibility. The dispatcher retains
        eager/origin, metadata-context, slot and padding guards, and validates
        any historical prefix using validate() before executing that prefix.
        """
        self._validate_capacity(tokens, heads, kv_heads, dim, self.geometry.tokens)

    def _validate_capacity(self, tokens, heads, kv_heads, dim, capacity):
        g = self.geometry
        if (type(tokens) is not int or not 1 <= tokens <= capacity or
                (heads, kv_heads, dim) != (g.query_heads, g.kv_heads, g.head_dim)):
            raise OscarReadinessError(
                f"OSCAR fixed workspace mismatch: tokens={tokens}/{capacity}, "
                f"Hq/Hkv/D={(heads, kv_heads, dim)}/{(g.query_heads, g.kv_heads, g.head_dim)}")

    def projection(self, tensor, destination, tokens, heads, dim):
        view = tensor.view(tokens, heads, dim)
        if view.is_contiguous():
            return view
        destination[:tokens].copy_(view)
        return destination[:tokens]

    def partial_views(self, tokens, splits):
        """Reinterpret fixed flat arenas for this graph's static split count."""
        g = self.geometry
        if (type(tokens) is not int or type(splits) is not int or
                not 1 <= tokens <= g.history_tokens or not 1 <= splits <= 32 or
                tokens * splits > g.history_tokens * g.splits):
            raise OscarReadinessError("attention splits exceed the preallocated workspace")
        rows = tokens * g.query_heads * 3 * splits
        partial = self.partial.view(-1)[:rows * g.head_dim].view(
            tokens, g.query_heads, 3 * splits, g.head_dim)
        lse = self.partial_lse.view(-1)[:rows].view(tokens, g.query_heads, 3 * splits)
        return partial, lse


@dataclass(frozen=True)
class LayerState:
    raw: Any
    packed: Any
    spec: Any
    snapshots: SnapshotLayout | RequestWindowLayout
    window_key: Any
    window_value: Any
    window_tags: Any
    rotation_k: Any
    rotation_v: Any
    rotation_k_transpose: Any
    rotation_v_transpose: Any
    hadamard: bool
    num_blocks: int
    workspace: GraphWorkspace
    cache_format: str = "canonical_v1"
    window_map: Any = None
    window_status: Any = None


@dataclass(frozen=True)
class RequestWindowSidecar:
    layout: RequestWindowLayout
    raw: Any
    key: Any
    value: Any
    tags: Any
    status: Any


class AscendRuntimeProvider:
    """Production provider installed lazily by the external plugin factory."""
    def __init__(self, config=None):
        path = Path(os.environ.get("OSCAR_TARGET_CONFIG", Path(__file__).resolve().parents[1] / "configs/target.json"))
        supplied = dict(config) if config is not None else json.loads(path.read_text())
        try:
            from tools.serving_variants import production_runtime_config
            self.config = production_runtime_config(supplied)
            require_bf16_history_production(self.config)
        except ValueError as error:
            raise OscarReadinessError(str(error)) from error
        self._request_windows = self.config.get("experimental_request_windows", False)
        if type(self._request_windows) is not bool:
            raise OscarReadinessError("experimental_request_windows must be an explicit boolean")
        self._history_token_capacity = self.config.get("history_token_capacity")
        if self._history_token_capacity is not None and (
                type(self._history_token_capacity) is not int or self._history_token_capacity <= 0):
            raise OscarReadinessError("history token capacity must be an explicit positive integer or None")
        if type(self.config.get("experimental_history_reuse", False)) is not bool:
            raise OscarReadinessError("experimental_history_reuse must be an explicit boolean")
        if type(self.config.get("experimental_fast_unpack", False)) is not bool:
            raise OscarReadinessError("experimental_fast_unpack must be an explicit boolean")
        if type(self.config.get("experimental_mixed_cv", False)) is not bool:
            raise OscarReadinessError("experimental_mixed_cv must be an explicit boolean")
        if type(self.config.get("experimental_striped_cache", False)) is not bool:
            raise OscarReadinessError("experimental_striped_cache must be an explicit boolean")
        if type(self.config.get("experimental_current_only", False)) is not bool:
            raise OscarReadinessError("experimental_current_only must be an explicit boolean")
        for flag in ("experimental_first_mtp_current_fia", "experimental_compact_later_mtp",
                     "experimental_mixed_decode_split", "experimental_decode_bundle",
                     "experimental_quantization_bundle", "experimental_c4_member_pipeline",
                     "experimental_mdl_decode"):
            if type(self.config.get(flag, False)) is not bool:
                raise OscarReadinessError(f"{flag} must be an explicit boolean")
        if self.config.get("experimental_fast_unpack", False) and not self.config.get("experimental_history_reuse", False):
            raise OscarReadinessError("fast unpack requires explicit candidate history configuration")
        if self.config.get("experimental_mixed_cv", False) and not (
                self.config.get("experimental_fast_unpack", False) and
                self.config.get("experimental_history_reuse", False)):
            raise OscarReadinessError("mixed CV requires explicit fast/history candidate configuration")
        if self.config.get("experimental_striped_cache", False) and not (
                self.config.get("experimental_mixed_cv", False) and
                self.config.get("experimental_fast_unpack", False) and
                self.config.get("experimental_history_reuse", False)):
            raise OscarReadinessError("striped cache requires explicit mixed/fast/history candidate configuration")
        self._cache_format = ("striped_v1" if self.config.get("experimental_striped_cache", False)
                              else "canonical_v1")
        if self.config.get("experimental_decode_bundle", False) and self._cache_format != "striped_v1":
            raise OscarReadinessError("decode bundle requires the explicit striped cache bundle")
        if self.config.get("experimental_mdl_decode", False) and not self.config.get("experimental_decode_bundle", False):
            raise OscarReadinessError("MDL decode requires the explicit decode bundle")
        if self.config.get("experimental_mdl_decode", False) and self._request_windows:
            raise OscarReadinessError("MDL decode requires physical-page snapshots; request-window MDL reader is not qualified")
        if self.config.get("experimental_c4_member_pipeline", False) and not self.config.get("experimental_decode_bundle", False):
            raise OscarReadinessError("C4 member pipeline requires the explicit decode bundle")
        if self.config.get("experimental_mixed_decode_split", False) and not self.config.get("experimental_decode_bundle", False):
            raise OscarReadinessError("mixed decode partition requires the optimized decoder bundle")
        if self.config.get("experimental_mixed_decode_split", False) and not self.config.get("experimental_first_mtp_current_fia", False):
            raise OscarReadinessError("mixed decode partition requires the qualified first-MTP current path")
        if self._request_windows:
            if self._cache_format != "striped_v1":
                raise OscarReadinessError("request windows require the explicit striped candidate bundle")
            if self.config.get("experimental_decode_bundle") is not True:
                raise OscarReadinessError("request windows require the complete decode reader bundle")
            if self.config.get("scheduler_cls") is not None:
                raise OscarReadinessError("request windows retain the native scheduler without a custom scheduler_cls")
            if self._history_token_capacity is not None:
                raise OscarReadinessError("request windows use actual native token capacity, without a history override")
            if (self.config.get("experimental_bounded_whole_prefill", False) or
                    self.config.get("experimental_whole_prefill", False)):
                raise OscarReadinessError("request windows cannot enable a whole-prefill budget policy")
            request_capacity = self.config.get("max_num_seqs")
            if type(request_capacity) is not int or not 1 <= request_capacity <= 128:
                raise OscarReadinessError("request-window owner kernel requires max_num_seqs in [1,128]")
            if (type(self.config.get("max_model_len")) is not int or
                    not 1 <= self.config["max_model_len"] <= 1 << 31):
                raise OscarReadinessError("request-window tag writer supports up to 31 logical position bits")
            self._cache_format = "striped_request_window_v1"
        self._request_window_config = self._window_config_signature() if self._request_windows else None
        self._request_window_layers: dict[str, RequestWindowSidecar] = {}
        self._request_window_binding = None
        self._request_window_leases = None
        self._request_window_stream = None
        self._request_window_runner = None
        self._request_window_prep_runner = None
        self._request_windows_sealed = False
        self._ready = False
        self.layers: dict[str, LayerState] = {}
        self.workspaces: dict[tuple, GraphWorkspace] = {}
        self.rotations: dict[tuple, dict] = {}
        self.device_cores: dict[str, int] = {}
        self.ops = None

    def assert_ready(self):
        self._check_bf16_history_production()
        if (type(self.config.get("experimental_request_windows", False)) is not bool or
                self.config.get("experimental_request_windows", False) != self._request_windows):
            raise OscarReadinessError("request-window storage changed after runtime construction")
        if self.config.get("experimental_striped_cache", False) != self._cache_format.startswith("striped_"):
            raise OscarReadinessError("cache format changed after runtime construction")
        if self._request_windows and self._window_config_signature() != self._request_window_config:
            raise OscarReadinessError("request-window geometry changed after runtime construction")
        if self._ready:
            return
        from .ops.loader import require_production_ops
        devices = self.config.get("devices")
        if not isinstance(devices, list) or not devices:
            raise OscarReadinessError("configs/target.json must select current-task devices before NPU initialization")
        expected = ",".join(str(x) for x in devices)
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != expected:
            raise OscarReadinessError("visible NPU selection does not match the current target configuration")
        require_production_ops()
        if self.config.get("experimental_quantization_bundle", False):
            from .ops.loader import require_capabilities
            require_capabilities({"rotate_vector_init_out"})
        # Only current production readers are mandatory. Retired FP32 readers
        # remain compiled for independent diagnostics, never serving selection.
        from .ops.loader import require_capabilities
        from .ops.contracts import BF16_HISTORY_CAPABILITIES
        required = set(BF16_HISTORY_CAPABILITIES) | {"rotate_clip_store_striped_out"}
        if self.config.get("experimental_current_only", False):
            required.add("copy_validate_current_out")
        require_capabilities(required)
        import torch
        if not torch.npu.is_available():
            raise OscarReadinessError("OSCAR production requires an available NPU")
        self.ops = torch.ops.oscar_ascend_ops
        self._ready = True

    def _check_bf16_history_production(self):
        try:
            validate_bf16_history(self.config, production=True)
        except ValueError as error:
            raise OscarReadinessError(str(error)) from error

    def get_impl_cls(self):
        from .integration.impl import OscarAttentionImpl
        return OscarAttentionImpl

    @property
    def request_windows_enabled(self):
        return self._request_windows

    def _window_config_signature(self):
        return tuple(self.config[name] for name in
                     ("max_num_seqs", "sink_tokens", "recent_tokens", "max_model_len")) + (
            self.config["speculative_config"]["num_speculative_tokens"],)

    def _allocate_request_window_sidecar(self, layout):
        """Called inside native Attention construction and model memory profiler."""
        import torch
        device = torch.device("npu", torch.npu.current_device())
        raw = torch.empty(layout.required_bytes, dtype=torch.uint8, device=device)
        key, value, tags = layout.views(raw)
        tags.fill_(INVALID_TAG)
        status = torch.zeros(layout.max_num_seqs, dtype=torch.int32, device=device)
        return RequestWindowSidecar(layout, raw, key, value, tags, status)

    def _allocate_request_window_map(self, layout):
        import torch
        device = torch.device("npu", torch.npu.current_device())
        host = torch.empty(layout.map_shape, dtype=torch.int64, device="cpu", pin_memory=True)
        target = torch.empty(layout.map_shape, dtype=torch.int64, device=device)
        host[:, 0].fill_(-1); host[:, 1].zero_()
        target.copy_(host, non_blocking=False)  # initialization only
        self._request_window_stream = torch.npu.current_stream()
        return FixedWindowMap(layout, host, target)

    def register_request_window_layer(self, layer):
        """The native Attention has already registered its exact layer identity."""
        if not self._request_windows:
            return
        if self._request_windows_sealed:
            raise OscarReadinessError("FULL layer registered after request-window budget was sealed")
        name = layer.layer_name
        canonical_rotation_name(name)
        if name in self._request_window_layers:
            raise OscarReadinessError(f"request-window layer registered twice: {name}")
        impl = layer.impl
        if getattr(impl, "provider", None) is not self or impl.head_size != 256:
            raise OscarReadinessError("request windows require the actual OSCAR D256 FULL layer")
        layout = RequestWindowLayout.from_config(self.config, num_kv_heads=impl.num_kv_heads,
                                                head_dim=impl.head_size)
        sidecar = self._allocate_request_window_sidecar(layout)
        if self._request_window_binding is None:
            self._request_window_binding = self._allocate_request_window_map(layout)
            self._request_window_leases = layout.make_leases()
        elif (self._request_window_binding.layout.map_shape != layout.map_shape or
              self._request_window_binding.layout.position_bits != layout.position_bits):
            raise OscarReadinessError("FULL request-window identity geometry differs across layers")
        self._request_window_layers[name] = sidecar

    def seal_request_windows(self, runner):
        """After target + draft load: verify coverage, never allocate missing layers."""
        if not self._request_windows:
            return
        from .plugin import (request_window_runner_hooks_present,
                             request_window_scheduler_hooks_present)
        if not request_window_runner_hooks_present(runner):
            raise OscarReadinessError("request-window runner lifecycle hooks are incomplete")
        cache, parallel = runner.vllm_config.cache_config, runner.vllm_config.parallel_config
        scheduler_config = runner.vllm_config.scheduler_config
        scheduler = scheduler_config.get_scheduler_cls()
        selected_scheduler = f"{scheduler.__module__}.{scheduler.__qualname__}"
        if (scheduler_config.scheduler_cls is not None or
                self.config.get("scheduler_cls") is not None or
                selected_scheduler not in {
                    "vllm.v1.core.sched.scheduler.Scheduler",
                    "vllm.v1.core.sched.async_scheduler.AsyncScheduler"} or
                not request_window_scheduler_hooks_present(scheduler)):
            raise OscarReadinessError("request windows require the native scheduler and complete preemption notification hooks")
        if (runner.vllm_config.scheduler_config.max_num_seqs != self.config["max_num_seqs"] or
                runner.model_config.max_model_len != self.config["max_model_len"]):
            raise OscarReadinessError("native request or logical position capacity differs from sidecar reservation")
        if cache.enable_prefix_caching is not False or cache.mamba_cache_mode != "none":
            raise OscarReadinessError("request windows require prefix caching disabled and Mamba mode none")
        if (runner.vllm_config.kv_transfer_config is not None or
                getattr(runner.vllm_config, "ec_transfer_config", None) is not None) or any(
                getattr(parallel, name, 1) != 1 for name in
                ("pipeline_parallel_size", "data_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")):
            raise OscarReadinessError("request-window publication requires local TP with PP/DP/CP=1")
        if getattr(parallel, "enable_dbo", False):
            raise OscarReadinessError("request-window publication has no DBO stream contract")
        if runner.use_async_scheduling and runner.prepare_inputs_event is None:
            raise OscarReadinessError("async request-window publication lacks native input-prep event")
        context = runner.compilation_config.static_forward_context
        expected = {name for name, layer in context.items()
                    if getattr(getattr(layer, "impl", None), "provider", None) is self}
        if not expected or expected != set(self._request_window_layers):
            raise OscarReadinessError("registered FULL layer sidecars do not cover native target/draft layers")
        if (runner.vllm_config.speculative_config is not None and
                not any(canonical_rotation_name(name) is None for name in expected)):
            raise OscarReadinessError("native MTP FULL layer has no pre-profile request-window sidecar")
        self._request_window_runner = runner
        self._request_windows_sealed = True

    def begin_request_window_input_prep(self, runner):
        if not self._request_windows:
            return
        if (not self._request_windows_sealed or runner is not self._request_window_runner or
                self._request_window_prep_runner is not None):
            raise OscarReadinessError("request-window input-prep scope is missing or reentrant")
        self._request_window_prep_runner = runner

    def end_request_window_input_prep(self, runner):
        if self._request_windows:
            if self._request_window_prep_runner is not runner:
                raise OscarReadinessError("request-window input-prep ownership changed")
            self._request_window_prep_runner = None

    def publish_request_window_map(self, runner, scheduler_output):
        """After native condense/reorder, inside its protected input-prep scope."""
        if not self._request_windows:
            return
        if self._request_window_prep_runner is not runner:
            raise OscarReadinessError("request-window map publication requires native input-prep scope")
        import torch
        if torch.npu.current_stream() != self._request_window_stream:
            raise OscarReadinessError("request-window map changed compute stream without an event contract")
        batch = self._request_window_leases.update(
            scheduler_output, runner.input_batch.req_ids, padded_rows=self.config["max_num_seqs"])
        self._request_window_binding.write_host(batch)
        # Native synchronize_input_prep records its event after this H2D. It
        # protects pinned host reuse; same-stream order protects old KV writes.
        self._request_window_binding.device_map.copy_(self._request_window_binding.host_map,
                                                     non_blocking=runner.use_async_scheduling)

    def request_window_map(self, rows):
        if not self._request_windows:
            return None
        if (not self._request_windows_sealed or type(rows) is not int or
                not 0 <= rows <= self.config["max_num_seqs"]):
            raise OscarReadinessError("request-window metadata rows exceed sealed capacity")
        return self._request_window_binding.device_map[:rows]

    def ensure_workspace(self, heads, kv_heads, dim, device):
        """Reserve scratch during model loading, before native KV profiling."""
        self._check_bf16_history_production()
        try:
            bf16_history = validate_bf16_history(self.config, head_dim=dim,
                                                 cache_format=self._cache_format)
        except ValueError as error:
            raise OscarReadinessError(str(error)) from error
        configured_history = self.config.get("history_token_capacity")
        if (type(configured_history) is not type(self._history_token_capacity) or
                configured_history != self._history_token_capacity):
            raise OscarReadinessError("history token capacity changed after runtime construction")
        if self._cache_format.startswith("striped_") and dim != 256:
            raise OscarReadinessError("striped INT2 cache is valid only for D256")
        captures = self.config.get("compilation_config", {}).get("cudagraph_capture_sizes", [])
        capacity = max(int(self.config["max_num_batched_tokens"]), max(captures, default=0))
        geometry = WorkspaceGeometry(capacity, heads, kv_heads, dim,
                                     int(self.config.get("attention_splits", 1)),
                                     self._device_cube_cores(device),
                                     16 if self.config.get("experimental_mixed_cv", False) else
                                     4 if self.config.get("experimental_history_reuse", False) else 1,
                                     history_token_capacity=self._history_token_capacity,
                                     c4_member_pipeline=self.config.get("experimental_c4_member_pipeline", False),
                                     bf16_history=bf16_history)
        key = (str(device), geometry)
        if key not in self.workspaces:
            self.workspaces[key] = GraphWorkspace(geometry, device)
        self._prepare_rotations(dim, device)
        return self.workspaces[key]

    def _device_cube_cores(self, device):
        """Read native device properties once, never guess the active SoC.

        The native Triton utility queries the driver for hardware properties;
        it does not execute an OSCAR numerical kernel or change its AscendC
        implementation. The same initialized property source drives native
        GDN operators (triton_utils.py:46-67).
        """
        key = str(device)
        if key in self.device_cores:
            return self.device_cores[key]
        import torch
        from vllm_ascend.ops.triton.triton_utils import (
            get_aicore_num, get_vectorcore_num, init_device_properties_triton)
        selected = torch.device(device)
        if selected.type != "npu" or selected.index != torch.npu.current_device():
            raise OscarReadinessError("core discovery must use the current worker NPU")
        init_device_properties_triton()
        aic, aiv = int(get_aicore_num()), int(get_vectorcore_num())
        if aic <= 0 or aiv < 2 * aic:
            raise OscarReadinessError(f"OSCAR A2 CV requires two Vector cores per Cube: AIC={aic}, AIV={aiv}")
        requested = self.config.get("cube_cores")
        cores = min(aic, 32) if requested is None else requested
        if type(cores) is not int or not 1 <= cores <= min(aic, 32):
            raise OscarReadinessError(f"cube_cores={cores!r} exceeds measured AIC count {aic} or ABI maximum 32")
        self.device_cores[key] = cores
        from .telemetry import emit_once
        emit_once("device_cores", key=key, device=key, measured_aic=aic,
                  measured_aiv=aiv, selected_cube_cores=cores)
        return cores

    def transform_kv_cache_specs(self, runner, native_specs):
        from .integration.specs import transform_native_specs
        if self._request_windows:
            if not self._request_windows_sealed or runner is not self._request_window_runner:
                raise OscarReadinessError("request-window sidecars must be registered during model load before KV planning")
            converted = transform_native_specs(native_specs, request_windows=True)
        else:
            converted = transform_native_specs(native_specs)
        # Validate precise snapshot geometry before the scheduler budgets its
        # pool. A page advertised as compressed must also preserve its window.
        from .integration.specs import OscarFullAttentionSpec
        full_names = set()
        for name, spec in converted.items():
            if isinstance(spec, OscarFullAttentionSpec):
                full_names.add(name)
                if self._request_windows:
                    sidecar = self._request_window_layers.get(name)
                    if (sidecar is None or spec.head_size != spec.head_size_v or
                            sidecar.layout.num_kv_heads != spec.num_kv_heads or
                            sidecar.layout.head_dim != spec.head_size):
                        raise OscarReadinessError("native FULL spec differs from its pre-profile sidecar geometry")
                else:
                    self._snapshot_layout(spec)
        if self._request_windows and full_names != set(self._request_window_layers):
            raise OscarReadinessError("native FULL specs omit or add registered request-window layers")
        return converted

    def _snapshot_layout(self, spec):
        if spec.head_size != spec.head_size_v:
            raise OscarReadinessError("target CV FULL spec requires equal K/V head dimensions")
        return SnapshotLayout(
            spec.layout, int(self.config["sink_tokens"]), int(self.config["recent_tokens"]),
            int(self.config.get("speculative_config", {}).get("num_speculative_tokens", 0)))

    def allocate_kv_cache_tensors(self, runner, config, native_allocate):
        """Allocate each native shared tensor exactly once as raw bytes."""
        import torch
        from vllm.v1.kv_cache_interface import MambaSpec
        from .integration.specs import OscarFullAttentionSpec
        if torch.device(runner.device).type != "npu":
            raise OscarReadinessError("production KV allocation requires NPU")
        if runner.vllm_config.kv_transfer_config is not None:
            raise OscarReadinessError("target local-hybrid cache cannot be used by a KV transfer connector")
        specs = runner._get_layer_kv_cache_specs(config)
        raw = {}
        for tensor in config.kv_cache_tensors:
            if not all(isinstance(specs[name], (MambaSpec, OscarFullAttentionSpec)) for name in tensor.shared_by):
                raise OscarReadinessError("target shared cache pool contains an undeclared cache spec")
            allocation = torch.zeros(tensor.size, dtype=torch.int8, device=runner.device)
            for name in tensor.shared_by:
                if name in raw:
                    raise OscarReadinessError(f"layer appears in multiple native allocations: {name}")
                raw[name] = allocation
        runner.hybrid_with_attn_and_mamba = True
        return raw

    def _prepare_rotations(self, dim, device):
        import torch
        from .rotations import load_artifact, prepare_device_rotations, DeviceRotation
        cache_key = (str(device), dim)
        if cache_key in self.rotations:
            return self.rotations[cache_key]
        model = Path(self.config["model"])
        contents = (model / "config.json").read_bytes()
        model_config = json.loads(contents)
        text = model_config.get("text_config", model_config)
        layer_types = text.get("layer_types")
        count = text["num_hidden_layers"]
        if layer_types is None:
            interval = text.get("full_attention_interval")
            if type(interval) is not int or interval <= 0:
                raise OscarReadinessError("model config must declare FULL layer identities")
            layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                           for i in range(count)]
        if len(layer_types) != count:
            raise OscarReadinessError("model layer_types differs from num_hidden_layers")
        target = [f"model.layers.{i}.self_attn.attn" for i, kind in enumerate(layer_types)
                  if kind == "full_attention"]
        fingerprint = hashlib.sha256(contents)
        index = model / "model.safetensors.index.json"
        if index.is_file():
            fingerprint.update(index.read_bytes())
        artifact_path = self.config.get("rotation_artifact") or os.environ.get("OSCAR_ROTATION_ARTIFACT")
        if artifact_path is None:
            artifact_path = Path(__file__).resolve().parents[1] / "artifacts/rotations" / fingerprint.hexdigest() / "hadamard.pt"
        artifact = load_artifact(artifact_path, layer_names=target, head_dim=dim,
                                 model_fingerprint=fingerprint.hexdigest(), device=device)
        prepared = prepare_device_rotations(artifact, device=device)
        identity = torch.eye(dim, device=device, dtype=torch.float32)
        prepared[None] = DeviceRotation(identity, identity, identity, False)
        self.rotations[cache_key] = prepared
        return prepared

    def _rotations(self, specs, device):
        result = {}
        for name, spec in specs.items():
            canonical = canonical_rotation_name(name)
            prepared = self._prepare_rotations(spec.head_size, device)
            if canonical is None:
                # This is an explicit PR rule for the native MTP layer, whose
                # covariance is absent from the target-model calibration.
                warnings.warn(f"OSCAR MTP rotation is explicit PR identity: {name}", RuntimeWarning)
            if canonical not in prepared:
                raise OscarReadinessError(f"FULL layer has no matching rotation: {name}")
            result[name] = prepared[canonical]
        return result

    def reshape_kv_cache_tensors(self, runner, config, raw, native_reshape):
        import torch
        from .integration.specs import OscarFullAttentionSpec, packed_view
        specs = runner._get_layer_kv_cache_specs(config)
        full = {name: spec for name, spec in specs.items() if isinstance(spec, OscarFullAttentionSpec)}
        # Reuse the exact native GDN reshape on its original objects and bytes.
        # The native skip set is an existing initializer seam, restored even
        # if reshape raises. No GDN dimensions or dtype are reconstructed here.
        previous = runner.runner_only_attn_layers
        runner.runner_only_attn_layers = set(previous) | set(full)
        try:
            views = native_reshape(config, raw)
        finally:
            runner.runner_only_attn_layers = previous
        rotations = self._rotations(full, runner.device)
        scheduler = runner.vllm_config.scheduler_config
        capture_sizes = runner.vllm_config.compilation_config.cudagraph_capture_sizes or []
        token_capacity = max(scheduler.max_num_batched_tokens, max(capture_sizes, default=0))
        heads = runner.model_config.get_num_attention_heads(runner.parallel_config)
        for name, spec in full.items():
            allocation = raw[name]
            if not torch.is_tensor(allocation) or allocation.numel() % spec.page_size_bytes:
                raise OscarReadinessError(f"invalid native raw allocation for {name}")
            blocks = allocation.numel() // spec.page_size_bytes
            if blocks < config.num_blocks:
                raise OscarReadinessError("worker physical blocks are fewer than scheduler blocks")
            byte_raw = allocation.view(torch.uint8)
            if bool(getattr(spec, "request_window_storage", False)) != self._request_windows:
                raise OscarReadinessError("FULL spec and runtime disagree on request-window storage")
            if self._request_windows:
                if not self._request_windows_sealed:
                    raise OscarReadinessError("request-window storage was not budgeted during model load")
                sidecar = self._request_window_layers.get(name)
                if sidecar is None:
                    raise OscarReadinessError(f"FULL layer lacks pre-profile request-window allocation: {name}")
                packed = byte_raw  # genuine opaque storage owner; never a fake strided dual view
                snapshots = sidecar.layout
                wk, wv, tags = sidecar.key, sidecar.value, sidecar.tags
                window_map = self._request_window_binding.device_map
                window_status = sidecar.status
            else:
                packed = packed_view(byte_raw, blocks, spec.layout)
                snapshots = self._snapshot_layout(spec)
                wk, wv, tags = snapshots.views(byte_raw, blocks)
                tags.fill_(-1)
                window_map = window_status = None
            workspace = self.ensure_workspace(heads, spec.num_kv_heads, spec.head_size, runner.device)
            if workspace.geometry.tokens < token_capacity:
                raise OscarReadinessError("native token capacity exceeds the profiled OSCAR workspace")
            rotation = rotations[name]
            rk = rotation.key_transposed.T
            rv = rotation.inverse_value_transposed
            self.layers[name] = LayerState(byte_raw, packed, spec, snapshots, wk, wv, tags,
                                          rk, rv, rotation.key_transposed, rotation.value_transposed,
                                          rotation.hadamard, blocks, workspace, self._cache_format,
                                          window_map, window_status)
            views[name] = packed
            from .telemetry import emit_once
            emit_once("cache_layout", key=name, layer=name, physical_blocks=blocks,
                      physical_block_tokens=spec.block_size, page_bytes=spec.page_size_bytes,
                      cache_format=self._cache_format,
                      snapshot_bytes_per_page=0 if self._request_windows else snapshots.required_bytes,
                      request_window_sidecar_bytes=snapshots.required_bytes if self._request_windows else 0,
                      scratch_bytes=workspace.geometry.total_bytes,
                      raw_storage_ptr=byte_raw.data_ptr(), gdn_reshape="native")
        return views

    def layer_state(self, name):
        self._check_bf16_history_production()
        try:
            state = self.layers[name]
        except KeyError as error:
            raise OscarReadinessError(f"FULL layer cache was not initialized: {name}") from error
        if state.cache_format != self._cache_format:
            raise OscarReadinessError("FULL layer cache format differs from the runtime's immutable format")
        return state


def create_runtime():
    return AscendRuntimeProvider()
