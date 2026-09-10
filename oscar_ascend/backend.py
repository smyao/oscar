"""oscar_ascend.backend — AscendOSCAR 注意力实现（继承原生 impl，类外科手术后生效）。

零侵入接入点（见 plugin.py / plan §5.5）：插件把 FULL 层 impl 的 __class__ 换成
`AscendOscarAttentionBackendImpl`；本类只重写 forward / do_kv_cache_update / 存储与
读取路径，**不动** backend 类、metadata builder、allocator、页表、GDN 路径。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module, util

import torch

from .config import OscarAscendConfig
from .format import K_IDX_OFF, VALUES_PER_BYTE, check_d
from .kernels.decode_kernel import oscar_decode_ref
from .kernels.dequant_kernel import oscar_full_dequant
from .kernels.dequant_kernel import triton as _k_triton
from .kernels.prefill import (
    npu_prefill_prepared_batch,
    npu_prefill_packed,
    oscar_prefill,
    oscar_prefill_prepared,
)
from .kernels.prepare_kv import prepare_native_kv, prepare_native_kv_batch
from .kernels.store_kernel import oscar_store_ref
from .rotation import get_layer_rotation


def _cache_identity(value):
    """Return an identity token without reading forbidden inference state.

    ``Tensor._version`` raises (rather than returning no value) for tensors
    created in ``torch.inference_mode``. vLLM constructs attention metadata in
    that mode. The metadata object is the scheduler-step owner of these caches,
    so object identity remains the correct fallback for inference tensors.
    """
    try:
        version = value._version
    except (AttributeError, RuntimeError):
        version = None
    return id(value), version


def metadata_batch_lists(attn_metadata) -> tuple[list, list]:
    """提取 (query_start_loc, seq_lens) 的 host 列表——**禁止 Tensor 作布尔值**。

    真机 2026-09-04 05:21 教训：`(seq_lens_cpu or seq_lens)` 在 seq_lens_cpu 为
    **多元素 Tensor** 时抛 `Boolean value of Tensor with more than one value is
    ambiguous`（MTP 目标层 decode 走 __prefill_attention__ 时元数据为 chunked-prefill 形）。
    兼容：Tensor(任意设备，tolist 自动同步)、None 回退、_seq_lens_cpu、裸 list。
    """

    def _as_list(x, name: str) -> list:
        if x is None:
            raise RuntimeError(f"attn_metadata.{name} 缺失")
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x)

    source_key = (
        getattr(attn_metadata, "num_actual_tokens", None),
        _cache_identity(getattr(attn_metadata, "query_start_loc_cpu", None)),
        _cache_identity(getattr(attn_metadata, "actual_seq_lengths_q", None)),
        _cache_identity(getattr(attn_metadata, "query_start_loc", None)),
        _cache_identity(getattr(attn_metadata, "seq_lens_list", None)),
        _cache_identity(getattr(attn_metadata, "seq_lens_cpu", None)),
        _cache_identity(getattr(attn_metadata, "_seq_lens_cpu", None)),
        _cache_identity(getattr(attn_metadata, "seq_lens", None)),
    )
    cached = getattr(attn_metadata, "_oscar_batch_lists", None)
    if cached is not None and cached[0] == source_key:
        return cached[1]
    q_sl_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    ends = getattr(attn_metadata, "actual_seq_lengths_q", None)
    if q_sl_cpu is not None:
        qsl = _as_list(q_sl_cpu, "query_start_loc_cpu")
    elif ends is not None:
        qsl = [0] + _as_list(ends, "actual_seq_lengths_q")
    else:
        qsl = _as_list(attn_metadata.query_start_loc, "query_start_loc")
    seq_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    if seq_cpu is None:
        seq_cpu = getattr(attn_metadata, "_seq_lens_cpu", None)
    seq_list = getattr(attn_metadata, "seq_lens_list", None)
    seqs = _as_list(seq_list if seq_list is not None else (
        seq_cpu if seq_cpu is not None else attn_metadata.seq_lens), "seq_lens")
    result = (qsl, seqs)
    try:
        attn_metadata._oscar_batch_lists = (source_key, result)
    except Exception:
        pass
    return result


def metadata_token_positions(attn_metadata, device, n: int):
    """Cache token positions used identically by every attention layer."""
    qsl, seqs = metadata_batch_lists(attn_metadata)
    key = (str(device), n, tuple(qsl), tuple(seqs))
    cached = getattr(attn_metadata, "_oscar_token_positions", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    lengths = [max(0, min(end, n) - min(start, n))
               for start, end in zip(qsl, qsl[1:])]
    req = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        torch.tensor(lengths, device=device), output_size=n,
    )
    seq = torch.tensor(seqs[:len(lengths)], device=device)
    ends = torch.tensor(qsl[1:], device=device)
    token_seq = seq[req]
    pos = token_seq - ends[req] + torch.arange(n, device=device)
    result = (pos, token_seq)
    try:
        attn_metadata._oscar_token_positions = (key, result)
    except Exception:
        pass
    return result

class _CPUAttentionState(Enum):
    """Reference-only states when neither serving package is installed."""

    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


def _load_attention_types():
    if all(util.find_spec(name) is None for name in ("vllm", "vllm_ascend")):
        return object, _CPUAttentionState
    try:
        # Resolve the lazy platform before entering the native attention import
        # graph. Installed but broken vendor dependencies must never become CPU
        # placeholders, even when the failure is a transitive ImportError.
        _ = import_module("vllm.platforms").current_platform
        native = import_module("vllm_ascend.attention.attention_v1")
        return native.AscendAttentionBackendImpl, native.AscendAttentionState
    except Exception as exc:
        raise RuntimeError(
            "OSCAR could not import the installed Ascend attention backend; "
            "see the original exception above (CPU placeholders are disabled)."
        ) from exc


AscendAttentionBackendImpl, AscendAttentionState = _load_attention_types()


def staging_order(seats, capacity):
    # Seats are bounded by the arena, not by physical cache block IDs.
    # Every integer below 2**24 is exact in fp32. Ascend integer argsort
    # falls back to AiCPU; retain int64 for exceptionally large arenas.
    keys = seats.float() if capacity <= 2**24 else seats
    try:
        return torch.argsort(keys, stable=True)
    except TypeError:  # torch < 1.8, CPU-only compatibility path
        order = sorted(range(seats.numel()), key=lambda i: (seats[i].item(), i))
        return torch.tensor(order, dtype=torch.int64, device=seats.device)


def metadata_staging_plan(attn_metadata, device, n, block_size, stage_rows,
                          sink_tokens, recent_tokens):
    """Build layer-independent staging indices once per scheduler step.

    FULL layers have different staged values but share slot mapping, token
    positions and collision winners. Caching this plan avoids repeating the
    same nonzero/sort/window bookkeeping on every layer.
    """
    slot_source = attn_metadata.slot_mapping
    qsl, seqs = metadata_batch_lists(attn_metadata)
    key = (
        str(device), n, block_size, stage_rows, sink_tokens, recent_tokens,
        _cache_identity(slot_source),
        tuple(qsl), tuple(seqs),
    )
    cached = getattr(attn_metadata, "_oscar_staging_plan", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    slot = slot_source[:n].to(device=device, dtype=torch.int64)
    pos, token_seq = metadata_token_positions(attn_metadata, device, n)
    keep = ((pos >= token_seq - recent_tokens) | (pos < sink_tokens))
    valid = torch.nonzero(slot >= 0, as_tuple=True)[0]
    capacity = stage_rows * block_size
    seats = ((slot[valid] // block_size % stage_rows) * block_size
             + slot[valid] % block_size)
    order = staging_order(seats, capacity)
    sorted_seats = seats[order]
    last = torch.cat([
        sorted_seats[1:] != sorted_seats[:-1],
        torch.ones(min(1, sorted_seats.numel()), device=device, dtype=torch.bool),
    ])
    selected = valid[order[last]]
    seats = sorted_seats[last]
    rows, offsets = seats // block_size, seats % block_size
    retained = keep[selected]
    # Dense token -> staging-seat routing is shared by every FULL layer in the
    # scheduler step.  Building it here once lets the INT2 store kernel write
    # the retained fp32 value without a second gather/scatter kernel per layer.
    stage_seats = torch.full((n,), -1, dtype=torch.int64, device=device)
    stage_seats[selected[retained]] = seats[retained]
    result = (
        rows, offsets,
        torch.where(retained, slot[selected] // block_size, -1),
        selected[retained], rows[retained], offsets[retained], stage_seats,
    )
    try:
        attn_metadata._oscar_staging_plan = (key, result)
    except Exception:
        pass
    return result


def metadata_short_decode_layout(attn_metadata, device, actual, max_query_len=8):
    """Return per-query block-table rows and causal ends for short queries."""
    qsl, seqs = metadata_batch_lists(attn_metadata)
    qlens = [max(0, min(b, actual) - min(a, actual))
             for a, b in zip(qsl, qsl[1:])]
    active = [i for i, length in enumerate(qlens) if length]
    if (not active or max(qlens) > max_query_len
            or sum(qlens) != actual):
        return None
    bt = attn_metadata.block_tables
    key = (
        str(device), actual, max_query_len, tuple(qsl), tuple(seqs),
        _cache_identity(bt),
    )
    cached = getattr(attn_metadata, "_oscar_short_decode_layout", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    request_ids, causal_ends, fresh_starts, prefixes = [], [], [], []
    for i in active:
        q_len = qlens[i]
        prefix = max(0, int(seqs[i]) - q_len)
        request_ids.extend([i] * q_len)
        causal_ends.extend(prefix + j + 1 for j in range(q_len))
        fresh_starts.extend([qsl[i]] * q_len)
        prefixes.extend([prefix] * q_len)
    request_ids = torch.tensor(request_ids, dtype=torch.int64, device=bt.device)
    result = (
        bt.index_select(0, request_ids).contiguous(),
        torch.tensor(causal_ends, dtype=torch.int32, device=device),
        max(causal_ends),
        torch.tensor(fresh_starts, dtype=torch.int32, device=device),
        torch.tensor(prefixes, dtype=torch.int32, device=device),
    )
    try:
        attn_metadata._oscar_short_decode_layout = (key, result)
    except Exception:
        pass
    return result


def metadata_native_prepare(attn_metadata, active, prefixes, kv_ends, fresh_starts,
                            device):
    """Cache grouped prepare routing tensors once per scheduler step."""
    key = (str(device), tuple(active), tuple(prefixes), tuple(kv_ends),
           tuple(fresh_starts), _cache_identity(attn_metadata.block_tables))
    cache = getattr(attn_metadata, "_oscar_native_prepare", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    rows = attn_metadata.block_tables[active].contiguous()
    starts = [0] + list(kv_ends[:-1])
    tensors = (
        torch.tensor(prefixes, dtype=torch.int32, device=device),
        torch.tensor(starts + [kv_ends[-1]], dtype=torch.int32, device=device),
        torch.tensor(fresh_starts, dtype=torch.int32, device=device),
    )
    result = rows, tensors
    try:
        attn_metadata._oscar_native_prepare = (key, result)
    except Exception:
        pass
    return result


def metadata_decode_buckets(attn_metadata, device):
    """Host-metadata length buckets with cached device row indices."""
    _, seqs = metadata_batch_lists(attn_metadata)
    key = (str(device), tuple(seqs))
    cached = getattr(attn_metadata, "_oscar_decode_buckets", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    grouped = {}
    for row, length in enumerate(seqs):
        splits = 1 if length <= 2048 else 4 if length <= 8192 else 8 if length <= 32768 else 16
        grouped.setdefault(splits, []).append(row)
    result = tuple(
        (splits, torch.tensor(rows, dtype=torch.int64, device=device))
        for splits, rows in sorted(grouped.items())
    )
    try:
        attn_metadata._oscar_decode_buckets = (key, result)
    except Exception:
        pass
    return result


def metadata_grouped_mtp_layout(attn_metadata, device, q_starts, seq_lens):
    """Cache the four device vectors consumed by every grouped-MTP layer."""
    starts = tuple(int(x) for x in q_starts)
    seqs = tuple(int(x) for x in seq_lens)
    lengths = tuple(b - a for a, b in zip(starts, starts[1:]))
    key = (str(device), starts, seqs)
    cached = getattr(attn_metadata, "_oscar_grouped_mtp_layout", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    prefixes = tuple(s - n for s, n in zip(seqs, lengths))
    result = (
        torch.tensor(starts[:-1], dtype=torch.int32, device=device),
        torch.tensor(lengths, dtype=torch.int32, device=device),
        torch.tensor(prefixes, dtype=torch.int32, device=device),
        torch.tensor(
            [prefix + j + 1 for prefix, n in zip(prefixes, lengths)
             for j in range(n)],
            dtype=torch.int32, device=device,
        ),
    )
    try:
        attn_metadata._oscar_grouped_mtp_layout = (key, result)
    except Exception:
        pass
    return result


@dataclass(frozen=True)
class OscarExecutionPlan:
    """Layer-independent routing for one scheduler step.

    Request classification stays on the host; device row tensors are cached
    once on metadata and shared by every FULL attention layer.
    """

    actual_tokens: int
    q_starts: tuple[int, ...]
    seq_lens: tuple[int, ...]
    decode_requests: tuple[int, ...]
    multi_query_requests: tuple[int, ...]
    empty_requests: tuple[int, ...]
    decode_rows: torch.Tensor
    decode_tokens: torch.Tensor
    multi_query_rows: torch.Tensor

    @property
    def all_active_are_decode(self) -> bool:
        return bool(self.decode_requests) and not self.multi_query_requests


def metadata_execution_plan(attn_metadata, device) -> OscarExecutionPlan:
    """Build and cache the common decode/MTP/prefill routing decision."""
    actual = int(attn_metadata.num_actual_tokens)
    qsl, seqs = metadata_batch_lists(attn_metadata)
    q_starts = tuple(int(x) for x in qsl)
    seq_lens = tuple(int(x) for x in seqs)
    key = (str(device), actual, q_starts, seq_lens)
    cached = getattr(attn_metadata, "_oscar_execution_plan", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    decode, multi, empty = [], [], []
    for request, (start, end) in enumerate(zip(q_starts, q_starts[1:])):
        q_len = max(0, min(end, actual) - min(start, actual))
        (empty if q_len == 0 else decode if q_len == 1 else multi).append(request)
    plan = OscarExecutionPlan(
        actual_tokens=actual,
        q_starts=q_starts,
        seq_lens=seq_lens,
        decode_requests=tuple(decode),
        multi_query_requests=tuple(multi),
        empty_requests=tuple(empty),
        decode_rows=torch.tensor(decode, dtype=torch.int64, device=device),
        decode_tokens=torch.tensor(
            [q_starts[i] for i in decode], dtype=torch.int64, device=device
        ),
        multi_query_rows=torch.tensor(multi, dtype=torch.int64, device=device),
    )
    try:
        attn_metadata._oscar_execution_plan = (key, plan)
    except Exception:
        pass
    return plan


class AscendOscarAttentionBackendImpl(AscendAttentionBackendImpl):  # type: ignore[misc]
    """OSCAR INT2 FULL 层注意力实现（参考 OSCAR PR oscar_attn.py 的 impl 部分）。"""

    # ------------------------------------------------------------------ setup
    def _oscar_setup(self) -> None:
        if getattr(self, "_oscar_cfg", None) is not None:
            return
        cfg = OscarAscendConfig.from_env(head_dim=self.head_size)
        check_d(cfg.head_dim)
        if self.num_heads % self.num_kv_heads:
            raise ValueError("OSCAR requires Hq divisible by Hk")
        self._oscar_cfg = cfg
        self._oscar_rot_ready = False
        self._oscar_stage_ready = False
        self._oscar_warned_quantile = False
        self._oscar_use_triton = (
            self._oscar_cfg.use_triton and _k_triton is not None
        )
        # ★ 自证点 2：配置生效摘要（每层首次 setup 打一次）
        print(
            f"[oscar-ascend] ★ OSCAR 配置生效: D={self.head_size}, 逻辑槽=160B "
            f"(K 96B+V 64B), K旋转={'已加载' if self._oscar_cfg.k_rotation_path else '单位阵(未加载)'}, "
            f"V旋转={'已加载' if self._oscar_cfg.v_rotation_path else '单位阵(未加载)'}, "
            f"路径={self._oscar_cfg.k_rotation_path or '-'}, "
            f"triton={'启用' if self._oscar_use_triton else 'torch参考路径'}, "
            f"窗口(sink={self._oscar_cfg.sink_tokens}, recent={self._oscar_cfg.recent_tokens})"
        )
        self._oscar_stats = {"writes": 0, "kv_bytes_written": 0, "reads": 0}

    @property
    def _oscar(self) -> OscarAscendConfig:
        return self._oscar_cfg

    # ------------------------------------------------------------------ 旋转/裁剪
    def _clip_rotated(self, x_rot: torch.Tensor, clip_ratio: float) -> torch.Tensor:
        if clip_ratio > 0.0:
            if not self._oscar_warned_quantile:
                self._oscar_warned_quantile = True
                print(
                    "[oscar-ascend] 裁剪实现 = top-k分位数（阈值语义对齐论文内核；"
                    "不物化完整排序结果）"
                )
            try:
                # 论文内核同款：idx = int(ratio*D)，阈值 = 第 idx 大 |x|（per-vector）
                D = x_rot.shape[-1]
                idx = min(int(clip_ratio * D), D - 1)
                # Only the threshold is needed. For production ratios 0.96/0.92,
                # topk retains 11/21 values at D=256 instead of materializing a
                # complete sorted vector for every token and head.
                tail = max(1, D - idx)
                thr = torch.topk(x_rot.abs(), tail, dim=-1, largest=True,
                                 sorted=True).values[..., -1:]
                x_rot = torch.clamp(x_rot, -thr, thr)
            except Exception as e:  # pragma: no cover
                print(f"[oscar-ascend] sort 裁剪不可用，跳过: {e}")
        return x_rot

    def _rotate_clip(self, x: torch.Tensor, R: torch.Tensor, clip_ratio: float) -> torch.Tensor:
        return self._clip_rotated(torch.matmul(x.float(), R), clip_ratio)

    # ------------------------------------------------------------------ 写路径
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        slot_mapping: torch.Tensor,
        rotated: tuple[torch.Tensor, torch.Tensor] | None = None,
        staging=None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        slot_mapping = slot_mapping.to(device=key.device, dtype=torch.int64) if key is not None else slot_mapping
        N = slot_mapping.shape[0]
        if N <= 0 or key is None or value is None:
            return None
        D = self.head_size
        Hk = self.num_kv_heads
        self._set_caches(kv_cache)
        k_cache, v_cache = self.key_cache, self.value_cache
        k = key[:N].view(N, Hk, D)
        v = value[:N].view(N, Hk, D)
        raw_k, raw_v = rotated or (self._rotate_k(k, layer), self._rotate_v(v, layer))
        # The production Triton store performs clipping and quantization
        # reductions in its single write kernel.  Keep torch clipping only for
        # the numerical fallback path.
        k_rot, v_rot = raw_k, raw_v
        if not getattr(layer, "_oscar_wrote_once", False):
            layer._oscar_wrote_once = True
            # ★ 自证点 3：INT2 写路径真实执行（每层首写一次日志 + 字节统计）
            print(
                f"[oscar-ascend] ★ INT2 写路径首次执行: {layer.layer_name} "
                f"tokens={N} heads={Hk} — 每 token·head 逻辑槽 {160}B (原生KV {4 * D}B, "
                f"写入开销 -{(1 - 160 / (4 * D)) * 100:.1f}%)"
            )
        self._oscar_stats["writes"] += 1
        self._oscar_stats["kv_bytes_written"] += N * Hk * 160
        layer._oscar_store_staged = False
        if self._oscar_use_triton:
            try:
                from .kernels.store_kernel import oscar_store_triton

                stage_args = None
                if staging is not None:
                    stage_k, stage_v, owner, stage_seats = staging
                    stage_args = (
                        raw_k, raw_v, stage_k, stage_v, owner, stage_seats
                    )
                oscar_store_triton(
                    k_rot, v_rot, k_cache, v_cache, slot_mapping,
                    staging=stage_args,
                    k_clip_ratio=self._oscar.k_clip_ratio,
                    v_clip_ratio=self._oscar.v_clip_ratio,
                )
                layer._oscar_store_staged = stage_args is not None
                return raw_k, raw_v
            except Exception as e:  # pragma: no cover — triton 编译/执行失败则回退
                print(f"[oscar-ascend] triton store 失败，回退 torch 参考路径: {e}")
        k_rot = self._clip_rotated(raw_k, self._oscar.k_clip_ratio)
        v_rot = self._clip_rotated(raw_v, self._oscar.v_clip_ratio)
        oscar_store_ref(k_rot, v_rot, k_cache, v_cache, slot_mapping)
        return raw_k, raw_v

    def _set_caches(self, kv_cache) -> None:
        if isinstance(kv_cache, (tuple, list)) and len(kv_cache) >= 2:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        elif isinstance(kv_cache, torch.Tensor) and kv_cache.dim() > 0:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        assert self.key_cache is not None and self.value_cache is not None
        # 几何对账（DESIGN-20260904-E，每 impl 一次）：槽宽只允许
        #   head_size（packed×2：int8 几何，512B/token·head，block_size=1536）或
        #   2×head_size（legacy：bf16 几何，1024B/token·head，block_size=768）；
        # 且槽内落位（K 96B / V 64B）不得越界。fork 漂移/参数漏传在此拦截。
        if not getattr(self, "_oscar_geo_ok", False):
            k8 = self.key_cache.view(torch.uint8)
            v8 = self.value_cache.view(torch.uint8)
            if k8.ndim != 4 or v8.ndim != 4 or not k8.is_contiguous() or not v8.is_contiguous():
                raise ValueError("OSCAR requires contiguous [blocks, tokens, heads, bytes] caches")
            slot_k, slot_v = int(k8.stride(2)), int(v8.stride(2))
            need_k = K_IDX_OFF + self.head_size // VALUES_PER_BYTE  # 32+64=96
            need_v = self.head_size // VALUES_PER_BYTE  # 64
            assert slot_k in (self.head_size, 2 * self.head_size), (
                f"[oscar-ascend] 几何对账失败: k 槽宽 {slot_k}B（预期 "
                f"{self.head_size}=packed×2 或 {2 * self.head_size}=legacy）"
            )
            assert need_k <= slot_k and need_v <= slot_v, (
                f"[oscar-ascend] 槽容量不足: K 需 {need_k}B/V 需 {need_v}B "
                f"vs 槽 {slot_k}/{slot_v}B（布局契约 N-01 越界）"
            )
            self._oscar_geo_ok = True
            mode = (
                "packed×2（int8 几何：512B/token·head，block_size=1536，FULL 密度×2）"
                if slot_k == self.head_size
                else "legacy（bf16 几何：1024B/token·head，block_size=768，无显存收益）"
            )
            print(
                f"[oscar-ascend] ★ 几何对账: K 槽 {slot_k}B / V 槽 {slot_v}B → {mode}；"
                f"槽内落位 K{need_k}B+V{need_v}B 无越界"
            )

    def _layer_rots(self, layer: torch.nn.Module, device: torch.device):
        if not getattr(layer, "_oscar_rots", None):
            cfg = self._oscar
            rk = get_layer_rotation(cfg.k_rotation_path, layer.layer_name, self.head_size, device, mode="k", strict=True)
            rv = get_layer_rotation(cfg.v_rotation_path, layer.layer_name, self.head_size, device, mode="v", strict=True)
            layer._oscar_rots = (rk, rv)
            layer._oscar_rkT = rk.t().contiguous()
            layer._oscar_rvT = rv.t().contiguous()
            # Empty paths are defined by rotation.py as exact identities.  Do
            # not launch four GEMMs per layer merely to multiply by I.
            layer._oscar_identity_rk = not bool(cfg.k_rotation_path)
            layer._oscar_identity_rv = not bool(cfg.v_rotation_path)
        elif not hasattr(layer, "_oscar_identity_rk"):
            # Tests and external integrations may inject rotations directly;
            # conservatively treat those as real matrices.
            layer._oscar_identity_rk = False
            layer._oscar_identity_rv = False
            rk, rv = layer._oscar_rots
            layer._oscar_rkT = rk.t().contiguous()
            layer._oscar_rvT = rv.t().contiguous()
        return layer._oscar_rots

    def _rotate_k(self, value: torch.Tensor, layer) -> torch.Tensor:
        rk, _ = self._layer_rots(layer, value.device)
        return value if layer._oscar_identity_rk else value.float() @ rk

    def _rotate_v(self, value: torch.Tensor, layer) -> torch.Tensor:
        _, rv = self._layer_rots(layer, value.device)
        return value if layer._oscar_identity_rv else value.float() @ rv

    def _restore_v(self, value: torch.Tensor, layer,
                   dtype: torch.dtype | None = None) -> torch.Tensor:
        self._layer_rots(layer, value.device)
        out = value if layer._oscar_identity_rv else value.float() @ layer._oscar_rvT
        return out if dtype is None or out.dtype == dtype else out.to(dtype)

    # ------------------------------------------------------------------ 主入口
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("OSCAR fused output quantization is unsupported")
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        self._set_caches(kv_cache)
        state = getattr(attn_metadata, "attn_state", AscendAttentionState.ChunkedPrefill)
        execution = metadata_execution_plan(attn_metadata, query.device)

        # 1) 写路径：本步新 token K/V → INT2（decode/prefill 均写；前缀命中时旧前缀已在缓存）
        rotated = None
        if key is not None and value is not None:
            staging = None
            if self._oscar.window_enabled:
                self._ensure_staging(layer, kv_cache)
                plan = metadata_staging_plan(
                    attn_metadata, key.device, attn_metadata.num_actual_tokens,
                    self.stage_block, layer._oscar_stage_rows, self.sink_eff,
                    self._oscar.recent_tokens,
                )
                rows, offsets, owners = plan[:3]
                if not self._oscar_use_triton:
                    layer._oscar_slot_owner[rows, offsets] = owners
                staging = (
                    layer._oscar_stage_k, layer._oscar_stage_v,
                    layer._oscar_slot_owner, plan[6]
                )
            rotated = self.do_kv_cache_update(
                layer, key, value, kv_cache,
                attn_metadata.slot_mapping[: attn_metadata.num_actual_tokens],
                staging=staging,
            )
            if self._oscar.window_enabled and not getattr(
                    layer, "_oscar_store_staged", False):
                self._staging_write(layer, key, value, attn_metadata, rotated=rotated)

        # True decode has one query per request and can read paged INT2 KV
        # directly.  Multi-token MTP must stay grouped by request below so its
        # long history is not reread once per speculative query.
        if key is not None and value is not None and self._oscar_use_triton:
            layout = (metadata_short_decode_layout(
                attn_metadata, query.device, execution.actual_tokens,
                max_query_len=1,
            ) if execution.all_active_are_decode else None)
            if layout is not None:
                try:
                    attn_out = self._short_query_attention(
                        query[:attn_metadata.num_actual_tokens], layout, layer,
                        rotated=rotated,
                    )
                except Exception as exc:  # pragma: no cover - vendor/runtime fallback
                    if not getattr(self, "_oscar_warned_short_decode", False):
                        self._oscar_warned_short_decode = True
                        print(f"[oscar-ascend] short INT2 decode 失败，回退 dense FIA: {exc}")
                else:
                    output[:attn_metadata.num_actual_tokens] = attn_out.reshape(
                        output[:attn_metadata.num_actual_tokens].shape
                    ).to(output.dtype)
                    if attn_metadata.num_actual_tokens < num_tokens:
                        output[attn_metadata.num_actual_tokens:num_tokens].zero_()
                    return output

        # MTP/common speculative step: one request program handles q<=4 and a
        # GQA tile together, so every historical INT2 K/V vector is loaded and
        # dequantized once rather than once per speculative row.
        if (key is not None and value is not None and self._oscar_use_triton
                and self._oscar.use_grouped_mtp
                and execution.multi_query_requests
                and not execution.decode_requests and not execution.empty_requests
                # On the target 910B a single 15K request spent more than two
                # minutes in the 16 FULL-layer pure-Vector grouped path.  Past
                # the measured crossover, reconstruct once per request/layer
                # and use native FIA (Cube) instead of stalling generation.
                and max(execution.seq_lens, default=0)
                    <= self._oscar.grouped_mtp_max_seq_len
                and max(b - a for a, b in zip(
                    execution.q_starts, execution.q_starts[1:])) <= 4):
            try:
                from .kernels.paged_attention import oscar_grouped_mtp_attention_triton
                stage = None
                if self._oscar.window_enabled and self._oscar_stage_ready:
                    stage = (layer._oscar_stage_k, layer._oscar_stage_v,
                             layer._oscar_slot_owner)
                q_rot = self._rotate_k(
                    query[:execution.actual_tokens], layer
                )
                grouped = oscar_grouped_mtp_attention_triton(
                    q_rot, rotated[0][:execution.actual_tokens],
                    rotated[1][:execution.actual_tokens],
                    self.key_cache, self.value_cache, attn_metadata.block_tables,
                    list(execution.q_starts), list(execution.seq_lens),
                    self.scale, stage,
                    prepared_metadata=metadata_grouped_mtp_layout(
                        attn_metadata, query.device, execution.q_starts,
                        execution.seq_lens,
                    ),
                )
                attn_out = self._restore_v(grouped, layer, query.dtype)
            except Exception as exc:  # pragma: no cover - device gate/fallback
                if not getattr(self, "_oscar_warned_grouped_mtp", False):
                    self._oscar_warned_grouped_mtp = True
                    print(f"[oscar-ascend] grouped MTP paged 失败，回退 dense FIA: {exc}")
            else:
                output[:execution.actual_tokens] = attn_out.reshape(
                    output[:execution.actual_tokens].shape
                ).to(output.dtype)
                if execution.actual_tokens < num_tokens:
                    output[execution.actual_tokens:num_tokens].zero_()
                return output

        # DecodeOnly still supplies the current token's K/V in vLLM. It has
        # already been stored above, so route one-query requests directly to
        # paged INT2 decode instead of rebuilding dense history for FIA.
        if state == getattr(AscendAttentionState, "DecodeOnly", None):
            actual = execution.actual_tokens
            if execution.all_active_are_decode:
                attn_out = self._decode_attention(
                    query[:actual], kv_cache, attn_metadata, layer
                )
                output[:actual] = attn_out.reshape(output[:actual].shape).to(output.dtype)
                if actual < num_tokens:
                    output[actual:num_tokens].zero_()
                return output

        # Mixed batches are partitioned by the shared execution plan.  The
        # current paged kernel is safe here only for q_len=1; multi-query rows
        # stay on grouped FIA until the request/KV-head tiled kernel is ready.
        if (self._oscar_use_triton and key is not None and value is not None
                and execution.decode_requests
                and execution.multi_query_requests):
            out_short = self._decode_attention_rows(
                query.index_select(0, execution.decode_tokens),
                execution.decode_rows, execution.decode_requests,
                attn_metadata, layer,
            )
            attn_out = self._prefill_attention(
                query, key, value, kv_cache, attn_metadata, layer,
                request_indices=list(execution.multi_query_requests),
                rotated=rotated,
            )
            attn_out.index_copy_(0, execution.decode_tokens, out_short)
            output.copy_(attn_out.reshape(output.shape).to(output.dtype))
            return output

        # The vector paged kernel took ~4.35 s over 16 layers for a 4-token
        # MTP step on the target NPU. With fresh K/V, use dense reconstruction
        # plus native fused attention by default, including DecodeOnly.
        if key is not None and value is not None:
            attn_out = self._prefill_attention(
                query, key, value, kv_cache, attn_metadata, layer, rotated=rotated
            )
        elif state == getattr(AscendAttentionState, "DecodeOnly", None):
            attn_out = self._decode_attention(query, kv_cache, attn_metadata, layer)
        else:
            attn_out = self._prefill_attention(query, key, value, kv_cache, attn_metadata, layer)

        if output.ndim == 3:
            output[:num_tokens] = attn_out[:num_tokens].to(output.dtype)
        else:
            output[:num_tokens] = attn_out.reshape(num_tokens, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ decode
    def _decode_attention(self, query, kv_cache, attn_metadata, layer) -> torch.Tensor:
        if not getattr(layer, "_oscar_read_once", False):
            layer._oscar_read_once = True
            self._oscar_stats["reads"] += 1
            print(
                f"[oscar-ascend] ★ INT2 读路径(decode) 首次执行: {layer.layer_name} "
                f"seq_len={int(attn_metadata.seq_lens.max()) if attn_metadata.seq_lens.numel() else 0}, "
                f"窗口={'开' if self._oscar.window_enabled and self._oscar_stage_ready else '关(纯INT2)'}"
            )
        if (self._oscar.window_enabled and self._oscar_stage_ready
                and not self._oscar_use_triton):
            return self._decode_attention_windowed(query, kv_cache, attn_metadata, layer)
        q_rot = self._rotate_k(query, layer)
        bt = attn_metadata.block_tables
        seq = attn_metadata.seq_lens.to(device=q_rot.device, dtype=torch.int32)
        if self._oscar_use_triton:
            try:
                from .kernels.decode_kernel import oscar_decode_triton
                stage = None
                if self._oscar.window_enabled and self._oscar_stage_ready:
                    stage = (layer._oscar_stage_k, layer._oscar_stage_v,
                             layer._oscar_slot_owner)
                # Requests with very different lengths must not all inherit the
                # longest request's split/reducer cost.
                buckets = metadata_decode_buckets(attn_metadata, q_rot.device)
                if len(buckets) == 1:
                    out_rot, _ = oscar_decode_triton(
                        q_rot, self.key_cache, self.value_cache, bt, seq,
                        self.scale, self.num_kv_heads, self.head_size,
                        max_num_kv_splits=buckets[0][0], stage=stage,
                    )
                else:
                    out_rot = torch.empty_like(q_rot)
                    for bucket_splits, rows in buckets:
                        bucket_out, _ = oscar_decode_triton(
                            q_rot[rows], self.key_cache, self.value_cache,
                            bt[rows], seq[rows], self.scale, self.num_kv_heads,
                            self.head_size, max_num_kv_splits=bucket_splits,
                            stage=stage,
                        )
                        out_rot[rows] = bucket_out
            except Exception as e:  # pragma: no cover
                print(f"[oscar-ascend] triton decode 失败，回退 torch: {e}")
                out_rot, _ = oscar_decode_ref(
                    q_rot, self.key_cache, self.value_cache, bt, seq,
                    self.scale, self.num_kv_heads, self.head_size,
                )
        else:
            out_rot, _ = oscar_decode_ref(
                q_rot, self.key_cache, self.value_cache, bt, seq,
                self.scale, self.num_kv_heads, self.head_size,
            )
        return self._restore_v(out_rot, layer, query.dtype)

    def _decode_attention_rows(self, query, request_rows, host_rows,
                               attn_metadata, layer) -> torch.Tensor:
        """Decode selected q=1 requests without the legacy paged-row kernel."""
        from .kernels.decode_kernel import oscar_decode_triton

        q_rot = self._rotate_k(query, layer)
        bt = attn_metadata.block_tables.index_select(0, request_rows)
        seq = attn_metadata.seq_lens.to(
            device=q_rot.device, dtype=torch.int32
        ).index_select(0, request_rows)
        host_seqs = metadata_batch_lists(attn_metadata)[1]
        longest = max((int(host_seqs[i]) for i in host_rows), default=1)
        splits = (1 if longest <= 2048 else 2 if longest <= 8192
                  else 4 if longest <= 16384 else 8 if longest <= 32768 else 16)
        stage = None
        if self._oscar.window_enabled and self._oscar_stage_ready:
            stage = (layer._oscar_stage_k, layer._oscar_stage_v,
                     layer._oscar_slot_owner)
        out_rot, _ = oscar_decode_triton(
            q_rot, self.key_cache, self.value_cache, bt, seq,
            self.scale, self.num_kv_heads, self.head_size,
            max_num_kv_splits=splits, stage=stage,
        )
        return self._restore_v(out_rot, layer, query.dtype)

    def _short_query_attention(self, query, layout, layer, rotated=None):
        """True q=1 decode without dense historical KV reconstruction."""
        from .kernels.decode_kernel import oscar_decode_triton

        bt, seq, longest, fresh_starts, prefixes = layout
        q_rot = self._rotate_k(query, layer)
        # `longest` comes from cached host metadata, so split selection does
        # not synchronize the NPU seq tensor.
        splits = min(16, max(1, 1 << max(0, (longest - 1).bit_length() - 10)))
        stage = None
        if self._oscar.window_enabled and self._oscar_stage_ready:
            stage = (layer._oscar_stage_k, layer._oscar_stage_v,
                     layer._oscar_slot_owner)
        out_rot, _ = oscar_decode_triton(
            q_rot, self.key_cache, self.value_cache, bt, seq,
            self.scale, self.num_kv_heads, self.head_size,
            max_num_kv_splits=splits, stage=stage,
            fresh=None if rotated is None else (
                rotated[0][:query.shape[0]], rotated[1][:query.shape[0]],
                fresh_starts, prefixes,
            ),
        )
        return self._restore_v(out_rot, layer, query.dtype)

    # ------------------------------------------------------------------ prefill
    def _prefill_attention(self, query, key, value, kv_cache, attn_metadata, layer,
                           request_indices=None, rotated=None, _grouped=False,
                           _output=None, _q_rotated=None) -> torch.Tensor:
        N, Hq, D = query.shape
        Hk = self.num_kv_heads
        qsl_list, seq_lens_list = metadata_batch_lists(attn_metadata)
        output = (_output if _output is not None else
                  torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype))
        actual = min(N, getattr(attn_metadata, "num_actual_tokens", N))
        num_reqs = len(qsl_list) - 1
        indices = list(range(num_reqs) if request_indices is None else request_indices)
        self._layer_rots(layer, query.device)
        q_rotated = (_q_rotated if _q_rotated is not None else
                     self._rotate_k(query, layer).to(query.dtype))
        if (query.device.type == "npu" and self._oscar.use_batched_native
                and not _grouped and indices):
            groups, group, tokens = [], [], 0
            for i in indices:
                kv_tokens = max(0, int(seq_lens_list[i]))
                if group and tokens + kv_tokens > self._oscar.native_group_kv_tokens:
                    groups.append(group)
                    group, tokens = [], 0
                group.append(i)
                tokens += kv_tokens
            if group:
                groups.append(group)
            for group in groups:
                self._prefill_attention(
                    query, key, value, kv_cache, attn_metadata, layer,
                    request_indices=group, rotated=rotated, _grouped=True,
                    _output=output, _q_rotated=q_rotated,
                )
            return output
        rotated_k = rotated[0] if rotated is not None else self._rotate_k(key, layer)
        rotated_v = rotated[1] if rotated is not None else self._rotate_v(value, layer)
        if (query.device.type == "npu" and self._oscar.use_batched_native
                and _grouped and indices):
            q_parts, k_parts, v_parts = [], [], []
            prefixes, q_ends, kv_ends, active = [], [], [], []
            # Convert the whole group once. Per-request casts create two tiny
            # NPU kernels per sequence and are especially visible during MTP.
            native_k = rotated_k.to(query.dtype)
            native_v = rotated_v.to(query.dtype)
            for i in indices:
                start, end = qsl_list[i], min(qsl_list[i + 1], actual)
                if end <= start:
                    continue
                active.append(i)
                q_parts.append(q_rotated[start:end])
                k_parts.append(native_k[start:end])
                v_parts.append(native_v[start:end])
                prefix = max(0, int(seq_lens_list[i]) - (end - start))
                prefixes.append(prefix)
                q_ends.append((q_ends[-1] if q_ends else 0) + end - start)
                kv_ends.append(
                    (kv_ends[-1] if kv_ends else 0) + prefix + end - start
                )
            if q_parts:
                stage = None
                if self._oscar.window_enabled and self._oscar_stage_ready:
                    stage = (layer._oscar_stage_k, layer._oscar_stage_v,
                             layer._oscar_slot_owner)
                first_q = qsl_list[active[0]]
                last_q = min(qsl_list[active[-1] + 1], actual)
                contiguous_q = last_q - first_q == q_ends[-1]
                if contiguous_q and not any(prefixes):
                    # First prefill already has packed contiguous current K/V.
                    # Avoid allocating k_all/v_all and copying the whole prompt.
                    k_all = native_k[first_q:last_q]
                    v_all = native_v[first_q:last_q]
                    # With no historical prefix the Q and KV packed boundaries
                    # are identical; kv_ends was already built above.
                else:
                    fresh_starts = [qsl_list[i] for i in active]
                    rows, prep_metadata = metadata_native_prepare(
                        attn_metadata, active, prefixes, kv_ends,
                        fresh_starts, query.device,
                    )
                    k_all, v_all, kv_ends = prepare_native_kv_batch(
                        self.key_cache, self.value_cache, rows, prefixes,
                        k_parts, v_parts, stage,
                        use_triton=self._oscar_use_triton,
                        fresh_source=(native_k, native_v),
                        fresh_starts=fresh_starts,
                        prepared_metadata=prep_metadata,
                    )
                q_all = (q_rotated[first_q:last_q] if contiguous_q else
                         torch.cat(q_parts, dim=0).contiguous())
                out_all = npu_prefill_packed(
                    q_all, k_all, v_all, q_ends, kv_ends, self.scale, Hk, D
                )
                out_all = self._restore_v(out_all, layer, query.dtype)
                if contiguous_q:
                    output[first_q:last_q] = out_all
                else:
                    offset = 0
                    for i, q_part in zip(active, q_parts):
                        start = qsl_list[i]
                        length = q_part.shape[0]
                        output[start:start + length] = out_all[offset:offset + length]
                        offset += length
            return output
        prepared = []
        for i in indices:
            q_start, q_end = qsl_list[i], min(qsl_list[i + 1], actual)
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = rotated_k[q_start:q_end].to(query.dtype)
            v_seq = rotated_v[q_start:q_end].to(query.dtype)
            cached_len = seq_len - q_len
            if cached_len <= 0:
                q_rot = q_rotated[q_start:q_end]
                if query.device.type == "npu" and self._oscar.use_batched_native:
                    prepared.append((i, q_start, q_end, q_rot, k_seq.contiguous(), v_seq.contiguous()))
                    continue
                out = oscar_prefill(
                    q_rot, k_seq, v_seq,
                    torch.zeros(0, Hk, D, device=query.device),
                    torch.zeros(0, Hk, D, device=query.device),
                    self.scale, Hk, D,
                )
            elif self._oscar.use_fused_prep:
                stage = None
                if self._oscar.window_enabled and self._oscar_stage_ready:
                    stage = (layer._oscar_stage_k, layer._oscar_stage_v, layer._oscar_slot_owner)
                k_full, v_full = prepare_native_kv(
                    self.key_cache, self.value_cache, attn_metadata.block_tables[i], cached_len,
                    k_seq, v_seq,
                    stage, use_triton=self._oscar_use_triton,
                )
                prepared.append((i, q_start, q_end, q_rotated[q_start:q_end], k_full, v_full))
                continue
            else:
                bt_row = attn_metadata.block_tables[i]
                k_cached, v_cached = oscar_full_dequant(
                    self.key_cache, self.value_cache, bt_row, cached_len, Hk, D,
                    use_triton=self._oscar_use_triton,
                )
                if self._oscar.window_enabled and self._oscar_stage_ready:
                    k_cached, v_cached = self._stage_splice(
                        layer, bt_row, cached_len, k_cached, v_cached
                    )
                q_rot = q_rotated[q_start:q_end]
                if query.device.type == "npu" and self._oscar.use_batched_native:
                    prepared.append((
                        i, q_start, q_end, q_rot,
                        torch.cat((k_cached.to(query.dtype), k_seq), dim=0),
                        torch.cat((v_cached.to(query.dtype), v_seq), dim=0),
                    ))
                    continue
                out = oscar_prefill(
                    q_rot, k_seq, v_seq,
                    k_cached.to(query.dtype), v_cached.to(query.dtype),
                    self.scale, Hk, D,
                )
            output[q_start:q_end] = self._restore_v(out, layer, query.dtype)
        if prepared:
            groups, group, tokens = [], [], 0
            for item in prepared:
                kv_tokens = item[4].shape[0]
                if group and tokens + kv_tokens > self._oscar.native_group_kv_tokens:
                    groups.append(group)
                    group, tokens = [], 0
                group.append(item)
                tokens += kv_tokens
            if group:
                groups.append(group)
            for group in groups:
                if query.device.type == "npu" and self._oscar.use_batched_native and len(group) > 1:
                    outs = npu_prefill_prepared_batch(
                        [x[3] for x in group], [x[4] for x in group], [x[5] for x in group],
                        self.scale, Hk, D,
                    )
                else:
                    outs = [oscar_prefill_prepared(x[3], x[4], x[5], self.scale, Hk, D)
                            for x in group]
                for item, out in zip(group, outs):
                    output[item[1]:item[2]] = self._restore_v(
                        out, layer, query.dtype
                    )
        return output

    # ------------------------------------------------------------------ 窗口（BF16 sink/recent staging，port PR oscar_attn.py:245-336/618-750）
    def _ensure_staging(self, layer: torch.nn.Module, kv_cache) -> None:
        if getattr(layer, "_oscar_stage_ready", False):
            return
        bs = kv_cache[0].shape[1]
        cfg = self._oscar
        self.stage_block = bs
        self.sink_eff = (cfg.sink_tokens // bs) * bs
        self.sink_pages = self.sink_eff // bs
        self.tail_pages = (cfg.recent_tokens + bs - 1) // bs + 1
        rows = max(
            (cfg.staging_tokens + bs - 1) // bs,
            self.sink_pages + self.tail_pages + 2,
        )
        dev = kv_cache[0].device
        layer._oscar_stage_k = torch.zeros(rows, bs, self.num_kv_heads, self.head_size, dtype=torch.float32, device=dev)
        layer._oscar_stage_v = torch.zeros_like(layer._oscar_stage_k)
        layer._oscar_slot_owner = torch.full((rows, bs), -1, dtype=torch.int64, device=dev)
        layer._oscar_stage_rows = rows
        layer._oscar_stage_ready = True
        self._oscar_stage_ready = True

    def _staging_write(self, layer, key, value, attn_metadata, rotated=None) -> None:
        N = attn_metadata.num_actual_tokens
        if N == 0:
            return
        dev = key.device
        bs = self.stage_block
        rows, offsets, owner, selected, value_rows, value_offsets, _ = metadata_staging_plan(
            attn_metadata, dev, N, bs, layer._oscar_stage_rows,
            self.sink_eff, self._oscar.recent_tokens,
        )
        layer._oscar_slot_owner[rows, offsets] = owner
        if selected.numel() == 0:
            return
        # Preserve the model's unquantized K/V in fp32 rotated space. No clipping
        # or INT2 rounding is applied to staging, and historical tokens never
        # need to be rotated again on the read path.
        if rotated is None:
            k_rot = self._rotate_k(
                key[:N].reshape(N, self.num_kv_heads, self.head_size), layer
            )
            v_rot = self._rotate_v(
                value[:N].reshape(N, self.num_kv_heads, self.head_size), layer
            )
        else:
            k_rot, v_rot = rotated
        layer._oscar_stage_k[value_rows, value_offsets] = k_rot[selected]
        layer._oscar_stage_v[value_rows, value_offsets] = v_rot[selected]

    def _stage_splice(self, layer, bt_row, cached_len, k_cached, v_cached):
        bs = self.stage_block
        rows_total = layer._oscar_stage_rows
        dev = k_cached.device
        npg = (cached_len + bs - 1) // bs
        blk = bt_row[:npg].to(torch.int64)
        rows = blk % rows_total
        staged = layer._oscar_slot_owner[rows] == blk.unsqueeze(-1)
        pos = (torch.arange(npg, device=dev) * bs).unsqueeze(-1) + torch.arange(bs, device=dev)
        staged = (staged & (pos < cached_len)).reshape(-1)[:cached_len]
        ks = layer._oscar_stage_k[rows].reshape(npg * bs, self.num_kv_heads, -1)
        vs = layer._oscar_stage_v[rows].reshape(npg * bs, self.num_kv_heads, -1)
        m = staged.view(-1, 1, 1)
        k_out = torch.where(m, ks[:cached_len].to(k_cached.dtype), k_cached)
        v_out = torch.where(m, vs[:cached_len].to(v_cached.dtype), v_cached)
        return k_out, v_out

    def _decode_attention_windowed(self, query, kv_cache, attn_metadata, layer) -> torch.Tensor:
        B = query.shape[0]
        Hq, Hk, D = self.num_heads, self.num_kv_heads, self.head_size
        g = Hq // Hk
        bs = self.stage_block
        R = layer._oscar_stage_rows
        dev = query.device
        owner = layer._oscar_slot_owner
        bt = attn_metadata.block_tables
        seq = attn_metadata.seq_lens.to(device=dev, dtype=torch.int64)
        maxpg = bt.shape[1]
        S_nb, TP, S_eff = self.sink_pages, self.tail_pages, self.sink_eff
        W = self._oscar.recent_tokens
        offs = torch.arange(bs, device=dev)

        if S_nb > 0:
            sblk = bt[:, :S_nb].to(torch.int64)
            sown = owner[(sblk % R).unsqueeze(-1), offs.view(1, 1, bs)]
            s_staged = sown == sblk.unsqueeze(-1)
            sink_active = (seq > S_eff) & s_staged.reshape(B, -1).all(dim=1)
            spos = (torch.arange(S_nb, device=dev) * bs).view(1, S_nb, 1) + offs.view(1, 1, bs)
            s_valid = sink_active.view(B, 1, 1) & (spos < S_eff)
            s_valid = s_valid.expand(B, S_nb, bs)
        else:
            sblk = torch.zeros(B, 0, dtype=torch.int64, device=dev)
            s_valid = torch.zeros(B, 0, bs, dtype=torch.bool, device=dev)
            sink_active = torch.zeros(B, dtype=torch.bool, device=dev)
        si = torch.where(sink_active, torch.full_like(seq, S_nb), torch.zeros_like(seq))

        last_page = (seq - 1) // bs
        pg = (last_page - (TP - 1)).unsqueeze(1) + torch.arange(TP, device=dev).unsqueeze(0)
        pg_ok = pg >= 0
        tblk = torch.gather(bt.to(torch.int64), 1, pg.clamp(0, maxpg - 1))
        town = owner[(tblk % R).unsqueeze(-1), offs.view(1, 1, bs)]
        t_staged = (town == tblk.unsqueeze(-1)) & pg_ok.unsqueeze(-1)
        pos = (pg * bs).unsqueeze(-1) + offs.view(1, 1, bs)
        t0 = torch.maximum(seq - W, si * bs)
        inrange = (pos >= t0.view(B, 1, 1)) & (pos < seq.view(B, 1, 1))
        ok = torch.where(inrange, t_staged, torch.ones_like(t_staged))
        sv = (torch.flip(torch.cumprod(torch.flip(ok.reshape(B, -1).long(), [1]), 1), [1]) > 0)
        posf = pos.reshape(B, -1)
        cand = sv & inrange.reshape(B, -1)
        big = torch.iinfo(torch.int64).max
        cut = torch.where(cand, posf, torch.full_like(posf, big)).amin(1)
        cut = torch.minimum(cut, seq)
        t_valid = t_staged & inrange & (pos >= cut.view(B, 1, 1))

        # INT2 中段 [sink, cut)：块表按 sink 页平移（块表为 kernel 粒度，bs 即 kernel 块）
        seq_eff = (cut - si * bs).to(torch.int32)
        gidx = (torch.arange(maxpg, device=dev).unsqueeze(0) + si.unsqueeze(1)).clamp(max=maxpg - 1)
        bt_eff = torch.gather(bt, 1, gidx)
        q_rot = self._rotate_k(query, layer)
        out1, lse1 = self._oscar_int2_decode(q_rot, bt_eff, seq_eff.clamp(min=0))
        o1 = out1
        empty1 = (seq_eff <= 0).view(B, 1)
        lse1 = torch.where(
            empty1 | ~torch.isfinite(lse1),
            torch.full_like(lse1, float("-inf")), lse1,
        )
        o1 = torch.nan_to_num(o1)

        all_blk = torch.cat([sblk, tblk], dim=1)
        valid = torch.cat([s_valid, t_valid], dim=1)
        P = all_blk.shape[1]
        L = P * bs
        rowsP = all_blk % R
        kseg = layer._oscar_stage_k[rowsP].reshape(B, L, Hk, D).float()
        vseg = layer._oscar_stage_v[rowsP].reshape(B, L, Hk, D).float()
        vmask = valid.reshape(B, L)
        qh = q_rot.view(B, Hk, g, D)
        sc = torch.einsum("bkgd,blkd->bkgl", qh, kseg) * self.scale
        sc = sc.masked_fill(~vmask.view(B, 1, 1, L), float("-inf"))
        m2 = sc.amax(dim=-1)
        m2s = torch.where(torch.isfinite(m2), m2, torch.zeros_like(m2))
        p2 = torch.exp(sc - m2s.unsqueeze(-1))
        p2 = torch.where(vmask.view(B, 1, 1, L), p2, torch.zeros_like(p2))
        s2 = p2.sum(-1)
        o2 = torch.einsum("bkgl,blkd->bkgd", p2, vseg) / s2.clamp_min(1e-38).unsqueeze(-1)
        lse2 = torch.where(
            s2 > 0, m2s + torch.log(s2.clamp_min(1e-38)),
            torch.full_like(s2, float("-inf")),
        )
        o2 = o2.reshape(B, Hq, D)
        lse2 = lse2.reshape(B, Hq)

        new_lse = torch.logaddexp(lse1, lse2)
        safe_lse = torch.where(torch.isfinite(new_lse), new_lse, torch.zeros_like(new_lse))
        w1 = torch.exp(lse1 - safe_lse).unsqueeze(-1)
        w2 = torch.exp(lse2 - safe_lse).unsqueeze(-1)
        return self._restore_v(
            o1 * w1 + torch.nan_to_num(o2) * w2, layer, query.dtype
        )

    def _oscar_int2_decode(self, q_rot, bt_eff, seq_eff):
        """INT2 段 decode → (out_rot[B,Hq,D], lse[B,Hq])；seq_eff<=0 时返回空。"""
        if seq_eff.numel() == 0:
            B, Hq, D = q_rot.shape
            o = torch.zeros(B, Hq, D, device=q_rot.device)
            l = torch.full((B, Hq), float("-inf"), device=q_rot.device)
            return o, l
        if self._oscar_use_triton:
            try:
                from .kernels.decode_kernel import oscar_decode_triton

                return oscar_decode_triton(
                    q_rot, self.key_cache, self.value_cache, bt_eff, seq_eff,
                    self.scale, self.num_kv_heads, self.head_size,
                )
            except Exception:  # pragma: no cover
                pass
        return oscar_decode_ref(
            q_rot, self.key_cache, self.value_cache, bt_eff, seq_eff,
            self.scale, self.num_kv_heads, self.head_size,
        )
