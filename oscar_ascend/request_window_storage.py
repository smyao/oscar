"""Host geometry for exact request-window sidecars; no serving/kernel route.

Archive #26-49/#69: exact BF16 K/V follows stable request ownership, not batch
rows, physical KV pages, or mRoPE coordinates. #127-129: actual capacities and
metadata128 are separate. #137/#140: byte/host proofs are not device speed.
#155: tags and raw BF16 bits must not be compared with lossy float equality.

D.4: this addresses bounded window/metadata storage; the historical full
dequant6499.8-6655.1ms, prepare~725ms and stores~209ms must not return. Layout
is O(max_num_seqs*(sink+recent+spec)*Hkv*D), independent of history length.
No KV values are allocated, copied, dequantized or computed by this module.
The only writable tensor helper fills a small CPU metadata map. It does not
publish to NPU or prove GPU stream, graph, MTP numerical or service safety.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .request_windows import RequestWindowLeases, WindowBatch, WindowLeaseError

INT64_MAX = (1 << 63) - 1
INVALID_TAG = -1


def _integer(name: str, value: int, low: int, high: int = INT64_MAX) -> int:
    if type(value) is not int or not low <= value <= high:
        raise WindowLeaseError(f"{name} must be an integer in [{low},{high}]")
    return value


@dataclass(frozen=True)
class WindowAddress:
    """Byte offsets inside one FULL layer's sidecar raw allocation."""

    row: int
    key_offset: int
    value_offset: int
    tag_offset: int


@dataclass(frozen=True)
class RequestWindowLayout:
    max_num_seqs: int
    sink_tokens: int
    recent_tokens: int
    speculative_tokens: int
    num_kv_heads: int
    head_dim: int
    max_model_len: int

    def __post_init__(self):
        for name in ("max_num_seqs", "recent_tokens", "num_kv_heads", "max_model_len"):
            _integer(name, getattr(self, name), 1)
        for name in ("sink_tokens", "speculative_tokens"):
            _integer(name, getattr(self, name), 0)
        if type(self.head_dim) is not int or self.head_dim not in (64, 128, 256):
            raise WindowLeaseError("exact-window head_dim must be 64, 128 or 256")
        if self.position_bits >= 63:
            raise WindowLeaseError("logical position leaves no positive epoch bits in int64")
        if self.required_bytes > INT64_MAX or self.map_bytes > INT64_MAX:
            raise WindowLeaseError("sidecar byte addressing exceeds signed int64")

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, num_kv_heads: int, head_dim: int):
        return cls(config["max_num_seqs"], config["sink_tokens"], config["recent_tokens"],
                   config["speculative_config"]["num_speculative_tokens"],
                   num_kv_heads, head_dim, config["max_model_len"])

    @property
    def ring_tokens(self) -> int:
        return self.recent_tokens + self.speculative_tokens

    @property
    def rows(self) -> int:
        return self.sink_tokens + self.ring_tokens

    @property
    def kv_shape(self) -> tuple[int, int, int, int]:
        return self.max_num_seqs, self.rows, self.num_kv_heads, self.head_dim

    @property
    def tag_shape(self) -> tuple[int, int]:
        return self.max_num_seqs, self.rows

    @property
    def map_shape(self) -> tuple[int, int]:
        return self.max_num_seqs, 2  # columns: lease slot, epoch; both int64

    @property
    def kv_row_bytes(self) -> int:
        return self.num_kv_heads * self.head_dim * 2

    @property
    def kv_slot_bytes(self) -> int:
        return self.rows * self.kv_row_bytes

    @property
    def key_bytes(self) -> int:
        return self.max_num_seqs * self.kv_slot_bytes

    @property
    def tag_bytes(self) -> int:
        return self.max_num_seqs * self.rows * 8

    @property
    def required_bytes(self) -> int:
        return 2 * self.key_bytes + self.tag_bytes

    @property
    def map_bytes(self) -> int:
        return self.max_num_seqs * 2 * 8

    @property
    def position_bits(self) -> int:
        return max(1, (self.max_model_len - 1).bit_length())

    @property
    def position_stride(self) -> int:
        return 1 << self.position_bits

    @property
    def max_epoch(self) -> int:
        return INT64_MAX >> self.position_bits

    def make_leases(self) -> RequestWindowLeases:
        return RequestWindowLeases(self.max_num_seqs, max_epoch=self.max_epoch)

    def row_for_position(self, position: int) -> int:
        _integer("logical position", position, 0, self.max_model_len - 1)
        return position if position < self.sink_tokens else self.sink_tokens + position % self.ring_tokens

    def encode_tag(self, epoch: int, position: int) -> int:
        _integer("epoch", epoch, 1, self.max_epoch)
        _integer("logical position", position, 0, self.max_model_len - 1)
        return epoch * self.position_stride + position

    def decode_tag(self, tag: int) -> tuple[int, int]:
        _integer("tag", tag, 1, INT64_MAX)
        epoch, position = divmod(tag, self.position_stride)
        # Reject unused position bit patterns for non-power-of-two max lengths.
        if self.encode_tag(epoch, position) != tag:
            raise WindowLeaseError("invalid exact-window tag")
        return epoch, position

    def address(self, slot: int, position: int, head: int = 0) -> WindowAddress:
        _integer("lease slot", slot, 0, self.max_num_seqs - 1)
        _integer("KV head", head, 0, self.num_kv_heads - 1)
        row = self.row_for_position(position)
        key = slot * self.kv_slot_bytes + row * self.kv_row_bytes + head * self.head_dim * 2
        return WindowAddress(row, key, self.key_bytes + key,
                             2 * self.key_bytes + (slot * self.rows + row) * 8)

    def retained_ranges(self, materialized_end: int) -> tuple[range, range]:
        """Exclusive logical end; at most S+R+spec exact positions survive.

        Diagnostic/address algebra, never an online writer. A device writer
        must select only these positions *among the current chunk* and publish
        after its attention reads finish; otherwise ring writes can collide.
        """
        _integer("materialized_end", materialized_end, 0, self.max_model_len)
        return (range(min(self.sink_tokens, materialized_end)),
                range(max(self.sink_tokens, materialized_end - self.ring_tokens), materialized_end))

    def map_rows(self, batch: WindowBatch) -> tuple[tuple[int, int], ...]:
        """Validate host identities, then fill the complete fixed map shape."""
        if not isinstance(batch, WindowBatch):
            raise WindowLeaseError("window map requires a WindowBatch")
        actual = len(batch.request_ids)
        if (any(not isinstance(key, str) or not key for key in batch.request_ids) or
                len(set(batch.request_ids)) != actual or
                len(batch.slots) != len(batch.epochs) or
                not actual <= len(batch.slots) <= self.max_num_seqs):
            raise WindowLeaseError("invalid WindowBatch shape/identities")
        rows = []
        used = set()
        for index, (slot, epoch) in enumerate(batch.pairs):
            if index < actual:
                _integer("lease slot", slot, 0, self.max_num_seqs - 1)
                _integer("epoch", epoch, 1, self.max_epoch)
                if slot in used:
                    raise WindowLeaseError("two batch requests share an exact-window slot")
                used.add(slot)
            elif (type(slot) is not int or type(epoch) is not int or (slot, epoch) != (-1, 0)):
                raise WindowLeaseError("window map padding must be (-1,0)")
            rows.append((slot, epoch))
        rows.extend([(-1, 0)] * (self.max_num_seqs - len(rows)))
        return tuple(rows)

    def views(self, raw, *, testing: bool = False):
        """Zero-copy BF16 K/V and int64 tag views; no allocation or fill.

        The caller must budget and allocate raw once before cache profiling,
        and initialize tags to INVALID_TAG once. Never regenerate tags from
        compressed history or clear a newly leased slot over in-flight writes.
        """
        import torch
        if type(testing) is not bool:
            raise WindowLeaseError("testing must be an explicit boolean")
        if not torch.is_tensor(raw) or raw.dtype != torch.uint8 or raw.ndim != 1 or not raw.is_contiguous():
            raise WindowLeaseError("sidecar raw must be a contiguous one-dimensional uint8 tensor")
        if not testing and raw.device.type != "npu":
            raise WindowLeaseError("production sidecar views require NPU storage")
        if raw.numel() != self.required_bytes or raw.storage_offset() % 8:
            raise WindowLeaseError("sidecar raw byte size/alignment differs from layout")
        key = raw[:self.key_bytes].view(torch.bfloat16).view(self.kv_shape)
        value = raw[self.key_bytes:2 * self.key_bytes].view(torch.bfloat16).view(self.kv_shape)
        tags = raw[2 * self.key_bytes:].view(torch.int64).view(self.tag_shape)
        return key, value, tags


class FixedWindowMap:
    """Bind preallocated CPU staging and NPU [max_num_seqs,2] int64 maps.

    This helper only writes the CPU staging tensor. The runner must call it
    after native _update_states, inside synchronize_input_prep, then enqueue
    device_map.copy_(host_map, non_blocking=True) on the SAME current stream
    as every prior/next window kernel. The native context records its event
    after those copies and waits before the CPU staging memory is reused.
    Neither this class nor epoch tags establish that stream ordering.
    """

    def __init__(self, layout: RequestWindowLayout, host_map, device_map, *, testing: bool = False):
        import torch
        if type(testing) is not bool:
            raise WindowLeaseError("testing must be an explicit boolean")
        self.layout, self.host_map, self.device_map = layout, host_map, device_map
        for name, tensor in (("host", host_map), ("device", device_map)):
            if (not torch.is_tensor(tensor) or tensor.dtype != torch.int64 or
                    tuple(tensor.shape) != layout.map_shape or tensor.stride() != (2, 1)):
                raise WindowLeaseError(f"{name} window map must be contiguous int64{layout.map_shape}")
        if host_map.device.type != "cpu":
            raise WindowLeaseError("window-map staging must be CPU metadata")
        if not testing and (not host_map.is_pinned() or device_map.device.type != "npu"):
            raise WindowLeaseError("production map requires pinned CPU staging and NPU destination")
        if host_map.device == device_map.device and host_map.data_ptr() == device_map.data_ptr():
            raise WindowLeaseError("window-map source and destination must not alias")
        self._identities = self._signature()

    def _signature(self):
        return tuple((id(t), t.data_ptr(), tuple(t.shape), t.stride(), t.dtype, str(t.device))
                     for t in (self.host_map, self.device_map))

    def write_host(self, batch: WindowBatch) -> None:
        """One bounded metadata fill; caller owns native CPU reuse synchronization."""
        if self._signature() != self._identities:
            raise WindowLeaseError("fixed window-map address/shape changed")
        rows = self.layout.map_rows(batch)  # all errors before any buffer write
        import torch
        # One bounded CPU metadata staging tensor; no NumPy dependency, no
        # device readback, and no numerical KV processing on the host.
        self.host_map.copy_(torch.tensor(rows, dtype=torch.int64, device="cpu"))
