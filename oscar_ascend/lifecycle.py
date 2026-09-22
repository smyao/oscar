"""Physical-page precise snapshots; archive #26/#37-49/#69.

Native block_pool.py/cache_full_blocks and FullAttentionManager cache only
complete immutable prefix pages. Native page IDs, rather than transient batch
rows, therefore own both INT2 data and the exact page-boundary snapshot.
No request lifecycle is inferred from a guessed materialized-length counter.
"""
from dataclasses import dataclass

from .layout import HybridPageLayout


@dataclass(frozen=True)
class SnapshotLayout:
    page: HybridPageLayout
    sink_tokens: int
    recent_tokens: int
    speculative_tokens: int

    def __post_init__(self):
        for name in ("sink_tokens", "recent_tokens", "speculative_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.ring_tokens <= 0:
            raise ValueError("precise recent ring must contain at least one token")
        if self.sink_tokens > self.page.block_size:
            raise ValueError("sink must fit the first logical FULL page")
        if self.ring_tokens > self.page.block_size:
            raise ValueError("recent plus speculative retention must fit a FULL page")
        if self.required_bytes > self.page.native_padding_bytes:
            raise ValueError(
                f"exact prefix snapshots need {self.required_bytes} bytes/page; "
                f"native padding provides {self.page.native_padding_bytes}")
        if self.page.native_padding_bytes % 8:
            raise ValueError("snapshot page stride must be int64 aligned")

    @property
    def ring_tokens(self):
        # At most speculative_tokens rejected writes may advance the physical
        # high-water mark beyond the next committed context. Retain them in
        # addition to R so that rollback never destroys the committed window.
        return self.recent_tokens + self.speculative_tokens

    @property
    def rows(self):
        return self.sink_tokens + self.ring_tokens

    @property
    def key_bytes(self):
        slot = self.page.slot_layout
        return self.rows * slot.num_kv_heads * slot.head_size * 2

    @property
    def value_bytes(self):
        slot = self.page.slot_layout
        return self.rows * slot.num_kv_heads * slot.head_size_v * 2

    @property
    def required_bytes(self):
        return self.key_bytes + self.value_bytes + self.rows * 8

    def base(self, num_blocks):
        if type(num_blocks) is not int or num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        # Native GDN views are SoA: all conv pages, then all SSM pages,
        # then the unused native padding. Never use physical_page * P.
        return num_blocks * (self.page.conv_bytes + self.page.ssm_bytes)

    def interval(self, num_blocks, physical_page):
        self.page._check_page(num_blocks, physical_page)
        start = self.base(num_blocks) + physical_page * self.page.native_padding_bytes
        return start, start + self.required_bytes

    def row_for_position(self, position):
        """Debug/address algebra; production mapping is performed on device."""
        if position < 0:
            raise ValueError("negative logical token position")
        if position < self.sink_tokens:
            return position
        return self.sink_tokens + (position % self.page.block_size) % self.ring_tokens

    def views(self, raw, num_blocks):
        """Make zero-copy typed views; never create a second BF16 KV pool."""
        import torch
        if raw.ndim != 1 or raw.stride() != (1,) or raw.element_size() != 1:
            raise ValueError("snapshot storage must be contiguous flat bytes")
        if raw.numel() != num_blocks * self.page.page_size_bytes:
            raise ValueError("snapshot allocation does not match the native pool")
        slot = self.page.slot_layout
        padding = self.page.native_padding_bytes
        base = raw.storage_offset() + self.base(num_blocks)
        half = raw.view(torch.bfloat16)
        keys = half.as_strided(
            (num_blocks, self.rows, slot.num_kv_heads, slot.head_size),
            (padding // 2, slot.num_kv_heads * slot.head_size, slot.head_size, 1),
            base // 2)
        values = half.as_strided(
            (num_blocks, self.rows, slot.num_kv_heads, slot.head_size_v),
            (padding // 2, slot.num_kv_heads * slot.head_size_v, slot.head_size_v, 1),
            (base + self.key_bytes) // 2)
        tags = raw.view(torch.int64).as_strided(
            (num_blocks, self.rows), (padding // 8, 1),
            (base + self.key_bytes + self.value_bytes) // 8)
        return keys, values, tags


def causal_segments(context: int, query_position: int, sink: int, recent: int):
    """Host algebra for independent tests/diagnostics, never a serving route.

    The current chunk is exact BF16 as in PR _prefill_attention. Cached
    context alone splits into global sink/history/recent at each query.
    """
    if min(context, query_position, sink, recent) < 0 or context > query_position:
        raise ValueError("invalid causal context/query geometry")
    sink_end = min(sink, context)
    cut = min(context, max(sink, query_position + 1 - recent))
    return ((0, sink_end), (sink_end, cut), (cut, context),
            (context, query_position + 1))
