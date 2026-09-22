"""Shape-only metadata validation; archive #34/#36/#37–49/#77/#78.

No .item(), .cpu(), .tolist(), or per-request assembly is used. In
particular the deprecated CommonAttentionMetadata CPU properties are
never read. Tensor values remain on the native device and stream.
"""

from dataclasses import dataclass
from typing import Any


class OscarMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class OscarMetadata:
    query_start_loc: Any
    seq_lens: Any
    block_tables: Any
    slot_mapping: Any
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    positions: Any = None
    capture_origin: bool = False

    @property
    def block_table(self):
        return self.block_tables


@dataclass(frozen=True)
class MetadataCapacity:
    rows: int
    query_offsets: int
    token_slots: int
    block_columns: int

    @classmethod
    def from_native_buffers(cls, query_start_loc, seq_lens, slot_mapping, block_tables):
        if len(block_tables.shape) != 2:
            raise OscarMetadataError("native block table must have rank 2")
        if any(len(t.shape) != 1 for t in (query_start_loc, seq_lens, slot_mapping)):
            raise OscarMetadataError("native lengths, query starts, and slots must have rank 1")
        return cls(min(seq_lens.shape[0], block_tables.shape[0]), query_start_loc.shape[0],
                   slot_mapping.shape[0], block_tables.shape[1])

    def validate(self, *, rows: int, tokens: int, columns: int) -> None:
        if rows < 0 or tokens < 0 or columns < 0:
            raise OscarMetadataError("negative metadata dimension")
        if rows > self.rows or rows + 1 > self.query_offsets or tokens > self.token_slots or columns > self.block_columns:
            raise OscarMetadataError(
                "OSCAR metadata capacity exceeded: "
                f"rows={rows}/{self.rows}, query_offsets={rows + 1}/{self.query_offsets}, "
                f"tokens={tokens}/{self.token_slots}, block_columns={columns}/{self.block_columns}")


def from_common(common: Any, *, capacity: MetadataCapacity | None = None, capture_origin: bool = False) -> OscarMetadata:
    if not common.causal:
        raise OscarMetadataError("OSCAR FULL attention requires causal decoder metadata")
    tensors = (common.query_start_loc, common.seq_lens, common.slot_mapping, common.block_table_tensor)
    current = MetadataCapacity.from_native_buffers(*tensors)
    # Physical padding rows can exceed num_reqs. Preserve their complete native
    # buffers; kernels use num_reqs/num_actual_tokens and negative slot masking.
    current.validate(rows=common.num_reqs, tokens=common.num_actual_tokens, columns=current.block_columns)
    if capacity is not None:
        capacity.validate(rows=current.rows, tokens=current.token_slots, columns=current.block_columns)
        if current.query_offsets > capacity.query_offsets:
            raise OscarMetadataError("native query-start buffer exceeds fixed graph capacity")
    return OscarMetadata(
        query_start_loc=common.query_start_loc, seq_lens=common.seq_lens,
        block_tables=common.block_table_tensor, slot_mapping=common.slot_mapping,
        num_reqs=common.num_reqs, num_actual_tokens=common.num_actual_tokens,
        max_query_len=common.max_query_len, max_seq_len=common.max_seq_len,
        positions=getattr(common, "positions", None), capture_origin=capture_origin)
