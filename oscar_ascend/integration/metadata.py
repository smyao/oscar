"""Native metadata validation; archive #34/#36/#37–49/#55–69/#77/#78/#140/#142.

The native runner already maintains query_start_loc_cpu. Carry that CPU mirror
for causal current-chunk FIA without an NPU readback; device metadata and
physical page tables remain on the native stream.
"""

from dataclasses import dataclass, replace
from typing import Any

from .dummy_context import is_native_dummy_run


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
    num_input_tokens: int | None = None
    draft_index: int = 0
    query_start_loc_cpu: Any = None
    attn_state: Any = None
    is_draft: bool = False
    current_cumulative: tuple[int, ...] | None = None
    dummy_origin: bool = False

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


def from_common(common: Any, *, capacity: MetadataCapacity | None = None,
                capture_origin: bool = False, is_draft: bool = False) -> OscarMetadata:
    if not common.causal:
        raise OscarMetadataError("OSCAR FULL attention requires causal decoder metadata")
    tensors = (common.query_start_loc, common.seq_lens, common.slot_mapping, common.block_table_tensor)
    current = MetadataCapacity.from_native_buffers(*tensors)
    query_start_loc = common.query_start_loc
    rows = common.num_reqs
    if (rows == current.rows + 1 and current.query_offsets == current.rows + 2
            and common.seq_lens.shape[0] == common.block_table_tensor.shape[0]):
        # Native FIA may append one synthetic request to qstarts after the
        # seq_lens/table row capacity is exhausted (attention_v1.py:323-338).
        # Its slots are negative; the device prepare kernel masks these rows
        # even when their token index is past the final real request offset.
        # A nonnegative slot outside this prefix is a device metadata error,
        # so truncating the synthetic offset cannot hide a real request.
        rows = current.rows
        query_start_loc = query_start_loc[:rows + 1]
    # Physical padding rows can exceed num_reqs. Preserve their complete native
    # buffers; kernels use num_reqs/num_actual_tokens and negative slot masking.
    current.validate(rows=rows, tokens=common.num_actual_tokens, columns=current.block_columns)
    if capacity is not None:
        capacity.validate(rows=current.rows, tokens=current.token_slots, columns=current.block_columns)
        if current.query_offsets > capacity.query_offsets:
            raise OscarMetadataError("native query-start buffer exceeds fixed graph capacity")
    state = getattr(common, "attn_state", None)
    dummy_origin = is_native_dummy_run()
    main_prefill = (not capture_origin and not dummy_origin and not is_draft and
                    type(state).__name__ == "AscendAttentionState" and
                    state.name in {"PrefillNoCache", "ChunkedPrefill", "PrefillCacheHit"})
    # Only this eager main-model stage needs CPU qstarts. Existing graph/draft
    # and decode callers never touch the CPU property (#34/#36/#140).
    cpu_starts = getattr(common, "query_start_loc_cpu", None) if main_prefill else None
    metadata = OscarMetadata(
        query_start_loc=query_start_loc, seq_lens=common.seq_lens,
        block_tables=common.block_table_tensor, slot_mapping=common.slot_mapping,
        num_reqs=rows, num_actual_tokens=common.num_actual_tokens,
        max_query_len=common.max_query_len, max_seq_len=common.max_seq_len,
        positions=getattr(common, "positions", None), capture_origin=capture_origin,
        num_input_tokens=getattr(common, "num_input_tokens", common.slot_mapping.shape[0]),
        query_start_loc_cpu=cpu_starts[:rows + 1] if cpu_starts is not None else None,
        attn_state=state, is_draft=is_draft, dummy_origin=dummy_origin)
    if main_prefill:
        # The native model runner constructs a new CommonAttentionMetadata per
        # step (model_runner_v1.py:3161). Every FULL builder sees that same
        # immutable step object; derive the short CPU list once, then share it
        # across layers. Never cache by tensor address, which is reused later.
        cumulative = getattr(common, "_oscar_current_cumulative", None)
        if cumulative is None:
            from .current_attention import current_cumulative_lengths
            cumulative = tuple(current_cumulative_lengths(metadata, common.num_actual_tokens))
            common._oscar_current_cumulative = cumulative
        if (not isinstance(cumulative, tuple) or not cumulative or
                cumulative[-1] != common.num_actual_tokens):
            raise OscarMetadataError("cached current sequence lengths do not match this native step")
        metadata = replace(metadata, current_cumulative=cumulative)
    return metadata


def buffer_signature(metadata: OscarMetadata):
    """Address evidence only; no metadata tensor values are read by Python."""
    rows = metadata.num_reqs
    tokens = metadata.num_input_tokens
    slots = metadata.slot_mapping if tokens is None else metadata.slot_mapping[:tokens]
    return tuple((tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
                 for tensor in (metadata.query_start_loc[:rows + 1], metadata.seq_lens[:rows],
                                metadata.block_tables[:rows], slots))


class GraphMetadataBindings:
    """The native runner updates these device buffers before each replay.

    Unlike FIA, direct kernels have no host list attributes to update after
    replay. Requiring the captured native tensor addresses ensures that an
    apparent successful graph replay cannot silently consume an older batch.
    """
    def __init__(self):
        self.captured = {}

    def bind(self, metadata):
        key = (metadata.num_input_tokens, metadata.num_reqs)
        signature = buffer_signature(metadata)
        previous = self.captured.get(key)
        if previous is not None and previous != signature:
            raise OscarMetadataError(
                f"native graph metadata buffers changed for tokens/requests={key}; "
                "OSCAR graph replay requires the captured device addresses")
        if metadata.capture_origin:
            self.captured[key] = signature
        return metadata
