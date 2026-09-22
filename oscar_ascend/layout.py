"""Host-only cache geometry; no tensor values are inspected.

Archive #17–20: byte offsets and fp16 metadata must agree with the PR.
Archive #34/#36: graph and speculative capacities come from real buffers.
Archive #37–49: physical identity is separate from batch-row identity.
Native evidence: model_runner_v1.py:4696 stores GDN state as SoA, not AoS.
Archive #127: uncached GDN's request-span block is not a FULL page alignment.
"""

from dataclasses import dataclass
from math import lcm


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div needs a nonnegative value and positive divisor")
    return (value + divisor - 1) // divisor


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class SlotLayout:
    """PR INT2 format, one independent quantizer per token and local head.

    Each K/V region contains codes, little-endian fp16 scale, fp16 zero.
    This describes storage only; numerical rounding belongs in the kernels.
    """

    head_size: int
    head_size_v: int | None = None
    num_kv_heads: int = 1

    def __post_init__(self) -> None:
        _positive("head_size", self.head_size)
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)
        _positive("head_size_v", self.head_size_v)
        _positive("num_kv_heads", self.num_kv_heads)

    @property
    def k_code_bytes(self) -> int:
        return ceil_div(self.head_size, 4)

    @property
    def v_code_bytes(self) -> int:
        return ceil_div(self.head_size_v, 4)

    @property
    def k_scale_offset(self) -> int:
        return self.k_code_bytes

    @property
    def k_zero_offset(self) -> int:
        return self.k_code_bytes + 2

    @property
    def v_codes_offset(self) -> int:
        return self.k_code_bytes + 4

    @property
    def v_scale_offset(self) -> int:
        return self.v_codes_offset + self.v_code_bytes

    @property
    def v_zero_offset(self) -> int:
        return self.v_scale_offset + 2

    @property
    def slot_bytes(self) -> int:
        return self.k_code_bytes + self.v_code_bytes + 8

    @property
    def token_bytes(self) -> int:
        return self.num_kv_heads * self.slot_bytes

    @property
    def bf16_token_bytes(self) -> int:
        return 2 * self.num_kv_heads * (self.head_size + self.head_size_v)


@dataclass(frozen=True)
class HybridPageLayout:
    """Packed FULL pages occupy their own native SSM page's byte interval.

    The raw tensor remains ``nb * native_page_size_bytes`` bytes. GDN
    physical page p uses two ranges: ``[p*C,(p+1)*C)`` and
    ``[nb*C+p*M,nb*C+(p+1)*M)``. FULL page p uses only the latter.
    Different groups must receive disjoint physical page IDs from the
    native block pool; sharing a tensor does not permit sharing a live ID.
    """

    slot_layout: SlotLayout
    conv_bytes: int
    ssm_bytes: int
    native_mamba_block_size: int
    kernel_block_size: int = 128
    native_page_size_bytes: int | None = None
    native_mamba_cache_mode: str = "align"

    def __post_init__(self) -> None:
        for name in ("conv_bytes", "ssm_bytes", "native_mamba_block_size", "kernel_block_size"):
            _positive(name, getattr(self, name))
        if self.native_mamba_cache_mode not in {"none", "align", "all"}:
            raise ValueError(f"unsupported native Mamba cache mode: {self.native_mamba_cache_mode!r}")
        if self.native_page_size_bytes is None:
            object.__setattr__(self, "native_page_size_bytes", self.conv_bytes + self.ssm_bytes)
        _positive("native_page_size_bytes", self.native_page_size_bytes)
        if self.native_page_size_bytes < self.conv_bytes + self.ssm_bytes:
            raise ValueError("native padded page is smaller than its conv/SSM states")
        if self.block_size == 0:
            raise ValueError(
                "native SSM page cannot contain one aligned OSCAR block: "
                f"ssm_bytes={self.ssm_bytes}, token_bytes={self.slot_layout.token_bytes}, "
                f"alignment_tokens={self.alignment_tokens}"
            )

    @property
    def alignment_tokens(self) -> int:
        # In mode=none the native manager keeps one running state (+MTP
        # states) for a whole request. That span is NOT the token capacity of
        # one compressed FULL page. Native grouping still owns scheduler LCM.
        if self.native_mamba_cache_mode == "none":
            return self.kernel_block_size
        return lcm(self.native_mamba_block_size, self.kernel_block_size)

    @property
    def block_size(self) -> int:
        unit = self.slot_layout.token_bytes * self.alignment_tokens
        return (self.ssm_bytes // unit) * self.alignment_tokens

    @property
    def page_size_bytes(self) -> int:
        return self.native_page_size_bytes

    @property
    def native_padding_bytes(self) -> int:
        return self.native_page_size_bytes - self.conv_bytes - self.ssm_bytes

    @property
    def payload_bytes(self) -> int:
        return self.block_size * self.slot_layout.token_bytes

    @property
    def unused_ssm_bytes(self) -> int:
        return self.ssm_bytes - self.payload_bytes

    @property
    def virtual_blocks_per_page(self) -> int:
        return self.block_size // self.kernel_block_size

    @property
    def scheduler_block_size(self) -> int:
        return lcm(self.block_size, self.native_mamba_block_size)

    def _check_page(self, num_blocks: int, physical_block: int) -> None:
        _positive("num_blocks", num_blocks)
        if not 0 <= physical_block < num_blocks:
            raise ValueError(f"physical block {physical_block} outside [0,{num_blocks})")

    def ssm_interval(self, num_blocks: int, physical_block: int) -> tuple[int, int]:
        self._check_page(num_blocks, physical_block)
        start = num_blocks * self.conv_bytes + physical_block * self.ssm_bytes
        return start, start + self.ssm_bytes

    def conv_interval(self, num_blocks: int, physical_block: int) -> tuple[int, int]:
        self._check_page(num_blocks, physical_block)
        start = physical_block * self.conv_bytes
        return start, start + self.conv_bytes

    def packed_interval(self, num_blocks: int, physical_block: int) -> tuple[int, int]:
        start, _ = self.ssm_interval(num_blocks, physical_block)
        return start, start + self.payload_bytes

    def address(self, num_blocks: int, physical_block: int, token: int, head: int = 0) -> int:
        if not 0 <= token < self.block_size:
            raise ValueError(f"token {token} outside physical block capacity {self.block_size}")
        if not 0 <= head < self.slot_layout.num_kv_heads:
            raise ValueError(f"head {head} outside local heads {self.slot_layout.num_kv_heads}")
        start, _ = self.ssm_interval(num_blocks, physical_block)
        return start + token * self.slot_layout.token_bytes + head * self.slot_layout.slot_bytes

    def virtual_to_physical(self, virtual_block: int, kernel_offset: int = 0) -> tuple[int, int]:
        if virtual_block < 0 or not 0 <= kernel_offset < self.kernel_block_size:
            raise ValueError("invalid virtual block or kernel token offset")
        physical, virtual_offset = divmod(virtual_block, self.virtual_blocks_per_page)
        return physical, virtual_offset * self.kernel_block_size + kernel_offset

    def slot_to_physical(self, slot: int) -> tuple[int, int]:
        if slot < 0:
            raise ValueError("negative slots denote padding and have no storage address")
        return divmod(slot, self.block_size)

    def address_from_slot(self, num_blocks: int, slot: int, head: int = 0) -> int:
        physical, token = self.slot_to_physical(slot)
        return self.address(num_blocks, physical, token, head)


@dataclass(frozen=True)
class MemoryBudget:
    """Count physical pools once, plus all externally allocated runtime bytes.

    Window bytes represent a declared bounded BF16 arena, not an implicit
    reserve. Its ownership, rollback, and prefix protocol must be separately
    proved. This budget never claims such a protocol has been implemented.
    """

    layout: HybridPageLayout
    num_blocks: int
    shared_tensors: int
    full_layers: int
    max_num_reqs: int
    sink_tokens: int
    recent_tokens: int
    speculative_tokens: int
    rotation_bytes: int = 0
    workspace_bytes: int = 0
    metadata_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("num_blocks", "shared_tensors", "full_layers", "max_num_reqs"):
            _positive(name, getattr(self, name))
        for name in ("sink_tokens", "recent_tokens", "speculative_tokens", "rotation_bytes", "workspace_bytes", "metadata_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @property
    def raw_pool_bytes(self) -> int:
        return self.num_blocks * self.shared_tensors * self.layout.page_size_bytes

    @property
    def window_bytes(self) -> int:
        return (self.full_layers * self.max_num_reqs
                * (self.sink_tokens + self.recent_tokens + self.speculative_tokens)
                * self.layout.slot_layout.bf16_token_bytes)

    @property
    def runtime_bytes(self) -> int:
        return self.window_bytes + self.rotation_bytes + self.workspace_bytes + self.metadata_bytes

    @property
    def total_bytes(self) -> int:
        return self.raw_pool_bytes + self.runtime_bytes

    @property
    def full_token_capacity_upper_bound(self) -> int:
        """An upper bound before GDN groups, null block, and request rounding.

        A shared physical block pool also serves GDN. Multiplying all blocks
        by FULL block size must not be reported as usable request capacity.
        """
        return self.num_blocks * self.layout.block_size

    def blocks_for_contexts(self, lengths: tuple[int, ...], mamba_blocks_per_request: int, mamba_groups: int) -> int:
        if mamba_blocks_per_request < 0 or mamba_groups < 0:
            raise ValueError("Mamba reservations must be nonnegative")
        return (sum(ceil_div(length, self.layout.block_size) for length in lengths)
                + len(lengths) * mamba_blocks_per_request * mamba_groups + 1)

    def blocks_within_budget(self, total_hbm_bytes: int) -> int:
        available = total_hbm_bytes - self.runtime_bytes
        if available < 0:
            return 0
        return available // (self.shared_tensors * self.layout.page_size_bytes)
