"""Host-only ABI contracts; importing this module never imports torch/NPU.

Archive #17/#22/#34/#36/#87/#108/#111: dtype, capacity and ABI boundaries.

Device status is an asynchronous output, never an implicit CPU synchronization.
Probes must synchronize then require all status values to be zero. Production
cannot use these components until the complete capability/acceptance gate passes.
"""
from dataclasses import dataclass

ABI_VERSION = 1
SOURCE_CAPABILITIES = frozenset({"store_int2_out", "merge_lse_out", "rotate_out",
    "rotate_clip_store_out", "prepare_attention_tasks_out", "attention_cv_out", "status_guard"})
# Exact callable symbols, not names of abstract components. Compilation,
# CPU simulation, NPU execution and service acceptance remain separate gates.
PRODUCTION_CAPABILITIES = frozenset({"store_int2_out", "merge_lse_out", "rotate_out",
    "rotate_clip_store_out", "prepare_attention_tasks_out", "attention_cv_out", "status_guard"})
SUPPORTED_HEAD_DIMS = (64, 128, 256)
MAX_SPLITS = 128
DEVICE_STATUS = {
    0: "ok (including padding store slots and all-empty merge rows)",
    1: "slot_mapping exceeds the physical page pool",
    2: "non-finite quantizer extrema or invalid partial LSE",
    3: "fp16 quantizer scale is zero/non-finite, or merge denominator is invalid",
    4: "negative logical token position in a valid write slot",
}


class OperatorContractError(ValueError):
    pass


@dataclass(frozen=True)
class PackedStorage:
    """Packed FULL rows in one GDN SSM SoA region; all sizes are bytes."""
    physical_block_tokens: int
    physical_num_blocks: int
    heads: int
    head_dim: int
    raw_ssm_offset: int
    physical_page_stride: int
    raw_num_bytes: int

    def __post_init__(self):
        fields = tuple(self.__dict__.values())
        if any(type(x) is not int for x in fields):
            raise OperatorContractError("storage geometry must use integer byte counts")
        if self.head_dim not in SUPPORTED_HEAD_DIMS or self.heads <= 0:
            raise OperatorContractError("supported D is 64/128/256 and heads must be positive")
        if self.physical_block_tokens <= 0 or self.physical_block_tokens % 128:
            raise OperatorContractError("physical B must be a positive multiple of virtual block 128")
        if self.physical_num_blocks <= 0 or self.raw_ssm_offset < 0:
            raise OperatorContractError("invalid block count or SSM offset")
        if self.physical_page_stride < self.physical_block_tokens * self.token_bytes:
            raise OperatorContractError("packed rows exceed one physical SSM page")
        if self.raw_ssm_offset + self.physical_num_blocks * self.physical_page_stride > self.raw_num_bytes:
            raise OperatorContractError("SSM page pool exceeds raw storage allocation")
        if any(x > (1 << 63) - 1 or x < 0 for x in fields):
            raise OperatorContractError("storage ABI requires nonnegative signed int64 geometry")

    @property
    def head_bytes(self):
        return self.head_dim // 2 + 8

    @property
    def token_bytes(self):
        return self.heads * self.head_bytes

    @property
    def field_offsets(self):
        data = self.head_dim // 4
        return {"k_codes": 0, "k_scale": data, "k_zero": data + 2,
                "v_codes": data + 4, "v_scale": 2 * data + 4,
                "v_zero": 2 * data + 6}

    def slot_offset(self, slot: int, head: int = 0):
        """Debug/capacity algebra only; never loop over tokens in production."""
        if type(slot) is not int or type(head) is not int:
            raise OperatorContractError("slot and head must be integer indices")
        if slot < 0:
            return None
        if slot >= self.physical_block_tokens * self.physical_num_blocks:
            raise OperatorContractError("slot exceeds physical pool")
        if not 0 <= head < self.heads:
            raise OperatorContractError("head exceeds local KV head count")
        block, token = divmod(slot, self.physical_block_tokens)
        return (self.raw_ssm_offset + block * self.physical_page_stride
                + token * self.token_bytes + head * self.head_bytes)


def merge_shapes(rows: int, splits: int, dim: int):
    if any(type(x) is not int for x in (rows,splits,dim)):
        raise OperatorContractError("merge geometry must use integer dimensions")
    if rows < 0 or not 1 <= splits <= MAX_SPLITS or dim not in SUPPORTED_HEAD_DIMS:
        raise OperatorContractError("merge requires R>=0, 1<=S<=128, D in 64/128/256")
    return {"partial_out": (rows, splits, dim), "partial_lse": (rows, splits),
            "output": (rows, dim), "lse": (rows,), "status": (rows,)}


def missing_capabilities(available, required=PRODUCTION_CAPABILITIES):
    return tuple(sorted(set(required) - set(available)))
