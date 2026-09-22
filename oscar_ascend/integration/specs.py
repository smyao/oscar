"""Native cache-spec bridge, imported only after native initialization.

Archive #28/#77/#78: register() never imports this heavy module.
Archive #17–20/#34/#36: packed byte geometry is explicit and preserved
across spec merging. GDN shapes/dtypes/spec instances remain unchanged.
"""

from dataclasses import dataclass
from math import prod

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from ..layout import HybridPageLayout, SlotLayout


@dataclass(frozen=True, kw_only=True)
class OscarFullAttentionSpec(FullAttentionSpec):
    conv_bytes: int
    ssm_bytes: int
    native_mamba_block_size: int
    kernel_block_size: int = 128
    native_page_size_bytes: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.sliding_window is not None or self.attention_chunk_size is not None:
            raise ValueError("OscarFullAttentionSpec represents full causal attention")
        if int(self.kv_quant_mode) != 0:
            raise ValueError("native KV quantization cannot be combined with OSCAR INT2")
        layout = self.layout
        if self.block_size != layout.block_size:
            raise ValueError(f"OSCAR block_size={self.block_size} differs from physical layout={layout.block_size}")
        if self.page_size_padded is not None and self.page_size_padded != layout.page_size_bytes:
            raise ValueError("OSCAR page size must exactly match the native GDN allocation")

    @property
    def layout(self) -> HybridPageLayout:
        return HybridPageLayout(
            SlotLayout(self.head_size, self.head_size_v, self.num_kv_heads),
            self.conv_bytes, self.ssm_bytes, self.native_mamba_block_size, self.kernel_block_size,
            self.native_page_size_bytes)

    @property
    def real_page_size_bytes(self) -> int:
        return self.layout.payload_bytes

    @property
    def page_size_bytes(self) -> int:
        return self.layout.page_size_bytes

    @classmethod
    def merge(cls, specs):
        if not specs or any(type(spec) is not cls for spec in specs):
            raise ValueError("OSCAR cache group must contain OSCAR specs")
        if any(spec != specs[0] for spec in specs[1:]):
            raise ValueError("OSCAR cache group contains incompatible geometry")
        return specs[0]


def register_oscar_spec() -> None:
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=OscarFullAttentionSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=OscarFullAttentionSpec)


def transform_native_specs(native_specs: dict) -> dict:
    """Initialization-only conversion; no online tensor data is touched.

    Exact type matching leaves MLA/sliding-window/hidden-state cache specs
    intact. This target integration requires a native hybrid GDN page; a
    pure dense model needs its own explicit physical-allocation contract.
    """
    mamba_specs = [spec for spec in native_specs.values() if isinstance(spec, MambaSpec)]
    if not mamba_specs:
        raise ValueError("OSCAR hybrid layout requires native GDN/Mamba state specs")
    geometry = set()
    for spec in mamba_specs:
        if len(spec.shapes) != 2 or len(spec.dtypes) != 2:
            raise ValueError("OSCAR GDN layout requires exactly conv and SSM states")
        conv = prod(spec.shapes[0]) * get_dtype_size(spec.dtypes[0])
        ssm = prod(spec.shapes[1]) * get_dtype_size(spec.dtypes[1])
        if conv + ssm > spec.page_size_bytes:
            raise ValueError("native GDN page budget cannot contain conv and SSM states")
        geometry.add((conv, ssm, spec.block_size, spec.page_size_bytes))
    if len(geometry) != 1:
        raise ValueError(f"OSCAR requires uniform GDN page geometry, got {sorted(geometry)}")
    conv, ssm, mamba_block, native_page = next(iter(geometry))
    converted = {}
    for name, spec in native_specs.items():
        if type(spec) is not FullAttentionSpec:
            converted[name] = spec
            continue
        layout = HybridPageLayout(SlotLayout(spec.head_size, spec.head_size_v, spec.num_kv_heads),
                                  conv, ssm, mamba_block, native_page_size_bytes=native_page)
        converted[name] = OscarFullAttentionSpec(
            block_size=layout.block_size, num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size, head_size_v=spec.head_size_v, dtype=spec.dtype,
            kv_quant_mode=spec.kv_quant_mode, page_size_padded=layout.page_size_bytes,
            sliding_window=spec.sliding_window, attention_chunk_size=spec.attention_chunk_size,
            conv_bytes=conv, ssm_bytes=ssm, native_mamba_block_size=mamba_block,
            native_page_size_bytes=native_page)
    if not any(isinstance(spec, OscarFullAttentionSpec) for spec in converted.values()):
        raise ValueError("no native FULL attention layer was converted to OSCAR")
    return converted


def packed_view(raw, num_blocks: int, layout: HybridPageLayout):
    """Create a byte view without allocation, copying, or changing GDN views."""
    if raw.element_size() != 1 or raw.ndim != 1 or raw.stride() != (1,):
        raise ValueError("OSCAR raw shared tensor must be a contiguous rank-one byte tensor")
    if raw.numel() != num_blocks * layout.page_size_bytes:
        raise ValueError("OSCAR raw tensor size does not match the shared page budget")
    slot = layout.slot_layout
    return raw.as_strided(
        size=(num_blocks, layout.block_size, slot.num_kv_heads, slot.slot_bytes),
        stride=(layout.ssm_bytes, slot.token_bytes, slot.slot_bytes, 1),
        storage_offset=raw.storage_offset() + num_blocks * layout.conv_bytes)
