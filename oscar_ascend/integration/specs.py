"""Native cache-spec bridge, imported only after native initialization.

Archive #28/#77/#78: register() never imports this heavy module.
Archive #17–20/#34/#36: packed byte geometry is explicit and preserved
across spec merging. GDN shapes/dtypes/spec instances remain unchanged.
Archive #27: merge follows native grouping's AssertionError protocol.
Archive #127: preserve GDN's none/align mode when deriving FULL page capacity.
Archive #128: native metadata copies use virtual128, not a new allocation.
Archive #37-49/#127-129/#155: explicit request-window storage reuses two
native-owned SoA segments, never an overlapping page snapshot or fake view.
"""

from dataclasses import dataclass, replace
from math import prod

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from ..layout import HybridPageLayout, SlotLayout
from ..dual_region_layout import DualRegionPageLayout
from .prefix_cache import GuardedHashes, GuardedRequest, cacheable_tokens


@dataclass(frozen=True, kw_only=True)
class OscarFullAttentionSpec(FullAttentionSpec):
    conv_bytes: int
    ssm_bytes: int
    native_mamba_block_size: int
    kernel_block_size: int = 128
    native_page_size_bytes: int | None = None
    native_mamba_cache_mode: str = "align"
    metadata_block_view: bool = False
    request_window_storage: bool = False
    mtp_prefix_guard: bool = False

    def __post_init__(self):
        super().__post_init__()
        if type(self.request_window_storage) is not bool:
            raise ValueError("request_window_storage must be boolean")
        if type(self.mtp_prefix_guard) is not bool:
            raise ValueError("mtp_prefix_guard must be boolean")
        if self.mtp_prefix_guard and (self.request_window_storage or self.native_mamba_cache_mode != "align"):
            raise ValueError("MTP prefix guards require native align physical snapshots")
        if self.sliding_window is not None or self.attention_chunk_size is not None:
            raise ValueError("OscarFullAttentionSpec represents full causal attention")
        if int(self.kv_quant_mode) != 0:
            raise ValueError("native KV quantization cannot be combined with OSCAR INT2")
        layout = self.layout
        if type(self.metadata_block_view) is not bool:
            raise ValueError("metadata_block_view must be boolean")
        expected_block_size = self.kernel_block_size if self.metadata_block_view else layout.block_size
        if self.block_size != expected_block_size:
            raise ValueError(f"OSCAR block_size={self.block_size} differs from physical layout={layout.block_size}")
        if self.page_size_padded is not None and self.page_size_padded != layout.page_size_bytes:
            raise ValueError("OSCAR page size must exactly match the native GDN allocation")

    @property
    def layout(self) -> HybridPageLayout | DualRegionPageLayout:
        native = HybridPageLayout(
            SlotLayout(self.head_size, self.head_size_v, self.num_kv_heads),
            self.conv_bytes, self.ssm_bytes, self.native_mamba_block_size, self.kernel_block_size,
            self.native_page_size_bytes, self.native_mamba_cache_mode)
        return DualRegionPageLayout(native) if self.request_window_storage else native

    @property
    def real_page_size_bytes(self) -> int:
        return self.layout.payload_bytes

    @property
    def page_size_bytes(self) -> int:
        return self.layout.page_size_bytes

    @property
    def storage_block_size(self) -> int:
        return self.layout.block_size

    def copy_with_new_block_size(self, block_size: int):
        """Return the native builder's virtual-block view without resizing storage.

        AttentionGroup creates this copy for draft metadata. The pool/group
        spec remains physical; all byte geometry and GDN state stay intact.
        Only this explicit native seam enables the supported kernel view.
        """
        if type(block_size) is not int or block_size not in {self.storage_block_size, self.kernel_block_size}:
            raise ValueError(f"unsupported OSCAR metadata block size {block_size!r}; "
                             f"expected {self.kernel_block_size} or {self.storage_block_size}")
        return replace(self, block_size=block_size,
                       metadata_block_view=block_size != self.storage_block_size)

    @classmethod
    def merge(cls, specs):
        # Native kv_cache_utils.is_kv_cache_spec_uniform deliberately tries
        # merging mixed groups and catches AssertionError to partition them.
        # A ValueError here aborts initialization before hybrid grouping runs.
        if not specs or any(type(spec) is not cls for spec in specs):
            raise AssertionError("OSCAR cache group must contain OSCAR specs")
        if any(spec != specs[0] for spec in specs[1:]):
            raise AssertionError("OSCAR cache group contains incompatible geometry")
        return specs[0]


class OscarFullAttentionManager(FullAttentionManager):
    """Native ownership with a hash-bound MTP next-token dependency.

    Archive #37-49/#127/#128: caching only immutable complete FULL pages.
    The guard is part of the *native pool key*, so forks with a different
    next hash cannot reuse a shifted draft KV page. No auxiliary page-ID map
    can survive eviction/reset or accidentally attach to a recycled page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_blocks_this_step = set()

    def new_step_starts(self):
        super().new_step_starts()
        self.cached_blocks_this_step.clear()

    def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                                   total_computed_tokens, num_tokens_main_model,
                                   apply_admission_cap=False):
        # Native allocate_slots publishes hashes before the model runs. OSCAR
        # writes FULL KV after attention, so a newly published page cannot be
        # consumed by another request in the same forward. GDN's identical
        # admission rule alone is insufficient: its checkpoint may predate the
        # newly computed FULL page. Keep native scheduling/ownership intact.
        if any(block.block_hash in self.cached_blocks_this_step
               for block in new_computed_blocks if not block.is_null):
            return self.block_pool.num_gpu_blocks + 1
        return super().get_num_blocks_to_allocate(request_id, num_tokens,
            new_computed_blocks, total_computed_tokens, num_tokens_main_model,
            apply_admission_cap)

    def cache_blocks(self, request, num_tokens, retention_interval=None):
        before = self.num_cached_block.get(request.request_id, 0)
        if self.kv_cache_spec.mtp_prefix_guard:
            step = self.block_pool.hash_block_size
            if self.block_size % step:
                raise ValueError("OSCAR FULL block is not divisible by native hash block")
            request = GuardedRequest(request, self.block_size // step)
            num_tokens = min(cacheable_tokens(num_tokens, step), len(request.block_hashes) * step)
        super().cache_blocks(request, num_tokens, retention_interval)
        after = self.num_cached_block.get(request.request_id, 0)
        self.cached_blocks_this_step.update(
            block.block_hash for block in self.req_to_blocks[request.request_id][before:after]
            if not block.is_null and block.block_hash is not None)

    @classmethod
    def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                              block_pool, kv_cache_spec, drop_eagle_block,
                              alignment_tokens, dcp_world_size=1, pcp_world_size=1):
        if kv_cache_spec.mtp_prefix_guard:
            if dcp_world_size != 1 or pcp_world_size != 1:
                raise ValueError("OSCAR prefix guard requires local TP cache ownership")
            from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
            raw = getattr(block_hashes, "block_hashes", block_hashes)
            step = block_pool.hash_block_size
            guarded = GuardedHashes(raw, kv_cache_spec.storage_block_size // step)
            block_hashes = BlockHashListWithBlockSize(guarded, step, kv_cache_spec.storage_block_size)
            max_length = min(max_length, len(guarded) * step)
            # The next-token dependency is checked in each immutable key.
            # This replaces the whole-FULL-page lookahead with one hash block.
            drop_eagle_block = False
        return super().find_longest_cache_hit(block_hashes, max_length, kv_cache_group_ids,
            block_pool, kv_cache_spec, drop_eagle_block, alignment_tokens,
            dcp_world_size, pcp_world_size)


def register_oscar_spec() -> None:
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=OscarFullAttentionSpec,
        manager_class=OscarFullAttentionManager,
        uniform_type_base_spec=OscarFullAttentionSpec)


def transform_native_specs(native_specs: dict, *, request_windows: bool = False,
                           mtp_prefix_guard: bool = False) -> dict:
    """Initialization-only conversion; no online tensor data is touched.

    Exact type matching leaves MLA/sliding-window/hidden-state cache specs
    intact. This target integration requires a native hybrid GDN page; a
    pure dense model needs its own explicit physical-allocation contract.
    """
    if type(request_windows) is not bool:
        raise ValueError("request_windows must be boolean")
    if type(mtp_prefix_guard) is not bool:
        raise ValueError("mtp_prefix_guard must be boolean")
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
        geometry.add((conv, ssm, spec.block_size, spec.page_size_bytes, spec.mamba_cache_mode))
    if len(geometry) != 1:
        raise ValueError(f"OSCAR requires uniform GDN page geometry, got {sorted(geometry)}")
    conv, ssm, mamba_block, native_page, mamba_mode = next(iter(geometry))
    converted = {}
    for name, spec in native_specs.items():
        if type(spec) is not FullAttentionSpec:
            converted[name] = spec
            continue
        layout = HybridPageLayout(SlotLayout(spec.head_size, spec.head_size_v, spec.num_kv_heads),
                                  conv, ssm, mamba_block, native_page_size_bytes=native_page,
                                  native_mamba_cache_mode=mamba_mode)
        if request_windows:
            layout = DualRegionPageLayout(layout)
        converted[name] = OscarFullAttentionSpec(
            block_size=layout.block_size, num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size, head_size_v=spec.head_size_v, dtype=spec.dtype,
            kv_quant_mode=spec.kv_quant_mode, page_size_padded=layout.page_size_bytes,
            sliding_window=spec.sliding_window, attention_chunk_size=spec.attention_chunk_size,
            conv_bytes=conv, ssm_bytes=ssm, native_mamba_block_size=mamba_block,
            native_page_size_bytes=native_page, native_mamba_cache_mode=mamba_mode,
            request_window_storage=request_windows, mtp_prefix_guard=mtp_prefix_guard)
    if not any(isinstance(spec, OscarFullAttentionSpec) for spec in converted.values()):
        raise ValueError("no native FULL attention layer was converted to OSCAR")
    return converted


def packed_view(raw, num_blocks: int, layout: HybridPageLayout | DualRegionPageLayout):
    """Create a byte view without allocation, copying, or changing GDN views."""
    if raw.element_size() != 1 or raw.ndim != 1 or raw.stride() != (1,):
        raise ValueError("OSCAR raw shared tensor must be a contiguous rank-one byte tensor")
    if raw.numel() != num_blocks * layout.page_size_bytes:
        raise ValueError("OSCAR raw tensor size does not match the shared page budget")
    if isinstance(layout, DualRegionPageLayout):
        # Native attention binding transports this tensor opaquely. The two
        # separated SoA segments cannot be represented by a single strided
        # [page,token,head,byte] view. New kernels receive the actual owner.
        return raw
    slot = layout.slot_layout
    return raw.as_strided(
        size=(num_blocks, layout.block_size, slot.num_kv_heads, slot.slot_bytes),
        stride=(layout.ssm_bytes, slot.token_bytes, slot.slot_bytes, 1),
        storage_offset=raw.storage_offset() + num_blocks * layout.conv_bytes)
