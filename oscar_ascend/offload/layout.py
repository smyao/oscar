"""Connector-only physical byte spans; no model tensor or numerical changes.

Archive #34/#36/#127/#128: physical storage is distinct from metadata128.
#37-49/#157/#163/#171: complete window and GDN state components travel with
the page, including a partial token tail; callers own commit/pin barriers.
Native evidence: ascend_store/pool_worker.py _get_cache_block_metadata and
_as_cache_tuple support independent component lengths and physical strides.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, Iterable, Mapping, Sequence


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class StorageRegion:
    address: int
    length: int


@dataclass(frozen=True)
class PageSpan:
    component: str
    page_id: int
    address: int
    length: int
    storage_offset: int
    payload_offset: int
    valid_tokens: int | None = None


@dataclass(frozen=True)
class CacheComponent:
    name: str
    tensor: Any = field(repr=False, compare=False)
    shape: tuple[int, ...]
    dtype: str
    storage_ptr: int
    storage_nbytes: int
    storage_offset: int
    page_stride_bytes: int
    page_bytes: int

    @classmethod
    def from_tensor(cls, name: str, tensor: Any) -> CacheComponent:
        import torch

        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"{name}: cache component must be a strided tensor")
        shape = tuple(tensor.shape)
        if len(shape) < 2 or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError(f"{name}: cache component needs nonempty physical pages")
        # Native direct-copy supports strided pages, not holes inside a page.
        expected = 1
        for size, stride in zip(reversed(shape[1:]), reversed(tensor.stride()[1:])):
            if size != 1 and stride != expected:
                raise ValueError(f"{name}: each page must be contiguous")
            expected *= size
        itemsize = tensor.element_size()
        length = prod(shape[1:]) * itemsize
        stride = tensor.stride(0) * itemsize
        if stride < length:
            raise ValueError(f"{name}: physical pages overlap")
        storage = tensor.untyped_storage()
        start = tensor.storage_offset() * itemsize
        if start < 0 or start + (shape[0] - 1) * stride + length > storage.nbytes():
            raise ValueError(f"{name}: physical page exceeds underlying storage")
        if tensor.data_ptr() != storage.data_ptr() + start:
            raise ValueError(f"{name}: tensor/storage byte offsets disagree")
        return cls(name, tensor, shape, str(tensor.dtype).removeprefix("torch."),
                   storage.data_ptr(), storage.nbytes(), start, stride, length)

    def span(self, page_id: int, *, payload_offset: int = 0,
             valid_tokens: int | None = None) -> PageSpan:
        _integer(page_id, "physical page id")
        if page_id >= self.shape[0]:
            raise ValueError(f"{self.name}: physical page id exceeds pool capacity")
        start = self.storage_offset + page_id * self.page_stride_bytes
        return PageSpan(self.name, page_id, self.storage_ptr + start, self.page_bytes,
                        start, payload_offset, valid_tokens)

    def geometry(self) -> dict[str, Any]:
        # Allocation address, pool capacity and storage offset are local only.
        return {"name": self.name, "shape": list(self.shape[1:]), "dtype": self.dtype,
                "page_bytes": self.page_bytes, "page_stride_bytes": self.page_stride_bytes}


@dataclass(frozen=True)
class LayerCacheExport:
    layer_name: str
    cache_family: str
    storage_block_size: int
    components: tuple[CacheComponent, ...]
    native_cache: Any = field(repr=False, compare=False)
    format_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def connector_cache(self) -> tuple[Any, ...]:
        return tuple(component.tensor for component in self.components)

    @property
    def num_pages(self) -> int:
        return self.components[0].shape[0]

    @property
    def page_bytes(self) -> int:
        return sum(component.page_bytes for component in self.components)

    def page_spans(self, page_ids: Sequence[int], *,
                   valid_tokens: Sequence[int] | None = None) -> tuple[PageSpan, ...]:
        pages = tuple(page_ids)
        counts = (None,) * len(pages) if valid_tokens is None else tuple(valid_tokens)
        if len(counts) != len(pages):
            raise ValueError("valid token counts must match physical pages")
        result = []
        offset = 0
        for page, count in zip(pages, counts):
            if count is not None:
                _integer(count, "valid tokens")
                if count > self.storage_block_size:
                    raise ValueError("valid tokens exceed physical block size")
            for component in self.components:
                result.append(component.span(page, payload_offset=offset, valid_tokens=count))
                offset += component.page_bytes
        return tuple(result)

    def geometry(self) -> dict[str, Any]:
        return {"layer_name": self.layer_name, "cache_family": self.cache_family,
                "storage_block_size": self.storage_block_size,
                "components": [component.geometry() for component in self.components],
                "format": dict(self.format_metadata)}


def _same_view(left: Any, right: Any) -> bool:
    return (left is right or (hasattr(left, "data_ptr") and hasattr(right, "data_ptr")
            and getattr(left, "shape", None) == getattr(right, "shape", None)
            and left.dtype == right.dtype and left.device == right.device
            and left.data_ptr() == right.data_ptr() and left.stride() == right.stride()))


def export_layer_cache(layer_name: str, native_cache: Any, *, state: Any | None = None,
                       spec: Any | None = None) -> LayerCacheExport:
    """Export complete components without replacing the native attention view.

    FULL state is the existing runtime LayerState; GDN spec and tuple are the
    native reshape outputs. Request-window research has a separate per-request
    identity and is deliberately rejected, never treated as physical snapshots.
    """
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError("layer_name is required")
    if state is not None:
        spec = state.spec if spec is None else spec
        if getattr(spec, "request_window_storage", False) or getattr(state, "window_map", None) is not None:
            raise ValueError("offload requires physical-page windows; request-window research is unsupported")
        if getattr(spec, "metadata_block_view", False):
            raise ValueError("offload requires the physical FULL spec, not a metadata128 view")
        block = _integer(spec.storage_block_size, "storage block size", minimum=1)
        if spec.block_size != block:
            raise ValueError("FULL group spec differs from physical storage block size")
        if not _same_view(native_cache, state.packed):
            raise ValueError("native attention view differs from runtime packed storage")
        tensors = (state.packed, state.window_key, state.window_value, state.window_tags)
        names = ("packed", "window_k", "window_v", "tags")
        family = "oscar_full"
        metadata = {"cache_format": state.cache_format, "metadata_bytes": spec.metadata_bytes,
                    "kernel_block_size": spec.kernel_block_size,
                    "native_mamba_cache_mode": spec.native_mamba_cache_mode,
                    "sink_tokens": state.snapshots.sink_tokens,
                    "recent_tokens": state.snapshots.recent_tokens,
                    "speculative_tokens": state.snapshots.speculative_tokens,
                    "tag_semantics": "physical_page_local_token_offset"}
        expected_shapes = ((block, spec.num_kv_heads, spec.layout.slot_layout.slot_bytes),
                           (state.snapshots.rows, spec.num_kv_heads, spec.head_size),
                           (state.snapshots.rows, spec.num_kv_heads, spec.head_size_v),
                           (state.snapshots.rows,))
        expected_dtypes = ("uint8", "bfloat16", "bfloat16", "int64")
    else:
        if spec is None or not hasattr(spec, "shapes") or not hasattr(spec, "dtypes"):
            raise ValueError("GDN export requires its native state spec")
        if not isinstance(native_cache, (tuple, list)) or len(native_cache) != 2:
            raise ValueError("GDN export requires the complete native conv/SSM tuple")
        if len(spec.shapes) != 2 or len(spec.dtypes) != 2:
            raise ValueError("GDN state spec must declare conv and SSM")
        tensors = tuple(native_cache)
        names = ("conv", "ssm")
        block = _integer(spec.block_size, "GDN block size", minimum=1)
        family = "gdn"
        metadata = {"mamba_cache_mode": spec.mamba_cache_mode}
        expected_shapes = tuple(tuple(shape) for shape in spec.shapes)
        expected_dtypes = tuple(str(dtype).removeprefix("torch.") for dtype in spec.dtypes)
    components = tuple(CacheComponent.from_tensor(name, tensor) for name, tensor in zip(names, tensors))
    if len({component.shape[0] for component in components}) != 1:
        raise ValueError("cache components disagree on physical pool capacity")
    if len({component.tensor.device for component in components}) != 1:
        raise ValueError("cache components must reside on one worker device")
    for component, shape, dtype in zip(components, expected_shapes, expected_dtypes):
        if component.shape[1:] != shape or component.dtype != dtype:
            raise ValueError(f"{layer_name}/{component.name}: native component geometry differs from spec")
    if state is not None and components[0].shape[0] != state.num_blocks:
        raise ValueError("FULL pool capacity differs from runtime state")
    if state is not None:
        raw = state.raw
        layout = spec.layout
        offset = raw.storage_offset() * raw.element_size()
        snapshot_base = offset + state.snapshots.base(state.num_blocks)
        starts = (offset + state.num_blocks * layout.conv_bytes, snapshot_base,
                  snapshot_base + state.snapshots.key_bytes,
                  snapshot_base + state.snapshots.key_bytes + state.snapshots.value_bytes)
        strides = (layout.ssm_bytes,) + (layout.native_padding_bytes,) * 3
        for component, start, stride in zip(components, starts, strides):
            if (component.storage_ptr != raw.untyped_storage().data_ptr()
                    or component.storage_offset != start or component.page_stride_bytes != stride):
                raise ValueError(f"{layer_name}/{component.name}: physical snapshot byte layout differs")
    return LayerCacheExport(layer_name, family, block, components, native_cache, metadata)


def export_kv_caches(native_caches: Mapping[str, Any], layers: Mapping[str, Any],
                     specs: Mapping[str, Any]) -> dict[str, LayerCacheExport]:
    """Return a new connector map; never mutate the model's cache mapping."""
    missing = set(native_caches) - set(specs)
    if missing:
        raise ValueError(f"cache specs missing layers: {sorted(missing)}")
    result = {name: export_layer_cache(name, cache, state=layers.get(name), spec=specs[name])
              for name, cache in native_caches.items()}
    if len({component.tensor.device for export in result.values()
            for component in export.components}) > 1:
        raise ValueError("registered cache layers must reside on one worker device")
    return result


def _exports(exports: Mapping[str, LayerCacheExport] | Iterable[LayerCacheExport]):
    return exports.values() if isinstance(exports, Mapping) else exports


def registration_regions(exports: Mapping[str, LayerCacheExport] | Iterable[LayerCacheExport], *,
                         alignment: int = 2 * 1024 * 1024
                         ) -> tuple[StorageRegion, ...]:
    """Merge actual spans per owner and align within its allocated storage.

    Runtime raw tensors are aligned slices of a larger allocation. The storage
    base itself need not be aligned; use the native worker's align-down rule.
    CPU-only callers can explicitly select alignment=1.
    """
    _integer(alignment, "registration alignment", minimum=1)
    owners: dict[int, tuple[int, int, int]] = {}
    for export in _exports(exports):
        for component in export.components:
            start = component.storage_ptr + component.storage_offset
            end = start + (component.shape[0] - 1) * component.page_stride_bytes + component.page_bytes
            previous = owners.get(component.storage_ptr)
            if previous is not None:
                old_start, old_end, size = previous
                if size != component.storage_nbytes:
                    raise ValueError("aliased storage owners disagree on allocation size")
                start, end = min(start, old_start), max(end, old_end)
            owners[component.storage_ptr] = (start, end, component.storage_nbytes)
    result = []
    for base, (start, end, size) in sorted(owners.items()):
        aligned = start // alignment * alignment
        if aligned < base or end > base + size:
            raise ValueError("aligned registration exceeds owner storage; allocate an aligned raw slice")
        result.append(StorageRegion(aligned, end - aligned))
    return tuple(result)


def unique_page_spans(exports: Mapping[str, LayerCacheExport],
                      page_ids_by_layer: Mapping[str, Sequence[int]]) -> tuple[PageSpan, ...]:
    """Deduplicate exact aliases, rejecting conflicting overlapping live pages.

    Native FULL/GDN groups may share raw storage but must own different page
    IDs. Merging overlapping nonidentical ranges would hide an ownership bug.
    """
    spans: dict[tuple[int, int], PageSpan] = {}
    for name, pages in page_ids_by_layer.items():
        for span in exports[name].page_spans(pages):
            previous = spans.setdefault((span.address, span.length), span)
            if previous.component != span.component:
                raise ValueError("different live cache components alias physical storage")
    ordered = sorted(spans.values(), key=lambda span: span.address)
    for left, right in zip(ordered, ordered[1:]):
        if left.address + left.length > right.address:
            raise ValueError("different live cache components overlap in physical storage")
    result = []
    offset = 0
    for span in ordered:
        result.append(replace(span, payload_offset=offset))
        offset += span.length
    return tuple(result)
