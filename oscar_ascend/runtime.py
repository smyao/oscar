"""Concrete AscendC runtime, allocation and per-layer bindings.

Archive #27/#31-49: real operators, module-owned state, physical-page lifetime,
and fixed workspaces. #28/#77/#78: imported only after native initialization.
#86/#97: fingerprinted target rotations and explicit PR draft identity.
#69/#111: NPU-only production; exact BF16 projection and FP32 rotation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import warnings
from typing import Any

from .integration.runtime_api import OscarReadinessError
from .lifecycle import SnapshotLayout


def canonical_rotation_name(name: str) -> str | None:
    """Qwen3.5 multimodal wrapper prefixes do not alter layer identity."""
    target = re.search(r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.attn$", name)
    if target:
        return f"model.layers.{int(target[1])}.self_attn.attn"
    if re.search(r"(?:^|\.)mtp\.layers\.\d+\.self_attn\.attn$", name):
        return None
    raise OscarReadinessError(f"unrecognized target/draft FULL layer identity: {name}")


@dataclass(frozen=True)
class WorkspaceGeometry:
    tokens: int
    query_heads: int
    kv_heads: int
    head_dim: int
    # Capacity multiplier, not the per-call split count. Small captured token
    # shapes can use otherwise idle arena rows without enlarging this budget.
    splits: int = 1
    cube_cores: int = 1

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in self.__dict__.values()):
            raise ValueError("workspace dimensions must be positive integers")
        if self.query_heads % self.kv_heads or self.query_heads // self.kv_heads > 16:
            raise ValueError("CV query grouping requires integral GQA ratio <= 16")
        if self.head_dim not in (64, 128, 256) or self.splits > 32 or self.cube_cores > 32:
            raise ValueError("unsupported CV workspace geometry")

    @property
    def task_count(self):
        return self.tokens * self.kv_heads * 3 * self.splits

    def splits_for_tokens(self, tokens: int) -> int:
        """Choose parallelism from captured shape and measured Cube count.

        Archive #34/#36/#70-73: no seq_len readback, length-based route switch
        or new decode-only allocation. The same shape always chooses the same
        S, and tokens*S never exceeds the preallocated arena capacity.
        """
        if type(tokens) is not int or not 1 <= tokens <= self.tokens:
            raise ValueError(f"active tokens must be in [1,{self.tokens}]")
        query_tile = 64 // (self.query_heads // self.kv_heads)
        groups = ((tokens + query_tile - 1) // query_tile) * self.kv_heads
        cube_parallelism = (self.cube_cores + groups - 1) // groups
        arena_limit = self.tokens * self.splits // tokens
        return max(1, min(32, cube_parallelism, arena_limit))

    @property
    def cv_bytes(self):
        # Archive #71/#140, D.4: one 128-token tile, never a full history.
        # Q 64D + K 128D + V 128D + score/P 64x128 + PV/rotation 64D.
        return self.cube_cores * (384 * self.head_dim + 8192) * 4

    @property
    def total_bytes(self):
        n, h, d, p = self.tokens, self.query_heads, self.head_dim, 3 * self.splits
        return (n * h * d * 4 * 2 + n * (h + 2 * self.kv_heads) * d * 2
                + n * h * p * d * 4 + n * h * p * 4
                + n * h * 8 + self.task_count * (128 + 8)
                + n * self.kv_heads * 4 + n * h * 4 + n * 16 + self.cv_bytes)


class GraphWorkspace:
    """One stream-ordered scratch arena reused across sequential FULL layers."""
    def __init__(self, geometry: WorkspaceGeometry, device):
        import torch
        if torch.device(device).type != "npu":
            raise OscarReadinessError("production graph workspace requires NPU")
        self.geometry = geometry
        n, h, d, p = geometry.tokens, geometry.query_heads, geometry.head_dim, 3 * geometry.splits
        self.query_rot = torch.empty((n, h, d), dtype=torch.float32, device=device)
        # qkv.split leaves a strided value projection in the native model.
        # These bounded current-step copies are reused; no historical KV is
        # ever copied here. Contiguous projections use their original storage.
        self.query_input = torch.empty((n, h, d), dtype=torch.bfloat16, device=device)
        self.key_input = torch.empty((n, geometry.kv_heads, d), dtype=torch.bfloat16, device=device)
        self.value_input = torch.empty_like(self.key_input)
        self.partial = torch.empty((n, h, p, d), dtype=torch.float32, device=device)
        self.partial_lse = torch.empty((n, h, p), dtype=torch.float32, device=device)
        self.output = torch.empty((n, h, d), dtype=torch.float32, device=device)
        self.lse = torch.empty((n, h), dtype=torch.float32, device=device)
        self.merge_status = torch.empty((n, h), dtype=torch.int32, device=device)
        self.rotate_status = torch.empty((n, h), dtype=torch.int32, device=device)
        self.store_status = torch.empty((n, geometry.kv_heads), dtype=torch.int32, device=device)
        self.tasks = torch.empty((geometry.task_count, 16), dtype=torch.int64, device=device)
        self.attention_status = torch.empty((geometry.task_count, 2), dtype=torch.int32, device=device)
        self.positions = torch.empty((n,), dtype=torch.int64, device=device)
        self.slots = torch.empty((n,), dtype=torch.int64, device=device)
        self.cv = torch.empty((geometry.cv_bytes,), dtype=torch.uint8, device=device)

    def validate(self, tokens, heads, kv_heads, dim):
        g = self.geometry
        if tokens > g.tokens or (heads, kv_heads, dim) != (g.query_heads, g.kv_heads, g.head_dim):
            raise OscarReadinessError(
                f"OSCAR fixed workspace mismatch: tokens={tokens}/{g.tokens}, "
                f"Hq/Hkv/D={(heads, kv_heads, dim)}/{(g.query_heads, g.kv_heads, g.head_dim)}")

    def projection(self, tensor, destination, tokens, heads, dim):
        view = tensor.view(tokens, heads, dim)
        if view.is_contiguous():
            return view
        destination[:tokens].copy_(view)
        return destination[:tokens]

    def partial_views(self, tokens, splits):
        """Reinterpret fixed flat arenas for this graph's static split count."""
        g = self.geometry
        if (type(tokens) is not int or type(splits) is not int or
                not 1 <= tokens <= g.tokens or not 1 <= splits <= 32 or
                tokens * splits > g.tokens * g.splits):
            raise OscarReadinessError("attention splits exceed the preallocated workspace")
        rows = tokens * g.query_heads * 3 * splits
        partial = self.partial.view(-1)[:rows * g.head_dim].view(
            tokens, g.query_heads, 3 * splits, g.head_dim)
        lse = self.partial_lse.view(-1)[:rows].view(tokens, g.query_heads, 3 * splits)
        return partial, lse


@dataclass
class LayerState:
    raw: Any
    packed: Any
    spec: Any
    snapshots: SnapshotLayout
    window_key: Any
    window_value: Any
    window_tags: Any
    rotation_k: Any
    rotation_v: Any
    rotation_k_transpose: Any
    rotation_v_transpose: Any
    hadamard: bool
    num_blocks: int
    workspace: GraphWorkspace


class AscendRuntimeProvider:
    """Production provider installed lazily by the external plugin factory."""
    def __init__(self, config=None):
        path = Path(os.environ.get("OSCAR_TARGET_CONFIG", Path(__file__).resolve().parents[1] / "configs/target.json"))
        self.config = dict(config) if config is not None else json.loads(path.read_text())
        self._ready = False
        self.layers: dict[str, LayerState] = {}
        self.workspaces: dict[tuple, GraphWorkspace] = {}
        self.rotations: dict[tuple, dict] = {}
        self.device_cores: dict[str, int] = {}
        self.ops = None

    def assert_ready(self):
        if self._ready:
            return
        from .ops.loader import require_production_ops
        devices = self.config.get("devices")
        if not isinstance(devices, list) or not devices:
            raise OscarReadinessError("configs/target.json must select current-task devices before NPU initialization")
        expected = ",".join(str(x) for x in devices)
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != expected:
            raise OscarReadinessError("visible NPU selection does not match the current target configuration")
        require_production_ops()
        import torch
        if not torch.npu.is_available():
            raise OscarReadinessError("OSCAR production requires an available NPU")
        self.ops = torch.ops.oscar_ascend_ops
        self._ready = True

    def get_impl_cls(self):
        from .integration.impl import OscarAttentionImpl
        return OscarAttentionImpl

    def ensure_workspace(self, heads, kv_heads, dim, device):
        """Reserve scratch during model loading, before native KV profiling."""
        captures = self.config.get("compilation_config", {}).get("cudagraph_capture_sizes", [])
        capacity = max(int(self.config["max_num_batched_tokens"]), max(captures, default=0))
        geometry = WorkspaceGeometry(capacity, heads, kv_heads, dim,
                                     int(self.config.get("attention_splits", 1)),
                                     self._device_cube_cores(device))
        key = (str(device), geometry)
        if key not in self.workspaces:
            self.workspaces[key] = GraphWorkspace(geometry, device)
        self._prepare_rotations(dim, device)
        return self.workspaces[key]

    def _device_cube_cores(self, device):
        """Read native device properties once, never guess the active SoC.

        The native Triton utility queries the driver for hardware properties;
        it does not execute an OSCAR numerical kernel or change its AscendC
        implementation. The same initialized property source drives native
        GDN operators (triton_utils.py:46-67).
        """
        key = str(device)
        if key in self.device_cores:
            return self.device_cores[key]
        import torch
        from vllm_ascend.ops.triton.triton_utils import (
            get_aicore_num, get_vectorcore_num, init_device_properties_triton)
        selected = torch.device(device)
        if selected.type != "npu" or selected.index != torch.npu.current_device():
            raise OscarReadinessError("core discovery must use the current worker NPU")
        init_device_properties_triton()
        aic, aiv = int(get_aicore_num()), int(get_vectorcore_num())
        if aic <= 0 or aiv < 2 * aic:
            raise OscarReadinessError(f"OSCAR A2 CV requires two Vector cores per Cube: AIC={aic}, AIV={aiv}")
        requested = self.config.get("cube_cores")
        cores = min(aic, 32) if requested is None else requested
        if type(cores) is not int or not 1 <= cores <= min(aic, 32):
            raise OscarReadinessError(f"cube_cores={cores!r} exceeds measured AIC count {aic} or ABI maximum 32")
        self.device_cores[key] = cores
        from .telemetry import emit_once
        emit_once("device_cores", key=key, device=key, measured_aic=aic,
                  measured_aiv=aiv, selected_cube_cores=cores)
        return cores

    def transform_kv_cache_specs(self, runner, native_specs):
        from .integration.specs import transform_native_specs
        converted = transform_native_specs(native_specs)
        # Validate precise snapshot geometry before the scheduler budgets its
        # pool. A page advertised as compressed must also preserve its window.
        from .integration.specs import OscarFullAttentionSpec
        for spec in converted.values():
            if isinstance(spec, OscarFullAttentionSpec):
                self._snapshot_layout(spec)
        return converted

    def _snapshot_layout(self, spec):
        if spec.head_size != spec.head_size_v:
            raise OscarReadinessError("target CV FULL spec requires equal K/V head dimensions")
        return SnapshotLayout(
            spec.layout, int(self.config["sink_tokens"]), int(self.config["recent_tokens"]),
            int(self.config.get("speculative_config", {}).get("num_speculative_tokens", 0)))

    def allocate_kv_cache_tensors(self, runner, config, native_allocate):
        """Allocate each native shared tensor exactly once as raw bytes."""
        import torch
        from vllm.v1.kv_cache_interface import MambaSpec
        from .integration.specs import OscarFullAttentionSpec
        if torch.device(runner.device).type != "npu":
            raise OscarReadinessError("production KV allocation requires NPU")
        if runner.vllm_config.kv_transfer_config is not None:
            raise OscarReadinessError("target local-hybrid cache cannot be used by a KV transfer connector")
        specs = runner._get_layer_kv_cache_specs(config)
        raw = {}
        for tensor in config.kv_cache_tensors:
            if not all(isinstance(specs[name], (MambaSpec, OscarFullAttentionSpec)) for name in tensor.shared_by):
                raise OscarReadinessError("target shared cache pool contains an undeclared cache spec")
            allocation = torch.zeros(tensor.size, dtype=torch.int8, device=runner.device)
            for name in tensor.shared_by:
                if name in raw:
                    raise OscarReadinessError(f"layer appears in multiple native allocations: {name}")
                raw[name] = allocation
        runner.hybrid_with_attn_and_mamba = True
        return raw

    def _prepare_rotations(self, dim, device):
        import torch
        from .rotations import load_artifact, prepare_device_rotations, DeviceRotation
        cache_key = (str(device), dim)
        if cache_key in self.rotations:
            return self.rotations[cache_key]
        model = Path(self.config["model"])
        contents = (model / "config.json").read_bytes()
        model_config = json.loads(contents)
        text = model_config.get("text_config", model_config)
        layer_types = text.get("layer_types")
        count = text["num_hidden_layers"]
        if layer_types is None:
            interval = text.get("full_attention_interval")
            if type(interval) is not int or interval <= 0:
                raise OscarReadinessError("model config must declare FULL layer identities")
            layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                           for i in range(count)]
        if len(layer_types) != count:
            raise OscarReadinessError("model layer_types differs from num_hidden_layers")
        target = [f"model.layers.{i}.self_attn.attn" for i, kind in enumerate(layer_types)
                  if kind == "full_attention"]
        fingerprint = hashlib.sha256(contents)
        index = model / "model.safetensors.index.json"
        if index.is_file():
            fingerprint.update(index.read_bytes())
        artifact_path = self.config.get("rotation_artifact") or os.environ.get("OSCAR_ROTATION_ARTIFACT")
        if artifact_path is None:
            artifact_path = Path(__file__).resolve().parents[1] / "artifacts/rotations" / fingerprint.hexdigest() / "hadamard.pt"
        artifact = load_artifact(artifact_path, layer_names=target, head_dim=dim,
                                 model_fingerprint=fingerprint.hexdigest(), device=device)
        prepared = prepare_device_rotations(artifact, device=device)
        identity = torch.eye(dim, device=device, dtype=torch.float32)
        prepared[None] = DeviceRotation(identity, identity, identity, False)
        self.rotations[cache_key] = prepared
        return prepared

    def _rotations(self, specs, device):
        result = {}
        for name, spec in specs.items():
            canonical = canonical_rotation_name(name)
            prepared = self._prepare_rotations(spec.head_size, device)
            if canonical is None:
                # This is an explicit PR rule for the native MTP layer, whose
                # covariance is absent from the target-model calibration.
                warnings.warn(f"OSCAR MTP rotation is explicit PR identity: {name}", RuntimeWarning)
            if canonical not in prepared:
                raise OscarReadinessError(f"FULL layer has no matching rotation: {name}")
            result[name] = prepared[canonical]
        return result

    def reshape_kv_cache_tensors(self, runner, config, raw, native_reshape):
        import torch
        from .integration.specs import OscarFullAttentionSpec, packed_view
        specs = runner._get_layer_kv_cache_specs(config)
        full = {name: spec for name, spec in specs.items() if isinstance(spec, OscarFullAttentionSpec)}
        # Reuse the exact native GDN reshape on its original objects and bytes.
        # The native skip set is an existing initializer seam, restored even
        # if reshape raises. No GDN dimensions or dtype are reconstructed here.
        previous = runner.runner_only_attn_layers
        runner.runner_only_attn_layers = set(previous) | set(full)
        try:
            views = native_reshape(config, raw)
        finally:
            runner.runner_only_attn_layers = previous
        rotations = self._rotations(full, runner.device)
        scheduler = runner.vllm_config.scheduler_config
        capture_sizes = runner.vllm_config.compilation_config.cudagraph_capture_sizes or []
        token_capacity = max(scheduler.max_num_batched_tokens, max(capture_sizes, default=0))
        heads = runner.model_config.get_num_attention_heads(runner.parallel_config)
        for name, spec in full.items():
            allocation = raw[name]
            if not torch.is_tensor(allocation) or allocation.numel() % spec.page_size_bytes:
                raise OscarReadinessError(f"invalid native raw allocation for {name}")
            blocks = allocation.numel() // spec.page_size_bytes
            if blocks < config.num_blocks:
                raise OscarReadinessError("worker physical blocks are fewer than scheduler blocks")
            byte_raw = allocation.view(torch.uint8)
            packed = packed_view(byte_raw, blocks, spec.layout)
            snapshots = self._snapshot_layout(spec)
            wk, wv, tags = snapshots.views(byte_raw, blocks)
            tags.fill_(-1)
            workspace = self.ensure_workspace(heads, spec.num_kv_heads, spec.head_size, runner.device)
            if workspace.geometry.tokens < token_capacity:
                raise OscarReadinessError("native token capacity exceeds the profiled OSCAR workspace")
            rotation = rotations[name]
            rk = rotation.key_transposed.T
            rv = rotation.inverse_value_transposed
            self.layers[name] = LayerState(byte_raw, packed, spec, snapshots, wk, wv, tags,
                                          rk, rv, rotation.key_transposed, rotation.value_transposed,
                                          rotation.hadamard, blocks, workspace)
            views[name] = packed
            from .telemetry import emit_once
            emit_once("cache_layout", key=name, layer=name, physical_blocks=blocks,
                      physical_block_tokens=spec.block_size, page_bytes=spec.page_size_bytes,
                      snapshot_bytes_per_page=snapshots.required_bytes,
                      scratch_bytes=workspace.geometry.total_bytes,
                      raw_storage_ptr=byte_raw.data_ptr(), gdn_reshape="native")
        return views

    def layer_state(self, name):
        try:
            return self.layers[name]
        except KeyError as error:
            raise OscarReadinessError(f"FULL layer cache was not initialized: {name}") from error


def create_runtime():
    return AscendRuntimeProvider()
