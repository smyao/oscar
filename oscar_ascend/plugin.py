"""vllm.general_plugins entry point, stdlib-only and reversible.

Archive #28/#77/#78: register must not initialize torch/NPU/native modules.
Archive #27/#31–33: enabled routing requires an external ready runtime;
never attach undeclared fields to VllmConfig or silently keep native FULL.
Archive #34/#36: native dummy warmup must not masquerade as eager prefill.
"""

import functools
import importlib.abc
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .integration.runtime_api import require_runtime


class OscarHookConflictError(RuntimeError):
    """Another integration changed a descriptor owned by this plugin."""


@dataclass
class _Patch:
    owner: type
    name: str
    original: Any
    replacement: Any
    had_own_attribute: bool


_patches: list[_Patch] = []
_finder = None
_enabled = False


def _clear_loaded_selector_cache() -> None:
    # _cached_get_attn_backend memoizes platform decisions. An existing native
    # result must not outlive hook installation, or an OSCAR result restoration.
    module = sys.modules.get("vllm.v1.attention.selector")
    if module is None:
        return
    selector = getattr(module, "_cached_get_attn_backend", None)
    clear = getattr(selector, "cache_clear", None)
    if clear is not None:
        clear()


def _patch(owner: type, name: str, builder) -> None:
    for existing in _patches:
        if existing.owner is owner and existing.name == name:
            if owner.__dict__.get(name) is not existing.replacement:
                raise OscarHookConflictError(f"{owner.__name__}.{name} changed after OSCAR installation")
            return
    had_own = name in owner.__dict__
    descriptor = next((base.__dict__[name] for base in owner.__mro__ if name in base.__dict__), None)
    if descriptor is None:
        raise OscarHookConflictError(f"required native seam missing: {owner.__name__}.{name}")
    if isinstance(descriptor, classmethod):
        function = descriptor.__func__
        replacement = classmethod(functools.wraps(function)(builder(function)))
    elif isinstance(descriptor, staticmethod):
        function = descriptor.__func__
        replacement = staticmethod(functools.wraps(function)(builder(function)))
    else:
        function = descriptor
        replacement = functools.wraps(function)(builder(function))
    setattr(owner, name, replacement)
    _patches.append(_Patch(owner, name, descriptor, replacement, had_own))


def _patch_platform(module: ModuleType) -> None:
    cls = module.NPUPlatform

    def route(original):
        def wrapped(platform_cls, selected_backend, attn_selector_config, num_heads=None):
            native = original(platform_cls, selected_backend, attn_selector_config, num_heads)
            # GDN has its own get_mamba_attn_backend and never enters this seam.
            # MLA/sparse/encoder attention are outside the dense FULL target.
            # Route by selector semantics, not one backend path string: choosing
            # native FA3 must not silently bypass an explicitly enabled OSCAR.
            if (getattr(attn_selector_config, "use_mla", False)
                    or getattr(attn_selector_config, "use_sparse", False)
                    or getattr(attn_selector_config, "use_compress", False)
                    or getattr(attn_selector_config, "attn_type", "decoder") != "decoder"):
                return native
            require_runtime()
            return "oscar_ascend.integration.backend.OscarAttentionBackend"
        return wrapped

    def register_specs(original):
        def wrapped(platform_cls, vllm_config):
            original(platform_cls, vllm_config)
            from .integration.specs import register_oscar_spec
            register_oscar_spec()
        return wrapped

    _patch(cls, "get_attn_backend_cls", route)
    _patch(cls, "register_custom_kv_cache_specs", register_specs)
    _clear_loaded_selector_cache()


def _patch_runner(module: ModuleType) -> None:
    cls = module.NPUModelRunner

    def specs(original):
        def wrapped(runner):
            provider = require_runtime()
            return provider.transform_kv_cache_specs(runner, original(runner))
        return wrapped

    def allocate(original):
        def wrapped(runner, kv_cache_config):
            provider = require_runtime()
            return provider.allocate_kv_cache_tensors(
                runner, kv_cache_config, lambda config: original(runner, config))
        return wrapped

    def reshape(original):
        def wrapped(runner, kv_cache_config, kv_cache_raw_tensors):
            provider = require_runtime()
            return provider.reshape_kv_cache_tensors(
                runner, kv_cache_config, kv_cache_raw_tensors,
                lambda config, raw: original(runner, config, raw))
        return wrapped

    def dummy_run(original):
        def wrapped(runner, *args, **kwargs):
            # Native model_runner_v1.py warmup may use ordinary builder.build
            # with ChunkedPrefill even though every dummy slot is -1. Keep an
            # exact, nested scope around the original call; the builder marks
            # dummy_origin without reading slots back from NPU. Import only
            # when this deferred native seam actually executes (#77/#78).
            from .integration.dummy_context import native_dummy_run
            with native_dummy_run():
                return original(runner, *args, **kwargs)
        return wrapped

    _patch(cls, "get_kv_cache_spec", specs)
    _patch(cls, "_allocate_kv_cache_tensors", allocate)
    _patch(cls, "_reshape_kv_cache_tensors", reshape)
    _patch(cls, "_dummy_run", dummy_run)


_callbacks = {
    "vllm_ascend.platform": _patch_platform,
    "vllm_ascend.worker.model_runner_v1": _patch_runner,
}


def _patch_graph_evidence(module):
    # Native acl_graph.py:138-257 owns the capture/replay branch. Hook only
    # when explicitly collecting probe evidence, not in performance runs.
    if not os.environ.get("OSCAR_TRACE_DIR"):
        return
    def observe(original):
        def wrapped(wrapper,*args,**kwargs):
            context=module.get_forward_context()
            active=context.cudagraph_runtime_mode==wrapper.runtime_mode
            descriptor=context.batch_descriptor
            entry=wrapper.concrete_aclgraph_entries.get(descriptor) if active else None
            replay=entry is not None and entry.aclgraph is not None
            result=original(wrapper,*args,**kwargs)
            if active:
                entry=wrapper.concrete_aclgraph_entries.get(descriptor)
                if entry is not None and entry.aclgraph is not None:
                    from .telemetry import emit_once, emit_throttled
                    emit_once("graph_replay_launch_return" if replay else "graph_capture_return",
                              key=(id(wrapper),str(descriptor),replay),
                              descriptor=str(descriptor),mode=str(wrapper.runtime_mode),
                              device_completion="not_established_by_launch")
                    if replay:
                        # FULL_DECODE_ONLY replay bypasses the Python attention
                        # path, so this is the only decode-phase liveness record.
                        emit_throttled("graph_replay_progress", key=id(wrapper),
                                       descriptor=str(descriptor),mode=str(wrapper.runtime_mode),
                                       device_completion="not_established_by_launch")
            return result
        return wrapped
    _patch(module.ACLGraphWrapper,"__call__",observe)


_callbacks["vllm_ascend.compilation.acl_graph"]=_patch_graph_evidence


def _apply_callback(callback, module: ModuleType) -> None:
    first = len(_patches)
    try:
        callback(module)
    except BaseException:
        # A future import can fail after one descriptor was patched. Restore
        # only this callback's changes before propagating that import failure.
        for patch in reversed(_patches[first:]):
            if patch.owner.__dict__.get(patch.name) is not patch.replacement:
                raise OscarHookConflictError(f"partial install changed externally: {patch.name}")
            if patch.had_own_attribute:
                setattr(patch.owner, patch.name, patch.original)
            else:
                delattr(patch.owner, patch.name)
        del _patches[first:]
        raise


class _AfterExecLoader(importlib.abc.Loader):
    def __init__(self, loader, callback):
        self.loader = loader
        self.callback = callback

    def create_module(self, spec):
        creator = getattr(self.loader, "create_module", None)
        return creator(spec) if creator is not None else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        if _enabled:
            _apply_callback(self.callback, module)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _NativeSeamFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        callback = _callbacks.get(fullname)
        if callback is None or not _enabled:
            return None
        # Preserve the host's loader choice, including editable install hooks.
        # Only finders after us are considered to avoid re-entering ourselves.
        following = False
        for finder in tuple(sys.meta_path):
            if finder is self:
                following = True
                continue
            if not following:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None:
                if spec.loader is None or not hasattr(spec.loader, "exec_module"):
                    raise OscarHookConflictError(f"native seam has unsupported loader: {fullname}")
                spec.loader = _AfterExecLoader(spec.loader, callback)
                return spec
        return None


def register() -> None:
    """Install deferred hooks only when explicitly enabled by deployment.

    This function imports no torch, torch_npu, vllm, or vllm_ascend module.
    It also performs no operator loading or device initialization.
    """
    global _enabled, _finder
    if os.environ.get("OSCAR_ENABLED") != "1":
        return
    if _enabled:
        return
    from .integration.runtime_api import ensure_default_runtime_factory
    ensure_default_runtime_factory()
    _enabled = True
    _finder = _NativeSeamFinder()
    sys.meta_path.insert(0, _finder)
    try:
        for name, callback in _callbacks.items():
            module = sys.modules.get(name)
            # Never patch an in-flight module: our loader wraps future imports,
            # and fully initialized existing modules can be patched immediately.
            if module is not None:
                spec = getattr(module, "__spec__", None)
                if getattr(spec, "_initializing", False):
                    raise OscarHookConflictError(f"register called while native seam is importing: {name}")
                _apply_callback(callback, module)
    except BaseException:
        unregister()
        raise


def unregister() -> None:
    """Restore exact original descriptors; refuse to erase another plugin."""
    global _enabled, _finder
    for patch in _patches:
        if patch.owner.__dict__.get(patch.name) is not patch.replacement:
            raise OscarHookConflictError(f"cannot restore externally changed {patch.owner.__name__}.{patch.name}")
    for patch in reversed(_patches):
        if patch.had_own_attribute:
            setattr(patch.owner, patch.name, patch.original)
        else:
            delattr(patch.owner, patch.name)
    _patches.clear()
    if _finder in sys.meta_path:
        sys.meta_path.remove(_finder)
    _finder = None
    _enabled = False
    _clear_loaded_selector_cache()


def hook_status() -> dict:
    return {"enabled": _enabled, "patched": [f"{p.owner.__name__}.{p.name}" for p in _patches]}
