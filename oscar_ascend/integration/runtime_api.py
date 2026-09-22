"""Deferred runtime readiness contract.

Archive #27: an installed plugin is not a complete service implementation.
Archive #31–33: runtime state belongs here, never on VllmConfig.
Archive #28/#77/#78: importing this module imports no torch or native engine.
"""

from collections.abc import Callable
from typing import Any, Protocol


class OscarReadinessError(RuntimeError):
    """OSCAR was requested but its concrete runtime is not ready."""


class RuntimeProvider(Protocol):
    def assert_ready(self) -> None: ...
    def get_impl_cls(self) -> type: ...
    def transform_kv_cache_specs(self, runner: Any, native_specs: dict) -> dict: ...
    def allocate_kv_cache_tensors(self, runner: Any, config: Any, native_allocate: Callable) -> dict: ...
    def reshape_kv_cache_tensors(self, runner: Any, config: Any, raw: dict, native_reshape: Callable) -> dict: ...


_provider: RuntimeProvider | None = None
_factory: Callable[[], RuntimeProvider] | None = None
_constructing = False
_required = ("assert_ready", "get_impl_cls", "transform_kv_cache_specs",
             "allocate_kv_cache_tensors", "reshape_kv_cache_tensors")


def install_runtime(provider: RuntimeProvider) -> None:
    global _provider
    missing = [name for name in _required if not callable(getattr(provider, name, None))]
    if missing:
        raise OscarReadinessError(f"runtime provider lacks required methods: {missing}")
    if _provider is not None and _provider is not provider:
        raise OscarReadinessError("a different OSCAR runtime provider is already installed")
    _provider = provider


def install_runtime_factory(factory: Callable[[], RuntimeProvider]) -> None:
    global _factory
    if not callable(factory):
        raise TypeError("runtime factory must be callable")
    if _factory is not None and _factory is not factory:
        raise OscarReadinessError("a different OSCAR runtime factory is already installed")
    _factory = factory


def clear_runtime() -> None:
    global _provider, _factory
    _provider = None
    _factory = None


def require_runtime() -> RuntimeProvider:
    global _constructing
    if _provider is None and _factory is not None:
        if _constructing:
            raise OscarReadinessError("recursive runtime initialization")
        _constructing = True
        try:
            install_runtime(_factory())
        finally:
            _constructing = False
    if _provider is None:
        raise OscarReadinessError(
            "OSCAR_ENABLED=1 but no runtime provider is installed; "
            "fused AscendC operators, cache lifecycle and graph workspace must be ready "
            "before FULL attention can be routed"
        )
    _provider.assert_ready()
    return _provider
