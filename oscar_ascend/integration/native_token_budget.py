"""Archive #34/#36/#127/#128: reserve the engine's actual token/graph capacity.

Native model_loader/utils.py:initialize_model installs the current VllmConfig
before constructing Attention/OSCAR implementations, inside model loading and
before KV profiling. No native configuration or numerical operator is changed.
"""
from ..runtime import AscendRuntimeProvider
from .runtime_api import OscarReadinessError


class NativeTokenBudgetRuntime(AscendRuntimeProvider):
    def __init__(self, config):
        super().__init__(config)
        if (self.config.get("use_native_token_budget") is not True or
                "max_num_batched_tokens" in self.config or
                self.config.get("scheduler_cls") is not None or
                self.config.get("experimental_whole_prefill", False) or
                self.config.get("experimental_bounded_whole_prefill", False) or
                self.config.get("history_token_capacity") is not None):
            raise OscarReadinessError("native token budget requires an omitted budget and native scheduler")
        self._native_token_budget = None
        self._native_token_capacity = None

    def _bind_native_capacity(self, native):
        budget = native.scheduler_config.max_num_batched_tokens
        captures = list(native.compilation_config.cudagraph_capture_sizes or [])
        if (type(budget) is not int or budget <= 0 or
                any(type(size) is not int or size <= 0 for size in captures)):
            raise OscarReadinessError("native token/graph capacity must be positive integers")
        capacity = max(budget, max(captures, default=0))
        if self._native_token_budget is not None:
            if budget != self._native_token_budget or capacity > self._native_token_capacity:
                raise OscarReadinessError("native token capacity changed after pre-profile reservation")
            return
        self._native_token_budget = budget
        self._native_token_capacity = capacity
        # Runtime-local copy only. The effective target file still omits the
        # CLI argument; neither the native scheduler nor VllmConfig is mutated.
        self.config["max_num_batched_tokens"] = budget
        self.config["compilation_config"] = {
            **self.config.get("compilation_config", {}), "cudagraph_capture_sizes": captures}
        print(f"[oscar] NATIVE_TOKEN_BUDGET max_num_batched_tokens={budget} "
              f"workspace_token_capacity={capacity} source=native_scheduler", flush=True)

    def ensure_workspace(self, heads, kv_heads, dim, device):
        if self._native_token_budget is None:
            from vllm.config import get_current_vllm_config
            self._bind_native_capacity(get_current_vllm_config())
        if self.config["max_num_batched_tokens"] != self._native_token_budget:
            raise OscarReadinessError("runtime token budget changed after native resolution")
        return super().ensure_workspace(heads, kv_heads, dim, device)

    def reshape_kv_cache_tensors(self, runner, config, raw, native_reshape):
        if self._native_token_budget is None:
            raise OscarReadinessError("native token capacity was not reserved during model loading")
        self._bind_native_capacity(runner.vllm_config)
        return super().reshape_kv_cache_tensors(runner, config, raw, native_reshape)
