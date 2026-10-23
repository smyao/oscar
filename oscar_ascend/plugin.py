"""vllm.general_plugins entry point, stdlib-only and reversible.

Archive #28/#77/#78: register must not initialize torch/NPU/native modules.
Archive #27/#31–33: enabled routing requires an external ready runtime;
never attach undeclared fields to VllmConfig or silently keep native FULL.
Archive #34/#36: native dummy warmup must not masquerade as eager prefill.
Archive #37-49/#127-129/#142/#155: request-window lease notifications wrap
native lifecycle methods without changing scheduler selection or token budget.
Archive #157-160: greedy sampling hooks keep masked rows outside the writer
without changing random probabilities or the frozen device kernels.
Archive #26/#127/#128: prefix checkpoints follow actual FULL/GDN alignment,
without changing state layouts, scheduler class, token budget or native files.
"""

import functools
import importlib.abc
import inspect
import json
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .integration.runtime_api import require_runtime


class OscarHookConflictError(RuntimeError):
    """Another integration changed a descriptor owned by this plugin."""


@dataclass
class _Patch:
    owner: type | ModuleType
    name: str
    original: Any
    replacement: Any
    had_own_attribute: bool


_patches: list[_Patch] = []
_finder = None
_enabled = False
_passive_only = False


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

    def classified_dispatch(original):
        signature = inspect.signature(original)

        def wrapped(runner, *args, **kwargs):
            from .integration.prefill_state_contract import force_prefill_eager
            bound = signature.bind(runner, *args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            if force_prefill_eager(
                    runner, values["num_reqs"], values["num_scheduled_tokens_np"],
                    values["max_num_scheduled_tokens"], values.get("force_uniform_decode")):
                values["force_eager"] = True
                if not getattr(runner, "_oscar_prefill_state_logged", False):
                    print("[oscar] PREFILL_STATE_FIX cache_mode=none "
                          "route=native_prefill mtp=enabled decode_graph=retained", flush=True)
                    runner._oscar_prefill_state_logged = True
            return original(*bound.args, **bound.kwargs)
        return wrapped

    if hasattr(cls, "_determine_batch_execution_and_padding"):
        _patch(cls, "_determine_batch_execution_and_padding", classified_dispatch)

    # Validate target identity before Attention constructors select rotations;
    # request-window sidecars are sealed after native loading. Registration is light.
    def loaded(original):
        def wrapped(runner, *args, **kwargs):
            provider = require_runtime()
            # #86/#97: raw `vllm serve` can differ from OSCAR_TARGET_CONFIG.
            # Attention constructors load rotations inside original(), so the
            # model identity must agree before entering native model loading.
            from .integration.model_identity import validate_runner_model_identity
            validate_runner_model_identity(provider.config, runner)
            result = original(runner, *args, **kwargs)
            if getattr(provider, "request_windows_enabled", False):
                provider.seal_request_windows(runner)
            return result
        return wrapped

    def prepared(original):
        @contextmanager
        def wrapped(runner, *args, **kwargs):
            provider = require_runtime()
            enabled = getattr(provider, "request_windows_enabled", False)
            with original(runner, *args, **kwargs):
                if enabled:
                    provider.begin_request_window_input_prep(runner)
                try:
                    yield
                finally:
                    if enabled:
                        provider.end_request_window_input_prep(runner)
        return wrapped

    def updated(original):
        def wrapped(runner, scheduler_output):
            if os.environ.get("OSCAR_TIMER_CONTROL"):
                with _observer_timer().phase("host_update_states"):
                    result = original(runner, scheduler_output)
            else:
                result = original(runner, scheduler_output)
            provider = require_runtime()
            if getattr(provider, "request_windows_enabled", False):
                provider.publish_request_window_map(runner, scheduler_output)
            return result  # native deferred MTP corrections must survive
        return wrapped

    for name, wrapper in (("load_model", loaded), ("synchronize_input_prep", prepared),
                          ("_update_states", updated)):
        if hasattr(cls, name):
            _patch(cls, name, wrapper)
    _patch_passive_runner(module)


def request_window_runner_hooks_present(runner):
    names = {patch.name for patch in _patches if patch.owner in type(runner).__mro__}
    return {"load_model", "synchronize_input_prep", "_update_states"} <= names


def _request_windows_configured():
    # EngineCore has no NPU runtime provider. Read only the explicit target
    # flag; do not initialize torch or require_runtime in scheduler hooks.
    path = Path(os.environ.get("OSCAR_TARGET_CONFIG",
                Path(__file__).resolve().parents[1] / "configs/target.json"))
    enabled = json.loads(path.read_text()).get("experimental_request_windows", False)
    if type(enabled) is not bool:
        raise OscarHookConflictError("experimental_request_windows must be an explicit boolean")
    return enabled


def _patch_request_window_scheduler(module):
    """Keep native schedule/preempt bodies; add missing forced-reset notices."""
    if not _request_windows_configured():
        return

    def preempt(original):
        def wrapped(scheduler, request, timestamp):
            result = original(scheduler, request, timestamp)
            from .request_windows import note_preemption
            note_preemption(scheduler, request.request_id)
            return result
        return wrapped

    def schedule(original):
        def wrapped(scheduler, *args, **kwargs):
            result = original(scheduler, *args, **kwargs)
            from .request_windows import publish_preemptions
            return publish_preemptions(scheduler, result)
        return wrapped

    _patch(module.Scheduler, "_preempt_request", preempt)
    _patch(module.Scheduler, "schedule", schedule)


def request_window_scheduler_hooks_present(scheduler_cls):
    names = {patch.name for patch in _patches
             if patch.owner in scheduler_cls.__mro__ and
             getattr(scheduler_cls, patch.name) is patch.replacement}
    return {"_preempt_request", "schedule"} <= names


def _patch_prefix_checkpoint_scheduler(module):
    """Bind physical group boundaries while retaining the native scheduler."""
    def initialized(original):
        def wrapped(scheduler, *args, **kwargs):
            result = original(scheduler, *args, **kwargs)
            from .integration.prefix_checkpoint import policy_for_scheduler
            policy = policy_for_scheduler(scheduler)
            scheduler._oscar_prefix_checkpoint_policy = policy
            if policy is not None:
                print("[oscar] PREFIX_CHECKPOINT_FIX "
                      f"joint_tokens={policy.alignment_tokens} "
                      f"gdn_tokens={policy.state_block_tokens} "
                      "terminal=last_token_and_full_drop low_budget=native_state_blocks",
                      flush=True)
            return result
        return wrapped

    def aligned(original):
        def wrapped(scheduler, request, num_new_tokens,
                    num_new_local_computed_tokens=0, num_external_computed_tokens=0):
            if scheduler._oscar_prefix_checkpoint_policy is not None:
                from .integration.prefix_checkpoint import split_prefix_checkpoint
                planned = split_prefix_checkpoint(scheduler, request, num_new_tokens,
                                                  num_new_local_computed_tokens,
                                                  num_external_computed_tokens)
                if planned is not None:
                    return planned
            return original(scheduler, request, num_new_tokens,
                            num_new_local_computed_tokens, num_external_computed_tokens)
        return wrapped

    _patch(module.Scheduler, "__init__", initialized)
    _patch(module.Scheduler, "_mamba_block_aligned_split", aligned)


def _patch_scheduler(module):
    _patch_prefix_checkpoint_scheduler(module)
    _patch_request_window_scheduler(module)


def _patch_attention_registration(module: ModuleType) -> None:
    """Allocate sidecars while native load_model still measures persistent bytes."""
    def registered(original):
        def wrapped(layer, *args, **kwargs):
            result = original(layer, *args, **kwargs)
            provider = getattr(getattr(layer, "impl", None), "provider", None)
            if provider is not None and getattr(provider, "request_windows_enabled", False):
                provider.register_request_window_layer(layer)
            return result
        return wrapped
    _patch(module.Attention, "__init__", registered)


def _patch_passive_runner(module: ModuleType) -> None:
    """Observe native runner boundaries without touching their arguments."""
    if not _observer_requested():
        return
    cls = module.NPUModelRunner

    def execute(original):
        def wrapped(runner, scheduler_output, *args, **kwargs):
            timer = _observer_timer()
            timer.begin_step(scheduler_output)
            try:
                result = original(runner, scheduler_output, *args, **kwargs)
            except BaseException as error:
                timer.end_step(error)
                raise
            if result is not None:
                timer.end_step()
            return result
        return wrapped

    def sample(original):
        def wrapped(runner, *args, **kwargs):
            timer = _observer_timer()
            try:
                with timer.phase("sample_tokens"):
                    result = original(runner, *args, **kwargs)
            except BaseException as error:
                timer.end_step(error)
                raise
            timer.end_step()
            return result
        return wrapped

    def forward(original):
        def wrapped(runner, *args, **kwargs):
            with _observer_timer().phase("target_forward"):
                return original(runner, *args, **kwargs)
        return wrapped

    def draft(original):
        def wrapped(runner, *args, **kwargs):
            with _observer_timer().phase("draft_proposal"):
                return original(runner, *args, **kwargs)
        return wrapped

    _patch(cls, "execute_model", execute)
    _patch(cls, "sample_tokens", sample)
    _patch(cls, "_model_forward", forward)
    _patch(cls, "propose_draft_token_ids", draft)
    # These CPU preparation seams exist in the pinned native runner. Observe
    # availability without requiring optional seams in old host fixtures.
    if _diagnostic_requested():
        def host_scope(name):
            def decorate(original):
                def wrapped(runner, *args, **kwargs):
                    with _observer_timer().phase(name):
                        return original(runner, *args, **kwargs)
                return wrapped
            return decorate
        for name in ("_update_states", "_prepare_inputs", "_build_attention_metadata", "_bookkeeping_sync",
                     "_update_full_graph_params_if_needed"):
            if hasattr(cls, name):
                _patch(cls, name, host_scope("host" + name))


def _diagnostic_requested():
    path = os.environ.get("OSCAR_TIMER_CONTROL")
    return bool(path and Path(path).is_file())


def _observer_requested():
    return bool(os.environ.get("OSCAR_PASSIVE_TIMING_CONTROL")) or _diagnostic_requested()


def _observer_timer():
    # Installation checked the file once; hot scopes must not stat a file.
    if os.environ.get("OSCAR_TIMER_CONTROL"):
        from . import diagnostic_timer
        return diagnostic_timer
    from . import passive_timing
    return passive_timing


def _patch_mtp_forward(module: ModuleType) -> None:
    """Observe MTP and optionally compact explicitly enabled later eager calls.

    #140/#142/#148/#155: __call__ must wrap the compile decorator so its
    documented skip_compiled branch is reached before any compiled graph.
    """
    cls = getattr(module, "Qwen3_5MTP", None)
    if cls is None:
        return
    def draft_model(original):
        def wrapped(model, *args, **kwargs):
            with _observer_timer().phase("draft_forward"):
                return original(model, *args, **kwargs)
        return wrapped
    if _observer_requested():
        _patch(cls, "forward", draft_model)
    if not _passive_only:
        def compact(original):
            def wrapped(model, *args, **kwargs):
                provider = require_runtime()
                enabled = getattr(provider, "config", {}).get("experimental_compact_later_mtp", False)
                if type(enabled) is not bool:
                    raise OscarHookConflictError("experimental_compact_later_mtp must be an explicit boolean")
                if not enabled:
                    return original(model, *args, **kwargs)
                from .integration.compact_later_mtp import compact_later_mtp_call
                return compact_later_mtp_call(original, model, args, kwargs, provider)
            return wrapped
        _patch(cls, "__call__", compact)


def _patch_gdn_prefill_classification(module: ModuleType) -> None:
    """#163: preserve none-mode prefill state; GDN kernels remain native."""
    def retain_prefill(original):
        def wrapped(builder, common_attn_metadata, spec_sequence_masks_cpu, num_accepted_tokens):
            from .integration.prefill_state_contract import keep_prefill_state
            return keep_prefill_state(builder, original, common_attn_metadata,
                                      spec_sequence_masks_cpu, num_accepted_tokens)
        return wrapped
    _patch(module.AscendGDNAttentionMetadataBuilder,
           "_fold_spec_sized_prefill_chunks_into_spec", retain_prefill)


def _patch_mtp_proposer(module: ModuleType) -> None:
    """Archive #157: bind explicit native discard rows without altering GDN.

    Both construction paths run before the native model/graph call. The
    context is reset on exceptions and never persists a prior batch's mask.
    """
    cls = module.AscendSpecDecodeBaseProposer

    def scoped(dummy):
        def decorate(original):
            def wrapped(proposer, *args, **kwargs):
                if proposer.method != "mtp":
                    return original(proposer, *args, **kwargs)
                from .integration.draft_discard import native_draft_scope
                common = None if dummy else kwargs.get(
                    "common_attn_metadata", args[5] if len(args) > 5 else None)
                rejected = None if dummy else kwargs.get(
                    "num_rejected_tokens_gpu", args[15] if len(args) > 15 else None)
                with native_draft_scope(proposer, dummy=dummy, common=common, rejected_tokens=rejected):
                    return original(proposer, *args, **kwargs)
            return wrapped
        return decorate

    _patch(cls, "_propose", scoped(False))
    _patch(cls, "dummy_run", scoped(True))


def _patch_native_attention(module: ModuleType) -> None:
    """Record native attention, including CP inheritance and C8 override."""
    if not _observer_requested():
        return
    def observe(original):
        def wrapped(impl, *args, **kwargs):
            with _observer_timer().phase("native_attention", backend=type(impl).__name__):
                return original(impl, *args, **kwargs)
        return wrapped
    _patch(module.AscendAttentionBackendImpl, "forward", observe)
    _patch(module.AscendC8AttentionBackendImpl, "forward", observe)


def _patch_native_gdn(module: ModuleType) -> None:
    """#140/#143: observe native GDN separately from FULL; no data changes."""
    if not _diagnostic_requested():
        return
    def observe(original):
        def wrapped(layer, *args, **kwargs):
            with _observer_timer().phase("native_gdn", backend=type(layer).__name__):
                return original(layer, *args, **kwargs)
        return wrapped
    _patch(module.AscendGatedDeltaNetAttention, "forward", observe)


def _patch_module_sampling_function(module: ModuleType, name: str, builder) -> None:
    """Track a module function in the same reversible registry as class seams."""
    for existing in _patches:
        if existing.owner is module and existing.name == name:
            if module.__dict__.get(name) is not existing.replacement:
                raise OscarHookConflictError(f"{module.__name__}.{name} changed after OSCAR installation")
            return
    current = module.__dict__.get(name)
    if not callable(current):
        raise OscarHookConflictError(f"required native seam missing: {module.__name__}.{name}")
    # A deferred sampler import may already have imported our defining-module
    # wrapper. Restore the true native function in both modules on unregister.
    original = getattr(current, "_oscar_native_sampling_original", current)
    replacement = functools.wraps(original)(builder(original))
    replacement._oscar_native_sampling_original = original
    setattr(module, name, replacement)
    _patches.append(_Patch(module, name, original, replacement, True))


def _patch_native_greedy_sampling(module: ModuleType) -> None:
    """Patch both greedy aliases and guard native first-row cumulative reads."""
    if _passive_only or os.environ.get("OSCAR_ENABLED") != "1":
        return
    if module.__name__ == "vllm_ascend.sample.rejection_sampler" and getattr(module, "HAS_TRITON", None) is False:
        # The native PyTorch writer does not execute either unsafe Triton load.
        # Respect that native capability choice; no numerical failure fallback.
        return

    def greedy_writer(original):
        def replacement(*args, **kwargs):
            from .integration.greedy_sampling import rejection_greedy_sample_with_triton
            return rejection_greedy_sample_with_triton(*args, **kwargs)
        return replacement

    _patch_module_sampling_function(module, "rejection_greedy_sample_with_triton", greedy_writer)
    if module.__name__ == "vllm_ascend.sample.rejection_sampler":
        already_installed = any(p.owner is module and p.name == "rejection_sample" for p in _patches)
        def guarded_sampling(original):
            def replacement(draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
                            draft_probs, target_logits_or_tuple, bonus_token_ids, sampling_metadata,
                            synthetic_mode=False, synthetic_conditional_rates=None, ori_target_logits=None):
                from .integration.greedy_sampling import rejection_sample_with_guarded_cu
                return rejection_sample_with_guarded_cu(
                    original, draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
                    draft_probs, target_logits_or_tuple, bonus_token_ids, sampling_metadata,
                    synthetic_mode, synthetic_conditional_rates, ori_target_logits)
            return replacement
        _patch_module_sampling_function(module, "rejection_sample", guarded_sampling)
        if not already_installed:
            print("[oscar] PERF_SAMPLER_FIX greedy_rows=isolated cumulative_prefix_bytes=32", flush=True)


def _patch_compiler_cache(module):
    # Archive #169: native cache load must not call inner AOT code with the
    # outer graph's argument list. Installed only for active OSCAR routing.
    def checked_load(original):
        def wrapped(compiler, handle, graph, example_inputs, graph_index, compile_range):
            from .integration.compile_cache import load_compatible
            return load_compatible(original, compiler, handle, graph,
                                   example_inputs, graph_index, compile_range)
        return wrapped

    _patch(module.AscendCompiler, "load", checked_load)


_callbacks = {
    "vllm_ascend.compilation.compiler_interface": _patch_compiler_cache,
    "vllm_ascend.platform": _patch_platform,
    "vllm_ascend.worker.model_runner_v1": _patch_runner,
    "vllm_ascend.attention.attention_v1": _patch_native_attention,
    "vllm.model_executor.models.qwen3_5_mtp": _patch_mtp_forward,
    "vllm_ascend.spec_decode.llm_base_proposer": _patch_mtp_proposer,
    "vllm.model_executor.layers.attention.attention": _patch_attention_registration,
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
    "vllm_ascend.ops.triton.reject_sample": _patch_native_greedy_sampling,
    "vllm_ascend.sample.rejection_sampler": _patch_native_greedy_sampling,
    "vllm_ascend.ops.gdn_attn_builder": _patch_gdn_prefill_classification,
    "vllm_ascend.ops.gdn": _patch_native_gdn,
}


def _patch_graph_evidence(module):
    # Native acl_graph.py:138-257 owns the capture/replay branch. Hook only
    # when explicitly collecting probe evidence, not in performance runs.
    if not os.environ.get("OSCAR_TRACE_DIR") and not _diagnostic_requested():
        return
    def observe(original):
        def wrapped(wrapper,*args,**kwargs):
            context=module.get_forward_context()
            active=context.cudagraph_runtime_mode==wrapper.runtime_mode
            descriptor=context.batch_descriptor
            entry=wrapper.concrete_aclgraph_entries.get(descriptor) if active else None
            replay=entry is not None and entry.aclgraph is not None
            # #133/#142-143: arm only at the model boundary after the primary
            # batch. Captured kernels stay unchanged; time whole replay from
            # outside its graph. Eager attention has its own phase events.
            from . import device_timing
            observer = _observer_timer()
            from .integration.dummy_context import is_native_dummy_run
            if not is_native_dummy_run():
                device_timing.refresh()
            if active and not replay and os.environ.get("OSCAR_TIMER_CONTROL") and observer.active():
                observer.note_graph_capture()
            if replay and observer.active():
                with observer.phase("graph_replay", tokens=getattr(descriptor, "num_tokens", None),
                                          requests=getattr(descriptor, "num_reqs", None),
                                          graph_scope="whole_model_graph"):
                    result=original(wrapper,*args,**kwargs)
            elif replay and device_timing.active():
                from .timing import phase
                with phase("graph_replay", tokens=getattr(descriptor, "num_tokens", None),
                           requests=getattr(descriptor, "num_reqs", None),
                           graph_scope="whole_model_graph"):
                    result=original(wrapper,*args,**kwargs)
            else:
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
    if _passive_only:
        if callback is _patch_runner:
            callback = _patch_passive_runner
        elif callback not in {_patch_graph_evidence, _patch_mtp_forward,
                              _patch_native_attention, _patch_native_gdn}:
            return
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
        # Spawned native TP workers invoke this general-plugin entry point.
        # They must install only observation, never the OSCAR runtime factory.
        if _diagnostic_requested():
            register_passive_observer()
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


def register_passive_observer() -> None:
    """Instrument an explicitly native service without routing OSCAR FULL."""
    global _enabled, _finder, _passive_only
    if not _observer_requested() or _enabled:
        return
    _passive_only = True
    _enabled = True
    _finder = _NativeSeamFinder()
    sys.meta_path.insert(0, _finder)
    try:
        for name, callback in _callbacks.items():
            module = sys.modules.get(name)
            if module is not None:
                _apply_callback(callback, module)
    except BaseException:
        unregister()
        raise


def unregister() -> None:
    """Restore exact original descriptors; refuse to erase another plugin."""
    global _enabled, _finder, _passive_only
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
    _passive_only = False
    _clear_loaded_selector_cache()


def hook_status() -> dict:
    return {"enabled": _enabled, "patched": [f"{p.owner.__name__}.{p.name}" for p in _patches]}
