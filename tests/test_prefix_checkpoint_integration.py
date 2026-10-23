"""Archive #26/#37-49/#127/#128: prefix scheduler seams and real budget use.

Pinned Scheduler.schedule/AsyncScheduler/splitter bodies execute unchanged.
Initialization and allocation are host fixtures; cache geometry is native.
No NPU, state payload, graph, model quality or performance evidence is made.
The fourteen warm-tail fixtures exercise remaining budget with the actual
global limit fixed at 16384, without changing a serving argument.
"""

import inspect
import json
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from oscar_ascend import plugin
from oscar_ascend.integration import prefix_checkpoint as checkpoint
from test_bounded_whole_prefill_native import FixtureBase, Request, native_class, STATUS
from test_prefix_caching import _coordinator_with_hashing
from test_runtime_native import definitions, native_groups, native_specs
from test_stable_launch_frozen import FROZEN, authorized_model_arguments
from tools import install_serve, observe_serve


@pytest.fixture(autouse=True)
def restore_hooks():
    plugin.unregister()
    yield
    plugin.unregister()


def scheduler_class(native_groups, monkeypatch):
    """Use real schedule bodies; inject geometry before the plugin init seam."""
    cls = native_class()
    base = cls.__mro__[1]
    coordinator = _coordinator_with_hashing(native_groups, monkeypatch)
    split = definitions(
        "references/vllm/vllm/v1/core/sched/scheduler.py", "prefix_integration_native_split",
        set(), {}, monkeypatch, methods=[("Scheduler", "_mamba_block_aligned_split")]
    )._mamba_block_aligned_split

    def initialized(self, running=(), waiting=()):
        FixtureBase.__init__(self, running, waiting)
        self.cache_config = native_groups[4].cache_config
        self.block_size = coordinator.scheduler_block_size
        self.kv_cache_manager.coordinator = coordinator
        self.need_mamba_block_aligned_split = (
            self.cache_config.enable_prefix_caching
            and self.cache_config.mamba_cache_mode == "align")
        self.max_num_scheduled_tokens = 16384

    base.__init__ = initialized
    base._mamba_block_aligned_split = split
    return cls, base, coordinator


def test_real_hook_binds_once_preserves_geometry_and_restores(native_groups, monkeypatch, capsys):
    cls, base, coordinator = scheduler_class(native_groups, monkeypatch)
    original_init, original_split = base.__init__, base._mamba_block_aligned_split
    config = native_groups[4].cache_config
    config_before = vars(config).copy()
    specs = tuple(manager.kv_cache_spec for manager in coordinator.single_type_managers)
    hashes = coordinator.hash_block_size, coordinator.block_pool.hash_block_size
    calls = []
    original_policy = checkpoint.policy_for_scheduler

    def counted(scheduler):
        calls.append(scheduler)
        return original_policy(scheduler)

    monkeypatch.setattr(checkpoint, "policy_for_scheduler", counted)
    monkeypatch.setattr(plugin, "_request_windows_configured", lambda: False)
    module = NS(Scheduler=base)
    plugin._callbacks["vllm.v1.core.sched.scheduler"](module)
    installed = base.__init__, base._mamba_block_aligned_split
    plugin._callbacks["vllm.v1.core.sched.scheduler"](module)
    assert (base.__init__, base._mamba_block_aligned_split) == installed
    assert inspect.signature(base.__init__) == inspect.signature(original_init)
    assert inspect.signature(base._mamba_block_aligned_split) == inspect.signature(original_split)
    scheduler = cls()
    request = Request("prefill", prompt=32768)
    assert scheduler._mamba_block_aligned_split(request, 16384) == 16128
    request.num_computed_tokens = 16128
    assert scheduler._mamba_block_aligned_split(request, 16384) == 13824
    assert calls == [scheduler]
    assert scheduler.cache_config is config and vars(config) == config_before
    assert all(m.kv_cache_spec is spec for m, spec in zip(coordinator.single_type_managers, specs))
    assert hashes == (coordinator.hash_block_size, coordinator.block_pool.hash_block_size)
    assert "PREFIX_CHECKPOINT_FIX joint_tokens=2304 gdn_tokens=768" in capsys.readouterr().out
    plugin.unregister()
    assert base.__init__ is original_init
    assert base._mamba_block_aligned_split is original_split


@pytest.mark.parametrize("scope", ["apc_off", "none", "native"])
def test_inactive_geometry_delegates_exact_original_signature(native_groups, monkeypatch, scope):
    cls, base, coordinator = scheduler_class(native_groups, monkeypatch)
    config = native_groups[4].cache_config
    if scope == "apc_off":
        config.enable_prefix_caching = False
    elif scope == "none":
        config.mamba_cache_mode = "none"
    else:
        interface = native_groups[0]
        old = coordinator.single_type_managers[0].kv_cache_spec
        coordinator.single_type_managers[0].kv_cache_spec = interface.FullAttentionSpec(
            block_size=768, num_kv_heads=1, head_size=256, dtype=old.dtype,
            page_size_padded=old.page_size_bytes)
    calls = []
    original = base._mamba_block_aligned_split

    def recorded(self, request, num_new_tokens, num_new_local_computed_tokens=0,
                 num_external_computed_tokens=0):
        calls.append((request, num_new_tokens, num_new_local_computed_tokens,
                      num_external_computed_tokens))
        return original(self, request, num_new_tokens, num_new_local_computed_tokens,
                        num_external_computed_tokens)

    base._mamba_block_aligned_split = recorded
    plugin._patch_prefix_checkpoint_scheduler(NS(Scheduler=base))
    scheduler = cls()
    request = Request("request", prompt=32768)
    assert scheduler._oscar_prefix_checkpoint_policy is None
    expected = original(scheduler, request, 16384, 768, 768)
    assert scheduler._mamba_block_aligned_split(request, 16384, 768, 768) == expected
    assert calls == [(request, 16384, 768, 768)]


def test_passive_native_registration_never_installs_prefix_hook(native_groups, monkeypatch):
    _, base, _ = scheduler_class(native_groups, monkeypatch)
    module = ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = base
    original = base.__init__, base._mamba_block_aligned_split
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("OSCAR_ENABLED", "0")
    monkeypatch.setenv("OSCAR_PASSIVE_TIMING_CONTROL", "/tmp/prefix-passive-unused.json")
    plugin.register_passive_observer()
    assert (base.__init__, base._mamba_block_aligned_split) == original


@pytest.mark.parametrize("mismatch", ["metadata128", "scheduler", "coordinator"])
def test_invalid_physical_geometry_is_rejected_at_hook_initialization(native_groups, monkeypatch, mismatch):
    cls, base, coordinator = scheduler_class(native_groups, monkeypatch)
    if mismatch == "metadata128":
        manager = coordinator.single_type_managers[0]
        manager.kv_cache_spec = manager.kv_cache_spec.copy_with_new_block_size(128)
        message = "physical OSCAR"
    elif mismatch == "coordinator":
        coordinator.lcm_block_size = 768
        message = "alignment disagree"
    else:
        coordinator.scheduler_block_size = 768
        message = "alignment disagree"
    plugin._patch_prefix_checkpoint_scheduler(NS(Scheduler=base))
    with pytest.raises(ValueError, match=message):
        cls()


def test_real_schedule_waiting_then_running_progresses_with_2048_remaining(native_groups, monkeypatch):
    cls, base, _ = scheduler_class(native_groups, monkeypatch)
    plugin._patch_prefix_checkpoint_scheduler(NS(Scheduler=base))
    target = Request("target", prompt=32768)
    scheduler = cls(waiting=[target])
    observed = []
    actual_split = checkpoint.split_prefix_checkpoint

    def record(scheduler, request, budget, local=0, external=0):
        if request is target:
            observed.append((request.status, budget))
        return actual_split(scheduler, request, budget, local, external)

    monkeypatch.setattr(checkpoint, "split_prefix_checkpoint", record)
    endpoints = []
    for step in range(4):
        # These warm tails have passed both the joint terminal=2304 and the
        # native terminal=3840. Original native finishing work consumes
        # 14*1024 tokens before the long target is seen; no new warm-q4 route.
        tails = [Request(f"tail-{step}-{i}", prompt=5000, computed=3976) for i in range(14)]
        scheduler.running = tails + ([target] if step else [])
        scheduler.requests.update({request.request_id: request for request in tails})
        result = scheduler.schedule()
        assert all(result.num_scheduled_tokens[request.request_id] == 1024 for request in tails)
        assert 0 < result.num_scheduled_tokens["target"] <= 2048
        assert result.total_num_scheduled_tokens == sum(result.num_scheduled_tokens.values()) <= 16384
        assert scheduler.max_num_scheduled_tokens == 16384
        endpoints.append(target.num_computed_tokens)
    assert observed == [(STATUS.WAITING, 2048), *[(STATUS.RUNNING, 2048)] * 3]
    assert endpoints == [1536, 2304, 3840, 4608]


@pytest.mark.parametrize("computed,budget,expected", [(29952, 4, 0), (29952, 2048, 1536),
                                                      (31488, 4, 4), (31488, 1280, 1280)])
def test_after_joint_terminal_preserves_real_native_tail_legality(
        native_groups, monkeypatch, computed, budget, expected):
    cls, base, _ = scheduler_class(native_groups, monkeypatch)
    original = base._mamba_block_aligned_split
    plugin._patch_prefix_checkpoint_scheduler(NS(Scheduler=base))
    scheduler = cls()
    request = Request("tail", prompt=32768, computed=computed)
    # The OSCAR readable joint terminal is 29952; native GDN's later terminal
    # is 31488. The hook must retain original chunk legality in this interval.
    assert scheduler._oscar_prefix_checkpoint_policy.terminal_position(32768) == 29952
    assert original(scheduler, request, budget) == expected
    assert scheduler._mamba_block_aligned_split(request, budget) == expected


@pytest.mark.parametrize("computed,budget,encoder,expected", [(2300, 4, False, 0),
                                                            (0, 15000, True, 14592)])
def test_partial_state_and_encoder_spans_delegate_to_native(
        native_groups, monkeypatch, computed, budget, encoder, expected):
    cls, base, _ = scheduler_class(native_groups, monkeypatch)
    original = base._mamba_block_aligned_split
    plugin._patch_prefix_checkpoint_scheduler(NS(Scheduler=base))
    scheduler = cls()
    request = Request("native-scope", prompt=32768, computed=computed)
    request.has_encoder_inputs = encoder
    assert checkpoint.split_prefix_checkpoint(scheduler, request, budget) is None
    assert original(scheduler, request, budget) == expected
    assert scheduler._mamba_block_aligned_split(request, budget) == expected


def test_aligned_local_external_prefix_offsets_reach_joint_endpoint(native_groups, monkeypatch):
    cls, base, _ = scheduler_class(native_groups, monkeypatch)
    plugin._patch_prefix_checkpoint_scheduler(NS(Scheduler=base))
    scheduler = cls()
    request = Request("prefix-offsets", prompt=32768)
    local, external = 768, 768
    scheduled = scheduler._mamba_block_aligned_split(request, 16384, local, external)
    assert scheduled == 14592
    assert request.num_computed_tokens + local + external + scheduled == 16128
    assert 16128 % scheduler._oscar_prefix_checkpoint_policy.alignment_tokens == 0
    assert request.num_computed_tokens == 0


def test_real_schedule_normal_decode_matches_unpatched_native(native_groups, monkeypatch):
    cls, base, _ = scheduler_class(native_groups, monkeypatch)
    control_request = Request("decode", prompt=32768, computed=32768, output=1)
    control_request.spec_token_ids = [-1] * 3
    control = cls(running=[control_request])
    expected = control.schedule()
    plugin._patch_prefix_checkpoint_scheduler(NS(Scheduler=base))
    request = Request("decode", prompt=32768, computed=32768, output=1)
    request.spec_token_ids = [-1] * 3
    scheduler = cls(running=[request])
    result = scheduler.schedule()
    assert result.num_scheduled_tokens == expected.num_scheduled_tokens == {"decode": 4}
    assert result.scheduled_spec_decode_tokens == expected.scheduled_spec_decode_tokens
    assert request.num_computed_tokens == control_request.num_computed_tokens
    assert request.num_output_placeholders == control_request.num_output_placeholders


@pytest.mark.parametrize("name,entry", [("direct", install_serve), ("observe", observe_serve)])
@pytest.mark.parametrize("enabled", [False, True])
def test_candidate_entrypoints_keep_frozen_argv_for_both_prefix_modes(name, entry, enabled, capsys):
    args = ["--variant", "candidate", "--plan"]
    if enabled:
        args.append("--enable-prefix-caching")
    assert entry.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    expected = FROZEN["cases"][f"{name}_prefix_{str(enabled).lower()}"]
    assert report["vllm_argv"] == authorized_model_arguments(expected["argv"])
    assert report["runtime_selection"]["prefix_caching"] is enabled
    assert "--scheduler-cls" not in report["vllm_argv"]
    assert plugin._callbacks["vllm.v1.core.sched.scheduler"] is plugin._patch_scheduler
