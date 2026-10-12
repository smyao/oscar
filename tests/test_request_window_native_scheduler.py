"""Archive #37-49/#127-129/#142/#155: native leases, budget and cleanup.

Uses pinned native reset/preempt/schedule/AsyncScheduler bodies. Host-only
evidence: it does not certify device streams, graph replay, accuracy or speed.
"""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from oscar_ascend import plugin, runtime
from oscar_ascend.integration.native_token_budget import NativeTokenBudgetRuntime
from oscar_ascend.integration.runtime_api import OscarReadinessError
from test_request_windows import NotificationTests
from test_request_window_integration import configuration
from test_bounded_whole_prefill_native import native_class, Request


class NativeNotificationTests(NotificationTests):
    """Repeat actual forced reset/error contracts without any mixin scheduler."""

    def setUp(self):
        plugin.unregister()
        self.flag = patch.object(plugin, "_request_windows_configured", return_value=True)
        self.flag.start()

    def tearDown(self):
        plugin.unregister()
        self.flag.stop()

    def make_scheduler(self):
        old = super().make_scheduler()
        native = type(old).__mro__[2]
        original_schedule, original_preempt = native.schedule, native._preempt_request
        plugin._patch_request_window_scheduler(NS(Scheduler=native))
        wrapped = native.schedule, native._preempt_request
        plugin._patch_request_window_scheduler(NS(Scheduler=native))
        self.assertEqual((native.schedule, native._preempt_request), wrapped)
        self.assertIs(native.schedule.__wrapped__, original_schedule)
        self.assertIs(native._preempt_request.__wrapped__, original_preempt)
        self.assertTrue(plugin.request_window_scheduler_hooks_present(native))
        return native()


@pytest.mark.parametrize("budget", [2048, 8192, 16384])
def test_real_native_scheduler_keeps_chunk_budget_and_async_results(monkeypatch, budget):
    plugin.unregister()
    cls = native_class()
    base = cls.__mro__[1]
    control = cls(waiting=[Request("a"), Request("b")])
    control.max_num_scheduled_tokens = budget
    expected = [control.schedule(), control.schedule()]
    monkeypatch.setattr(plugin, "_request_windows_configured", lambda: True)
    plugin._patch_request_window_scheduler(NS(Scheduler=base))
    actual = cls(waiting=[Request("a"), Request("b")])
    actual.max_num_scheduled_tokens = budget
    original_config = actual.scheduler_config
    try:
        for reference in expected:
            result = actual.schedule()
            assert result.num_scheduled_tokens == reference.num_scheduled_tokens
            assert result.total_num_scheduled_tokens <= budget
            assert result.scheduled_spec_decode_tokens == reference.scheduled_spec_decode_tokens
            assert result.preempted_req_ids == reference.preempted_req_ids
            assert actual.max_num_scheduled_tokens == budget
            assert actual.scheduler_config is original_config
        assert type(actual) is cls
    finally:
        plugin.unregister()


def test_disabled_flag_leaves_native_descriptors_identical(monkeypatch):
    plugin.unregister()
    cls = native_class().__mro__[1]
    before = cls.schedule, cls._preempt_request
    monkeypatch.setattr(plugin, "_request_windows_configured", lambda: False)
    plugin._patch_request_window_scheduler(NS(Scheduler=cls))
    assert (cls.schedule, cls._preempt_request) == before
    assert not plugin.request_window_scheduler_hooks_present(cls)


@pytest.mark.parametrize("budget,captures,expected", [(2048, [4096], 4096), (8192, [512], 8192)])
def test_request_window_workspace_follows_actual_native_capacity(monkeypatch, budget, captures, expected):
    config = configuration()
    config["use_native_token_budget"] = True
    config.pop("max_num_batched_tokens", None)
    provider = NativeTokenBudgetRuntime(config)
    native = NS(scheduler_config=NS(max_num_batched_tokens=budget),
                compilation_config=NS(cudagraph_capture_sizes=captures))
    provider._bind_native_capacity(native)
    monkeypatch.setattr(provider, "_device_cube_cores", lambda device: 20)
    monkeypatch.setattr(provider, "_prepare_rotations", lambda *args: None)
    monkeypatch.setattr(runtime, "GraphWorkspace", lambda geometry, device: geometry)
    workspace = provider.ensure_workspace(6, 1, 256, "test-device")
    assert workspace.tokens == expected
    assert workspace.history_tokens == expected
    assert native.scheduler_config.max_num_batched_tokens == budget
    assert native.compilation_config.cudagraph_capture_sizes == captures
    with pytest.raises(OscarReadinessError, match="changed after"):
        provider._bind_native_capacity(NS(scheduler_config=NS(max_num_batched_tokens=budget + 1),
                                         compilation_config=native.compilation_config))
