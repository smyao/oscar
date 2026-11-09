"""Composable external lifecycle hooks; never replace the native scheduler.

Archive #27/#31-33/#77/#78: lazy connector access, no config-object mutation.
Archive #37-49/#127/#128/#171: snapshot ownership before free/zero/reuse.
Archive #157/#163: worker accepted-state completion is the publication seam.
Existing plugin wrappers call these functions so the reversible patch registry
owns each native descriptor only once (prefix/window hooks remain intact).
"""

from contextlib import contextmanager


def _scheduler_connector(scheduler):
    connector = getattr(scheduler, "connector", None)
    return connector if getattr(connector, "_oscar_offload", False) else None


def scheduler_initialized(scheduler) -> None:
    connector = _scheduler_connector(scheduler)
    if connector is not None:
        connector.bind_scheduler(scheduler)


def before_preempt(scheduler, request):
    connector = _scheduler_connector(scheduler)
    if connector is not None:
        return connector.scheduler_state.before_preempt(request)
    return None


def after_schedule(scheduler, scheduler_output) -> None:
    connector = _scheduler_connector(scheduler)
    if connector is not None:
        connector.scheduler_state.begin_step(scheduler_output)


def after_output(scheduler, scheduler_output) -> None:
    connector = _scheduler_connector(scheduler)
    if connector is not None:
        connector.scheduler_state.after_output(scheduler_output)


def request_finished(scheduler, request_id: str) -> None:
    connector = _scheduler_connector(scheduler)
    if connector is not None:
        connector.scheduler_state.finish(request_id)


def request_branched(scheduler, request) -> None:
    connector = _scheduler_connector(scheduler)
    if connector is not None:
        connector.scheduler_state.branch(request)


@contextmanager
def resetting_cache(scheduler, *, reset_running_requests: bool = False):
    """Discard active snapshots only after a successful explicit forced reset.

    Clearing reusable prefix entries is independent of active or preempted
    request ownership. A normal reset may succeed with swappable requests
    waiting off-device; their committed snapshots must remain recoverable.
    Forced native preemption intentionally recomputes instead of saving, so
    suppress capture for that call, then commit invalidation only on success.
    """
    connector = _scheduler_connector(scheduler)
    if connector is None:
        yield lambda successful: None
        return
    state = connector.scheduler_state
    previous = state.suppress_preemption
    if reset_running_requests:
        state.suppress_preemption = True

    def complete(successful):
        if reset_running_requests and successful is True:
            state.reset()

    try:
        yield complete
    finally:
        # False returns and exceptions preserve the last recoverable ledger.
        # The caller must not report successful invalidation for either case.
        state.suppress_preemption = previous


def worker_step_completed(runner, output_token_ids, scheduler_output) -> None:
    """Called after the original native _update_states_after_model_execute.

    Preserve the native stream context: accepted-state copies and tracker
    event recording must be ordered after the original postprocess kernel.
    """
    from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

    if has_kv_transfer_group():
        connector = get_kv_transfer_group()
        if getattr(connector, "_oscar_offload", False):
            connector.observe_and_save(runner, output_token_ids, scheduler_output)
