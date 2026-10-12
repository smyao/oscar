"""Archive #26-49/#127-129/#155: host lifecycle contracts, not NPU evidence.

Native SchedulerOutput and the actual pinned _preempt_request/reset methods
provide the protocol oracle. No window values, device/graph or timing tests.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time
import unittest

from oscar_ascend.request_windows import (
    INVALID_EPOCH, INVALID_SLOT, MAX_EPOCH, PreemptionNotificationMixin,
    RequestWindowLeases, WindowLeaseError,
)

ROOT = Path(__file__).resolve().parents[1]


def output(*, new=(), resumed=(), finished=(), preempted=(), total_tokens=1):
    return NS(scheduled_new_reqs=[NS(req_id=key) for key in new],
              scheduled_cached_reqs=NS(resumed_req_ids=set(resumed)),
              finished_req_ids=set(finished), preempted_req_ids=set(preempted),
              total_num_scheduled_tokens=total_tokens)


def native_scheduler_methods():
    path = ROOT / "references/vllm/vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    names = {"_preempt_request", "reset_prefix_cache"}
    nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise AssertionError("pinned native scheduler methods changed")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"time": time, "RequestStatus": NS(RUNNING="running", PREEMPTED="preempted")}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class LeaseTests(unittest.TestCase):
    def test_config_capacity_reordering_padding_and_temporarily_unscheduled(self):
        config = json.loads((ROOT / "configs/target.json").read_text())
        self.assertEqual(RequestWindowLeases.from_config(config).capacity, config["max_num_seqs"])
        leases = RequestWindowLeases(3)
        first = leases.update(output(new=("a", "b")), ["a", "b"], padded_rows=3)
        self.assertEqual(first.pairs, ((0, 1), (1, 2), (INVALID_SLOT, INVALID_EPOCH)))
        second = leases.update(output(), ["b", "a"])
        self.assertEqual(second.pairs, ((1, 2), (0, 1)))
        leases.update(output(), ["b"])
        self.assertEqual(leases.active_count, 2)
        self.assertEqual(leases.update(output(), ["a"]).pairs, ((0, 1),))

    def test_normal_preemption_resume_and_resume_without_notification(self):
        leases = RequestWindowLeases(2)
        leases.update(output(new=("a", "b")), ["a", "b"])
        leases.update(output(preempted=("a",)), ["b"])
        resumed = leases.update(output(resumed=("a",)), ["a", "b"])
        self.assertEqual(resumed.pairs, ((0, 3), (1, 2)))
        # Native forced preemption can arrive first as a resumed ID. It must
        # invalidate the prior owner even when the tracker still holds it.
        again = leases.update(output(resumed=("a",)), ["b", "a"])
        self.assertEqual(again.pairs, ((1, 2), (0, 4)))

    def test_finished_id_resubmitted_in_same_step_and_streaming_continuation(self):
        leases = RequestWindowLeases(1)
        leases.update(output(new=("a",)), ["a"])
        self.assertEqual(leases.update(output(new=("a",)), ["a"]).epochs, (1,))
        fresh = leases.update(output(new=("a",), finished=("a",)), ["a"])
        self.assertEqual(fresh.pairs, ((0, 2),))

    def test_zero_token_events_release_and_unknown_finished_ids_are_harmless(self):
        leases = RequestWindowLeases(2)
        leases.update(output(new=("a", "b")), ["a", "b"])
        empty = leases.update(output(preempted=("a",), finished=("b", "never_seen"), total_tokens=0), [], padded_rows=2)
        self.assertEqual(empty.pairs, ((-1, 0), (-1, 0)))
        self.assertEqual(leases.active_count, 0)
        self.assertEqual(leases.update(output(new=("c",)), ["c"]).pairs, ((0, 3),))

    def test_capacity_failure_cannot_steal_unscheduled_owner_or_partly_commit(self):
        leases = RequestWindowLeases(2)
        leases.update(output(new=("a", "b")), ["a", "b"])
        before = leases.snapshot()
        with self.assertRaisesRegex(WindowLeaseError, "capacity exhausted"):
            leases.update(output(new=("c",)), ["c"])
        self.assertEqual(leases.snapshot(), before)
        with self.assertRaisesRegex(WindowLeaseError, "no lease"):
            leases.update(output(finished=("a",)), ["missing"])
        self.assertEqual(leases.snapshot(), before)
        self.assertEqual(leases.last_epoch, 2)

    def test_epoch_exhaustion_is_atomic_after_a_release(self):
        leases = RequestWindowLeases(1, max_epoch=2)
        leases.update(output(new=("a",)), ["a"])
        leases.update(output(resumed=("a",)), ["a"])
        before = leases.snapshot()
        with self.assertRaisesRegex(WindowLeaseError, "epoch exhausted"):
            leases.update(output(finished=("a",), new=("b",)), ["b"])
        self.assertEqual(leases.snapshot(), before)
        self.assertEqual(leases.last_epoch, 2)

    def test_malformed_events_and_uncondensed_batch_are_rejected(self):
        leases = RequestWindowLeases(2)
        for ids in ([None], ["a", "a"], "a"):
            with self.subTest(ids=ids), self.assertRaises(WindowLeaseError):
                leases.update(output(new=("a",)), ids)
        for value in (True, 0, -1, 1.5):
            with self.subTest(capacity=value), self.assertRaises(WindowLeaseError):
                RequestWindowLeases(value)
        for value in (True, 0, MAX_EPOCH + 1):
            with self.subTest(epoch=value), self.assertRaises(WindowLeaseError):
                RequestWindowLeases(1, max_epoch=value)
        bad = output(new=("a",))
        bad.finished_req_ids = ["a"]
        with self.assertRaisesRegex(WindowLeaseError, "must be a set"):
            leases.update(bad, ["a"])
        with self.assertRaisesRegex(WindowLeaseError, "absent"):
            leases.update(output(resumed=("a",)), [])
        leases.update(output(new=("a",)), ["a"])
        for event in (output(finished=("a",)), output(preempted=("a",))):
            with self.assertRaises(WindowLeaseError):
                leases.update(event, ["a"])
        with self.assertRaises(WindowLeaseError):
            leases.update(output(), ["a"], padded_rows=3)


class NotificationTests(unittest.TestCase):
    def make_scheduler(self):
        native = native_scheduler_methods()

        class Native:
            _preempt_request = native["_preempt_request"]
            reset_prefix_cache = native["reset_prefix_cache"]

            def __init__(self):
                self.running = []
                self.freed = []
                self.waiting_ids = []
                self._inflight_prefills = set()
                self.kv_cache_manager = NS(free=lambda req: self.freed.append(req.request_id), reset_prefix_cache=lambda: True)
                self.encoder_cache_manager = NS(free=lambda req: None)
                self.waiting = NS(prepend_request=lambda req: self.waiting_ids.insert(0, req.request_id))
                self.prev_step_scheduled_req_ids = set()
                self.log_stats = False
                self.fail_schedule = False
                self.preempt_during_schedule = None

            def schedule(self):
                if self.fail_schedule:
                    raise RuntimeError("native schedule failed")
                ordinary = set()
                if self.preempt_during_schedule is not None:
                    req = self.preempt_during_schedule
                    self.preempt_during_schedule = None
                    self._preempt_request(req, 1.0)
                    ordinary.add(req.request_id)
                return output(preempted=ordinary, total_tokens=0)

        class Scheduler(PreemptionNotificationMixin, Native):
            pass

        return Scheduler()

    def request(self, request_id):
        class Request:
            pass
        req = Request()
        req.request_id, req.status = request_id, "running"
        req.num_computed_tokens, req.spec_token_ids = 100, [1, 2, 3]
        req.num_preemptions, req.num_output_placeholders = 0, 4
        return req

    def test_actual_native_forced_reset_notifies_without_new_transport_field(self):
        scheduler = self.make_scheduler()
        a, b = self.request("a"), self.request("b")
        scheduler.running = [a, b]
        self.assertTrue(scheduler.reset_prefix_cache(reset_running_requests=True))
        self.assertEqual(scheduler.freed, ["b", "a"])
        self.assertEqual((a.status, a.num_computed_tokens, a.spec_token_ids, a.async_tokens_to_discard), ("preempted", 0, [], 4))
        notice = scheduler.schedule()
        self.assertEqual(notice.preempted_req_ids, {"a", "b"})
        self.assertEqual(scheduler.schedule().preempted_req_ids, set())
        # Draining future events must not mutate an already-returned output.
        self.assertEqual(notice.preempted_req_ids, {"a", "b"})
        leases = RequestWindowLeases(2)
        leases.update(output(new=("a", "b")), ["a", "b"])
        leases.update(notice, [])
        self.assertEqual(leases.active_count, 0)

    def test_normal_preemption_deduplicates_and_native_failure_is_not_notified(self):
        scheduler = self.make_scheduler()
        req = self.request("a")
        scheduler.preempt_during_schedule = req
        self.assertEqual(scheduler.schedule().preempted_req_ids, {"a"})
        with self.assertRaises(AssertionError):
            scheduler._preempt_request(req, 2.0)
        self.assertEqual(scheduler.schedule().preempted_req_ids, set())

    def test_failed_schedule_keeps_successful_preemption_pending(self):
        scheduler = self.make_scheduler()
        scheduler._preempt_request(self.request("a"), 1.0)
        scheduler.fail_schedule = True
        with self.assertRaisesRegex(RuntimeError, "native schedule failed"):
            scheduler.schedule()
        scheduler.fail_schedule = False
        self.assertEqual(scheduler.schedule().preempted_req_ids, {"a"})


if __name__ == "__main__":
    unittest.main()
