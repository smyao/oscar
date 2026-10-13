"""Archive #157-159: mutable pinned mask lifetime under native prep events.

Run the pinned native CpuGpuBuffer.copy_to_gpu and synchronize_input_prep
method bodies together with the real OSCAR scope/slot bridge. A delayed FIFO
models the stated stream/event contract; it is not torch-npu device evidence.
The no-event negative control must expose the mutable-source race.
"""
from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import torch

from oscar_ascend.integration.draft_discard import native_draft_scope, later_draft_slot_copy
from test_runtime_native import definitions


class Queue:
    def __init__(self):
        self.commands = []
        self.executed = 0

    def enqueue(self, fn):
        self.commands.append(fn)

    def finish(self, stop=None):
        stop = len(self.commands) if stop is None else stop
        while self.executed < stop:
            fn = self.commands[self.executed]
            self.executed += 1
            fn()


class Event:
    def __init__(self, queue):
        self.queue = queue
        self.point = 0

    def record(self):
        self.point = len(self.queue.commands)

    def synchronize(self):
        self.queue.finish(self.point)


class TensorView:
    def __init__(self, values, dtype, queue=None, offset=0, count=None):
        self.values, self.dtype, self.queue = values, dtype, queue
        self.offset = offset
        self.shape = (len(values) - offset if count is None else count,)
        self.ndim = 1
        self.device = "simulated_npu" if queue is not None else "cpu"

    def numel(self):
        return self.shape[0]

    def data_ptr(self):
        return id(self.values) + self.offset

    def stride(self):
        return (1,)

    def __getitem__(self, key):
        assert isinstance(key, slice)
        start, end, stride = key.indices(self.numel())
        assert stride == 1
        return TensorView(self.values, self.dtype, self.queue, self.offset + start, end - start)

    def read_at_execution(self):
        return self.values[self.offset:self.offset + self.numel()]

    def copy_(self, source, non_blocking=None):
        assert self.queue is not None and self.numel() == source.numel()
        def execute():
            self.values[self.offset:self.offset + self.numel()] = source.read_at_execution()
        self.queue.enqueue(execute)
        return self

    def masked_fill_(self, mask, value):
        assert mask.numel() == self.numel()
        def execute():
            for i, discarded in enumerate(mask.read_at_execution()):
                if discarded:
                    self.values[self.offset + i] = value
        self.queue.enqueue(execute)
        return self


def test_native_prep_event_protects_delayed_mask_copy_and_later_device_reads(monkeypatch):
    native = definitions("references/vllm/vllm/v1/utils.py", "native_delayed_mask_copy", set(),
        {"torch": torch}, monkeypatch, methods=[("CpuGpuBuffer", "copy_to_gpu")])
    runner_native = definitions("references/vllm/vllm/v1/worker/gpu_model_runner.py",
        "native_delayed_prep_event", set(), {"contextmanager": contextmanager}, monkeypatch,
        methods=[("GPUModelRunner", "synchronize_input_prep")])
    real_is_tensor = torch.is_tensor
    monkeypatch.setattr(torch, "is_tensor", lambda value: isinstance(value, TensorView) or real_is_tensor(value))
    monkeypatch.setattr(torch, "empty_like", lambda value: TensorView(
        [None] * value.numel(), value.dtype, value.queue))

    def execute(with_event):
        queue = Queue()
        mask = SimpleNamespace(cpu=TensorView([False, False], torch.bool),
                               gpu=TensorView([False, False], torch.bool, queue))
        mask.copy_to_gpu = MethodType(native.copy_to_gpu, mask)
        runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=0), discard_request_mask=mask,
            prepare_inputs_event=Event(queue) if with_event else None)
        runner.synchronize_input_prep = MethodType(runner_native.synchronize_input_prep, runner)
        proposer = SimpleNamespace(runner=runner, pcp_size=1, dcp_size=1,
            slot_mapping_group=[TensorView([-1] * 4, torch.int32, queue) for _ in range(3)])
        observed = []
        # The GPU executes as late as the native event protocol allows. Every
        # CPU mask update reuses the SAME mutable pinned source allocation.
        for discarded, live_slots in (([False], [1285]), ([True, False], [1290, 1420]), ([False], [1550])):
            size = len(discarded)
            with runner.synchronize_input_prep():
                runner.input_batch.num_reqs = size
                mask.cpu.values[:size] = discarded
                mask.copy_to_gpu(size)
            source = proposer.slot_mapping_group[1]
            source.copy_(TensorView(live_slots + [-1] * (4 - size), torch.int32))
            common = SimpleNamespace(slot_mapping=source, num_reqs=size,
                                     num_actual_tokens=size, max_query_len=1)
            with native_draft_scope(proposer):
                result = later_draft_slot_copy(common, draft_index=1)
            queue.enqueue(lambda result=result: observed.append(result.read_at_execution()))
        queue.finish()
        return observed

    expected = [[1285, -1, -1, -1], [-1, 1420, -1, -1], [1550, -1, -1, -1]]
    assert execute(with_event=True) == expected
    # Removing the native event while retaining async host advancement is
    # deliberately invalid. It proves the test can observe source corruption.
    unprotected = execute(with_event=False)
    assert unprotected != expected
    assert unprotected[1][0] == 1290
