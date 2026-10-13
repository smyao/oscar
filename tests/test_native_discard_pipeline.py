"""Archive #34/#36/#37-49/#142/#157: native discard-to-later-MTP evidence.

Compile the pinned native proposer methods unchanged. The runner publication
block is an unchanged AST slice of _prepare_inputs, not a reimplemented mask.
Only CPU NumPy buffer transport is substituted with torch in this host harness
(the local test environment has no NumPy). This does not validate NPU, Triton,
actual graph execution or the unobserved error code in the #157 target log.
"""
import ast
import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.draft_discard import native_draft_scope
from test_first_draft_current_only import builder as make_builder
from test_first_draft_padded_native import Common
from test_later_draft_slots import _prepare_status
from test_runtime_native import definitions


class _Buffer:
    def __init__(self, values):
        self.np = torch.as_tensor(values).clone()
        self.gpu = self.np.clone()

    def copy_to_gpu(self, count):
        self.gpu[:count].copy_(self.np[:count])


class _ArrayView:
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return _ArrayView(self.tensor[index])

    def numpy(self):
        return self.tensor


class _TorchTransport:
    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def from_numpy(array):
        return array


NP = SimpleNamespace(array=lambda values, dtype=None: torch.tensor(values, dtype=dtype),
                     int32=torch.int32, nonzero=lambda values: (torch.nonzero(values).flatten(),))


def _native_discard_publisher():
    path = Path(__file__).resolve().parents[1] / "references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_inputs")
    begin = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.ListComp)
                 and any(isinstance(t, ast.Name) and t.id == "num_tokens" for t in n.targets))
    end = next(i for i, n in enumerate(fn.body[begin:], begin) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == "copy_to_gpu"
               and isinstance(n.value.func.value, ast.Attribute)
               and n.value.func.value.attr == "discard_request_mask")
    wrapper = ast.parse("def publish(self, scheduler_output):\n    pass\n").body[0]
    wrapper.body = fn.body[begin:end + 1]
    ns = {"np": NP}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), "exec"), ns)
    return ns["publish"]


def _native(monkeypatch):
    device = definitions(
        "references/vllm-ascend/vllm_ascend/device/device_op.py", "discard_native_device", set(),
        {"torch": torch}, monkeypatch, methods=[("BaseDeviceAdaptor", "index_fill")])
    states = SimpleNamespace(SpecDecoding=object(), ChunkedPrefill=object())
    modes = SimpleNamespace(FULL=object(), NONE=object())
    native = definitions(
        "references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py", "discard_native_proposer", set(),
        {"torch": _TorchTransport(), "np": NP, "copy": copy, "HAS_TRITON": False,
         "DeviceOperator": device, "AscendCommonAttentionMetadata": Common,
         "AscendAttentionState": states, "CUDAGraphMode": modes, "PADDING_SLOT_ID": -1}, monkeypatch,
        methods=[("AscendSpecDecodeBaseProposer", name) for name in (
            "prepare_next_token_ids_padded", "prepare_inputs_padded", "shallow_copy_metadata",
            "attn_update_stack_num_spec_norm")])
    return native, states, modes


@pytest.mark.parametrize("async_spec", [False, True])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("discard_index", [0, 1, 2])
def test_pinned_native_discard_count_anchor_and_later_slots(monkeypatch, async_spec, mrope, discard_index):
    native, states, modes = _native(monkeypatch)
    builder = make_builder(monkeypatch)
    req_ids = ["left", "middle", "right"]
    lengths = torch.tensor([104, 104, 104], dtype=torch.int32)
    lengths[discard_index] = 5
    targets = lengths.clone()
    targets[discard_index] = 30000  # A chunked prefill still has unsampled input.
    requests = {name: SimpleNamespace(num_tokens=int(targets[i]), get_token_id=lambda pos: 700 + pos)
                for i, name in enumerate(req_ids)}
    batch = SimpleNamespace(num_reqs=3, req_ids=req_ids,
                            num_tokens_no_spec=targets, vocab_size=1000)
    runner = SimpleNamespace(input_batch=batch, requests=requests, pcp_size=1,
        optimistic_seq_lens_cpu=_ArrayView(lengths), discard_request_indices=_Buffer(torch.zeros(3, dtype=torch.int64)),
        discard_request_mask=_Buffer(torch.zeros(3, dtype=torch.bool)), pcp_manager=None,
        actual_seq_lengths_q=[4, 8, 12], attn_state=states.SpecDecoding, decode_token_per_req=4)
    _native_discard_publisher()(runner, SimpleNamespace())
    assert runner.discard_request_mask.gpu.tolist() == [i == discard_index for i in range(3)]

    proposer = SimpleNamespace(runner=runner, pcp_size=1, dcp_size=1, method="mtp",
        arange=torch.arange(16, dtype=torch.int32), token_arange_np=torch.arange(16, dtype=torch.int32),
        backup_next_token_ids=_Buffer(torch.zeros(3, dtype=torch.int64)), uses_mrope=mrope,
        has_gdn=True, kernel_block_size=128, max_model_len=262144, use_compress=False,
        slot_mapping_group=[torch.full((16,), -1, dtype=torch.int32) for _ in range(3)],
        seq_lens_group=[torch.zeros(3, dtype=torch.int32) for _ in range(3)],
        query_start_loc_group=[torch.zeros(4, dtype=torch.int32) for _ in range(3)])
    proposer.shallow_copy_metadata = MethodType(native.shallow_copy_metadata, proposer)
    sampled = torch.tensor([[11, 12, -1, -1], [21, 22, -1, -1], [31, 32, -1, -1]])
    _, counts = native.prepare_next_token_ids_padded(proposer, sampled, requests, batch,
        runner.discard_request_indices.gpu, runner.num_discarded_requests)
    assert counts.tolist() == [0 if i == discard_index else 2 for i in range(3)]
    assert sampled[:, :2].min() > 0  # Native clones instead of mutating sampler output.

    starts = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    common = Common(query_start_loc=starts, query_start_loc_cpu=starts.clone(),
        seq_lens=lengths.clone(), seq_lens_cpu=None if async_spec else lengths.clone(),
        _seq_lens_cpu=lengths.clone(), seq_lens_cpu_upper_bound=lengths.clone(),
        num_reqs=3, num_actual_tokens=12, num_input_tokens=12, max_query_len=4,
        num_computed_tokens_cpu=None, _num_computed_tokens_cpu=None,
        block_table_tensor=torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32),
        slot_mapping=proposer.slot_mapping_group[0], positions=torch.zeros(16, dtype=torch.int64),
        positions_cpu=None, attn_state=states.SpecDecoding, decode_token_per_req=4,
        is_prefilling=torch.tensor([i == discard_index for i in range(3)]), max_seq_len=104)
    spec = SimpleNamespace(cu_num_draft_tokens=torch.tensor([3, 6, 9], dtype=torch.int32))
    prepared, _, anchors, rejected = native.prepare_inputs_padded(proposer, common, spec, counts)
    assert rejected[discard_index] == 4
    assert anchors[discard_index] == starts[discard_index] - 1
    # Native positions is persistent and larger than the current real prefix.
    # For first request's -1 anchor it reads the stale tail. All live anchors
    # below address their own true current rows; a middle discard reads its
    # previous request. This is a constructed reachable contract case, not a
    # claim that #157 reported error code 5.
    positions = torch.full((16,), 103, dtype=torch.int64)
    for row in range(3):
        positions[row * 4:row * 4 + 4] = torch.arange(int(lengths[row]) - 4, int(lengths[row]))
    used = positions[anchors]
    if mrope:
        used = torch.stack((used, used, used))
    group = SimpleNamespace(get_metadata_builder=lambda: builder)
    original_slots = []
    with native_draft_scope(proposer):
        previous = prepared
        for draft_index in (1, 2):
            previous, metadata = native.attn_update_stack_num_spec_norm(
                proposer, draft_index, None, previous, 3, 12, used, modes.NONE, attn_group=group)
            native_slots = proposer.slot_mapping_group[draft_index]
            original_slots.append(native_slots.clone())
            assert _prepare_status(int(native_slots[discard_index]), int(previous.seq_lens[discard_index]),
                                   previous.block_table_tensor[discard_index].tolist()) == 5
            assert metadata.slot_mapping[discard_index] == -1
            for row in range(3):
                if row != discard_index:
                    assert metadata.slot_mapping[row] == native_slots[row]
                    assert _prepare_status(int(metadata.slot_mapping[row]), int(previous.seq_lens[row]),
                                           previous.block_table_tensor[row].tolist()) == 0
            assert torch.equal(native_slots, original_slots[-1])
            assert metadata.num_actual_tokens == 3 and metadata.max_query_len == 1
            assert (previous.seq_lens_cpu is None) is async_spec


@pytest.mark.parametrize("async_spec", [False, True])
@pytest.mark.parametrize("mrope", [False, True])
def test_native_reordered_shrunk_and_regrown_batches_refresh_fixed_draft_slots(monkeypatch, async_spec, mrope):
    """Exercise real native publication/gather through four scheduling turns.

    The native buffers retain capacity, including old mask values beyond B;
    native request ordering and B change. This checks proposal ownership, not
    simulated parallel stream ordering or actual NPU graph replay.
    """
    native, states, modes = _native(monkeypatch)
    builder = make_builder(monkeypatch)
    publisher = _native_discard_publisher()
    runner = SimpleNamespace(pcp_size=1, pcp_manager=None,
        discard_request_indices=_Buffer(torch.zeros(3, dtype=torch.int64)),
        discard_request_mask=_Buffer(torch.zeros(3, dtype=torch.bool)),
        attn_state=states.SpecDecoding, decode_token_per_req=4)
    proposer = SimpleNamespace(runner=runner, pcp_size=1, dcp_size=1, method="mtp",
        arange=torch.arange(16, dtype=torch.int32), token_arange_np=torch.arange(16, dtype=torch.int32),
        backup_next_token_ids=_Buffer(torch.zeros(3, dtype=torch.int64)), uses_mrope=mrope,
        has_gdn=True, kernel_block_size=128, max_model_len=262144, use_compress=False,
        slot_mapping_group=[torch.full((16,), -1, dtype=torch.int32) for _ in range(3)],
        seq_lens_group=[torch.zeros(3, dtype=torch.int32) for _ in range(3)],
        query_start_loc_group=[torch.zeros(4, dtype=torch.int32) for _ in range(3)])
    proposer.shallow_copy_metadata = MethodType(native.shallow_copy_metadata, proposer)
    groups = SimpleNamespace(get_metadata_builder=lambda: builder)
    addresses = {}
    # A discarded request becomes active later; rows reorder; tails shrink
    # then reappear. The last round retains no discarded row at all.
    rounds = [("abc", "b"), ("cb", "c"), ("b", ""), ("dba", "")]
    for names, discarded in rounds:
        size = len(names)
        lengths = torch.tensor([5 if name in discarded else 104 for name in names], dtype=torch.int32)
        targets = torch.tensor([30000 if name in discarded else 104 for name in names], dtype=torch.int32)
        runner.input_batch = SimpleNamespace(num_reqs=size, req_ids=list(names),
            num_tokens_no_spec=targets, vocab_size=1000)
        runner.requests = {name: SimpleNamespace(num_tokens=int(targets[i]), get_token_id=lambda pos: 700 + pos)
                           for i, name in enumerate(names)}
        runner.optimistic_seq_lens_cpu = _ArrayView(lengths)
        starts = torch.arange(size + 1, dtype=torch.int32) * 4
        runner.actual_seq_lengths_q = starts[1:].tolist()
        publisher(runner, SimpleNamespace())
        assert runner.discard_request_mask.gpu[:size].tolist() == [name in discarded for name in names]
        sampled = torch.tensor([[11, 12, -1, -1]] * size)
        _, counts = native.prepare_next_token_ids_padded(proposer, sampled, runner.requests,
            runner.input_batch, runner.discard_request_indices.gpu, runner.num_discarded_requests)
        table = torch.tensor([[10 * (ord(name) - ord("a") + 1), 10 * (ord(name) - ord("a") + 1) + 1]
                              for name in names], dtype=torch.int32)
        common = Common(query_start_loc=starts, query_start_loc_cpu=starts.clone(),
            seq_lens=lengths.clone(), seq_lens_cpu=None if async_spec else lengths.clone(),
            _seq_lens_cpu=lengths.clone(), seq_lens_cpu_upper_bound=lengths.clone(),
            num_reqs=size, num_actual_tokens=size * 4, num_input_tokens=size * 4, max_query_len=4,
            num_computed_tokens_cpu=None, _num_computed_tokens_cpu=None,
            block_table_tensor=table, slot_mapping=proposer.slot_mapping_group[0],
            positions=torch.zeros(16, dtype=torch.int64), positions_cpu=None,
            attn_state=states.SpecDecoding, decode_token_per_req=4,
            is_prefilling=torch.tensor([name in discarded for name in names]), max_seq_len=104)
        spec = SimpleNamespace(cu_num_draft_tokens=torch.arange(1, size + 1, dtype=torch.int32) * 3)
        previous, _, anchors, _ = native.prepare_inputs_padded(proposer, common, spec, counts)
        positions = torch.full((16,), 103, dtype=torch.int64)
        for row in range(size):
            positions[row * 4:row * 4 + 4] = torch.arange(int(lengths[row]) - 4, int(lengths[row]))
        used = positions[anchors]
        if mrope:
            used = torch.stack((used, used, used))
        with native_draft_scope(proposer):
            for index in (1, 2):
                previous, metadata = native.attn_update_stack_num_spec_norm(
                    proposer, index, None, previous, size, size * 4, used, modes.NONE, attn_group=groups)
                native_slots = proposer.slot_mapping_group[index]
                expected = native_slots.clone()
                expected[:size].masked_fill_(runner.discard_request_mask.gpu[:size], -1)
                assert torch.equal(metadata.slot_mapping, expected)
                assert metadata.slot_mapping[size:].eq(-1).all()
                for row, name in enumerate(names):
                    assert (metadata.slot_mapping[row] == -1) == (name in discarded)
                    assert _prepare_status(int(metadata.slot_mapping[row]), int(previous.seq_lens[row]),
                                           table[row].tolist()) == 0
                if index in addresses:
                    assert metadata.slot_mapping.data_ptr() == addresses[index]
                addresses[index] = metadata.slot_mapping.data_ptr()
    assert addresses[1] != addresses[2]
