"""Archive #157-160: fixed-shape greedy writer and native MTP consumers.

CPU tests execute the adapter and pinned native methods. The undefined-load
witness chooses permitted unspecified values in a source address model; it
does not establish the Ascend compiler's choice or a target failure's cause.
NPU execution, graphs, quality and performance remain separate evidence.
"""
import ast
import inspect
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch

from oscar_ascend import plugin
from oscar_ascend.integration.greedy_sampling import (
    rejection_greedy_sample_with_triton as greedy, rejection_sample_with_guarded_cu)
from test_first_draft_padded_native import Common
from test_later_draft_slots import _prepare_status
from test_native_discard_pipeline import _Buffer, _native
from test_runtime_native import definitions


ROOT = Path(__file__).resolve().parents[1]
NAME = "rejection_greedy_sample_with_triton"
MODULES = ("vllm_ascend.ops.triton.reject_sample", "vllm_ascend.sample.rejection_sampler")


def _inputs(counts, rejection, mixed=False):
    cu = torch.tensor(counts, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    drafts = torch.arange(sum(counts), dtype=torch.int32) + 10
    targets = drafts.clone()
    offset = 0
    for count in counts:
        if rejection is not None and rejection < count:
            targets[offset + rejection] = 999
        offset += count
    bonus = torch.arange(len(counts), dtype=torch.int32).unsqueeze(1) + 1000
    mask = torch.arange(len(counts)) % 2 == 0 if mixed else None
    return cu, drafts, targets, bonus, mask


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("rejection", [None, 0, 1, 2])
def test_matches_pinned_native_greedy_for_variable_drafts_and_shrink(monkeypatch, mixed, rejection):
    native = definitions(
        "references/vllm-ascend/vllm_ascend/sample/rejection_sampler.py", "safe_greedy_oracle",
        {"rejection_greedy_sample_pytorch"}, {"torch": torch}, monkeypatch)
    # Reuse one output allocation across B32 -> B12; random rows have data
    # owned by their writer, and must remain byte-for-byte unchanged.
    storage = torch.full((32, 4), -1, dtype=torch.int32)
    for batch in (32, 12):
        counts = ([0, 1, 2, 3] * 8)[:batch]
        cu, drafts, targets, bonus, mask = _inputs(counts, rejection, mixed)
        actual = storage[:batch]
        actual.fill_(-1)
        if mask is not None:
            actual[~mask] = 777
        expected = actual.clone()
        native.rejection_greedy_sample_pytorch(
            expected, cu, drafts, targets, bonus, counts, 3, mask)
        assert greedy(actual, counts, cu, drafts, targets, bonus, mask, 3, batch, 1) is None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("counts", [[0, 0], [1, 1], [2, 2], [3, 3]])
def test_homogeneous_counts_and_empty_flat_input(monkeypatch, counts):
    native = definitions(
        "references/vllm-ascend/vllm_ascend/sample/rejection_sampler.py", "safe_greedy_homogeneous",
        {"rejection_greedy_sample_pytorch"}, {"torch": torch}, monkeypatch)
    cu, drafts, targets, bonus, mask = _inputs(counts, None)
    width = max(counts) + 1
    actual = torch.full((len(counts), width), -1, dtype=torch.int32)
    expected = actual.clone()
    native.rejection_greedy_sample_pytorch(expected, cu, drafts, targets, bonus, counts, width - 1, mask)
    greedy(actual, counts, cu, drafts, targets, bonus, mask, width - 1, len(counts), 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("counts", [[0, 0], [3, 1]])
def test_no_host_readback_or_data_dependent_shapes_and_exact_random_preservation(monkeypatch, counts):
    cu, drafts, targets, bonus, mask = _inputs(counts, 0, mixed=True)
    width = max(counts) + 1
    output = torch.full((2, width), 777, dtype=torch.int32)

    def forbidden(*args, **kwargs):
        raise AssertionError("host readback or dynamic output shape is forbidden")

    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist", "numpy", "__bool__", "nonzero", "masked_select"):
            patch.setattr(torch.Tensor, name, forbidden)
        for name in ("nonzero", "masked_select", "repeat_interleave"):
            patch.setattr(torch, name, forbidden)
        greedy(output, counts, cu, drafts, targets, bonus, mask, width - 1, 2, 1)
    assert output[1].tolist() == [777] * width
    assert output[0].tolist() == ([1000] if counts[0] == 0 else [999, -1, -1, -1])


def _source_scalar_kernels():
    """Single-lane CPU address model of unchanged Triton function bodies."""
    class Pointer:
        def __init__(self, name, values, offset=0):
            self.name, self.values, self.offset = name, values, offset
        def __add__(self, value):
            return Pointer(self.name, self.values, self.offset + value)
        def __sub__(self, value):
            return self + (-value)

    class Language:
        constexpr = int
        pid = 0
        unspecified = iter(())
        invalid_reads = []
        @staticmethod
        def arange(start, end):
            assert (start, end) == (0, 1)
            return 0
        range = staticmethod(range)
        @staticmethod
        def where(condition, left, right):
            return left if condition else right
        def program_id(self, axis):
            return self.pid
        def load(self, ptr, mask=True, other=None):
            if not mask:
                return other if other is not None else next(self.unspecified)
            # The source also issues cu[-1] before tl.where chooses zero.
            # Its loaded value is unused here; do not claim compiler behavior.
            if ptr.offset == -1:
                self.invalid_reads.append((ptr.name, ptr.offset))
                return 0
            return ptr.values[ptr.offset]
        @staticmethod
        def store(ptr, value, mask=True):
            if mask:
                ptr.values[ptr.offset] = value

    language = Language()
    path = ROOT / "references/vllm-ascend/vllm_ascend/ops/triton/reject_sample.py"
    names = {"bonus_renew", "rejection_greedy_sample_triton", "rejection_random_sample_kernel"}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    namespace = {"tl": language, "get_element": lambda value, position: value}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return Pointer, language, namespace


def _run_random_source(pointer, language, namespace, values, greedy_mask=(1, 0)):
    # Original random writer: row 1 rejects its only draft, writing column 0.
    for row in range(2):
        language.pid = row
        language.unspecified = iter((0, 0))
        namespace["rejection_random_sample_kernel"](
            pointer("out", values), pointer("cu", [3, 4]), pointer("draft", [10, 11, 12, 13]),
            None, pointer("probs", [0.0] * 4 * 64), None, pointer("bonus", [20, 21]),
            pointer("recovered", [30] * 4), pointer("uniform", [0.5] * 4),
            pointer("greedy", list(greedy_mask)), 3, 64, 64, 2, None, True, True, False, False, 1)


def _later_native_status(monkeypatch, sampled):
    native, states, modes = _native(monkeypatch)
    lengths = torch.tensor([104, 102], dtype=torch.int32)
    qsl = torch.tensor([0, 4, 6], dtype=torch.int32)
    positions = torch.tensor([100, 101, 102, 103, 100, 101, 200, 200], dtype=torch.int64)
    batch = SimpleNamespace(num_reqs=2, req_ids=["a", "b"], num_tokens_no_spec=lengths,
                            vocab_size=64)
    requests = {name: SimpleNamespace(get_token_id=lambda pos: 10) for name in batch.req_ids}
    runner = SimpleNamespace(input_batch=batch, pcp_manager=None, actual_seq_lengths_q=[4, 6],
                             attn_state=states.SpecDecoding, decode_token_per_req=4)
    proposer = SimpleNamespace(runner=runner, pcp_size=1, dcp_size=1, method="mtp",
        arange=torch.arange(8, dtype=torch.int32), token_arange_np=torch.arange(8, dtype=torch.int32),
        backup_next_token_ids=_Buffer(torch.zeros(2, dtype=torch.int64)), uses_mrope=False,
        has_gdn=True, kernel_block_size=128, max_model_len=262144, use_compress=False,
        slot_mapping_group=[torch.full((8,), -1, dtype=torch.int32) for _ in range(3)],
        seq_lens_group=[torch.zeros(2, dtype=torch.int32) for _ in range(3)],
        query_start_loc_group=[torch.zeros(3, dtype=torch.int32) for _ in range(3)])
    proposer.shallow_copy_metadata = MethodType(native.shallow_copy_metadata, proposer)
    _, counts = native.prepare_next_token_ids_padded(proposer, sampled, requests, batch,
                                                    torch.empty(0, dtype=torch.int64), 0)
    common = Common(query_start_loc=qsl, query_start_loc_cpu=qsl.clone(), seq_lens=lengths.clone(),
        seq_lens_cpu=lengths.clone(), _seq_lens_cpu=lengths.clone(), seq_lens_cpu_upper_bound=lengths,
        num_reqs=2, num_actual_tokens=6, num_input_tokens=6, max_query_len=4,
        num_computed_tokens_cpu=None, _num_computed_tokens_cpu=None,
        block_table_tensor=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        slot_mapping=proposer.slot_mapping_group[0], positions=positions,
        positions_cpu=None, attn_state=states.SpecDecoding, decode_token_per_req=4,
        is_prefilling=torch.zeros(2, dtype=torch.bool), max_seq_len=104)
    spec = SimpleNamespace(cu_num_draft_tokens=torch.tensor([3, 4], dtype=torch.int32))
    prepared, _, anchors, rejected = native.prepare_inputs_padded(proposer, common, spec, counts)
    group = SimpleNamespace(get_metadata_builder=lambda: SimpleNamespace(build_for_drafting=lambda meta, index: meta))
    updated, _ = native.attn_update_stack_num_spec_norm(
        proposer, 1, None, prepared, 2, 6, positions[anchors], modes.NONE, attn_group=group)
    status = [_prepare_status(int(updated.slot_mapping[row]), int(updated.seq_lens[row]),
                              updated.block_table_tensor[row].tolist()) for row in range(2)]
    return counts.tolist(), rejected.tolist(), anchors.tolist(), status


def test_unspecified_masked_load_witness_reaches_native_anchor_and_slot_code5(monkeypatch):
    """Constructed source witness, not a replay of either reported NPU fault."""
    pointer, language, namespace = _source_scalar_kernels()
    old = [-1] * 8
    for row in range(2):
        language.pid = row
        language.unspecified = iter((0, 3))
        namespace["rejection_greedy_sample_triton"](
            pointer("out", old), pointer("cu", [3, 4]), pointer("draft", [10, 11, 12, 13]),
            pointer("target", [10, 11, 12, 13]), pointer("bonus", [20, 21]),
            pointer("greedy", [1, 0]), 2, 3, 1)
    _run_random_source(pointer, language, namespace, old)
    assert ("cu", -1) in language.invalid_reads
    assert old == [10, 11, 12, 20, 30, 11, 12, -1]
    assert _later_native_status(monkeypatch, torch.tensor(old).reshape(2, 4)) == (
        [4, 3], [0, -1], [3, 6], [0, 5])

    actual = torch.full((2, 4), -1, dtype=torch.int32)
    ids = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    greedy(actual, [3, 1], torch.tensor([3, 4]), ids, ids,
           torch.tensor([[20], [21]]), torch.tensor([True, False]), 3, 2, 1)
    new = actual.flatten().tolist()
    _run_random_source(pointer, language, namespace, new)
    assert new == [10, 11, 12, 20, 30, -1, -1, -1]
    assert _later_native_status(monkeypatch, torch.tensor(new).reshape(2, 4)) == (
        [4, 1], [0, 1], [3, 4], [0, 0])


def test_original_all_random_source_reads_before_cumulative_tensor():
    pointer, language, namespace = _source_scalar_kernels()
    language.invalid_reads = []
    output = [-1] * 8
    _run_random_source(pointer, language, namespace, output, greedy_mask=(0, 0))
    assert language.invalid_reads == [("cu", -1)]
    assert output == [30, -1, -1, -1, 30, -1, -1, -1]


@pytest.mark.parametrize("deferred", [False, True])
def test_plugin_patches_both_native_aliases_and_restores_exact_original(monkeypatch, deferred):
    modules = [ModuleType(name) for name in MODULES]
    calls = []
    def native(*args, **kwargs):
        raise AssertionError("unsafe native greedy writer was called")
    for module in modules:
        setattr(module, NAME, native)
    native_sample = lambda *args, **kwargs: None
    modules[1].rejection_sample = native_sample
    monkeypatch.setenv("OSCAR_ENABLED", "1")
    helper = sys.modules[greedy.__module__]
    monkeypatch.setattr(helper, NAME, lambda *args, **kwargs: calls.append((args, kwargs)))
    plugin.unregister()
    try:
        if deferred:
            plugin.register()
            plugin._apply_callback(plugin._callbacks[MODULES[0]], modules[0])
            # Model `from ...reject_sample import ...` after its callback ran.
            setattr(modules[1], NAME, getattr(modules[0], NAME))
            plugin._apply_callback(plugin._callbacks[MODULES[1]], modules[1])
        else:
            for module in modules:
                monkeypatch.setitem(sys.modules, module.__name__, module)
            plugin.register()
        for module in modules:
            installed = getattr(module, NAME)
            assert installed is not native
            assert inspect.signature(installed) == inspect.signature(native)
            installed("sentinel", key="value")
            plugin._apply_callback(plugin._callbacks[module.__name__], module)
            assert getattr(module, NAME) is installed
        assert calls == [(("sentinel",), {"key": "value"})] * 2
    finally:
        plugin.unregister()
    assert all(getattr(module, NAME) is native for module in modules)
    assert modules[1].rejection_sample is native_sample


@pytest.mark.parametrize("passive", [False, True])
def test_disabled_or_passive_native_service_does_not_patch_sampling(monkeypatch, passive):
    modules = [ModuleType(name) for name in MODULES]
    native = lambda: None
    for module in modules:
        setattr(module, NAME, native)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    modules[1].rejection_sample = native
    monkeypatch.setenv("OSCAR_ENABLED", "0")
    monkeypatch.setenv("OSCAR_PASSIVE_TIMING_CONTROL", "/tmp/greedy-sampling-passive-test.json")
    plugin.unregister()
    try:
        if passive:
            plugin.register_passive_observer()
        else:
            plugin.register()
        assert all(getattr(module, NAME) is native for module in modules)
        assert modules[1].rejection_sample is native
    finally:
        plugin.unregister()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("keyword", [False, True])
def test_outer_sampler_guards_first_row_pointer_without_changing_sampling(monkeypatch, dtype, keyword):
    cu = torch.tensor([3, 4], dtype=dtype)
    original_values = cu.clone()
    arguments = dict(draft_token_ids=object(), num_draft_tokens=[3, 1], max_spec_len=3,
        cu_num_draft_tokens=cu, draft_probs=object(), target_logits_or_tuple=object(),
        bonus_token_ids=object(), sampling_metadata=SimpleNamespace(all_random=True),
        synthetic_mode=True, synthetic_conditional_rates=object(), ori_target_logits=object())
    result = object()
    observed = {}

    def native(draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
               draft_probs, target_logits_or_tuple, bonus_token_ids, sampling_metadata,
               synthetic_mode=False, synthetic_conditional_rates=None, ori_target_logits=None):
        observed.update(locals())
        return result

    modules = [ModuleType(name) for name in MODULES]
    for module in modules:
        setattr(module, NAME, lambda *args: None)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    modules[1].rejection_sample = native
    monkeypatch.setenv("OSCAR_ENABLED", "1")
    plugin.unregister()
    try:
        plugin.register()
        wrapped = modules[1].rejection_sample
        assert inspect.signature(wrapped) == inspect.signature(native)
        if keyword:
            returned = wrapped(**arguments)
        else:
            returned = wrapped(*arguments.values())
        assert returned is result
    finally:
        plugin.unregister()
    assert modules[1].rejection_sample is native
    guarded = observed["cu_num_draft_tokens"]
    for name, value in arguments.items():
        if name != "cu_num_draft_tokens":
            assert observed[name] is value
    assert guarded.shape == cu.shape and guarded.stride() == cu.stride()
    assert guarded.dtype == cu.dtype and guarded.device == cu.device
    assert guarded.storage_offset() == 32 // cu.element_size()
    assert guarded.data_ptr() % 32 == 0
    assert guarded.untyped_storage().data_ptr() != cu.untyped_storage().data_ptr()
    prefix_tail = torch.as_strided(guarded, (1,), (1,), guarded.storage_offset() - 1)
    assert prefix_tail.tolist() == [0]
    with pytest.raises(RuntimeError, match="storage[ _]offset"):
        torch.as_strided(cu, (1,), (1,), cu.storage_offset() - 1)
    torch.testing.assert_close(guarded, cu, rtol=0, atol=0)
    torch.testing.assert_close(cu, original_values, rtol=0, atol=0)


def test_outer_sampler_guard_has_no_tensor_readback(monkeypatch):
    cu = torch.tensor([0, 1, 3, 6], dtype=torch.int32)
    observed = []
    def native(**kwargs):
        observed.append(kwargs["cu_num_draft_tokens"])
        return "native result"
    def forbidden(*args, **kwargs):
        raise AssertionError("host tensor readback is forbidden")
    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist", "numpy", "__bool__", "nonzero"):
            patch.setattr(torch.Tensor, name, forbidden)
        result = rejection_sample_with_guarded_cu(
            native, None, [0, 1, 2, 3], 3, cu, None, None, None, None)
    assert result == "native result"
    torch.testing.assert_close(observed[0], cu, rtol=0, atol=0)


def test_failed_outer_hook_restores_partial_greedy_install(monkeypatch):
    module = ModuleType(MODULES[1])
    native = lambda: None
    setattr(module, NAME, native)
    monkeypatch.setenv("OSCAR_ENABLED", "1")
    plugin.unregister()
    try:
        with pytest.raises(plugin.OscarHookConflictError, match="rejection_sample"):
            plugin._apply_callback(plugin._callbacks[module.__name__], module)
        assert getattr(module, NAME) is native
        assert not any(patch.owner is module for patch in plugin._patches)
    finally:
        plugin.unregister()


def test_native_without_triton_needs_no_triton_seams(monkeypatch):
    module = ModuleType(MODULES[1])
    module.HAS_TRITON = False
    native = lambda *args, **kwargs: "native PyTorch sampling"
    module.rejection_sample = native
    monkeypatch.setenv("OSCAR_ENABLED", "1")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    plugin.unregister()
    try:
        plugin.register()
        assert module.rejection_sample is native
        assert not hasattr(module, NAME)
        assert not any(patch.owner is module for patch in plugin._patches)
    finally:
        plugin.unregister()


def test_triton_enabled_missing_greedy_seam_fails_closed(monkeypatch):
    module = ModuleType(MODULES[1])
    module.HAS_TRITON = True
    module.rejection_sample = lambda *args, **kwargs: None
    monkeypatch.setenv("OSCAR_ENABLED", "1")
    plugin.unregister()
    try:
        with pytest.raises(plugin.OscarHookConflictError, match=NAME):
            plugin._apply_callback(plugin._callbacks[module.__name__], module)
    finally:
        plugin.unregister()
