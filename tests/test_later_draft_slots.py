"""Archive #27/#34/#36/#37-49/#157/#158: explicit discard, never inferred corruption.

The independent slot resolver follows attention_tasks.cpp integer bounds.
These are host contract tests, not NPU/graph or whole-service acceptance.
"""
from types import SimpleNamespace
import ast
from pathlib import Path

import pytest
import torch

from oscar_ascend.integration.draft_discard import (
    later_draft_slot_copy, later_draft_lengths_copy, native_draft_scope)
from oscar_ascend.integration.runtime_api import OscarReadinessError


def _case(slots=(1285, 99999, -1, -1), discard=(False, True), *, index=1):
    groups = [torch.full((len(slots),), -1, dtype=torch.int32) for _ in range(3)]
    groups[index].copy_(torch.tensor(slots, dtype=torch.int32))
    runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=len(discard)),
                             discard_request_mask=SimpleNamespace(gpu=torch.tensor(discard)))
    proposer = SimpleNamespace(runner=runner, slot_mapping_group=groups, pcp_size=1, dcp_size=1,
        max_model_len=262144, seq_lens_group=[torch.full((len(slots),), 128, dtype=torch.int32) for _ in range(3)])
    common = SimpleNamespace(slot_mapping=groups[index], num_reqs=len(discard),
                             num_actual_tokens=len(discard), max_query_len=1)
    return proposer, common


def _prepare_status(slot, length, table):
    if slot < 0:
        return 0
    matches = [j for j in range(min(len(table), (length + 127) // 128)) if table[j] == slot // 128]
    return 0 if len(matches) == 1 and 0 <= matches[0] * 128 + slot % 128 < length else 5


def test_only_explicit_discard_is_padded_and_native_slots_are_untouched():
    proposer, common = _case()
    original = common.slot_mapping.clone()
    with native_draft_scope(proposer):
        slots = later_draft_slot_copy(common, draft_index=1)
    assert slots.tolist() == [1285, -1, -1, -1]
    assert torch.equal(common.slot_mapping, original)
    assert slots.data_ptr() != common.slot_mapping.data_ptr()


@pytest.mark.parametrize("length,table,slot,expected", [
    (128, [10, 10], 1285, 0),
    (200, [10, 10], 1285, 5),
    (128, [10, 0], 1413, 5),
    (130, [10, 11], 1413, 5),
])
def test_live_valid_and_corrupt_slots_are_preserved(length, table, slot, expected):
    proposer, common = _case(slots=(slot,), discard=(False,))
    with native_draft_scope(proposer):
        actual = later_draft_slot_copy(common, draft_index=1)
    assert actual.tolist() == [slot]
    assert _prepare_status(int(actual[0]), length, table) == expected


def test_nonnegative_padding_tail_is_not_hidden():
    proposer, common = _case(slots=(1285, 99999, 77777), discard=(False, True))
    with native_draft_scope(proposer):
        slots = later_draft_slot_copy(common, draft_index=1)
    assert slots.tolist() == [1285, -1, 77777]


def test_dummy_capture_and_changed_real_batches_share_address_without_stale_mask():
    proposer, common = _case()
    with native_draft_scope(proposer, dummy=True):
        captured = later_draft_slot_copy(common, capture=True)
        address = captured.data_ptr()
        assert torch.equal(captured, common.slot_mapping)
    with native_draft_scope(proposer):
        first = later_draft_slot_copy(common, draft_index=1)
        assert first.data_ptr() == address
        assert first.tolist() == [1285, -1, -1, -1]
    common.slot_mapping[:2].copy_(torch.tensor([99999, 1285], dtype=torch.int32))
    proposer.runner.discard_request_mask.gpu.copy_(torch.tensor([True, False]))
    with native_draft_scope(proposer):
        second = later_draft_slot_copy(common, draft_index=1)
        assert second.data_ptr() == address
        assert second.tolist() == [-1, 1285, -1, -1]
    assert later_draft_slot_copy(common, draft_index=1) is common.slot_mapping


def test_scope_resets_after_exception_and_nested_proposals():
    first, first_common = _case()
    second, second_common = _case(discard=(True, False))
    with native_draft_scope(first):
        with pytest.raises(RuntimeError, match="abort"):
            with native_draft_scope(second):
                assert later_draft_slot_copy(second_common, draft_index=1)[0] == -1
                raise RuntimeError("abort")
        assert later_draft_slot_copy(first_common, draft_index=1)[0] == 1285
    assert later_draft_slot_copy(first_common, draft_index=1) is first_common.slot_mapping


def test_first_draft_is_unchanged_and_steps_do_not_alias():
    proposer, common = _case(index=0)
    with native_draft_scope(proposer):
        assert later_draft_slot_copy(common, draft_index=0) is common.slot_mapping
        assert later_draft_slot_copy(common, capture=True) is common.slot_mapping
        common.slot_mapping = proposer.slot_mapping_group[1]
        one = later_draft_slot_copy(common, draft_index=1)
        common.slot_mapping = proposer.slot_mapping_group[2]
        two = later_draft_slot_copy(common, draft_index=2)
    assert one.data_ptr() != two.data_ptr()


def test_missing_native_mask_or_wrong_row_contract_fails_closed():
    proposer, common = _case()
    proposer.runner.discard_request_mask = None
    with pytest.raises(OscarReadinessError, match="discard mask"):
        with native_draft_scope(proposer):
            pass
    proposer, common = _case()
    common.num_actual_tokens = 1
    with native_draft_scope(proposer), pytest.raises(OscarReadinessError, match="request order"):
        later_draft_slot_copy(common, draft_index=1)


def test_no_host_readback_or_block_table_dependency(monkeypatch):
    proposer, common = _case()
    def forbidden(*args, **kwargs):
        raise AssertionError("device readback is forbidden")
    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        with native_draft_scope(proposer):
            slots = later_draft_slot_copy(common, draft_index=1)
    assert slots.tolist() == [1285, -1, -1, -1]


def test_plugin_proposal_scope_reaches_real_metadata_builder(monkeypatch):
    from oscar_ascend import plugin
    from test_first_draft_current_only import builder as make_builder
    builder = make_builder(monkeypatch)
    proposer, common = _case()
    common.causal = True
    common.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    common.seq_lens = torch.tensor([128, 128], dtype=torch.int32)
    proposer.seq_lens_group[1] = common.seq_lens
    common.block_table_tensor = torch.tensor([[10, 10], [22, 0]], dtype=torch.int32)
    common.max_seq_len = 128
    common.num_input_tokens = 4

    class NativeProposer:
        method = "mtp"
        def _propose(self, metadata):
            return builder.build_for_drafting(metadata, 1)
        def dummy_run(self, metadata):
            return builder.build_for_graph_capture(metadata)

    instance = NativeProposer()
    instance.__dict__.update(vars(proposer))
    original = NativeProposer._propose
    try:
        plugin._patch_mtp_proposer(SimpleNamespace(AscendSpecDecodeBaseProposer=NativeProposer))
        metadata = instance._propose(common)
        assert metadata.slot_mapping.tolist() == [1285, -1, -1, -1]
        assert metadata.is_draft and metadata.draft_index == 1
        assert common.slot_mapping.tolist() == [1285, 99999, -1, -1]
        assert later_draft_slot_copy(common, draft_index=1) is common.slot_mapping
    finally:
        plugin.unregister()
    assert NativeProposer._propose is original


def test_real_mtp_builder_fails_closed_without_the_native_scope(monkeypatch):
    from test_first_draft_current_only import builder as make_builder
    builder = make_builder(monkeypatch)
    _, common = _case()
    common.causal = True
    common.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    common.seq_lens = torch.tensor([128, 128], dtype=torch.int32)
    common.block_table_tensor = torch.tensor([[10, 10], [22, 0]], dtype=torch.int32)
    common.max_seq_len = 128
    common.num_input_tokens = 4
    with pytest.raises(OscarReadinessError, match="outside the hooked proposer scope"):
        builder.build_for_drafting(common, 1)


@pytest.mark.parametrize("deferred", [False, True])
def test_pinned_native_eagle_mro_inherits_the_proposer_scope_hook(monkeypatch, capsys, deferred):
    from oscar_ascend import plugin
    from test_first_draft_current_only import builder as make_builder
    builder = make_builder(monkeypatch)
    base, common = _case()
    common.causal = True
    common.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    common.seq_lens = torch.tensor([128, 128], dtype=torch.int32)
    base.seq_lens_group[1] = common.seq_lens
    common.block_table_tensor = torch.tensor([[10, 10], [22, 0]], dtype=torch.int32)
    common.max_seq_len = 128
    common.num_input_tokens = 4

    class SpecDecodeBaseProposer:
        pass

    class AscendSpecDecodeBaseProposer(SpecDecodeBaseProposer):
        method = "mtp"
        def _propose(self, metadata):
            return builder.build_for_drafting(metadata, 1)
        def dummy_run(self, metadata):
            return builder.build_for_graph_capture(metadata)

    namespace = {"SpecDecodeBaseProposer": SpecDecodeBaseProposer,
                 "AscendSpecDecodeBaseProposer": AscendSpecDecodeBaseProposer,
                 "torch": torch, "VllmConfig": object}
    root = Path(__file__).resolve().parents[1]
    # Execute the real class declarations, including their inheritance order.
    # Numerical native method bodies are exercised in test_native_discard_pipeline.
    for path, name in (
        ("references/vllm/vllm/v1/spec_decode/eagle.py", "EagleProposer"),
        ("references/vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py", "AscendEagleProposer"),
    ):
        tree = ast.parse((root / path).read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(root / path), "exec"), namespace)
    actual_type = namespace["AscendEagleProposer"]
    assert actual_type.__mro__[:4] == (actual_type, namespace["EagleProposer"],
                                     AscendSpecDecodeBaseProposer, SpecDecodeBaseProposer)
    module = SimpleNamespace()
    original = AscendSpecDecodeBaseProposer._propose
    try:
        if deferred:
            class NativeLoader:
                def exec_module(self, target):
                    target.AscendSpecDecodeBaseProposer = AscendSpecDecodeBaseProposer
            monkeypatch.setattr(plugin, "_enabled", True)
            plugin._AfterExecLoader(NativeLoader(), plugin._patch_mtp_proposer).exec_module(module)
        else:
            module.AscendSpecDecodeBaseProposer = AscendSpecDecodeBaseProposer
            plugin._apply_callback(plugin._patch_mtp_proposer, module)
        instance = actual_type.__new__(actual_type)
        instance.__dict__.update(vars(base))
        one = instance._propose(common)
        two = instance._propose(common)
        assert one.slot_mapping.tolist() == two.slot_mapping.tolist() == [1285, -1, -1, -1]
        assert one.slot_mapping.data_ptr() == two.slot_mapping.data_ptr()
        assert instance._propose.__func__ is AscendSpecDecodeBaseProposer._propose
        assert instance._propose.__func__ is not original
        log = capsys.readouterr().out
        assert log.count("MTP_DISCARD_BRIDGE") == 1
        assert "proposer=AscendEagleProposer method=mtp" in log
        assert "source=native.discard_request_mask binding=queued" in log
    finally:
        plugin.unregister()
    assert AscendSpecDecodeBaseProposer._propose is original


@pytest.mark.parametrize("base,length,slot,discard,expected", [
    (262143, 1, 777777, False, 262144),  # true native wrap; corruption stays live
    (7, 1, 103, False, 1),              # unrelated lens=1 is not a wrap proof
    (262143, 2, 103, False, 2),         # only the explicit native wrap-to-one
    (262143, 1, -1, False, 1),          # already out of model: native padding
    (262143, 1, 103, True, 1),          # explicit discard never reactivated
])
def test_native_wrap_correction_is_narrow_and_keeps_invalid_live_slots(
        monkeypatch, base, length, slot, discard, expected):
    proposer, common = _case(slots=(slot,), discard=(discard,), index=2)
    proposer.seq_lens_group[0].fill_(base)
    proposer.seq_lens_group[2].fill_(length)
    common.seq_lens = proposer.seq_lens_group[2]
    original = common.seq_lens.clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("native wrap correction must not read device contents")
    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        with native_draft_scope(proposer):
            slots = later_draft_slot_copy(common, draft_index=2)
            lengths = later_draft_lengths_copy(common, slots, draft_index=2)
    assert lengths.tolist() == [expected]
    assert torch.equal(common.seq_lens, original)
    if not discard and slot >= 0:
        assert slots.tolist() == [slot]
        assert _prepare_status(slot, expected, [0, 0]) == 5
    else:
        assert slots.tolist() == [-1]


def test_proven_native_upper_bound_keeps_fixed_copy_without_device_predicates(monkeypatch):
    proposer, common = _case(slots=(1285,), discard=(False,), index=1)
    common.seq_lens = proposer.seq_lens_group[1]
    common.max_seq_len = 128
    proposer.seq_lens_group[0].fill_(127)
    def forbidden(*args, **kwargs):
        raise AssertionError("proven far-from-limit path must not launch device predicates")
    with native_draft_scope(proposer):
        with monkeypatch.context() as patch:
            patch.setattr(torch, "where", forbidden)
            first = later_draft_lengths_copy(common, common.slot_mapping, draft_index=1)
        pointer = first.data_ptr()
        assert first.data_ptr() != common.seq_lens.data_ptr()
        assert torch.equal(first, common.seq_lens)
        common.max_seq_len = 0  # unknown host bound: retain the full device test
        second = later_draft_lengths_copy(common, common.slot_mapping, draft_index=1)
        assert second.data_ptr() == pointer
        assert torch.equal(second, common.seq_lens)
