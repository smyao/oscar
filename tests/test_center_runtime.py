"""Archive G19-24/#86/#111/#148/#164: TP mean ownership and BF16 centering.

Host integration checks only; no NPU/model-quality or performance claim.
"""
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from oscar_ascend import rotations, runtime
from oscar_ascend.integration.official_math import rotate_inputs
from oscar_ascend.integration.cache_storage_ops import store_current
from oscar_ascend.integration.runtime_api import OscarReadinessError
from test_official_runtime import workspace, state, Ops


def centered_state(n=2, hk=1):
    w = workspace(n=n, hk=hk)
    w.centered_key_bf16 = torch.empty_like(w.key_input)
    s = state(w)
    s.key_mean_bf16 = torch.linspace(-2, 2, 256).bfloat16()
    return w, s


@pytest.mark.parametrize("current_only", [False, True])
def test_all_current_keys_center_in_bf16_while_queries_and_values_do_not(current_only):
    w, s = centered_state()
    q = torch.ones_like(w.query_input)
    k = torch.full_like(w.key_input, 1.015625)
    v = torch.full_like(w.value_input, -.75)
    original = k.clone()
    addresses = [x.data_ptr() for x in (w.centered_key_bf16, w.key_rot_bf16, w.value_rot_bf16)]
    qr, kr, vr = rotate_inputs(Ops(), s, q, k, v, torch.arange(2), current_only=current_only)
    expected_center = (k - s.key_mean_bf16).bfloat16()
    assert torch.equal(w.centered_key_bf16, expected_center)
    assert torch.equal(qr, q @ s.rotation_k_bf16)
    assert torch.equal(kr, expected_center @ s.rotation_k_bf16)
    assert torch.equal(vr, v @ s.rotation_v_bf16)
    assert torch.equal(k, original)
    assert addresses == [x.data_ptr() for x in (w.centered_key_bf16, w.key_rot_bf16, w.value_rot_bf16)]


def test_standalone_store_and_precomputed_store_center_exactly_once():
    w, s = centered_state()
    q, k, v = torch.ones_like(w.query_input), torch.ones_like(w.key_input), torch.full_like(w.value_input, 2)
    expected = (k - s.key_mean_bf16).bfloat16() @ s.rotation_k_bf16
    calls = []
    ops = Ops()
    def store(kr, vr, exact_k, exact_v, *args):
        assert kr.data_ptr() == exact_k.data_ptr() and vr.data_ptr() == exact_v.data_ptr()
        assert torch.equal(kr, expected)
        assert torch.equal(vr, v @ s.rotation_v_bf16)
        calls.append(kr.clone())
    ops.store_aligned_int2_out = store
    config = {"experimental_official_math": True}
    slots = torch.arange(2)
    store_current(ops, s, k, v, slots, slots, None, None, config)
    _, kr, vr = rotate_inputs(ops, s, q, k, v, slots)
    store_current(ops, s, k, v, slots, slots, None, None, config, rotated_kv=(kr, vr))
    assert len(calls) == 2 and torch.equal(calls[0], calls[1])


def test_actual_current_only_dispatch_passes_centered_key_to_fia_and_store(monkeypatch):
    from oscar_ascend.integration import current_only_dispatch as dispatch
    from oscar_ascend.integration.current_only_plan import plan_current_suffix
    from oscar_ascend.integration.metadata import OscarMetadata
    w, s = centered_state(n=1)
    w.slots = torch.empty(1, dtype=torch.int64)
    w.tasks = torch.empty(3, 16, dtype=torch.int64)
    w.attention_status = torch.empty(3, 2, dtype=torch.int32)
    w.positions = torch.empty(1, dtype=torch.int64)
    w.merge_status = torch.empty(1, 6, dtype=torch.int32)
    w.lse = torch.empty(1, 6)
    s.snapshots.recent_tokens = 256
    q, k, v = torch.zeros(1, 6 * 256).bfloat16(), torch.ones(1, 256).bfloat16(), torch.ones(1, 256).bfloat16()
    expected = (k.view(1, 1, 256) - s.key_mean_bf16).bfloat16() @ s.rotation_k_bf16
    seen = []
    class CurrentOps(Ops):
        def prepare_official_attention_tasks_out(self, starts, lengths, slots, tasks, positions, *geometry):
            tasks.zero_(); tasks[:, 1] = 1; positions.zero_()
        def store_aligned_int2_out(self, kr, vr, wk, wv, *args):
            assert torch.equal(kr, expected) and kr.data_ptr() == wk.data_ptr()
            args[6].zero_(); seen.append("store")
    def fia(qr, kr, vr, cumulative, **kwargs):
        assert torch.equal(kr, expected)
        assert torch.equal(qr, q.view(1, 6, 256) @ s.rotation_k_bf16)
        seen.append("fia")
        return vr.expand(1, 6, 256).contiguous(), torch.zeros(1, 6, 1)
    monkeypatch.setattr(dispatch, "native_current_partial", fia)
    monkeypatch.setattr(dispatch, "guard_current_slots", lambda ops, status, slots, n: status[:n].zero_())
    metadata = OscarMetadata(torch.tensor([0, 1], dtype=torch.int32), torch.tensor([1], dtype=torch.int32),
        torch.zeros(1, 1, dtype=torch.int32), torch.tensor([0], dtype=torch.int32), 1, 1, 1, 1,
        num_input_tokens=1, query_start_loc_cpu=torch.tensor([0, 1]), current_cumulative=(1,))
    impl = NS(num_heads=6, num_kv_heads=1, head_size=256, scale=1 / 16,
              provider=NS(ops=CurrentOps(), config={"experimental_official_math": True}, layer_state=lambda _: s))
    plan = plan_current_suffix((0, 1), (1,), actual_tokens=1, padded_tokens=1)
    dispatch.dispatch_current_suffix(impl, NS(layer_name="test"), q, k, v, object(), metadata, torch.empty_like(q), plan)
    assert seen == ["fia", "store"]


def test_perhead_mean_never_silently_broadcasts_to_multiple_local_heads():
    w, s = centered_state(hk=2)
    with pytest.raises(OscarReadinessError, match="one local KV head"):
        rotate_inputs(Ops(), s, torch.ones_like(w.query_input), torch.ones_like(w.key_input),
                      torch.ones_like(w.value_input), torch.arange(2))


def test_centering_requires_preprofile_scratch_and_ledger_counts_it():
    w, s = centered_state()
    assert w.geometry.buffer_bytes["centered_key_bf16"] == 2 * 1 * 256 * 2
    w.centered_key_bf16 = None
    with pytest.raises(OscarReadinessError, match="before KV profiling"):
        rotate_inputs(Ops(), s, torch.ones_like(w.query_input), torch.ones_like(w.key_input),
                      torch.ones_like(w.value_input), torch.arange(2))


def setup_provider(monkeypatch, *, world=4, rank=2, official=True):
    name = "model.layers.3.self_attn.attn"
    matrices = torch.stack([torch.eye(256).roll(h, 1) for h in range(4)])
    means = torch.stack([torch.full((256,), float(h + 1)) for h in range(4)])
    artifact = {"format_version": 2, "source_grouping": "layer_kv_head", "global_kv_heads": 4,
                "head_dim": 256, "layers": {name: {"Rk": matrices, "Rv": matrices, "k_mean": means}}}
    cfg = {"model": "/unused/test/model", "tensor_parallel_size": 4,
           "experimental_official_math": official, "experimental_aligned_math": True,
           "experimental_bf16_attention": True}
    provider = object.__new__(runtime.AscendRuntimeProvider)
    provider.config = cfg
    provider.rotations = {}; provider.bf16_key_rotations = {}; provider.bf16_rotations = {}; provider.bf16_key_means = {}
    native = ModuleType("vllm.distributed")
    native.get_tensor_model_parallel_rank = lambda: rank
    native.get_tensor_model_parallel_world_size = lambda: world
    monkeypatch.setitem(sys.modules, "vllm.distributed", native)
    monkeypatch.setattr(rotations, "checkpoint_identity", lambda path: ([name], 256, "test-fingerprint"))
    monkeypatch.setattr(rotations, "load_active_artifact", lambda *args, **kwargs: artifact)
    prepare = rotations.prepare_device_rotations
    monkeypatch.setattr(rotations, "prepare_device_rotations", lambda artifact, **kwargs: prepare(artifact, testing=True, **kwargs))
    return provider, name, matrices, means


def test_native_tp_rank_owns_its_matrix_mean_and_mtp_has_identity_zero(monkeypatch):
    provider, name, matrices, means = setup_provider(monkeypatch)
    prepared = provider._prepare_rotations(256, "cpu")
    assert torch.equal(prepared[name].key_transposed.T, matrices[2])
    assert torch.equal(provider.bf16_key_means[("cpu", 256)][name], means[2].bfloat16())
    assert torch.equal(prepared[None].key_transposed, torch.eye(256))
    assert torch.count_nonzero(provider.bf16_key_means[("cpu", 256)][None]) == 0
    assert provider._prepare_rotations(256, "cpu") is prepared


def test_native_tp_world_mismatch_is_rejected_before_rank_selection(monkeypatch):
    provider, *_ = setup_provider(monkeypatch, world=2)
    with pytest.raises(OscarReadinessError, match="native worker group"):
        provider._prepare_rotations(256, "cpu")


def test_center_artifact_cannot_enter_legacy_math_without_subtracting_mean(monkeypatch):
    provider, *_ = setup_provider(monkeypatch, official=False)
    with pytest.raises(OscarReadinessError, match="official candidate"):
        provider._prepare_rotations(256, "cpu")
