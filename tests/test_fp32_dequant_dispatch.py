"""Archive #126/#142/#148/#164: fixed reader policy, including capture.

CPU routing/oracle contracts only; no NPU graph or model-quality claim.
"""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.integration import aligned_math, official_math
from oscar_ascend.integration.metadata import OscarMetadata
from oscar_ascend.integration.mixed_decode_split import _slice_metadata
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.runtime import AscendRuntimeProvider


def config(policy="fp32_metadata"):
    return {"experimental_official_math": True, "experimental_aligned_math": True,
            "experimental_bf16_attention": True, "oscar_dequant_policy": policy}


@pytest.mark.parametrize("state", ["DecodeOnly", "SpecDecoding", "PrefillNoCache",
                                   "ChunkedPrefill", "PrefillCacheHit"])
@pytest.mark.parametrize("capture", [False, True])
def test_all_valid_states_and_capture_use_the_fixed_policy(state, capture):
    metadata = NS(attn_state=NS(name=state), capture_origin=capture)
    assert aligned_math.dequant_mode(metadata, config()) == 0
    legacy = 1 if capture or state in {"DecodeOnly", "SpecDecoding"} else 0
    assert aligned_math.dequant_mode(metadata) == legacy
    assert aligned_math.dequant_mode(metadata, config("native_phase")) == legacy


@pytest.mark.parametrize("fault", ["policy", "state", "official", "aligned", "bf16"])
def test_invalid_policy_or_missing_math_contract_does_not_hide_errors(fault):
    selected = config(); metadata = NS(attn_state=NS(name="DecodeOnly"))
    if fault == "policy": selected["oscar_dequant_policy"] = "unknown"
    elif fault == "state": metadata.attn_state = None
    else:
        key = {"official": "experimental_official_math", "aligned": "experimental_aligned_math",
               "bf16": "experimental_bf16_attention"}[fault]
        selected[key] = False
    with pytest.raises(OscarReadinessError):
        aligned_math.dequant_mode(metadata, selected)


@pytest.mark.parametrize("policy", ["fp32_metadata", "native_phase"])
@pytest.mark.parametrize("n,later,capture,reader", [
    (8, False, False, "attention_cv_official_decode_out"),
    (8, True, False, "attention_cv_official_decode_q1_out"),
    (256, False, False, "attention_cv_official_cluster4_out"),
    (8192, False, False, "attention_cv_official_cluster16_out"),
    (8, False, True, "attention_cv_official_decode_out"),
])
def test_real_forward_passes_policy_mode_to_every_reader_and_capture(monkeypatch, policy, n, later, capture, reader):
    import test_runtime
    initialize = AscendRuntimeProvider.__init__
    def init(provider, selected=None):
        initialize(provider, {**selected, "oscar_dequant_policy": policy})
    monkeypatch.setattr(AscendRuntimeProvider, "__init__", init)
    from_common = test_runtime.from_common
    monkeypatch.setattr(test_runtime, "from_common", lambda *a, **kw:
                        replace(from_common(*a, **kw), capture_origin=capture))
    rotate = official_math.rotate_inputs
    seen = []
    expected = 0 if policy == "fp32_metadata" or (n > 128 and not capture) else 1
    def inspect(ops, state, *args, **kwargs):
        original = getattr(ops, reader)
        def call(*values):
            assert values[-2] == expected
            seen.append(values[-2])
            return original(*values)
        monkeypatch.setattr(ops, reader, call)
        return rotate(ops, state, *args, **kwargs)
    monkeypatch.setattr(official_math, "rotate_inputs", inspect)
    test_runtime.test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        n, 1, 1, n, 16, later, True, True, True, _member_pipeline=True,
        _c16_l1=True, _bf16_expected=reader, _aligned=True, _official=True)
    assert seen == [expected]


def test_mixed_children_keep_request_boundaries_and_fixed_mode():
    starts = torch.tensor([0, 1, 129], dtype=torch.int32)
    parent = OscarMetadata(starts, torch.tensor([513, 640], dtype=torch.int32),
        torch.zeros(2, 5, dtype=torch.int32), torch.arange(129), 2, 129, 128, 640,
        num_input_tokens=129, query_start_loc_cpu=starts, current_cumulative=(1, 129),
        attn_state=NS(name="ChunkedPrefill"), mixed_decode_split=True)
    children = [_slice_metadata(parent, 0, 1, 0, 1, 129, "short_prefix"),
                _slice_metadata(parent, 1, 2, 1, 129, 129, "long_suffix")]
    assert [child.seq_lens.tolist() for child in children] == [[513], [640]]
    assert [aligned_math.dequant_mode(child, config()) for child in children] == [0, 0]
    assert aligned_math.dequant_mode(replace(children[0], attn_state=NS(name="DecodeOnly")), config()) == 0


@pytest.mark.parametrize("policy,mode", [("fp32_metadata", 0), ("native_phase", 1)])
def test_next_decode_oracle_receives_provider_policy(monkeypatch, policy, mode):
    from tools import probe_current_only as probe, official_runtime_oracle
    from tools.official_fixture import make_fixture
    shape = probe.reuse.Shape("policy", (1,), (0,), 256, 1, 1, False)
    fixture = make_fixture(torch, shape, rotation_mode="identity")
    data = fixture["cpu"]
    state = NS(raw=data["raw"][probe.reuse.PREFIX:], rotation_k_bf16=fixture["rotation_k"],
        rotation_v=data["rv"], rotation_v_bf16=data["rv"], window_key=data["wk"],
        window_value=data["wv"], window_tags=data["tags"])
    monkeypatch.setattr(official_runtime_oracle, "sampled_fixture_expected",
                        lambda fixture, **kwargs: kwargs["dequant_mode"])
    q, k, v = (fixture["original_inputs"][key] for key in ("q", "ck", "cv"))
    assert probe._next_bf16_decode_oracle(torch, fixture, state, q, k, v, config(policy)) == mode
    class Reached(Exception): pass
    selected = config(policy)
    impl = NS(provider=NS(config=selected))
    def next_oracle(*args):
        assert args[-1] is selected
        raise Reached
    monkeypatch.setattr(probe, "_next_bf16_decode_oracle", next_oracle)
    with pytest.raises(Reached):
        probe._decode_after_store(torch, fixture, data, impl, state, impl, state, "cpu", {})


@pytest.mark.parametrize("mode", [0, 1])
def test_prefix_reader_and_independent_oracle_receive_identical_mode(monkeypatch, mode):
    from tools import probe_prefix_cache as probe, official_reference, official_runtime_oracle
    state = probe.allocate(torch, "cpu", 1, bf16=True, aligned=True, official=True, dequant_mode=mode)
    logical = tuple((torch.zeros(320, 1, 256).bfloat16(), torch.ones(320, 1, 256).bfloat16()) for _ in range(2))
    inputs = probe.make_inputs(torch, logical, ((1,), (2,)), 17)
    class Reached(Exception): pass
    class Ops:
        def __getattr__(self, name):
            def call(*args):
                if name == "attention_cv_official_decode_out":
                    assert args[-2] == mode
                    raise Reached
            return call
    with pytest.raises(Reached):
        probe.execute(Ops(), state, inputs, "attention_cv_official_decode_out", "unused", 1)
    unpack = official_reference.unpack_kv
    seen = []
    def read(packed, dim, selected):
        seen.append(selected)
        return unpack(packed, dim, selected)
    def request(*args, **kwargs):
        assert kwargs["dequant_mode"] == mode
        return {i: (torch.zeros(6, 256), torch.zeros(6)) for i in range(4)}
    monkeypatch.setattr(official_reference, "unpack_kv", read)
    monkeypatch.setattr(official_runtime_oracle, "request_samples", request)
    probe.oracle(torch, logical, inputs, bf16=True, aligned=True, official=True, dequant_mode=mode)
    assert seen == [mode, mode]


@pytest.mark.parametrize("policy,mode", [("fp32_metadata", 0), ("native_phase", 1)])
def test_prefix_run_resolves_policy_before_allocating_capture_state(monkeypatch, policy, mode):
    from tools import probe_prefix_cache as probe
    class Reached(Exception): pass
    def allocate(*args, **kwargs):
        assert kwargs["dequant_mode"] == mode
        raise Reached
    monkeypatch.setattr(probe, "selected_reader", lambda _: "attention_cv_official_decode_out")
    monkeypatch.setattr(probe, "allocate", allocate)
    with pytest.raises(Reached):
        probe.run_device(NS(device=lambda _: "cpu"), object(), config(policy), 1, {})
