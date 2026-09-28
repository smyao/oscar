"""Model-free q4 diagnostic contracts; target NPU timing is a separate gate.

Archive #126/#129/#143/#148-150 and startup D.4: never equate BF16 native
history with OSCAR INT2 numerical output or an eager Event with graph replay.
"""

import pytest
import torch

from tools import probe_decode_hotpath as hotpath
from tools import probe_history_reuse as reuse


def test_native_builder_observes_same_logical_inputs_without_changing_fe0_fixture():
    spec = reuse.Shape("small_q4", (4, 4), (400, 401), 64, 1, 1, False)
    original = reuse.make_fixture(torch, spec)
    oscar, native = hotpath.build_paired_fixture(torch, spec)
    assert oscar["input_sha256"] == original["input_sha256"]
    assert oscar["tokens"] == 8
    assert native["q_cumulative"] == [4, 8]
    assert native["kv_lengths"] == [404, 405]
    assert native["table_scope"] == "minimal_legal_synthetic_width_not_service_stride"
    assert tuple(native["block_table"].shape) == (2, 4)
    assert sorted(native["block_table"].flatten().tolist()) == list(range(8))
    assert len(native["logical_input_sha256"]) == 64
    from oscar_ascend.ops.reference import attention
    for request, context in enumerate(spec.contexts):
        pages = native["block_table"][request].to(torch.int64)
        key = native["key"].index_select(0, pages).reshape(-1, 1, spec.dim)
        value = native["value"].index_select(0, pages).reshape(-1, 1, spec.dim)
        for local in range(4):
            token = 4 * request + local
            result = attention(native["query"][token:token + 1],
                               key[:context + local + 1], value[:context + local + 1],
                               scale=spec.dim ** -0.5, causal=False)
            torch.testing.assert_close(native["expected_output"][token], result.output[0],
                                       atol=0, rtol=0)
    kwargs, mask = hotpath.native_kwargs(torch, native, torch.device("cpu"), 64 ** -0.5)
    assert kwargs["input_layout"] == "TND" and kwargs["sparse_mode"] == 3
    assert kwargs["block_size"] == 128 and kwargs["pre_tokens"] == 2147483647
    assert kwargs["actual_seq_lengths"] == [4, 8]
    assert kwargs["actual_seq_lengths_kv"] == [404, 405]
    assert tuple(kwargs["key"].shape) == (8, 128, 64)
    assert tuple(mask.shape) == (2048, 2048)
    assert int(mask[0, 1]) == 1 and int(mask[1, 0]) == 0


def test_profile_selects_one_critical_actor_without_cross_core_field_maxima():
    values = [[[[0] * 24 for _ in range(4)] for _ in range(3)] for _ in range(2)]
    # Each source has work. The selected history actor has larger task span;
    # another actor has a larger isolated DMA field, which must not be mixed.
    for source in range(3):
        values[0][0][source][0] = 1
        values[0][0][source][3] = 10 + source
        values[0][0][source][5] = 3
        values[1][0][source][0] = 1
        values[1][0][source][3] = 20 + source
        values[1][0][source][5] = 1
        values[0][1][source][0] = 1
        values[0][1][source][3] = 30 + source
        values[0][1][source][6] = 4
        values[1][2][source][0] = 1
        values[1][2][source][3] = 40 + source
        values[1][2][source][6] = 2
    values[0][0][0][5] = 8
    for core in range(2):
        for engine in range(3):
            for field in range(23):
                values[core][engine][3][field] = sum(values[core][engine][source][field]
                                                     for source in range(3))
            values[core][engine][3][23] = (100 if engine == 0 else 70) + core
    report = hotpath._profile_summary(values, 2)
    history = report["critical_source_actors"]["history"]["AIC"]
    assert history["core"] == 1 and history["engine"] == "AIC"
    assert history["task_span_raw_ticks"] == 20
    assert history["fields_raw_ticks_or_counts"]["packed_dma"] == 1
    aiv = report["critical_source_actors"]["history"]["AIV"]
    assert aiv["core"] == 1 and aiv["engine"] == "AIV1"
    assert aiv["task_span_raw_ticks"] == 40
    assert aiv["fields_raw_ticks_or_counts"]["unpack_total"] == 2
    assert report["critical_compute_actor"] == {"raw_ticks": 101, "core": 1, "engine": "AIC"}
    assert "not kernel wall" in report["field_contract"]
    values[1][1][3][23] = 0
    with pytest.raises(hotpath.DecodeHotpathError, match="actor counters were not completed"):
        hotpath._profile_summary(values, 2)


def test_profile_rejects_total_mismatch_and_missing_source():
    values = [[[[0] * 24 for _ in range(4)] for _ in range(3)]]
    values[0][0][0][0] = 1
    values[0][0][0][3] = 5
    for engine in range(3):
        values[0][engine][3][23] = 100
    with pytest.raises(hotpath.DecodeHotpathError, match="total source"):
        hotpath._profile_summary(values, 1)
    values[0][0][3][0] = 1
    values[0][0][3][3] = 5
    with pytest.raises(hotpath.DecodeHotpathError, match="no live history AIV actor"):
        hotpath._profile_summary(values, 1)
