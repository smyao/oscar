"""Native prefill API contract using an independent CPU causal attention op."""

import sys
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.kernels.prefill import (
    _causal_mask,
    npu_prefill,
    npu_prefill_prepared,
    npu_prefill_prepared_batch,
    npu_prefill_packed,
)


@pytest.mark.parametrize("prefix,n", [(0, 17), (65, 1), (2049, 33)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("prepared", [False, True])
def test_native_prefill_gqa_and_right_aligned_causality(
    monkeypatch, prefix, n, dtype, prepared
):
    torch.manual_seed(17)
    q = torch.randn(n, 4, 64).to(dtype)
    k, v = [torch.randn(prefix + n, 1, 64).to(dtype) for _ in range(2)]
    calls = []

    def native(**kw):
        calls.append(kw)
        assert kw["input_layout"] == "TND" and kw["sparse_mode"] == 3
        assert kw["num_heads"] == 4 and kw["num_key_value_heads"] == 1
        assert kw["actual_seq_lengths"] == [n]
        assert kw["actual_seq_lengths_kv"] == [n + prefix]
        assert kw["atten_mask"].shape == (2048, 2048)
        assert kw["atten_mask"][0, 1] == 1 and kw["atten_mask"][1, 0] == 0
        visible = torch.arange(prefix + n)[None, :] <= prefix + torch.arange(n)[:, None]
        out = (
            torch.nn.functional.scaled_dot_product_attention(
                kw["query"].float().transpose(0, 1),
                kw["key"].float().transpose(0, 1).repeat_interleave(4, 0),
                kw["value"].float().transpose(0, 1).repeat_interleave(4, 0),
                attn_mask=visible,
                scale=kw["scale"],
            )
            .transpose(0, 1)
            .to(dtype)
        )
        return out, None

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=native),
    )
    actual = (
        npu_prefill(q, k[prefix:], v[prefix:], k[:prefix], v[:prefix], 0.125, 1, 64)
        if not prepared
        else npu_prefill_prepared(q, k, v, 0.125, 1, 64)
    )
    # Independently compute each query over its own visible prefix.
    expected = torch.stack(
        [
            (
                torch.softmax(
                    torch.einsum(
                        "hd,td->ht", q[i].float(), k[: prefix + i + 1, 0].float()
                    )
                    * 0.125,
                    -1,
                )
                @ v[: prefix + i + 1, 0].float()
            ).to(dtype)
            for i in range(n)
        ]
    )
    torch.testing.assert_close(actual, expected)
    if prefix == 0 or prepared:
        assert calls[0]["key"].data_ptr() == k.data_ptr()
    assert calls[0]["atten_mask"] is _causal_mask(q.device)


def test_native_prefill_does_not_silently_downcast_fp32(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    x = torch.zeros(1, 1, 64)
    with pytest.raises(ValueError, match="bf16/fp16"):
        npu_prefill(x, x, x, x[:0], x[:0], 0.125, 1, 64)


def test_native_prefill_batch_uses_cumulative_tnd_lengths(monkeypatch):
    q_parts = [torch.randn(2, 4, 64, dtype=torch.bfloat16),
               torch.randn(1, 4, 64, dtype=torch.bfloat16)]
    k_parts = [torch.randn(5, 1, 64, dtype=torch.bfloat16),
               torch.randn(8, 1, 64, dtype=torch.bfloat16)]
    v_parts = [torch.randn_like(x) for x in k_parts]
    seen = []

    def native(**kw):
        seen.append(kw)
        assert kw["actual_seq_lengths"] == [2, 3]
        assert kw["actual_seq_lengths_kv"] == [5, 13]
        # This test checks packing/API boundaries; numerical causality is
        # covered independently by the single-request contract above.
        return torch.zeros_like(kw["query"]), None

    monkeypatch.setitem(
        sys.modules, "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=native),
    )
    out = npu_prefill_prepared_batch(q_parts, k_parts, v_parts, 0.125, 1, 64)
    assert [x.shape[0] for x in out] == [2, 1]
    assert len(seen) == 1


def test_native_prefill_packed_preserves_input_allocations(monkeypatch):
    q = torch.randn(3, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(13, 1, 64, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    seen = []

    def native(**kw):
        seen.append((kw["query"].data_ptr(), kw["key"].data_ptr(),
                     kw["value"].data_ptr()))
        return torch.zeros_like(kw["query"]), None

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(
        npu_fused_infer_attention_score=native
    ))
    out = npu_prefill_packed(q, k, v, [2, 3], [5, 13], 0.125, 1, 64)
    assert out.shape == q.shape
    assert seen == [(q.data_ptr(), k.data_ptr(), v.data_ptr())]
