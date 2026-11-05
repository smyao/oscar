"""Archive #86/#127-128/#157/#163/#170-177: native OSCAR integration audit.

Execute pinned native Qwen forward/sharding statements unchanged with CPU
boundary doubles. These check ownership and operator order, not fused NPU
arithmetic, end-to-end accuracy, or performance. References are read-only.
"""
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.rotations import prepare_device_rotations
from test_runtime_native import definitions


@pytest.mark.parametrize("fused", [False, True])
def test_pinned_native_qwen_keeps_rope_gate_and_projection_around_attention(monkeypatch, fused):
    """Both native paths apply the gate and o_proj to the backend's output."""
    calls = []
    positions = torch.tensor([7])
    q = torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16)
    k = torch.tensor([[5, 6]], dtype=torch.bfloat16)
    v = torch.tensor([[7, 8]], dtype=torch.bfloat16)
    gate = torch.tensor([[-2, 0, 2, 4]], dtype=torch.bfloat16)
    interleaved = torch.cat((q.view(1, 2, 2), gate.view(1, 2, 2)), dim=-1).flatten(1)
    qkv = torch.cat((interleaved, k, v), dim=-1)
    original_qkv = qkv.clone()
    backend_output = torch.tensor([[2, -3, 5, -7]], dtype=torch.bfloat16)
    projection = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.bfloat16)

    def fused_op(**kwargs):
        calls.append("fused_qkv_norm_rope")
        assert kwargs["qkv"] is qkv
        assert kwargs["has_gate"] is True
        assert kwargs["num_q_heads"] == 2 and kwargs["num_kv_heads"] == 1
        assert torch.equal(kwargs["q_weight"], torch.full((2,), 2.0))
        assert torch.equal(kwargs["k_weight"], torch.full((2,), 3.0))
        # Values at this boundary are intentionally supplied; the fused
        # hardware operator itself is outside this CPU test's evidence.
        return q * 2 + 7, k * 3 - 7, v, gate

    class TorchBoundary:
        ops = NS(vllm=NS(triton_split_qkv_rmsnorm_mrope=fused_op))

        def __getattr__(self, name):
            return getattr(torch, name)

    native = definitions("references/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py",
        "native_qwen_precision_audit", set(), {"torch": TorchBoundary()}, monkeypatch,
        methods=[("AscendQwen3NextAttention", "forward")])

    class Norm:
        def __init__(self, name, scale):
            self.name, self.scale = name, scale
            self.weight = torch.full((2,), float(scale - 1))

        def __call__(self, value):
            calls.append(self.name)
            return value * self.scale

    class Rope:
        cos_sin_cache = torch.arange(32).view(16, 2).bfloat16()
        mrope_section = [1, 0, 0]
        mrope_interleaved = True
        rotary_dim = 2

        def __call__(self, native_positions, native_q, native_k):
            calls.append("rope")
            assert native_positions is positions
            return native_q + 7, native_k - 7

    def qkv_proj(hidden):
        calls.append("qkv")
        return qkv, None

    def attention(native_q, native_k, native_v):
        calls.append("attention_backend")
        assert torch.equal(native_q, q * 2 + 7)
        assert torch.equal(native_k, k * 3 - 7)
        assert torch.equal(native_v, v)
        return backend_output

    def output_proj(value):
        calls.append("o_proj")
        assert torch.equal(value, backend_output * torch.sigmoid(gate))
        return value @ projection, None

    model = NS(config=NS(model_type="qwen3_5_text" if fused else "qwen3_next", rms_norm_eps=1e-6),
        qkv_proj=qkv_proj, q_norm=Norm("q_norm", 2), k_norm=Norm("k_norm", 3),
        rotary_emb=Rope(), num_heads=2, num_kv_heads=1, head_dim=2,
        q_size=4, kv_size=2, attn_output_gate=True, attn=attention, o_proj=output_proj)
    output = torch.empty((1, 2), dtype=torch.bfloat16)
    native.forward(model, positions, output, torch.empty((1, 3), dtype=torch.bfloat16))
    assert torch.equal(output, (backend_output * torch.sigmoid(gate)) @ projection)
    assert torch.equal(qkv, original_qkv) and positions.tolist() == [7]
    middle = ["fused_qkv_norm_rope"] if fused else ["q_norm", "k_norm", "rope"]
    assert calls == ["qkv", *middle, "attention_backend", "o_proj"]


@pytest.mark.parametrize("rank", range(4))
def test_recipe3_head_mean_matches_pinned_native_tp_kv_weight_shard(monkeypatch, rank):
    """Physical device 4-7 still has logical TP ranks 0-3."""
    class BlockQuantScaleParameter:
        pass

    native = definitions("references/vllm/vllm/model_executor/layers/linear.py",
        "native_qkv_precision_audit", set(),
        {"torch": torch, "BlockQuantScaleParameter": BlockQuantScaleParameter}, monkeypatch,
        methods=[("QKVParallelLinear", "weight_loader")])
    dim, heads = 64, 4
    # qkv_proj has 12 local Q/gate heads for six actual Q heads in TP4.
    linear = NS(validate_shard_id=lambda _: None, num_heads=12, num_kv_heads=1,
        head_size=dim, v_head_size=dim, tp_rank=rank, tp_size=4, num_kv_head_replicas=1)
    weight = torch.arange(heads, dtype=torch.float32).repeat_interleave(dim).view(-1, 1)
    parameter = torch.nn.Parameter(torch.zeros((14 * dim, 1)), requires_grad=False)
    parameter.output_dim = 0
    for shard, begin in (("k", 12 * dim), ("v", 13 * dim)):
        native.weight_loader(linear, parameter, weight, shard)
        assert torch.equal(parameter[begin:begin + dim], weight[rank * dim:(rank + 1) * dim])
    matrices = torch.stack([torch.eye(dim).roll(head, 1) for head in range(heads)])
    means = torch.arange(heads, dtype=torch.float32)[:, None].expand(heads, dim).contiguous()
    artifact = {"head_dim": dim, "source_grouping": "layer_kv_head", "global_kv_heads": heads,
        "layers": {"model.layers.3.self_attn.attn": {"Rk": matrices, "Rv": matrices, "k_mean": means}}}
    prepared = prepare_device_rotations(artifact, device="cpu", testing=True, tp_rank=rank, tp_size=4)
    rotation = prepared["model.layers.3.self_attn.attn"]
    assert torch.equal(rotation.key_mean, parameter[12 * dim:13 * dim, 0])
    assert torch.equal(rotation.key_transposed.T, matrices[rank])
    assert torch.equal(rotation.inverse_value_transposed, matrices[rank])
    with pytest.raises(ValueError, match="one KV head per TP rank"):
        prepare_device_rotations(artifact, device="cpu", testing=True, tp_rank=rank, tp_size=8)
