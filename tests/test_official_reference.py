"""Archive #13-20/#126/#145/#148/#164: independent v4 host numeric gates.

Pure Torch functions below are unchanged extracts from FutureMLS-Lab/OSCAR,
Apache-2.0, commit 38680852e52c10bd6b9e9a64ed9bc815e742859b. They are test
fixtures only. Neither a production kernel nor a prior local oracle supplies
expected values. No GPU/NPU, model-quality or performance claim.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import official_reference as ref

ROOT = Path(__file__).resolve().parents[1]

UPSTREAM_FILE_PINS = {'python/sglang/srt/layers/attention/quantized_kv_prefill.py': 'bc3f1718015e7a353d0f32e6341446f6ab0294e5eab9fcf8625bc8af2645e081', 'python/sglang/srt/mem_cache/kv_quant_kernels.py': '60aaa142c34d5113cb9b5e4366b74b9c474120f6eb6405fe6972b4cec8eb53eb', 'python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py': '66c8ed67781777ec1ded2bcc7cb0d4c5a440aa1abe842fd62dfffe13d730531e'}
UPSTREAM_TORCH_SOURCE = r'''
def _pool_uses_oscar_rotation(kv_pool) -> bool:
    """True for the mixed HP+int2 unified pool that loads per-layer Oscar
    rotation matrices. Non-oscar int2 pools (MHA without mixed HP+int2
    storage) implicitly use the legacy segmented Hadamard rotation.
    """
    return getattr(kv_pool, "_R_k", None) is not None

def oscar_q_rotation(kv_pool, layer_idx: int) -> torch.Tensor:
    """Matrix the query is rotated with for layer ``layer_idx``: the key
    rotation itself for an orthogonal key transform, ``R_k^{-T}`` (shipped as
    ``q_rotation``) for a non-orthogonal one."""
    Q = getattr(kv_pool, "_Q_k", None)
    return (Q if Q is not None else kv_pool._R_k)[layer_idx]

def oscar_o_rotation(kv_pool, layer_idx: int) -> torch.Tensor:
    """Matrix ``O`` the attention output is un-rotated with (``o @ O.T``):
    the value rotation for an orthogonal value transform, ``R_v^{-T}``
    (shipped as ``o_rotation``) for a non-orthogonal one."""
    O = getattr(kv_pool, "_O_v", None)
    return (O if O is not None else kv_pool._R_v)[layer_idx]

def oscar_k_mean(kv_pool, layer_idx: int) -> Optional[torch.Tensor]:
    """Per-head key mean subtracted before the key rotation, or ``None``."""
    mean = getattr(kv_pool, "_k_mean", None)
    return None if mean is None else mean[layer_idx]

def _apply_oscar_rotation(
    tensor: torch.Tensor, R: torch.Tensor, kv_group_num: int = 1
) -> torch.Tensor:
    """Apply the Oscar rotation along the last dim; returns contiguous ``R.dtype``.

    ``R`` is ``[hd, hd]`` (V1 shared) or ``[kv_heads, hd, hd]`` (V2 per-head).
    In the per-head case ``tensor`` must be ``[tokens, heads, hd]``; with GQA the
    query tensor has ``kv_group_num`` query heads per KV head, so its rotations
    are repeat-interleaved to line up.
    """
    t = tensor.to(R.dtype)
    if R.dim() == 2:
        return (t @ R).contiguous()
    Rh = R if kv_group_num == 1 else R.repeat_interleave(kv_group_num, dim=0)
    return torch.einsum("thd,hde->the", t, Rh).contiguous()

def prepare_quantized_extend_qkv(
    kv_pool,
    layer: "RadixAttention",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_already_hadamard_transformed: bool = False,
    kv_already_hadamard_transformed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Apply the pool's active rotation to Q/K/V for int2 extend.

    Returns the (possibly) rotated ``q, k, v`` tensors and a flag indicating
    whether the attention output must be inverse-rotated afterwards. For
    non-int2 pools this is a no-op.
    """
    need_v_inverse = False
    kv_dtype = kv_pool.dtype
    if kv_dtype != "int2":
        return q, k, v, need_v_inverse

    if _pool_uses_oscar_rotation(kv_pool):
        layer_idx = layer.layer_id - kv_pool.start_layer
        R_k = kv_pool._R_k[layer_idx]
        R_v = kv_pool._R_v[layer_idx]
        Q_k = oscar_q_rotation(kv_pool, layer_idx)
        k_mean = oscar_k_mean(kv_pool, layer_idx)
        v_rotation_absorbed = bool(
            getattr(layer, "oscar_v_rotation_absorbed", False)
        )
        kv_group_num = (
            q.shape[-2] // k.shape[-2] if R_k.dim() == 3 and k.dim() >= 2 else 1
        )
        if not q_already_hadamard_transformed:
            q = _apply_oscar_rotation(q, Q_k, kv_group_num)
        if not kv_already_hadamard_transformed:
            if k_mean is not None:
                # Centering: the same mean the pool subtracts from every stored
                # key, so every logit of the request shifts by one per-head
                # constant and the softmax is unchanged.
                k = k - k_mean.to(k.dtype)
            k = _apply_oscar_rotation(k, R_k)
            if v_rotation_absorbed:
                v = v.to(R_v.dtype).contiguous()
            else:
                v = _apply_oscar_rotation(v, R_v)
        need_v_inverse = True
        return q, k, v, need_v_inverse

    if not q_already_hadamard_transformed:
        q = _apply_segmented_hadamard_transform(q)
    if not kv_already_hadamard_transformed:
        k = _apply_segmented_hadamard_transform(k)
        v = _apply_segmented_hadamard_transform(v)
    need_v_inverse = True
    return q, k, v, need_v_inverse

def apply_inverse_v_rotation(
    result: torch.Tensor,
    kv_pool,
    layer: "RadixAttention",
    need_v_inverse: bool,
) -> torch.Tensor:
    """Apply the inverse V rotation on an attention output tensor, when
    required. Hadamard is self-inverse (segmented FWHT), Oscar inverts as
    ``result @ R_v.T`` in ``R_v``'s dtype.

    ``result`` must have shape ``[..., v_head_dim]``; callers should reshape
    beforehand if their output is stored flattened.
    """
    if not need_v_inverse or kv_pool.dtype != "int2":
        return result
    if _pool_uses_oscar_rotation(kv_pool):
        layer_idx = layer.layer_id - kv_pool.start_layer
        R_v = oscar_o_rotation(kv_pool, layer_idx)
        if R_v.dim() == 2:
            return (result.to(R_v.dtype) @ R_v.T).contiguous()
        q_heads = result.shape[-2]
        Rh = R_v.repeat_interleave(max(1, q_heads // R_v.shape[0]), dim=0)
        return torch.einsum(
            "thd,hed->the", result.to(R_v.dtype), Rh
        ).contiguous()
    return _apply_segmented_hadamard_transform(result)

def _groupwise_affine_quantize(
    values: torch.Tensor,
    group_size: int,
    max_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, head_dim = values.shape
    num_groups = head_dim // group_size
    grouped = values.to(torch.float32).reshape(
        num_tokens, num_heads, num_groups, group_size
    )
    val_min = grouped.amin(dim=-1)
    val_max = grouped.amax(dim=-1)
    scale = (val_max - val_min).clamp_min(1e-8) / float(max_q)
    zero = -val_min / scale
    q = torch.clamp(
        grouped / scale.unsqueeze(-1) + zero.unsqueeze(-1) + 0.5,
        0,
        max_q,
    ).to(torch.uint8)
    return q.reshape(num_tokens, num_heads, head_dim), scale, zero

def _clip_index(clip_ratio: float, head_dim: int) -> int:
    """Resolve ``clip_ratio`` -> integer sort index, or ``-1`` to disable.

    The kernels skip the in-kernel ``tl.sort`` when ``CLIP_INDEX < 0``.
    """
    if clip_ratio <= 0.0:
        return -1
    idx = int(clip_ratio * head_dim)
    if idx >= head_dim:
        idx = head_dim - 1
    if idx < 0:
        idx = 0
    return idx
'''
LEGACY_PINS = {'configs/acceptance_aligned_bf16.json': 'abc5db538ad9963d0c41dadc025b963503ed95033169577fafb892850a4fb45e', 'configs/acceptance_bf16.json': '0e137f2233f5d76418e4291cb773f0a501b7a9e3f2d35c1c6d485a51551b9054', 'configs/acceptance.json': 'b044a5f485cd8cfed81dc76c7199717952853167b4e1d8e0aa9bff499b290811', 'tools/aligned_reference.py': 'f6a871ee8b2bbfd539e5e3b7269323cdafdaa68aca7a0c6510de3d809121446e', 'tools/bf16_reference.py': '7b9b71b1bfe6198f41e36962c10adcca1e6863b03b9a9ad391c1533aed0ec859', 'oscar_ascend/ops/reference.py': '8fa1d3a7912073d0e37fd3189a2467ea0d8e4ddeb4dfb7ea1cec3e6add43c20c'}


@pytest.fixture(scope="module")
def upstream():
    tree = ast.parse(UPSTREAM_TORCH_SOURCE)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *tree.body], type_ignores=[])),
                 "pinned-upstream-pure-torch-fixture", "exec"), namespace)
    return namespace


def rotation(repeat=False):
    block = torch.tensor([[.6, -.8], [.8, .6]])
    if repeat:
        return torch.block_diag(*([block] * 128)).bfloat16()
    result = torch.eye(256)
    result[:2, :2] = block
    return result.bfloat16()


def test_v4_contract_is_frozen_and_legacy_oracles_are_unchanged():
    contract = ref.load_acceptance()
    assert contract["slot_bytes_D256"] == 144
    assert contract["upstream_commit"] == ref.UPSTREAM_COMMIT
    assert contract["cache_format"] == "striped_rotated_fp32_v3"
    assert contract["fused_attention"] == {"atol": .005, "rtol": .005, "per_element": True}
    for relative, digest in LEGACY_PINS.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("ratio", [0., .92, .96, 1.])
def test_codec_against_pinned_official_torch_functions(upstream, ratio):
    generator = torch.Generator().manual_seed(20261021)
    rows = torch.randn((5, 2, 256), generator=generator).bfloat16()
    index = upstream["_clip_index"](ratio, 256)
    values = rows.float()
    if index >= 0:
        threshold = values.abs().sort(-1).values[..., index, None]
        values = torch.clamp(values, -threshold, threshold)
    expected = upstream["_groupwise_affine_quantize"](values, 256, 3)
    actual = ref.quantize_rows(rows, ratio)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want.squeeze(-1) if got.ndim != want.ndim else want)
    assert actual[1].dtype == actual[2].dtype == torch.float32


def test_half_ties_are_not_nearest_even_and_constant_scale_is_nonzero():
    row = torch.zeros((1, 256), dtype=torch.bfloat16)
    row[0, :5] = torch.tensor([0, .5, 1.5, 2.5, 3])
    codes, scale, zero = ref.quantize_rows(row)
    assert codes[0, :5].tolist() == [0, 1, 2, 3, 3]
    assert scale.item() == 1 and zero.item() == 0
    _, scale, zero = ref.quantize_rows(torch.zeros_like(row))
    assert scale.item() == torch.tensor(1e-8, dtype=torch.float32).div(3).item()
    assert zero.item() == 0


@pytest.mark.parametrize("mode", [0, 1])
def test_striped_code_layout_and_fp32_metadata(mode):
    generator = torch.Generator().manual_seed(91)
    key = torch.randn((7, 2, 256), generator=generator).bfloat16()
    value = torch.randn((7, 2, 256), generator=generator).bfloat16()
    packed = ref.pack_kv(key, value)
    assert packed.shape == (7, 2, 144)
    metadata = packed[..., 128:].contiguous().view(torch.float32)
    decoded = ref.unpack_kv(packed, 256, mode)
    for side, rows, ratio in [(0, key, .96), (1, value, .92)]:
        codes, scale, zero = ref.quantize_rows(rows, ratio)
        assert torch.equal(metadata[..., 2 * side], scale)
        assert torch.equal(metadata[..., 2 * side + 1], zero)
        assert torch.equal(decoded[side], ref.dequantize_rows(codes, scale, zero, mode))
        for word in (0, 31):
            expected = sum(codes[..., word + 32 * plane].long() << (2 * plane) for plane in range(8))
            offset = 64 * side + 2 * word
            actual = packed[..., offset].long() | packed[..., offset + 1].long() << 8
            assert torch.equal(actual, expected)


def test_prefill_decode_cast_orders_remain_distinct():
    code = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
    scale, zero = torch.tensor([.12345]), torch.tensor([2.3456])
    expected_prefill = ((code.float() - zero[:, None]) * scale[:, None]).bfloat16().float()
    expected_decode = ((code.bfloat16() - zero.bfloat16()[:, None]) * scale.bfloat16()[:, None]).float()
    assert torch.equal(ref.dequantize_rows(code, scale, zero, 0), expected_prefill)
    assert torch.equal(ref.dequantize_rows(code, scale, zero, 1), expected_decode)
    assert not torch.equal(expected_prefill, expected_decode)


@pytest.mark.parametrize("mode", [0, 1])
def test_equal_true_keys_no_longer_receive_mixed_domain_source_weights(mode):
    q = torch.ones((1, 1, 256), dtype=torch.bfloat16)
    rk = rotation(repeat=True)
    qr, kr = ref.rotate_bf16(q, rk), ref.rotate_bf16(q, rk)
    positive, negative = torch.ones_like(q), -torch.ones_like(q)
    history_k, history_v = ref.unpack_kv(ref.pack_kv(kr, positive), 256, mode)
    assert torch.equal(history_k, kr.float()) and torch.equal(history_v, positive.float())
    history = ref.attention_tiles(qr, history_k, history_v)
    current = ref.attention_tiles(qr, kr, negative)
    assert torch.equal(history.lse, current.lse)
    actual = ref.merge_and_inverse([history, current], torch.eye(256, dtype=torch.bfloat16))
    # Analytic golden: identical keys with +1/-1 values have equal weights.
    assert torch.equal(actual.output, torch.zeros_like(q))
    raw_wrong_domain = ref.attention_tiles(q, q, negative)
    wrong = ref.merge_and_inverse([history, raw_wrong_domain], torch.eye(256, dtype=torch.bfloat16))
    assert wrong.output[0, 0, 0].item() == .06884765625


@pytest.mark.parametrize("mode", [0, 1])
def test_three_plus_two_split_keeps_unrounded_partial_until_final_merge(mode):
    q = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    k = torch.zeros((5, 1, 256), dtype=torch.bfloat16)
    v = torch.ones_like(k)
    v[2].fill_(1.015625)
    dk, dv = ref.unpack_kv(ref.pack_kv(k, v), 256, mode)
    assert torch.equal(dv, v.float())
    parts = [ref.attention_tiles(q, dk[a:b], dv[a:b]) for a, b in [(0, 3), (3, 5)]]
    assert parts[0].output.dtype == parts[1].output.dtype == torch.float32
    assert parts[0].output[0, 0, 0].item() == torch.tensor((1 + 1 + 1.015625) / 3).item()
    result = ref.merge_and_inverse(parts, torch.eye(256, dtype=torch.bfloat16))
    # Analytic golden: round_BF16((4 + 1.015625) / 5) = 1, without a split cast.
    assert torch.equal(result.output, torch.ones_like(result.output))
    assert torch.equal(result.output, ref.merge_and_inverse(
        [ref.attention_tiles(q, dk, dv)], torch.eye(256, dtype=torch.bfloat16)).output)
    wrong_parts = [ref.AttentionResult(p.output.bfloat16().float(), p.lse) for p in parts]
    wrong = ref.merge_and_inverse(wrong_parts, torch.eye(256, dtype=torch.bfloat16))
    assert wrong.output[0, 0, 0].item() == 1.0078125


def test_exact_only_path_uses_official_rotations_even_without_int2(upstream):
    q = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    k, v = q.clone(), q.clone()
    v[..., :2] = 1
    r = rotation()
    pool = SimpleNamespace(dtype="int2", start_layer=0, _R_k=r[None], _R_v=r[None])
    layer = SimpleNamespace(layer_id=0, oscar_v_rotation_absorbed=False)
    oq, ok, ov, inverse = upstream["prepare_quantized_extend_qkv"](pool, layer, q, k, v)
    expected = upstream["apply_inverse_v_rotation"](ov, pool, layer, inverse)
    actual = ref.merge_and_inverse([ref.attention_tiles(ref.rotate_bf16(q, r),
        ref.rotate_bf16(k, r), ref.rotate_bf16(v, r))], r)
    assert torch.equal(actual.output, expected)
    assert expected.flatten()[:2].tolist() == [1.0078125, 1.0078125]
    assert not torch.equal(actual.output, v)


def test_final_inverse_matches_extracted_official_function(upstream):
    value = torch.zeros((1, 1, 256), dtype=torch.float32)
    value[..., :2] = torch.tensor([1.0052083730697632, -.337])
    r = rotation()
    pool = SimpleNamespace(dtype="int2", start_layer=0, _R_k=r[None], _R_v=r[None])
    expected = upstream["apply_inverse_v_rotation"](
        value.bfloat16(), pool, SimpleNamespace(layer_id=0), True)
    got = ref.finalize_output(ref.AttentionResult(value, torch.zeros((1, 1))), r)
    assert torch.equal(got.output, expected)


def test_native_current_bridge_keeps_the_rotated_partial_and_final_inverse_external(upstream):
    original = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    original[..., :2] = 1
    r = rotation()
    rotated = upstream["_apply_oscar_rotation"](original, r)
    part = ref.native_current_part(rotated, torch.zeros((1, 1)))
    assert part.output.dtype == torch.float32 and torch.equal(part.output, rotated.float())
    actual = ref.merge_and_inverse([part], r)
    assert actual.output.flatten()[:2].tolist() == [1.0078125, 1.0078125]
    with pytest.raises(ValueError, match="BF16"):
        ref.native_current_part(rotated.float(), part.lse)


def test_gqa_uses_shared_v1_rotation_without_cross_head_mixing(upstream):
    generator = torch.Generator().manual_seed(12)
    q = torch.randint(-4, 5, (2, 12, 256), generator=generator).bfloat16()
    k = torch.randint(-4, 5, (5, 2, 256), generator=generator).bfloat16()
    v = torch.randint(-4, 5, (5, 2, 256), generator=generator).bfloat16()
    r = rotation()
    qr, kr, vr = [ref.rotate_bf16(x, r) for x in (q, k, v)]
    assert torch.equal(qr, upstream["_apply_oscar_rotation"](q, r))
    got = ref.attention_tiles(qr, kr, vr)
    for head in range(12):
        expected = ref.attention_tiles(qr[:, head:head+1], kr[:, head//6:head//6+1], vr[:, head//6:head//6+1])
        assert torch.equal(got.output[:, head:head+1], expected.output)
        assert torch.equal(got.lse[:, head:head+1], expected.lse)


@pytest.mark.parametrize("empty", [False, True])
def test_empty_or_fully_masked_rows_publish_exact_padding(empty):
    q = torch.ones((2, 1, 256), dtype=torch.bfloat16)
    k = torch.ones((0 if empty else 3, 1, 256), dtype=torch.bfloat16)
    mask = torch.zeros((2, k.shape[0]), dtype=torch.bool)
    part = ref.attention_tiles(q, k, k, visible=mask)
    result = ref.merge_and_inverse([part], rotation())
    assert torch.equal(result.output, torch.zeros_like(q))
    assert torch.isneginf(result.lse).all()
    assert ref.exact_protocol({"output": result.output, "lse": result.lse},
                             {"output": torch.zeros_like(q), "lse": torch.full((2, 1), -math.inf)})


def test_exact_status_metadata_and_padding_do_not_use_numeric_tolerance():
    expected = {"status": torch.zeros(2, dtype=torch.int32), "metadata": torch.tensor([3, 128]),
                "padding": torch.zeros((1, 256), dtype=torch.bfloat16)}
    actual = {name: value.clone() for name, value in expected.items()}
    assert ref.exact_protocol(actual, expected)
    actual["padding"][0, 0] = .0001
    assert not ref.exact_protocol(actual, expected)
    actual = {name: value.clone() for name, value in expected.items()}
    actual["status"][0] = 1
    assert not ref.exact_protocol(actual, expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_rows_and_metadata_fail_even_if_attention_mask_is_empty(bad):
    q = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    k = q.clone()
    k[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        ref.attention_tiles(q, k, q, visible=torch.zeros((1, 1), dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        ref.dequantize_rows(torch.zeros((1, 256), dtype=torch.uint8), torch.tensor([1.]), torch.tensor([bad]), 0)


def test_v3_style_per_partial_inverse_argument_and_bf16_partials_are_rejected():
    q = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        ref.attention_tiles(q, q, q, value_rotation=torch.eye(256))
    with pytest.raises(ValueError, match="FP32"):
        ref.merge_parts([ref.AttentionResult(q, torch.zeros((1, 1)))])
    with pytest.raises(ValueError, match="V1"):
        ref.rotate_bf16(q, torch.eye(256)[None])


def test_tolerance_is_per_element_and_lse_special_values_are_strict():
    expected = torch.zeros(3)
    assert ref.error_metrics(torch.tensor([.004, 0., 0.]), expected)["passed"]
    assert not ref.error_metrics(torch.tensor([.006, 0., 0.]), expected)["passed"]
    assert not ref.error_metrics(torch.tensor([0.]), torch.tensor([-math.inf]))["passed"]


def test_multiple_online_tiles_and_initially_empty_query_rows():
    q = torch.zeros((3, 12, 256), dtype=torch.bfloat16)
    k = torch.zeros((300, 2, 256), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    v[:, 0] = (torch.arange(300) % 9)[:, None]
    v[:, 1] = (torch.arange(300) % 7 - 3)[:, None]
    visible = torch.zeros((3, 300), dtype=torch.bool)
    visible[0, 256:] = True
    visible[1, :257] = True
    visible[2] = True
    result = ref.attention_tiles(q, k, v, visible=visible)
    for row in range(3):
        for head in range(12):
            expected = v[visible[row], head // 6].float().mean(0)
            torch.testing.assert_close(result.output[row, head], expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(result.lse[row], torch.full((12,), math.log(int(visible[row].sum()))),
                                   atol=1e-6, rtol=0)


def test_empty_partial_poison_and_signed_padding_zero_are_protocol_errors():
    with pytest.raises(ValueError, match="empty partial"):
        ref.merge_parts([ref.AttentionResult(torch.ones((1, 1, 256)), torch.full((1, 1), -math.inf))])
    assert not ref.exact_protocol({"padding": torch.tensor([-0.], dtype=torch.bfloat16)},
                                  {"padding": torch.tensor([0.], dtype=torch.bfloat16)})


def test_mixed_history_cannot_hide_a_native_bf16_current_partial_cast():
    output = torch.ones((1, 1, 256), dtype=torch.bfloat16)
    native = ref.native_current_part(output, torch.zeros((1, 1)))
    history = ref.AttentionResult(-output.float(), torch.zeros((1, 1)))
    with pytest.raises(ValueError, match="single zero-history source"):
        ref.merge_and_inverse([history, native], rotation())


def fixed_map_fixture():
    context, cut, sink, recent, speculative = 70, 68, 64, 3, 3
    k = torch.zeros((128, 1, 256), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    v[68, 0, 0] = 1  # Quantized form clips this isolated coordinate to zero.
    q = torch.zeros((2, 1, 256), dtype=torch.bfloat16)
    ring = recent + speculative
    wk = torch.zeros((1, sink + ring, 1, 256), dtype=torch.bfloat16)
    wv = torch.zeros_like(wk)
    tags = torch.full((1, sink + ring), -1, dtype=torch.int64)
    positions = torch.cat((torch.arange(sink), torch.arange(cut, context)))
    rows = torch.where(positions < sink, positions, sink + positions % ring)
    wk[0, rows], wv[0, rows], tags[0, rows] = k[positions], v[positions], positions
    fixture = {"cpu": {"q": q, "qr": q.clone(), "ck": q.clone(), "cv": q.clone(),
                       "rv": rotation(), "raw": ref.pack_kv(k, v).flatten(),
                       "table": torch.tensor([[0]], dtype=torch.int32),
                       "wk": wk, "wv": wv, "tags": tags, "slots": torch.tensor([70, 71])},
               "spec": SimpleNamespace(kv_heads=1, splits=1, recent_tokens=recent),
               "cached_quant_ends": [cut], "window_ring_tokens": ring,
               "stride": 128 * 144, "blocks": 1, "scale": 256 ** -.5,
               "cache_format": ref.CACHE_FORMAT}
    tasks = torch.zeros((6, 16), dtype=torch.int64)
    tasks[0, :11] = torch.tensor([0, 2, 0, sink, cut, 0, 0, 0, context, 0, 0])
    tasks[1, :11] = torch.tensor([0, 2, 0, 0, sink + context - cut, 0, 1, 1, context, 0, 0])
    tasks[2, :11] = torch.tensor([0, 2, 0, context, context + 2, 0, 2, 2, context, 0, 0])
    return fixture, tasks


@pytest.mark.parametrize("mode", [0, 1])
def test_fixture_uses_one_explicit_prefix_tier_map_for_all_query_rows(upstream, mode):
    fixture, tasks = fixed_map_fixture()
    result = ref.fixture_tasks_reference(fixture, tasks, mode)
    # Equal logits: old prefix position 68 remains exact for BOTH queries.
    # The old query-dependent cut would demote it for query position 71.
    expected_merged = torch.zeros((2, 1, 256))
    expected_merged[0, 0, 0], expected_merged[1, 0, 0] = 1 / 71, 1 / 72
    torch.testing.assert_close(result["merged_rotated"], expected_merged, atol=2e-9, rtol=0)
    assert result["partial"].dtype == torch.float32
    assert result["pre_inverse_bf16"].dtype == result["output"].dtype == torch.bfloat16
    assert result["live_output_rows"] == 6
    pool = SimpleNamespace(dtype="int2", start_layer=0, _R_k=rotation()[None], _R_v=rotation()[None])
    expected = upstream["apply_inverse_v_rotation"](
        expected_merged.bfloat16(), pool, SimpleNamespace(layer_id=0), True)
    assert torch.equal(result["output"], expected)
    assert torch.equal(result["final_output"], expected)
    assert not torch.equal(result["output"].float(), result["merged_rotated"])


@pytest.mark.parametrize("case", ["missing_cut", "illegal_cut", "old_cache", "raw_query", "tag", "duplicate", "missing_owner", "error_task"])
def test_fixture_protocol_violations_cannot_be_relabelled_as_numeric_success(case):
    fixture, tasks = fixed_map_fixture()
    if case == "missing_cut":
        del fixture["cached_quant_ends"]
    elif case == "illegal_cut":
        fixture["cached_quant_ends"] = [71]
    elif case == "old_cache":
        fixture["cache_format"] = "striped_fp32_v2"
    elif case == "raw_query":
        fixture["cpu"]["q"][0, 0, 0] = 1
    elif case == "tag":
        fixture["cpu"]["tags"][0, 0] = 99
    elif case == "duplicate":
        tasks[3] = tasks[0]
    elif case == "missing_owner":
        tasks[2, 1] = 0
    else:
        tasks[0, 10] = 1
    with pytest.raises(ValueError):
        ref.fixture_tasks_reference(fixture, tasks, 0)
