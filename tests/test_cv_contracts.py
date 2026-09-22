"""Archive G26-G34/#12/#17-20/#34/#36/#53-69/#71/#92: CV acceptance.

CPU assertions enforce published ABI/workspace and genuine Cube structure.
Opt-in NPU cases execute the compiled operators against the independent PR
oracle, including GQA6 MTP4 and precise/history boundaries.
D.4: no full-history allocation in production; only this test oracle may
materialize history. Tests do not turn CPU results into NPU/performance evidence.
"""
import json
import math
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_cv_uses_cube_for_both_products_and_keeps_hf32_disabled():
    source = (ROOT / "csrc/kernels/attention_cv.cpp").read_text()
    assert "matmul::MatmulImpl" in source
    assert "KERNEL_TYPE_MIX_AIC_1_2" in source
    assert "Matmul(qOffset,kOffset,scoreOffset" in source
    assert "Matmul(pOffset,vOffset,pvOffset" in source
    assert "Matmul(qOffset,0,rotOffset" in source
    assert "SetHF32Mode(false)" in source
    assert "SetHF32Mode(true)" not in source
    assert "Gather(words,packed" in source
    assert "ShiftRight(planes" in source
    assert "DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>" in source


def test_mtp_target_rows_share_one_tile_and_workspace_is_bounded():
    source = (ROOT / "csrc/kernels/attention_cv.cpp").read_text()
    assert "kQueryRows=64" in source
    assert 4 * 6 <= 64  # Qwen3.5 TP4: Hq6/Hkv1; all verify queries in one tile.
    header = (ROOT / "csrc/include/oscar_attention_launch.h").read_text()
    assert "return (256 * dim + 4096) * 4" in header
    assert (256 * 256 + 4096) * 4 == 278528
    # Full-history length is deliberately absent from this exact allocation.
    assert "attention_workspace_per_core(int64_t dim)" in header


def test_cv_metadata_and_outputs_have_distinct_schema_aliases():
    source = (ROOT / "csrc/attention_bindings.cpp").read_text()
    assert "Tensor(a!) tasks, Tensor(b!) positions" in source
    assert "Tensor(a!) partial, Tensor(b!) lse, Tensor(c!) status" in source
    assert "Tensor(d!) workspace" in source
    assert 'status.size(1)==2' in source
    assert 'tasks.size(1)==16' in source


def _npu_ops():
    if os.environ.get("OSCAR_RUN_NPU_TESTS") != "1":
        pytest.skip("real NPU test is opt-in; CPU is not substituted")
    target = json.loads(Path(os.environ.get("OSCAR_TARGET_CONFIG",ROOT / "configs/target.json")).read_text())
    if not target["devices"]:
        pytest.fail("current-task devices must be explicitly selected before NPU tests")
    selected = ",".join(str(x) for x in target["devices"])
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == selected
    import torch_npu  # noqa: F401
    from oscar_ascend.ops.loader import load_extension
    load_extension()
    if not torch.npu.is_available():
        pytest.fail("NPU requested but unavailable; cannot substitute CPU")
    return torch.ops.oscar_ascend_ops


def _case(dim, qlen, context, hk=1):
    from oscar_ascend.ops.reference import attention, encode_kv, decode_kv
    generator = torch.Generator().manual_seed(47 + dim + context)
    hq, sink, recent, speculative = 6 * hk, 4, 32, 3
    block_tokens, blocks, splits, cores = 512, 2, 1, 2
    q = torch.randn(qlen, hq, dim, generator=generator).to(torch.bfloat16)
    oldk = torch.randn(context, hk, dim, generator=generator).to(torch.bfloat16)
    oldv = torch.randn(context, hk, dim, generator=generator).to(torch.bfloat16)
    currentk = torch.randn(qlen, hk, dim, generator=generator).to(torch.bfloat16)
    currentv = torch.randn(qlen, hk, dim, generator=generator).to(torch.bfloat16)
    rotation_generator = torch.Generator().manual_seed(1900 + dim)
    rk = torch.linalg.qr(torch.randn(dim, dim, generator=rotation_generator)).Q.contiguous()
    rv = torch.linalg.qr(torch.randn(dim, dim, generator=rotation_generator)).Q.contiguous()
    if context:
        packed = encode_kv(oldk.float() @ rk, oldv.float() @ rv)
        restoredk, restoredv = decode_kv(packed, dim)
    else:
        packed = torch.empty(0, hk, dim // 2 + 8, dtype=torch.uint8)
        restoredk, restoredv = oldk.float(), oldv.float()
    restoredk = restoredk @ rk.T
    restoredv = restoredv @ rv.T
    slot_bytes = dim // 2 + 8
    stride, prefix = block_tokens * slot_bytes * hk, 64
    raw = torch.full((prefix + blocks * stride,), 0xA5, dtype=torch.uint8)
    rows = sink + recent + speculative
    windowk = torch.full((blocks, rows, hk, dim), float("nan"), dtype=torch.bfloat16)
    windowv = windowk.clone()
    tags = torch.full((blocks, rows), -1, dtype=torch.int64)
    # Physically permuted pages, native virtual128 table (page 0 -> physical 1).
    table = torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3]], dtype=torch.int32)
    for p in range(context):
        physical, inpage = 1 - p // block_tokens, p % block_tokens
        address = prefix + physical * stride + inpage * slot_bytes * hk
        raw[address:address + slot_bytes * hk] = packed[p].flatten()
        row = p if p < sink else sink + inpage % (recent + speculative)
        windowk[physical, row] = oldk[p]
        windowv[physical, row] = oldv[p]
        tags[physical, row] = inpage
    expected = []
    expected_lse = []
    for t in range(qlen):
        old_position = torch.arange(context)
        history = (old_position >= sink) & (old_position < context + t + 1 - recent)
        selected_k = torch.where(history[:, None, None], restoredk, oldk.float())
        selected_v = torch.where(history[:, None, None], restoredv, oldv.float())
        keys = torch.cat((selected_k, currentk[:t + 1].float()))
        values = torch.cat((selected_v, currentv[:t + 1].float()))
        result = attention(q[t:t + 1], keys, values)
        expected.append(result.output)
        expected_lse.append(result.lse)
    args = dict(q=q, qr=q.float() @ rk, ck=currentk, cv=currentv, rv=rv,
                raw=raw, table=table, wk=windowk, wv=windowv, tags=tags,
                starts=torch.tensor([0, qlen], dtype=torch.int32),
                lens=torch.tensor([context + qlen], dtype=torch.int32),
                slots=torch.arange(context, context + qlen, dtype=torch.int64),
                dim=dim, qlen=qlen, hq=hq, hk=hk, sink=sink, recent=recent,
                speculative=speculative, blocks=blocks, block_tokens=block_tokens,
                splits=splits, cores=cores, prefix=prefix, stride=stride)
    return args, torch.cat(expected), torch.cat(expected_lse)


@pytest.mark.parametrize("dim,qlen,context", [(64, 1, 17), (64, 4, 65),
    (128, 4, 129), (256, 1, 401), (256, 4, 511), (256, 4, 0)])
def test_npu_cv_matches_independent_dense_pr_oracle(dim, qlen, context):
    ops = _npu_ops()
    data, expected, expected_lse = _case(dim, qlen, context)
    tensors = {key: value.npu() for key, value in data.items() if isinstance(value, torch.Tensor)}
    tasks = torch.empty((qlen * 3, 16), dtype=torch.int64, device="npu")
    positions = torch.empty(qlen, dtype=torch.int64, device="npu")
    partial = torch.full((qlen, 6, 3, dim), float("nan"), device="npu")
    lse = torch.full((qlen, 6, 3), float("nan"), device="npu")
    status = torch.full((qlen * 3, 2), -99, dtype=torch.int32, device="npu")
    workspace = torch.empty(2 * (256 * dim + 4096) * 4, dtype=torch.uint8, device="npu")
    ops.prepare_attention_tasks_out(tensors["starts"], tensors["lens"], tensors["slots"],
                                   tasks, positions, 6, 1, 4, 32, 1)
    ops.attention_cv_out(tensors["q"], tensors["qr"], tensors["ck"], tensors["cv"],
        tensors["rv"], tensors["raw"], tensors["table"], tensors["wk"], tensors["wv"],
        tensors["tags"], tasks, partial, lse, status, workspace, 512, 2,
        data["prefix"], data["stride"], 4, 32, 3, 1, dim ** -0.5, 2)
    torch.npu.synchronize()
    assert torch.count_nonzero(status).item() == 0
    torch.testing.assert_close(positions.cpu(), torch.arange(context, context + qlen))
    got_lse = torch.logsumexp(lse.float(), dim=-1)
    got = (partial * torch.exp(lse - got_lse[..., None])[..., None]).sum(dim=2)
    tolerance = json.loads((ROOT / "configs/acceptance.json").read_text())["fused_attention"]
    torch.testing.assert_close(got.cpu(), expected, **tolerance)
    torch.testing.assert_close(got_lse.cpu(), expected_lse, **tolerance)
    # Archive G30/#12: every segment, including empty history, wrote finite output.
    assert torch.isfinite(partial).all().item()
    assert not torch.isnan(lse).any().item()


def export_cpu_debug_goldens(directory):
    """Offline fixture writer; values come from independent dense PR oracle."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases=[]
    for dim, qlen, context, hk in [(64, 1, 17, 1), (64, 4, 65, 1), (64, 4, 0, 1),
                                  (128, 4, 129, 1), (256, 4, 511, 1),
                                  (64, 17, 65, 1), (64, 4, 65, 2),
                                  (64, 2, 320, 1), (128, 3, 511, 1)]:
        data, output, lse = _case(dim, qlen, context, hk)
        case = directory / (f"d{dim}_q{qlen}_c{context}" + (f"_hk{hk}" if hk!=1 else ""))
        case.mkdir(exist_ok=True)
        for key, value in {**data, "expected_output": output, "expected_lse": lse}.items():
            if isinstance(value, torch.Tensor):
                (case / f"{key}.bin").write_bytes(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))
        (case / "shape.txt").write_text(
            f"{qlen} {hk*6} {hk} {dim} {context} 4 32 3 512 2 64 {512*(dim//2+8)*hk} 2\n")
        cases.append({"op":"attention_cv","path":str(case)})
    for dim, qlen, context, splits in [(64, 4, 65, 2), (64, 4, 65, 20),
                                      (256, 4, 511, 20), (64, 4, 0, 20)]:
        data, output, lse = _case(dim, qlen, context)
        case = directory / f"d{dim}_q{qlen}_c{context}_s{splits}"
        case.mkdir(exist_ok=True)
        for key, value in {**data, "expected_output": output, "expected_lse": lse}.items():
            if isinstance(value, torch.Tensor):
                (case / f"{key}.bin").write_bytes(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))
        # Requests then split count extend the shape format; old cases default S=1.
        (case / "shape.txt").write_text(
            f"{qlen} 6 1 {dim} {context} 4 32 3 512 2 64 {512*(dim//2+8)} 20 1 {splits}\n")
        cases.append({"op": "attention_cv", "path": str(case)})
    first, first_output, first_lse = _case(64, 1, 17)
    second, second_output, second_lse = _case(64, 4, 65)
    mixed = {}
    for key in ("q", "qr", "ck", "cv", "wk", "wv", "tags"):
        mixed[key] = torch.cat((first[key], second[key]))
    mixed["rv"] = first["rv"]
    mixed["raw"] = torch.cat((first["raw"], second["raw"][64:]))
    mixed["table"] = torch.cat((first["table"], second["table"] + 8))
    mixed["starts"] = torch.tensor([0, 1, 5], dtype=torch.int32)
    mixed["lens"] = torch.tensor([18, 69], dtype=torch.int32)
    mixed["slots"] = torch.tensor([17, 65, 66, 67, 68], dtype=torch.int64)
    mixed["expected_positions"] = mixed["slots"]
    mixed["expected_output"] = torch.cat((first_output, second_output))
    mixed["expected_lse"] = torch.cat((first_lse, second_lse))
    case = directory / "d64_mixed_q1q4"
    case.mkdir(exist_ok=True)
    for key, value in mixed.items():
        (case / f"{key}.bin").write_bytes(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))
    (case / "shape.txt").write_text("5 6 1 64 0 4 32 3 512 4 64 20480 2 2\n")
    cases.append({"op":"attention_cv","path":str(case)})
    for mode in ("bad_tag","nan_query","bad_meta","nan_value"):
        cases.append({"op":mode,"path":str(directory / "d64_q4_c65")})
    (directory / "cases.json").write_text(json.dumps(cases,indent=2)+"\n")
    return directory


if __name__ == "__main__":
    import sys
    print(export_cpu_debug_goldens(sys.argv[1]))
