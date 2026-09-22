# 档案 G25–G34/#13–22/#53/#69/#98/#111/#118/#122：真设备完成和限定本任务PID的CANN诊断。
"""Real-NPU primitive probe. No CPU execution route for operators under test."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from .phase import atomic_json
from .plog import attach_plog


def probe() -> dict:
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("physical NPU ownership selection is required")
    import torch
    import torch_npu
    from oscar_ascend.ops.loader import require_capabilities
    from oscar_ascend.ops.reference import encode_kv
    require_capabilities({"store_int2_out", "merge_lse_out"})
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    cases = []
    generator = torch.Generator(device="cpu").manual_seed(46774)
    for dim in (64, 128, 256):
        for tokens in (1, 7, 129):
            heads, block, pages = 2, 128, 3
            slot_bytes = dim // 2 + 8
            stride = block * heads * slot_bytes + 256
            offset = pages * 256
            key = torch.randn(tokens, heads, dim, generator=generator)
            value = torch.randn(tokens, heads, dim, generator=generator)
            expected = encode_kv(key, value)
            slots = (torch.arange(tokens, dtype=torch.int64) * 2) % (pages * block)
            if tokens > 1:
                slots[-1] = -1
            raw_expected = torch.full((offset+pages*stride,), 173, dtype=torch.uint8)
            for row, slot in enumerate(slots.tolist()):
                if slot < 0:
                    continue
                physical, token = divmod(slot, block)
                start = offset+physical*stride+token*heads*slot_bytes
                raw_expected[start:start+heads*slot_bytes] = expected[row].flatten()
            raw = torch.full(raw_expected.shape, 173, dtype=torch.uint8, device=device)
            status = torch.full((tokens, heads), -123, dtype=torch.int32, device=device)
            torch.ops.oscar_ascend_ops.store_int2_out(key.to(device), value.to(device), slots.to(device),
                raw, status, block, pages, offset, stride)
            torch.npu.synchronize()
            # Debug-only transfers; production never imports this module.
            torch.testing.assert_close(status.cpu(), torch.zeros_like(status, device="cpu"), rtol=0, atol=0)
            torch.testing.assert_close(raw.cpu(), raw_expected, rtol=0, atol=0)
            cases.append({"op": "store", "dim": dim, "tokens": tokens, "exact_bytes": True,
                          "gdn_guard_untouched": True, "device_completion": "passed"})
        for block in (128, 256):
            for slot_dtype in (torch.int32, torch.int64):
                pages, heads = 3, 1
                stride = block * (dim//2+8) + 256
                offset = pages * 256
                # Distinct nonnegative destinations; duplicate indices would
                # race and are prohibited by the native slot ownership contract.
                slot_values = [-1, *sorted({0, 127, 128, block-1, block, pages*block-1}), pages*block]
                count = len(slot_values)
                key = torch.randn(count, heads, dim, generator=generator)
                value = torch.randn(count, heads, dim, generator=generator)
                expected = encode_kv(key, value)
                raw_expected = torch.full((offset+pages*stride,), 173, dtype=torch.uint8)
                expected_status = torch.zeros((count, heads), dtype=torch.int32)
                for row, slot in enumerate(slot_values):
                    if slot == pages*block:
                        expected_status[row] = 1
                    elif slot >= 0:
                        b, t = divmod(slot, block)
                        start = offset+b*stride+t*(dim//2+8)
                        raw_expected[start:start+dim//2+8] = expected[row].flatten()
                raw = torch.full(raw_expected.shape, 173, dtype=torch.uint8, device=device)
                status = torch.full((count, heads), -123, dtype=torch.int32, device=device)
                torch.ops.oscar_ascend_ops.store_int2_out(key.to(device), value.to(device),
                    torch.tensor(slot_values, dtype=slot_dtype, device=device), raw, status, block, pages, offset, stride)
                torch.npu.synchronize()
                torch.testing.assert_close(status.cpu(), expected_status, atol=0, rtol=0)
                torch.testing.assert_close(raw.cpu(), raw_expected, atol=0, rtol=0)
                cases.append({"op": "store_boundaries", "dim": dim, "block": block,
                              "slot_dtype": str(slot_dtype), "device_completion": "passed"})
        for side in ("key", "value"):
            for fault, error_code in (("constant", 3), ("narrow", 3), ("nan", 2), ("inf", 2), ("-inf", 2)):
                key = torch.randn(1, 1, dim, generator=generator)
                value = torch.randn(1, 1, dim, generator=generator)
                bad = key if side == "key" else value
                if fault == "constant":
                    bad.fill_(1)
                elif fault == "narrow":
                    bad.mul_(1e-9)
                else:
                    bad[0, 0, 0] = float(fault)
                raw = torch.full((128*(dim//2+8),), 173, dtype=torch.uint8, device=device)
                status = torch.full((1, 1), -123, dtype=torch.int32, device=device)
                torch.ops.oscar_ascend_ops.store_int2_out(key.to(device), value.to(device),
                    torch.zeros(1, dtype=torch.int64, device=device), raw, status, 128, 1, 0, raw.numel())
                torch.npu.synchronize()
                torch.testing.assert_close(status.cpu(), torch.full((1, 1), error_code, dtype=torch.int32), rtol=0, atol=0)
                torch.testing.assert_close(raw.cpu(), torch.full(raw.shape, 173, dtype=torch.uint8), rtol=0, atol=0)
                cases.append({"op": "store_domain_error", "dim": dim, "side": side, "fault": fault,
                              "expected_status": error_code, "device_completion": "passed"})
        # Reuse the identical device addresses with changed content. This
        # catches stale scalar DCache reads; it is NOT graph replay evidence.
        block, pages, stride, offset = 256, 2, 256*(dim//2+8)+256, 512
        key_cpu = torch.randn(3, 1, dim, generator=generator)
        value_cpu = torch.randn(3, 1, dim, generator=generator)
        key_device, value_device = key_cpu.to(device), value_cpu.to(device)
        slots_device = torch.empty(3, dtype=torch.int64, device=device)
        raw = torch.empty(offset+pages*stride, dtype=torch.uint8, device=device)
        status = torch.empty((3, 1), dtype=torch.int32, device=device)
        for iteration, slot_values in enumerate(([0, 128, 511], [256, -1, 127])):
            key_device.copy_(key_cpu + iteration)
            value_device.copy_(value_cpu - iteration)
            slots_device.copy_(torch.tensor(slot_values, dtype=torch.int64, device=device))
            raw.fill_(173)
            status.fill_(-123)
            torch.ops.oscar_ascend_ops.store_int2_out(key_device, value_device, slots_device,
                raw, status, block, pages, offset, stride)
            torch.npu.synchronize()
            expected = encode_kv(key_cpu+iteration, value_cpu-iteration)
            raw_expected = torch.full(raw.shape, 173, dtype=torch.uint8)
            for row, slot in enumerate(slot_values):
                if slot >= 0:
                    b, t = divmod(slot, block)
                    start = offset+b*stride+t*(dim//2+8)
                    raw_expected[start:start+dim//2+8] = expected[row].flatten()
            torch.testing.assert_close(raw.cpu(), raw_expected, rtol=0, atol=0)
            torch.testing.assert_close(status.cpu(), torch.zeros((3, 1), dtype=torch.int32), rtol=0, atol=0)
        cases.append({"op": "store_reused_addresses", "dim": dim, "device_completion": "passed", "graph_replay": "not_run"})
        for splits in (1, 3, 128):
            rows = 5
            partial = torch.randn(rows, splits, dim, generator=generator)
            lse_cpu = torch.randn(rows, splits, generator=generator) * 10
            lse_cpu[0] = -torch.inf
            partial[0] = torch.nan
            if splits > 1:
                lse_cpu[1, 0] = -torch.inf
                partial[1, 0] = torch.nan
            weights = torch.softmax(lse_cpu[1:], dim=1)
            safe_partial = torch.where(torch.isneginf(lse_cpu[1:])[..., None], 0, partial[1:])
            expected_out = (safe_partial * weights[..., None]).sum(1)
            expected_lse = torch.logsumexp(lse_cpu[1:], 1)
            output = torch.full((rows, dim), torch.nan, device=device)
            lse = torch.full((rows,), torch.nan, device=device)
            status = torch.full((rows,), -123, dtype=torch.int32, device=device)
            torch.ops.oscar_ascend_ops.merge_lse_out(partial.to(device), lse_cpu.to(device), output, lse, status)
            torch.npu.synchronize()
            torch.testing.assert_close(status.cpu(), torch.zeros(rows, dtype=torch.int32), rtol=0, atol=0)
            torch.testing.assert_close(output[1:].cpu(), expected_out, rtol=0.005, atol=0.005)
            torch.testing.assert_close(lse[1:].cpu(), expected_lse, rtol=0.005, atol=0.005)
            torch.testing.assert_close(output[0].cpu(), torch.zeros(dim), rtol=0, atol=0)
            assert torch.isneginf(lse[0].cpu()), "empty-row LSE must be negative infinity"
            cases.append({"op": "merge", "dim": dim, "splits": splits, "device_completion": "passed"})
        for fault in ("nan_lse", "inf_lse", "nan_output"):
            partial = torch.ones((1, 3, dim), device=device)
            partial_lse = torch.zeros((1, 3), device=device)
            if fault == "nan_lse":
                partial_lse[0, 1] = torch.nan
            elif fault == "inf_lse":
                partial_lse[0, 1] = torch.inf
            else:
                partial[0, 1, 0] = torch.nan
            output = torch.full((1, dim), torch.nan, device=device)
            lse = torch.full((1,), torch.nan, device=device)
            status = torch.full((1,), -123, dtype=torch.int32, device=device)
            torch.ops.oscar_ascend_ops.merge_lse_out(partial, partial_lse, output, lse, status)
            torch.npu.synchronize()
            torch.testing.assert_close(status.cpu(), torch.tensor([2], dtype=torch.int32), rtol=0, atol=0)
            cases.append({"op": "merge_invalid_input", "dim": dim, "fault": fault,
                          "expected_status": 2, "device_completion": "passed"})
        partial_cpu = torch.randn(2, 3, dim, generator=generator)
        partial = partial_cpu.to(device)
        partial_lse = torch.empty((2, 3), device=device)
        output = torch.empty((2, dim), device=device)
        lse, status = torch.empty(2, device=device), torch.empty(2, dtype=torch.int32, device=device)
        for iteration in (0, 1):
            lse_cpu = torch.tensor([[0., -10., 10.], [2., 4., -3.]]) * iteration
            partial_lse.copy_(lse_cpu.to(device))
            output.fill_(torch.nan)
            lse.fill_(torch.nan)
            status.fill_(-123)
            torch.ops.oscar_ascend_ops.merge_lse_out(partial, partial_lse, output, lse, status)
            torch.npu.synchronize()
            expected = (partial_cpu * torch.softmax(lse_cpu, 1)[..., None]).sum(1)
            torch.testing.assert_close(output.cpu(), expected, rtol=0.005, atol=0.005)
            torch.testing.assert_close(lse.cpu(), torch.logsumexp(lse_cpu, 1), rtol=0.005, atol=0.005)
            torch.testing.assert_close(status.cpu(), torch.zeros(2, dtype=torch.int32), rtol=0, atol=0)
        cases.append({"op": "merge_reused_addresses", "dim": dim, "device_completion": "passed", "graph_replay": "not_run"})
    return {"status": "primitive_probe_passed", "device": str(device), "device_name": torch.npu.get_device_name(0),
            "cases": cases, "graph_capture": "not_run", "graph_replay": "not_run",
            "full_service_acceptance": "not_run", "performance": "not_run"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    try:
        report = probe()
        code = 0
    except Exception as exc:
        # Diagnostics never switch the operator execution path.
        import traceback
        traceback.print_exc()
        report = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                  "device_completion": "not_established", "full_service_acceptance": "not_run"}
        code = 1
        attach_plog(report, started_at=started, owned_pids={os.getpid()})
    report["elapsed_seconds"] = time.time()-started
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
