#!/usr/bin/env python3
"""Real-NPU numerical gate for the standalone OSCAR AscendC operator."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--library",
        default=str(Path(__file__).resolve().parents[1]
                    / "build/ascendc/liboscar_ascend_torch.so"),
    )
    ap.add_argument("--long", action="store_true",
                    help="exercise only the long-context Cube tiling key")
    ap.add_argument("--long-length", type=int, default=16384)
    args = ap.parse_args()

    # op_api_common snapshots this path when the binding shared library loads.
    if not os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        raise RuntimeError("ASCEND_CUSTOM_OPP_PATH must be set before the probe")

    import torch
    import torch_npu  # noqa: F401

    from oscar_ascend.kernels.ascendc_attention import oscar_ascendc_attention
    from oscar_ascend.kernels.paged_attention import oscar_paged_attention_ref
    from oscar_ascend.kernels.store_kernel import oscar_store_ref

    library = Path(args.library)
    if not library.is_file():
        raise RuntimeError(f"Torch binding does not exist: {library}")
    torch.ops.load_library(str(library))
    if not hasattr(torch.ops.oscar_ascend, "int2_paged_attention"):
        raise RuntimeError("OSCAR Torch operator was not registered")
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")

    torch.manual_seed(20260914)
    device = torch.device("npu")
    d, hk, hq, bs = 256, 1, 8, 128
    scale = d ** -0.5
    cases = [(0, 1), (17, 4), (129, 4), (252, 4)]
    if args.long:
        if args.long_length <= 256:
            raise ValueError("--long-length must be greater than 256")
        # Long validation is deliberately independent of the scalar/reference
        # tiling key.  Production dispatch uses this operator for grouped MTP;
        # a short-path regression must not prevent us from observing the first
        # real Cube result (and vice versa).
        cases = [(args.long_length - 4, 4)]
    for prefix, q_len in cases:
        blocks = max(1, (prefix + bs - 1) // bs)
        q = torch.randn(q_len, hq, d, dtype=torch.float16) * 0.25
        k_new = torch.randn(q_len, hk, d, dtype=torch.float16) * 0.25
        v_new = torch.randn(q_len, hk, d, dtype=torch.float16) * 0.25
        old_k = torch.randn(prefix, hk, d, dtype=torch.float16) * 0.25
        old_v = torch.randn(prefix, hk, d, dtype=torch.float16) * 0.25
        k_cache = torch.zeros(blocks, bs, hk, d, dtype=torch.int8,
                              device=device)
        v_cache = torch.zeros_like(k_cache)
        if prefix:
            slots = torch.arange(prefix, dtype=torch.int64, device=device)
            oscar_store_ref(old_k.to(device), old_v.to(device),
                            k_cache, v_cache, slots)
        block_table = torch.arange(blocks, dtype=torch.int32).unsqueeze(0)
        q_starts = torch.tensor([0], dtype=torch.int32, device=device)
        q_lens = torch.tensor([q_len], dtype=torch.int32, device=device)
        prefixes = torch.tensor([prefix], dtype=torch.int32, device=device)
        actual = oscar_ascendc_attention(
            q.to(device), k_new.to(device), v_new.to(device),
            k_cache, v_cache, block_table.to(device), q_starts, q_lens,
            prefixes, scale,
            max_seq_len=prefix + q_len,
        )
        torch.npu.synchronize()
        expected = oscar_paged_attention_ref(
            q.float(), k_new.float(), v_new.float(), k_cache.cpu(),
            v_cache.cpu(), block_table, [0, q_len], [prefix + q_len], scale,
        )
        actual_cpu = actual.cpu().float()
        try:
            torch.testing.assert_close(actual_cpu, expected,
                                       atol=2e-2, rtol=2e-2,
                                       equal_nan=False)
        except AssertionError:
            diff = (actual_cpu - expected).abs()
            bad = diff > (2e-2 + 2e-2 * expected.abs())
            flat_bad = bad.flatten().nonzero()
            first = int(flat_bad[0].item()) if flat_bad.numel() else -1
            print(
                "ASCENDC FAIL "
                f"prefix={prefix} q_len={q_len} "
                f"actual_nonzero={(actual_cpu != 0).sum().item()}/"
                f"{actual_cpu.numel()} max_abs_diff={diff.max().item():.6g} "
                f"first_bad_flat={first}",
                flush=True,
            )
            for qi in range(q_len):
                qdiff = diff[qi]
                qbad = bad[qi]
                print(
                    f"  q[{qi}]: bad={qbad.sum().item()}/{qbad.numel()} "
                    f"max={qdiff.max().item():.6g} "
                    f"mean={qdiff.mean().item():.6g} "
                    f"actual_l1={actual_cpu[qi].abs().sum().item():.6g}",
                    flush=True,
                )
            raise
        if not torch.isfinite(actual).all().item():
            raise AssertionError("AscendC attention produced NaN/Inf")
        print(f"ASCENDC PASS prefix={prefix} q_len={q_len} hq={hq}",
              flush=True)
    print("ASCENDC NUMERICAL PROBE PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
