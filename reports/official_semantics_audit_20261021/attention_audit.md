# FULL attention mathematical alignment audit — 2026-10-21

Local production revision inspected: `7786910f`. Official `sglang-rebase` pin:
`38680852e52c10bd6b9e9a64ed9bc815e742859b`; URLs and source hashes are in
`upstream_manifest.json`. All upstream paths below are relative to the
`upstream/` snapshot beside this document. Only report artifacts were added.

**The current local implementation is not numerically equivalent to the
official all-rotated-source pipeline, even with a V1 shared orthogonal FP32
rotation artifact and a perfect INT2 roundtrip.** These are source-backed
mathematical differences. They do not establish how much of the user's
approximately five LongBench points they explain.

## Confirmed differences

| Item | Official pipeline | Local pipeline | Evidence and consequence |
| --- | --- | --- | --- |
| HP sink/recent and current KV coordinate frame | All Q/K/V are rotated; HP stores BF16 rotated K/V. | History uses BF16 Qrot/Krot; sink/recent/current use original-space BF16 Q/K/V. | Upstream `python/sglang/srt/layers/attention/quantized_kv_prefill.py:160-185`, `python/sglang/srt/mem_cache/unified_kv_pool.py:1384-1405,1638-1645`; local `csrc/kernels/store_aligned_int2.cpp:205-214`, `attention_cv_aligned_decode.cpp:250-265`; C4/C16 equivalents `attention_cv_aligned_cluster4.cpp:598-599`, `attention_cv_aligned_cluster16.cpp:701-702`. BF16 rotation does not preserve exact orthogonality, so source-dependent coordinate frames alter relative logits. |
| Zero-history prefill | Rotates the current Q/K/V, performs attention, inverse-rotates the output. | `current_only` calls native FIA directly on raw BF16 Q/K/V and copies its output. | Upstream `triton_backend.py:2854-2871,1450-1483,1561-1575`; local `oscar_ascend/integration/current_only_dispatch.py:91-108`. This shortcut preserves uncompressed native math but does not implement the official finite-precision OSCAR path. Turning off this shortcut alone is insufficient: ordinary local current-source attention is also original-space. |
| Position of inverse V rotation | HP and INT2 partials stay in rotated space and FP32; stage-2 merges them, stores BF16 O, then applies one BF16 inverse rotation. | Each history source/split normalizes in FP32, rounds its partial to BF16, applies Rv.T, and publishes an original-space FP32 partial; exact/current partials are already original-space; final LSE merge is FP32. | Upstream `triton_backend.py:1221-1229,3778-3797`, `triton_ops/decode_attention.py:2206-2224,2533-2594,2634-2650`; local `attention_cv_aligned_decode.cpp:649-666`, C4 `:572-583,984-998,1068-1083`, C16 `:675-686,1102-1116`, `oscar_ascend/integration/impl.py:305-311`. Linear inverse rotation can commute with exact weighted addition; **BF16 partial rounding cannot**. Local split selection can therefore add a numerical dependence on batch/capture shape. |
| Prefix precision within one multi-query prefill step | Prefix slots are classified HP versus INT2 by their slot IDs and dequantized once; the same resulting prefix K/V serves every query in that request's current chunk. | Prefix history/recent membership moves with each query's `position+1-recent` cut. | Upstream `quantized_kv_prefill.py:223-269`, `triton_backend.py:1485-1513`; local `attention_tasks.cpp:132-142`, `attention_cv_aligned_decode.cpp:501-538`. For context512/recent256, prefix token300 is local exact at query512 and local INT2 at query612. Official classification is fixed within that forward. The actual official allocation/flush plan decides which fixed tier is used; this audit does not guess that live plan. |

The official default fast rotation (`environ.py:1031`) does not remove these
differences. `python/sglang/QuantKernel/oscar_rotate_rows.py:42-62` loads BF16
operands, computes `tl.dot` with FP32 accumulation, and stores BF16. Its
documented contract matches the pure-torch rotation helper used in the CPU
examples.

## Reproducible CPU evidence

Run from this repository:

```bash
.venv/bin/python reports/official_semantics_audit_20261021/attention_repro.py
```

The script executes unchanged pure-torch rotation functions extracted by AST
from the pinned official source. Attention reductions use the same existing
declared BF16/FP32 offline oracle on both sides so the examples isolate the
placement/domain differences. It does not execute GPU/NPU kernels and does not
claim bitwise NVIDIA-versus-Ascend equivalence. Results and source hashes are
in `attention_repro.json`.

1. **One exact token, no INT2 compression.** A D256 rotation with a 2×2
   `[[.6,-.8],[.8,.6]]` block passes FP32 orthogonality at
   `1.1920928955078125e-7`; after BF16 rounding its Gram error is
   `0.0031280517578125`. Input V begins `[1,1]`. Local raw exact output is
   `[1,1]`; the official rotate/attention/inverse sequence produces
   `[1.0078125,1.0078125]`.
2. **Identical original K, different history/exact weights, zero INT2 error.**
   Repeat that 2×2 rotation across D256; Q and both original K vectors are
   all ones, Rv=I, historical V=+1 and exact/current V=−1. Current INT2
   packing/dequantization reproduces the rotated history K/V exactly in
   both prefill and decode modes. Official logits are equal at
   `16.1378173828125`, output0. Local historical logit is
   `16.1378173828125` but exact-source logit is16, output
   `+0.06884765625`. This is a source-weight mismatch, not merely an output
   cast difference.
3. **Inverse-before-merge rounding with Rv=I.** Five historical tokens with
   Q=K=0 and V values `[1,1,1.015625,1,1]` are exactly representable through
   the current constant-row codec. With local ceil-divided split lengths
   3+2, normalized partials are `1.005208373...` and1. Local round-each-
   partial-before-merge produces1.0078125. Merge in FP32 before the one
   BF16 output/identity inverse produces1.0; local single-split also
   produces1.0. Both codec modes reproduce this. The example uses the
   official postmerge algebra; it does not assert official five-token
   dispatch chooses two splits. The official one-split value is likewise1.
4. **Moving prefill precision cut.** For context512/sink64/recent256,
   token300 switches from exact to INT2 between query512 and query612.
   With zero Q/K and one nonzero V channel that the frozen `.92` clip
   removes, local late output becomes0. If the upstream fixed prefix slot
   is HP, it retains a nonzero contribution at both queries; if fixed
   INT2, it contributes0 at both. The report records both possibilities
   rather than inventing an allocator state.

These artificial examples show why the current independent v3 oracle can
pass while official implementation alignment remains incomplete: its
`tools/bf16_reference.py:90-99` explicitly models local inverse-before-merge,
and `tools/aligned_reference.py` models original-space exact sources.

## What was not found

- No omitted or duplicated causal key follows from the inspected local
  three-source masks: history+sink/recent+current form the causal set through
  the current query. The confirmed prefill issue is **which precision/frame
  supplies a prefix token**, rather than a demonstrated missing causal key.
- V1 shared-matrix GQA head grouping is consistent: local query index
  `kvhead*(hq/hk)+group_offset` matches official repeated KV-head matrices.
  The CPU case with12 Q heads,2 KV heads,ratio6 agrees. Official optional
  V2 per-head matrices, centering, and nonorthogonal Q/O transforms require
  a separate complete contract; absence of those capabilities is not proof
  that the V1 GQA index is wrong.
- This audit does not change native GDN, MTP, graph policy, clipping values,
  timer code, or production operators. No hardware or model quality gate
  was run.

## Alignment strategy and cost boundaries

1. **Unify the finite-precision coordinate frame.** For an official-aligned
   candidate, feed rotated Q/K/V to exact/current paths and store rotated
   HP window bytes as well. The local writer already materializes rotated
   current K/V for INT2, so those bounded buffers can be reused. Window
   byte capacity need not grow merely to change the frame. Version the
   window representation explicitly so an old raw-space snapshot cannot
   be read as rotated-space. Update current-only, mixed-current FIA,
   q4/q1/C4/C16, MTP and prefix restore together; changing one call is not
   sufficient.
2. **Merge all partials in rotated FP32 space, then apply one inverse.**
   Remove per-history-partial BF16 normalization publication solely for
   inverse GEMM; preserve output/error-domain checks. Merge's FP32
   max/LSE/sum remains unchanged, but its output must round to BF16 before
   the final BF16-operand/FP32-accumulator inverse, matching official order.
   A current-step output buffer is O(tokens×heads×D), already a bounded
   allocation; no history-sized BF16 tensor is required. This may reduce
   repeated history inverse GEMMs but adds final inverse work to exact-only
   requests. **No speed claim follows without target measurements.**
3. **Make source precision a faithful cache policy, not an independent
   per-query shortcut.** Official BF16 HP geometry gives N_Q=8 and a
   recent ring up to256+7 (`unified_kv_pool.py:125-140,405-408`); local
   recent visibility is exactly256, plus separate MTP storage capacity.
   Exact parity requires a specified per-step HP/quant plan compatible
   with local speculative rollback/prefix snapshots. It must not be
   approximated by increasing a window constant or changing masks in
   isolation. The upstream full-prefix dequantization is reference math,
   **not** an acceptable local production architecture.
4. Add a new official-semantic oracle alongside the existing local v3
   contract. Preserve prior frozen tolerances as historical evidence;
   do not silently rewrite an oracle to bless current production outputs.
   Use same captured model Q/K/V and real rotations for eventual target
   differential diagnosis. Artificial examples alone cannot select the
   dominant source of the user's score loss.

## Archive and D.4 boundaries

Archive #13–20 motivates separate LSE/output/metadata inspection; #126 and
#145 prohibit treating CPU arithmetic as asynchronous device evidence;
#148 prohibits treating an operator oracle pass as model quality acceptance;
#164 preserves explicit BF16 conversion contracts. The D.4 phase relevant
here is fused FIA and bounded new-token store: the old
6499.8–6655.1ms full-history restore and approximately209–215ms store are
failure examples, not current timing. Proposed alignment retains fixed-tile
INT2 fusion, original ownership/guard rules, and O(current-step) rotation
buffers. No device-time bound or speedup was measured in this audit.
