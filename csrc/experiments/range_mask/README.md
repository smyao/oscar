# Range mask microexperiment

Archive #126/#129/#145 and startup D.4 govern this isolated experiment.
Production CV, `ReduceSum`, online softmax, task scheduling, bindings, and
service routes remain untouched. D.4 showed the failed full-history restore
at 6499.8–6655.1 ms per card versus about 18.7 ms for native FIA; this helper
only overwrites masked columns of an existing 256-float score row in UB.

## Exact masking contract

The current `attention_cv_fast_cluster16.cpp::Softmax` derives at most two
visible half-open intervals, then scans all 256 columns with scalar
`SetValue(-inf)` for every masked element. This helper masks their complement
as at most three intervals. For each masked interval, no more than seven
scalar FP32 words precede and seven follow a 32-byte aligned `Duplicate`.
It writes the same negative-infinity bits and leaves visible scores, including
NaN payloads and signed zero, untouched. A completed V→S handoff is required
before scalar edges; one S→V handoff precedes the disjoint vector interiors;
the caller must complete the final V write before a scalar read or V2 use.
The official CPU-debug probe exhausts all 33,153 one-visible-interval pairs
`0 <= begin <= end <= 256` and nine two-interval edge cases, comparing all
256 FP32 bits. CPU-debug cannot establish NPU asynchronous ownership or speed.

## Synthetic operation ledger; not an AISBench trace

Assumptions: 127 independent requests; one local rank has six query heads;
16 FULL model layers and one MTP draft layer; 256 KV columns per tile; a
16,384-token scheduler budget; a 20,000- or 32,000-token prompt split into
`[16,384, remainder]`; no prefix hit; one long first draft per target chunk.
The target's current source uses native FIA, so its causal mask is **not**
counted as CV scalar work. The first draft is a single additional FULL-like
layer, not another 16-layer target pass. These are hypothetical counts from
the current mask loop, not measured request counts or NPU timings.

For a first-draft current chunk of length `q`, its causal frontier contributes
`sum(t=0..q-1, 255 - t mod 256)` scalar negative-infinity stores per query
head; fully visible older tiles contribute zero. For a continuation source1
window with sink64/recent256, task preparation updates `tailBegin` at each
21-query tile. With one split and a sufficiently long context, query offsets
`t=0..62` contribute `t+193` scalar stores across two KV tiles; `t=63..254`
contribute `t-63` in one tile; thereafter each row contributes 192 in the
sink-only tile. Source0 terminal tile and other tasks are excluded.

| Prompt | First-draft current, 127 req × 1 layer | Continuation window, 127 req × 16 target layers | Same window in one draft layer |
| --- | ---: | ---: | ---: |
| 20,000 | 1,945,831,008 | 8,263,249,920 | 516,453,120 |
| 32,000 | 3,108,960,000 | 36,353,617,920 | 2,272,101,120 |

This counts attempted scalar stores, not wall time or cache traffic. The
number of requests/chunks/drafts in the user's LongBench run is not known
from these assumptions. Historical diagnostic `PERF_CV_PROFILE` in
`reports/target_longbench_regression.txt` reports source0 main
`mask_finite_subset_max_raw_ticks=232125` and
`aiv_load_max_raw_ticks=5687002`; those are maxima that may come from
different owners and cannot be divided into an end-to-end percentage. Device
cycles from this exact isolated helper and the complete production mix are
still required before claiming a speed benefit.

## Evidence status

- Host source: independent microkernel and official CPU-debug harness.
- Real CANN `ascend910b4` compile: passed, isolated VM target
  `build/range-mask-device-release/oscar_range_mask_experimental`.
- Official CPU-debug bitwise: passed. Single visible interval checked
  8,487,168 FP32 words across 33,153 cases; nine representative two-interval
  cases checked another 2,304 words. Both had zero bitwise mismatches.
- NPU cycle A/B, graph, model quality, and end-to-end speed: not run.
