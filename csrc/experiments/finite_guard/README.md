# Isolated finite-score guard prototype

Archive #126/#129/#140–#153 and startup D.4 govern this experiment. It clones
the current fast CV kernel into this directory and changes only source0 score
validation when a conservative input bound proves every pre-mask score finite.
It is **not** linked into the production extension, capability manifest,
serving selector, or candidate variant. The original fast kernel is unchanged.

For valid INT2 codes in `[0,3]` and finite FP16 `scale>0`/`zero`, the existing
FP32 `Mul` then `Add` yields `|Kdeq| < 4*65504 < 2^18`. If each rotated query
element has magnitude at most `2^32` and `0 < scale <= 2^32`, the absolute
bound for one scaled 256-term QK score is below `2^90`; a 256-score ReduceSum
is below `2^98`, well inside finite FP32 range. The original LoadQueries
ReduceSum finite check remains, and an additional one-time per-query max-abs
test rejects huge finite Q.

For history tasks with `0 < real_rows < 128` and `real_rows % 64 != 0`, one
otherwise padded Q row is filled with `1.0` and included only in QK. It is
read after CubeReady and before any P write, then reduced once over KV256. A
nonfinite K element makes this sentinel score/reduction nonfinite; the check
merely chooses the complete old per-real-row validation and **never sets an
error code itself**. If sentinel and Q bound are safe, only the redundant
per-real-row score ReduceSum is omitted. Real-row P, PV, masks, accumulator,
rotation, output and status logic stay old; precise source1/window and
source2/current always use the old path. No cross-AIV health flag, extra GM
workspace, or new synchronization protocol is introduced. At rows=64/128,
QK M+1 could cross an SDK basicM boundary, so the prototype falls back.

The D.4 failed path restored full history in a separate dequant kernel for
6,499.8–6,655.1 ms/card while native FIA was 18.5–18.9 ms; this experiment
keeps fixed KV256 tiles and no history-sized BF16/FP32 buffer. The prospective
gain is one Vector ReduceSum plus a Vector-to-Scalar fence per visible query
row per history tile, traded against one-time Q max-abs checks, one extra QK
row, and one sentinel ReduceSum per history tile. It has **no** target NPU or
full-model speed evidence. The old q4 actor profile's entire `softmax_p`
stage was only about 7.6% of that actor and predates fast unpack; no large
end-to-end gain is inferred from instruction count.

Independent CANN build (VM only):

```bash
cmake -S csrc/experiments/finite_guard -B build/finite-guard-device -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
cmake --build build/finite-guard-device -j2
```

Official CPU-debug build (VM only):

```bash
cmake -S csrc/experiments/finite_guard -B build/finite-guard-cpu -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 -DOSCAR_FINITE_GUARD_CPU_DEBUG=ON
cmake --build build/finite-guard-cpu -j2
```

Run `build/finite-guard-cpu/oscar_finite_guard_cpu MODE GOLDEN_DIR` for:

| Mode | Golden directory | Required result |
| --- | --- | --- |
| `normal` | `reports/vm-history-cluster148/goldens/d256_q4_c511` | partial/LSE/status bitwise and frozen merged oracle |
| `poison` | D64 `reports/vm-history-cluster148/goldens/d64_q4_c65` and D256 `reports/vm-history-cluster148/goldens/d256_q4_c511` | NaN poison workspace/partial/LSE, then bitwise and oracle |
| `empty` | `reports/vm-history-cluster148/goldens/d64_q4_c0` | empty source0 zero/negative-infinity and oracle |
| `masked_empty` | `reports/vm-history-cluster148/goldens/d256_q4_c511` | live tile with first three queries unable to see its only key; old/new bitwise |
| `masked_bad_k` | `reports/vm-history-cluster148/goldens/d256_q4_c511` | bad K metadata in one live tile whose key is masked for every query; retain old metadata error priority |
| `nan_q`, `inf_q`, `nan_scale`, `inf_scale`, `neg_scale`, `huge_q`, `huge_cancel_q` | `reports/vm-history-cluster148/goldens/d256_q4_c511` | identical final error status; invalid NaN payloads compared by class only |
| `cross_lane_bad_k`, `cross_lane_bad_v` | `reports/vm-history-cluster148/goldens/d256_q4_c511` | AIV1 bad metadata must leave both AIV final statuses identical to old via the shared QK sentinel/fallback |
| `bad_page` | `reports/vm-history-cluster148/goldens/d256_q4_c511` | first virtual page invalid; #126 sentinel UB must not be reused before MTE3 finishes |

`huge_q` sets a finite `1e30` rotated query and finite maximal FP16 K
scale/zero with INT2 code 3 over the live source0 history. Individual scaled
QK scores remain finite, while the old 256-score ReduceSum overflows. The
prototype must fall back and report the same error as the old kernel.
`huge_cancel_q` alternates finite `+2e30/-2e30` rotated Q, so the old
LoadQueries sum is exactly zero. Matching signed K is made from valid
FP16 scale `+65504`, zero `-65504`, and exact INT2 codes `2/0`. Each scaled
score remains finite but their old 256-score ReduceSum overflows. This
isolates the new max-abs fallback from the old query-sum check.

CPU-debug is only a numerical/ordering gate. It cannot prove target Cube M+1
bitwise parity, nonfinite propagation, graph replay, model quality, or speed. Those remain
unverified until a separate target-device experiment is authorized.
