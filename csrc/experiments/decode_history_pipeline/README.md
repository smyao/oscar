# q4/q1 INT2 history pipeline experiment

This isolated copy of the current production SIMD kernel changes only the
nonempty source0 scheduling. It is not registered or selected by serving.
Archive #126/#129/#140/#145/#148/#154/#155 and the complete startup D.4 logs
govern its design. `tools/experiments/make_decode_pipeline.py` records the
baseline hash and generates the candidate; precise sources retain the old
implementation.

Two KV256 GM buffers alternate. Next K is decoded while Cube computes current
QK; its two packed HalfKv64 inputs stay in UB. Next V is decoded from those
same bytes while Cube computes current PV. The next QK can overlap consumption
of the previous PV. A later P-ready flag cannot occur until the previous PV
has been added to the original FP32 UB accumulator. QK/PV shapes, KV order,
softmax and alpha, normalization, final RvT, output and status publication
are unchanged. Future metadata errors are deferred and committed in the old
`Load/K0,V0,Load/K1,V1` order after the previous PV accumulation.

| Flag | Producer → consumer | Protected data |
| --- | --- | --- |
| 8 KV ready | two AIV → AIC | Q and complete current K/V buffer |
| 9 score ready | AIC FIX → both AIV | score/P GM plane |
| 10 P ready | two AIV MTE3 → AIC | probability rows; prior PV is already consumed |
| 11 PV ready | AIC FIX → both AIV | one PV result buffer |
| 12 norm ready | two AIV MTE3 → AIC | normalized accumulator in former Q plane |
| 13 rotation ready | AIC FIX → both AIV | final RvT result |
| 14 task done | two AIV MTE3 → AIC | output/LSE/status; all scratch can be reused |

The D256 GM workspace is 1,146,880 bytes/core, compared with 917,504 bytes/core
for the baseline. Extra UB is one 10,240-byte packed buffer per AIV; total
156,608 bytes remains below 188,416. Both capacities are independent of
historical length. This is bounded tile staging, not a full-history restore.

D.4 comparison: the old full restore cost 6,499.8–6,655.1 ms/card while FIA
was 18.5–18.9 ms. This experiment tries to overlap the existing bounded
producer and Cube work; it does not reduce FP32 FLOPs or claim any measured
speed. #145/#154 require explicit UB DMA/vector ownership; #148 requires
same task/error domains, not a unioned history mask. CPU, CANN, CAModel, NPU,
graph and whole-model evidence must be recorded separately.

The shared CPU harness in `../window_batch_decode/probe.cpp` compares full
partial/LSE/status bytes against the actual production SIMD kernel and uses
an independent frozen oracle. It permits independent workspace sizes through
compile definitions. Real CANN 9.1 ascend910b4 compilation and eight official
CPU-debug cases initially passed on 2026-10-06. Two additional multi-request S3
cases also passed, for ten cases total. They cover q4/q1, one/three/seven history
tiles, a future invalid scale, earlier-score/later-scale error precedence,
and an invalid unused tail. Every case matches baseline partial/LSE/status
bytes and preserves inputs and allocation guards; all valid cases also pass
the independent frozen oracle. See
`reports/decode_history_pipeline_validation_20261006.json`.

The complete 20K q4 CAModel pair subsequently passed: N128/Hq6/Hkv1/D256/S3,
20 Cube cores, one live request with all padding and all three sources retained.
Baseline 2,464,632 ticks versus pipeline 1,889,225 ticks (1.3046x). Every
partial/LSE/status byte matches and both arms pass the independent oracle.
This is an insufficient standalone gain for the 840s-to-480s objective, not a
32-request or service speed claim. The prior 120s CPU-debug baseline timeout
on this long shape remains recorded; actual prepared tasks were exported and
the complete CANN CAModel pair provided numerical evidence instead. See
`reports/decode_history_pipeline_long_camodel_20261006/`.
Real NPU, graph replay, and whole-service speed remain unverified.
