# Decode finite-result readback research

This directory compares each current production family with its own unchanged
source denominator: hybrid L1 q4 and MDL q1. It is not enabled in production.
`generate.py` pins both whole sources and the complete shared Softmax body,
restores every other byte except entrypoint/include relocation, preserves all
other methods and cross-core flags, and does not add GM, UB or L1 memory.

Only a complete kind0 KV256 tile visible to the earliest active query enters
the new branch. Every row still runs the same FP32 ReduceSum256 with tmp[16]
in the same order and PIPE_V ordering. Results occupy 16 aligned 32-byte
slots in the already-dead natural buffer; one V_S permits scalar checks, and
S_V closes those reads before Softmax alpha reuses this memory. Other source,
partial, empty and padding paths retain the original loop exactly.

The actual-method checker extracts each family's complete Softmax and
AddVisibleRange, with real decode capacities (dequant65536, natural34816,
score16384, accumulator16384, stats256, scratch2048 bytes). It covers active
16/8/6/0 rows, overflow, cancellation, NaN/Inf, original precise and partial
masks, padding, incoming metadata error, score NaN overriding error3, and the
unchanged Commit method restoring next-block metadata3 after that score. Each
family has15 actual-method cases. This is not a full-CV oracle.

The complete checker is an inverse-verified relocation of decode_mdl_load's
real prepare/CV/merge/oracle checker. q4 uses the same exact NZ Rv permutation
for both arms; q1 uses natural Rv for both. Valid CPU shape is R2/N8/C641,701;
long CAModel shape is R2/N8/C1344,1376/S1, all 10 history + 4 precise-window +
2 current KV units. Long metadata comes from real prepare. The long pair
checks all output/status bits, frozen independent oracle, immutable inputs,
GM guards and all task/AIC/AIV aggregate profiles. It does not request an
instruction trace or claim a phase share. The original replay's FD polling
thread is removed with the existing strict logging-only adapter; official
LOG mask0 is the sole simulator config change, parsim remains stock0.

Every CPU case is bounded by120s and each CAModel arm by600s. Failures stop
qualification; diagnostics use new receipt IDs and never count as speed.
The initial double-fault expectation failed in the original q4 baseline
before the candidate ran. Its raw input has qr NaN and a next-block zero K
scale; a separate CPU-only scalar-order diagnostic proved LoadQueries2, then
finite actual score sums, then next-block metadata3, then finite final state.
Original expectation2, failure, full status bytes and diagnostic are retained.
The corrected `query_metadata` fixture changes only its exact expected code
to3; all original input bytes are identical. Both complete q4/q1 pairs pass
that code and all output bytes. This observation does not infer target NPU
NaN propagation. Actual nonfinite-score/Commit ordering is covered separately
by extracted original methods, not mislabeled as this full-input witness.

Current evidence is in `reports/decode_finite_20261015/`. No NPU, graph,
whole-model quality, or native32K/64K end-to-end speed acceptance follows from
these local results. Kernel counters and local ratios are never multiplied
into the user's native end-to-end target.
