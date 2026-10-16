# BF16 long readers: correctness-only qualification

The user stopped local timing experiments on 2026-10-16. This experiment now
qualifies the C4 and C16 BF16 readers with CANN compilation and complete official
CPU-debug execution. No CAModel performance run is planned or claimed. Real NPU,
graph, service, model quality, and end-to-end performance remain user-side gates.

`tools/generate_bf16_clusters.py` strictly derives the two independent readers
from the fixed main7047ce36 FP32 mothers. Its counted edits reverse to the complete
original bytes. Original task ownership, cross-core flags, error paths, INT2
storage, FP16 metadata, FP32 unpack, online softmax, LSE, and merge are retained.
Q/K/V, online unnormalized P, normalized history and natural Rv round to BF16;
the single Cube Matmul type has BF16 inputs and FP32 output/accumulation. The
independent mathematical contract is `oscar_bf16_attention_v2`.

Each original FP32 Q slot is explicitly divided into a BF16 Q half and a
separate BF16 P half. P never overlaps a live FP32 score or another AIV lane.
C16 external B1 uses BF16 NZ with C0=16 and 128 KiB. Casts reuse dead natural
UB under explicit ownership fences; final normalized Q is published in two
32-row chunks. The original GM bounds remain C4 2494464 and C16 7487488 bytes
per core. Statically counted AIV UB is 157760 bytes, below the declared 184 KiB
limit. These are source/resource facts, not performance results.

`fixture.py` declares six complete cases: mature C4 and C16, two-KV C16,
unclustered C1 with two KV heads and three splits, metadata lane1 failure,
and a late-member QR failure. Actual frozen prepare output determines the
independent oracle. Canonical natural `raw.bin` and the reversible
`raw_striped.bin` are kept separately. All tasks, padding, status words, input
immutability and outer GM guards are checked. Only the declared long-prefill
cases suppress healthy source2 work, matching the production current path.

`harness.py` strictly adapts the independent decode checker, including its
actual prepare -> CV/QK/PV/Rv -> merge sequence. No candidate output is used
to construct a golden value. CPU phases have a fixed 120-second deadline;
compile phases have a fixed 180-second deadline and use two build jobs.
Failures are retained, and deadlines are not extended automatically.

`run_phase.sh` records the exact official-SDK phase commands used in the isolated
`/tmp/oscar-bf16-clusters-20261016` Lima stage (staged as `run-phase.sh`). Every
phase is invoked through `tools/experiments/run_camodel_arm.py`, used solely as
a bounded process supervisor here. Its historical filename does not mean a
CAModel execution. After prepare-only, export the independent golden with
`python -m tools.probe_bf16_attention export-reference --fixture-dir FIXTURE
--tasks ACTUAL_TASKS --output-dir NEW_ORACLE`, preserve the actual tasks alongside
that golden, then run the full CPU phase. Use the current S3-aware oracle: prepare
field6 is the output segment and field7 is the source kind.

The device CANN build and all six complete CPU cases passed in the isolated
Lima stage. Every case completed all 20 AIC and 40 AIV actors. Four healthy
cases checked every partial, LSE and actual merged output against the independent
BF16v2 oracle. The two fault cases checked all 2016 status words and additionally
verified exact NaN output domains with `check_fault_outputs.py`: bad metadata
poisoned only the declared 992 source0 lane rows; late-member QR poisoned only
its declared 64 rows. All other partial rows remained finite and other LSE rows
finite or empty. No status value was used to define that expected NaN domain.

The source manifest is intentionally static; it contains no inferred runtime
pass. The source-bound runtime results are in
`reports/bf16_clusters_20261016/validation_summary.json`, with the real controls,
logs, actual task hashes and kernel hashes. CPU wall times are control evidence,
not performance measurements. Raw checker scratch is archived locally, omitted
from Git delivery, and removed from the VM only after archive readback and
owned-PID exit checks. The VM has been released for the remaining system gates.
