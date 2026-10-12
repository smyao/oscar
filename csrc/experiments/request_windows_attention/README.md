# Isolated request-window attention gate

Archive #126–129/#145/#154/#155; startup D.4. This directory is not in production CMake or serving selection. Old arm is the **current `attention_cv_decode_bundle.cpp`**, including q4/q1. The five generated candidates retain the source mathematical helpers; this harness builds only decode/q1 initially.

## Artifacts and ABI

- `generate.py`: checked substitutions plus source/candidate hashes in `sources.json`. New map is immediately after tags; `windowCapacity, positionBits` follow scale.
- `fixture.py`: 11 logical BF16/independent PR oracle cases. Old per-page windows and new request windows are independently scattered, as are old single-segment and new dual-segment compressed pages. Requests can reorder while window slot/epoch and stored payload stay fixed.
- `cpu_probe.cpp`: independently executes **both real task preparation calls**, checks all metadata/positions bytes, runs old/new kernels, compares every partial/LSE/status byte on valid and mirrored tag-fault cases, and checks merged valid output against frozen 0.005/0.005 oracle. All GM buffers have guards; both cache inputs and prepared tasks must remain immutable.
- Map-only and position-range faults have no matching OLD ABI error. OLD remains oracle-valid; candidate must publish source1 status1 or4 with NaN outputs, while all unaffected source/padding bytes remain equal. The report explicitly says `all_ab_bytes_equal=false` for these cases.

`shape.txt` columns:

```text
N Hq Hkv D requests tableColumns oldB oldBlocks newB newBlocks prefix segmentStride windowCapacity positionBits sink recent speculative splits cores
```

`flags.txt`: `candidateFault mirroredOldFault expectedCandidateCode`. Workspace is 1,146,880 bytes/core for both current bundle and candidate. A case does not omit any source or task.

## Host preparation

```bash
.venv/bin/python csrc/experiments/request_windows_attention/generate.py
.venv/bin/python csrc/experiments/request_windows_attention/fixture.py --output artifacts/request_windows_attention_20261009
.venv/bin/python -m pytest -q tests/test_request_window_attention_fixture.py
```

Host tests include address reconstruction, exact packed metadata/code bytes, BF16 window bits, permutation ownership, untouched mathematical helpers, and C++ probe syntax/declared arity using an explicitly marked local header stub. That last check is **not CANN compilation**.

## Official CPU gate (root schedules exclusive VM access)

Configure this directory with `SOC_VERSION=ascend910b4`, actual `ASCEND_HOME_PATH`, and `OSCAR_REQUEST_WINDOWS_ATTENTION_CPU_DEBUG=ON`. Build target `oscar_request_window_attention_cpu`. Each invocation is:

```text
oscar_request_window_attention_cpu q4|q1 FIXTURE_DIR [EXISTING_EXPORT_DIR]
```

With an export directory, each completed arm saves partial/LSE/status/task bytes and prints its completion marker immediately. Do not increase an agreed timeout or report a host fixture as official CPU success.

Start with B512→1024 valid and fault boundaries. The separate q1 case uses **B2816→5632, segment stride393216, context5901, one live query plus negative padding**, so reads cross both the segment boundary and the candidate physical-page boundary. It is an address/numerical contract, not a64K/full-model performance sample. C16 and other generated readers require their own later gates.

## Generator fixes found before SDK execution

1. Out-of-range logical position returns expected tag−1; require expected≥0 before tag equality so a poison/empty tag−1 never validates.
2. Check request<windowCapacity before loading a map row, in addition to request<requests and validating the returned slot/epoch.

At preparation time, CANN/official CPU/NPU/graph results remain **not run**. VM work must be coordinated with root through the existing exclusive validation slot.

## Explicit C16 CPU extension

`c16_cpu_probe.cpp` shares the complete host checker with the decode probe. The
separate CMake target `oscar_request_window_c16_cpu` is excluded from the default
build. It links the current `attention_cv_window_range_cluster16.cpp` and its
request-window clone, using 4,997,120 bytes/core and aligned per-core counters.

Generate two cases with:

```text
.venv/bin/python csrc/experiments/request_windows_attention/fixture.py --output artifacts/request_windows_attention_c16_20261009 --cluster-only
```

Both are one request, Q336/GQA6/D256/C65/sink64/recent1/S1. After both real task
preparations compare equal, the exact production healthy-source2 end=begin
rewrite is applied to both complete task tables. No tasks are filtered.
The independent oracle explicitly merges **source0+source1 only**. All336 query
outputs, LSE, status and cluster counters must match; counters must prove one
actual C16 cluster and one shared history KV256 tile. The second case maps
logical virtual128 columns0–3 into physical B1024 quarters4–7, testing segment1
without growing context or current-attention work. Every completed arm saves its
own counters alongside outputs before the next arm starts.

The explicit CPU command uses the same locked/deadlined controller:

```text
python3 csrc/experiments/request_windows_attention/run_cpu.py --executable build/cpu/oscar_request_window_c16_cpu --fixtures artifacts/request_windows_attention_c16_20261009 --log-dir reports/request_windows_attention_20261009/c16-cpu --cases ALL
```

The VM staging additionally needs current `attention_fast_unpack.h`,
`attention_striped_unpack.h`, and `attention_cv_window_range_cluster16.cpp`.

### Resume a completed OLD arm without extending the deadline

When a paired C16 run hits120s after OLD has printed its completion marker,
keep that failed run intact. `--baseline-root` accepts only the same fixture
hash, a C16 command in the original report, the exact OLD completion marker
without preceding SDK checker errors, and five complete OLD files with exact
sizes (partial/LSE/status/tasks/cluster_stats). It records their hashes and
checks the original report, log and files remain unchanged. Use a new log dir:

```text
python3 csrc/experiments/request_windows_attention/run_cpu.py --executable build/cpu/oscar_request_window_c16_cpu --fixtures artifacts/request_windows_attention_c16_20261009 --log-dir reports/request_windows_attention_20261009/c16-cpu-resume --cases c16_history_window_segment0 --baseline-root reports/request_windows_attention_20261009/c16-cpu
```

The C++ `candidate-from BASELINE_OUTPUT_DIR` mode still executes BOTH real
metadata preparations and the production source2 rewrite, checks saved task
bytes against the newly prepared table, loads OLD outputs, validates every
OLD status and the C16 counters, and reruns the full independent source0+1
oracle plus GM guards **before** running the candidate. The witnessed OLD
buffers must remain immutable during the candidate. Full A/B bytes/counters
and candidate oracle checks remain mandatory. All of this is within the same
unchanged120s invocation limit, plus the existing5s forced-stop grace.

The original timed-out pair did not record an executable hash or finish its
OLD oracle gate. Reuse reports explicitly retain that provenance limitation;
they do not relabel the original failed run as passed or authenticate its
binary from source staging alone. Current runs record executable SHA256.

For the unrun segment1 case, an optional `--baseline-only` invocation makes a
new OLD-only checkpoint, with real status/counter/oracle/guard validation and
its own120s limit. Its status is `baseline_checkpoint_passed` and explicitly
`candidate_executed=false`; it is not an A/B gate pass. A later invocation can
use that checkpoint as `--baseline-root` and a separate new output directory.
This split is a numerical validation contract and provides no speed evidence.

## Complete real-NPU helper

`tools/request_window_reader_gate.py` exports `REQUIRED_CAPABILITIES`,
`run_reader_gate`, and strict `validate_evidence`. Its result has `accuracy`
and `graph` dictionaries for the main `probe_request_windows.py` to attach.
The caller supplies the already selected real NPU, actual Cube count, full
frozen acceptance object and signed manifest path. CPU/unknown/failure returns
failed; no performance result is inferred.

All six reader routes run full source0+source1+source2 mathematics. q4/q1 use
N128/B32/S3, including q1's96 negative padding rows. Plain/balanced and actual
C4/C16 clusters also execute live precise windows. Gates include both real task
preparations, all partial/LSE/status bytes, independent PR output/LSE for every
query, metadata error1, invalid map1, and cluster counters.

For every reader, NPUGraph captures all padding with unpublished map(-1,0),
then replays real data and an independently regenerated dataset with relocated
slots and new epochs at the same addresses. Old unused sidecar tags remain.
Each replay must match OLD eager, NEW eager, and the independent full oracle.
A further same-address epoch-only mismatch must reject stale tags with4, matching
independently invalidated OLD tags. Missing any reader/graph/error/oracle row
keeps the complete gate failed. These NPU paths are implemented but not run by
the host tests or the reduced C16 CPU experiment.
