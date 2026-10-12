# Bounded precise-window DMA experiment

Isolated prototype only. The denominator is the current production SIMD kernel,
`csrc/kernels/attention_cv_striped_decode_simd.cpp`. The live registration in
`csrc/striped_attention_bindings.cpp` maps both q4/q1 to its SIMD launch wrappers.
Production, GDN, references, serving variants and launch scripts are unchanged.

Archive #126/#129/#140/#145/#148/#155 and startup D.4 were read before this change.
D.4 four questions: this is precise-window source1 inside fused FIA; the failed
full-history restore consumed 6499.8–6655.1ms/card versus FIA 18.5–18.9ms, so this
experiment changes only bounded DMA traffic for current exact-window rows. It
does not restore history or change FP32 math. Fewer DMA operations are a source
property; they do not establish device latency, model quality or an eight-minute
service result.

## Change and preserved error domain

Each original HalfKv64 window load is partitioned into contiguous runs. A run
ends at the earliest live-row limit, sink-to-recent logical jump, virtual128
boundary, physical-page end or recent/speculative ring wrap. With the usual
sink64/recent256/spec3 geometry, the ring period is 259 and total exact-window
capacity is 323; neither value is hardcoded in the kernel.

The page table is read once per run. Tags are read in one bounded transfer.
Only contiguous subruns whose tags match the original in-page positions issue
BF16 K/V DMA; invalid or unvisited future rows are never fetched. The existing
FP32 destination is initialized to +0 as before. BF16-to-FP32 Cast keeps its dtype
and rounding mode; each valid row retains the original ReduceSum finite check.

Tag scanning for DMA does not publish errors. After loading, the second scalar
scan visits the original row order: bad tag sets error4; valid row performs the
same finite check and may set error2. Invalid physical runs set error1 at their
original position in this sequence. K is still processed before V, including
their independent metadata checks. This avoids reordering a later bad tag before
an earlier nonfinite row. Task scheduling, QK/PV, V2, LSE merge, source0 unpack,
source2 current loading, rotary output and workspace ABI remain the same.

## Buffer ownership and capacity

No UB allocation is added. Explicit D256 UB remains **146368B per AIV** (below
188416B); the unchanged production workspace ABI is used for both arms.

| Buffer | Existing capacity | Temporary source1 use |
|---|---:|---|
| packedBuf | 10240B | Up to 64 contiguous int64 tags, 512B |
| naturalBuf | 32768B | Up to 64×256 BF16 rows |
| dequantBuf | 65536B | Existing 64×256 FP32 destination |

Permanent metadata indices and unpack masks are untouched. Before MTE2 refills
the tag scratch, V→MTE2 and S→MTE2 retire prior vector/scalar users. A zeroed BF16
run waits V→MTE2 before DMA; Cast waits MTE2→V. The per-row finite loop ends in
V→MTE2, and scalar tag reads end in S→MTE2/S→V before the next role reuses that
buffer. Existing PublishKv MTE3→V and score/P ownership barriers stay intact.

## Required evidence

Both full kernels must receive identical input bytes, tasks and grid. The CPU
harness compares every partial/LSE/status byte and the independent merged oracle,
including poison, page/ring boundaries and ordered tag/finite/page failures.
CANN compilation, official CPU results, CAModel cycles, target NPU, graph,
quality and service performance are distinct gates. No NPU/graph/service result
is asserted by this source or a simulator result.

CANN 9.1 / ascend910b4 device build passed in the local Lima `oscar` VM (return
code 0). The shared library exports the four baseline/candidate q4/q1 launch
symbols. Official CPU-debug then passed **10/10 cases** at the same two-Cube
launch budget on both arms: six valid cases passed the frozen independent
0.005 output/LSE oracle, and four fault cases passed the original tag/finite/page
error-order checks. All partial/LSE/status bytes matched the current production
SIMD kernel. Input snapshots and both 64B GM guards passed. Cases covered q1/q4,
contexts511/641/773, ring259, sink/tail jump, virtual128/physical512 boundaries,
Hkv2 interleaving, and both orders of a V NaN adjacent to a bad tag.

Every case had the unchanged 120-second limit and its own temporary SDK trace
directory. The checker remained enabled; each directory was removed after saving
bounded trace head/tail excerpts and complete stdout. The first CPU build missed
an existing merge header in the isolated staging tree; transferring that header
fixed the build without changing source. Both the failure and retry are retained.

Source hashes, library/binary identity, individual case logs and the unchanged
production-body check are recorded in `evidence/validation.json`. The CPU export
at `artifacts/window_batch_decode_cpu_export/<case>` contains the actual prepared
task table and current production partial/LSE/status for a later matched CAModel
comparison. The two-Cube budget is a local small-fixture setting, not the target
service configuration. **CAModel, real NPU, graph, model quality and full-service
performance remain unverified.**
