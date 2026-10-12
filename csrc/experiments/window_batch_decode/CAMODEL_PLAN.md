# Matched full-operator CAModel decision

Archive #126/#129/#140/#145/#148/#155 and startup D.4. The numerical gates have
passed; fewer source1 DMAs are still a hypothesis until a complete matched
operator comparison finishes. The denominator is the registered production
SIMD decode/q1 kernel in the same freshly compiled CANN library.

1. Run `q4_c641_page512` baseline then candidate, followed by
   `q1_c773_ring259` baseline then candidate. Each retains its entire CPU-prepared
   task table, every history/window/current source, all padding, and the same
   two-Cube fixture budget. No task/source filtering or core-count override is
   exposed by the runner.
2. Each arm uses a fresh simulator process with the same SDK, preload, guards,
   source/input bytes and output initialization. Host monotonic time enforces
   the existing 600-second cap, with termination restricted to that run's
   verified process group. A completed arm immediately saves all outputs and
   a completion manifest before input download, oracle checks or model cleanup.
3. Every output is checked against the actual CPU production output bytes.
   Stable offline LSE merge is also checked against the independent PR golden
   with frozen 0.005/0.005 tolerance. Inputs and prefix/suffix guards must remain
   unchanged. Profiles require one complete task and the exact launched AIC/AIV
   owner set; actor duration includes waiting and is not a subphase timing.
4. `compare_camodel.py` requires successful host exit, no timeout/interruption,
   successful artifact fetch/extraction, both numerical/profile reports passed,
   saved byte hashes valid, identical inputs/library/attrs/core budget, and
   matching partial/LSE/status bytes. Any missing gate yields a null ratio.

These short contexts isolate the impact of the changed window loading within
the whole operator. Their ratio cannot represent 20–30K history, N128/B32/S3,
graph execution, or a full model run.

The next fixture family must preserve N128, 32 requests, Hq6/Hkv1/D256, S3,
20–30K contexts and the production core budget. Preparation may quantify bytes,
real task count and KV256 work from the complete metadata. Budget assessment
will compare those counts with measured short-case host/model evidence before
starting an arm. An estimated or observed inability to complete in 600 seconds
does not authorize smaller context, task filtering, extra cores or a longer
deadline. No full-shape ratio exists until both original-shape arms finish.
