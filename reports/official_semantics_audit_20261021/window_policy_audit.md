# Official precision-window lifecycle: source-derived host reference

Archive #37–49 requires distinguishing committed context from tentative materialization and retaining exact data across rejection; #86 separates target/MTP execution; #127/#128 prohibits equating logical quantum, metadata128, and native FULL physical pages. D.4's full-history dequant 6499.8–6655.1 ms and host prepare around 725 ms are not acceptable implementation templates. This work is bounded host integer algebra only and is not connected to production.

The source pin is `38680852e52c10bd6b9e9a64ed9bc815e742859b`. Exact source URLs/hashes are in the audit's `upstream_manifest.json`.

## Verified allocation policy

`unified_kv_pool.py:126–140,393–408` fixes BF16 `N_Q=8`, requires nominal sink/recent to be multiples of eight, and reserves a recent ring of `R+N_Q-1`. With the requested R=256, the ordinary ring is 263 slots; this is not a change of the nominal recent window to 263.

`allocation.py::_mixed_extend_layout_counts` has an explicit `is_final_chunk` input. `_alloc_for_extend_mixed` derives it from `sequence_end >= len(req.origin_input_ids)`.

- A non-final prompt chunk stores `[HP-prefix][INT2-all]` and **no HP-recent**. Its trailing new tokens are exact inputs for the current attention, but their persistent representation is INT2. They are not promoted back to HP when the next chunk begins.
- A final prompt chunk stores only its newly admitted tail in HP-recent: `H0 = max(0, N - max(C, N - min(max(N-min(N,S),0),R)))`, where `N=C+L`. For a long prompt, this reduces to `min(L,R)`. A final 128-token chunk therefore starts decode with 128 HP-recent tokens, not 256.
- The new-token quant allocation is `ceil(quant_count/8)*8`. This rounds the allocated page count; it does **not** round the logical INT2/HP boundary. For a one-chunk 1001-token prompt, S=64/R=256 gives 681 INT2 tokens, 688 allocated INT2 slots, HP-recent beginning at position 745, and countdown 7.
- Existing cached tiers come from their actual allocation/history. C and L alone do not tell the reader which earlier positions are HP. A fresh chunked prompt's earlier non-final chunks, or an official APC prefix that excludes HP-recent, have INT2 cached history outside the sink.

The entire current chunk remains unquantized for its attention, independently of where those rows are stored for subsequent forwards. Thus the cached tier map is fixed for all queries in that forward; causal masking still limits the current range for each query.

## Decode and flush phase

The pool seeds `counter = R+7-H0`. `advance_flush_counters` performs a flush when the old counter is zero, resets it to 7, and otherwise decrements it. `allocation.py::_alloc_for_decode_mixed` plans/applies/remaps the old HP rows before forward writes the new token. `gpu_flush_int2.py::_flush_plan_kernel` demotes the eight positions `[C-R-7, C-R+1)` for the flushing request, where C is the old context length.

Consequently the ordinary HP occupancy grows to 263; on the next step the oldest eight rows are demoted before the new row is written, leaving occupancy 256. Flush phase is relative to the actual initial HP boundary, not absolute `position % 8`.

Example: a prompt ending at N=1001 with H0=256 begins with cut 745. After seven decode tokens the context is 1008 and cut is still 745. Before token eight, positions 745–752 are demoted; after the new token, context is 1009 and cut is 753. Neither cut is eight-aligned.

Two histories at the same C=1001 can differ: final prefill at 1001 gives cut 745, while final prefill at 1000 followed by one decode gives cut 744. A correct exact snapshot must therefore retain the actual tier boundary/phase, not reconstruct it only from C. For a short prompt ending before S, the actual allocated HP-prefix extent is also part of this state; decode allocates new rows in HP-recent rather than silently growing the HP-prefix allocation.

## APC and native physical-page boundary

Official `mixed_kv_prefix_mixin.py::_mixed_kv_tier_cap` caps a shared prefix to `floor(min(key_len,max(S,key_len-(R+7)))/8)*8`. Its request-owned HP-recent ring never enters the shared radix tree. This is not the same protocol as this project's immutable physical-page exact snapshots.

The new host `prefix_match_cap` reproduces the official cap only as a reference. It is **not installed as a cap on native vLLM APC**. Preserving larger native hits requires the existing physical snapshot ownership plus explicit, validated representation state; neither a page ID nor metadata128 supplies the missing flush phase. No physical page size, GDN layout, native scheduler parameter, or APC setting is changed here.

## MTP scope and retained storage

The official `_is_mixed_kv_enabled` explicitly excludes speculative decoding. `_alloc_for_decode_mixed` accepts only `token_per_req=1`. There is therefore no source-supported official q4/MTP cut to copy. Repeating the pure q1 transition models an accepted serial stream, not the original framework's speculative frame semantics. A real MTP extension must define tentative view/commit behavior and retain the native accepted/rejected positions.

The host reference's `retention_capacity(3)=266` means official 263-token occupancy plus this project's existing allowance for three extra materialized writes beyond the committed endpoint. It does not change R=256 or MTP=3 and does not enable MTP in the official allocator. Production in this turn deliberately does **not** adopt the 8-step allocator lifecycle or this expanded ring.

The selected production contract is separately identified as a native-compatible fixed-forward cached map: `cut=min(C,max(S,C+1-R))`, `sink_end=min(S,C)`, fixed across that forward. It is not a full SGLang allocation-policy reproduction. With existing read-before-write ordering in `integration/impl.py`, this fixed map uses at most R-1 old recent positions, the same range already needed by the old first query. It therefore adds no retention pressure to the existing R+spec ring. Moving the cache write before attention, adopting a 263-token old recent view, or adding 8-step flush semantics would invalidate that safety argument and need separate storage/snapshot integration.

## Deliverables and checks

- `oscar_ascend/official_window.py`: immutable geometry/state and pure `extend_layout`, `plan_decode_step`, and `prefix_match_cap` reference APIs. No runtime import or dispatch integration.
- `tests/test_official_window.py`: 33 passing host cases, including direct execution of the pinned official extend-count function over 176 C/L/finality combinations; exact ragged final chunks, eight-step phase, equal-context/different-state examples, short sinks, snapshot roundtrip, prefix cap, and rollback retention checks.
- No target connection, NPU execution, model-quality test, graph acceptance, or performance experiment occurred. These checks establish source/host contract behavior only.
