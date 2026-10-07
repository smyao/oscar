# Fresh current-only primitive experiment

This is an isolated experiment under archive G30–G34/#4–16/#91/#126/#145
and startup D.4. Fresh context=0 has no compressed history or exact old
window; its only attention contribution is native current FIA. The existing
route still prepares empty CV tasks and merges three source slots. This
experiment checks the proposed one-kernel output validation/copy step; it
does not change any production route, native vLLM source, GDN or KV format.

`oscar_current_only_kernel` takes BF16 current output `[rows,D]` and FP32
current LSE `[rows]`, writes BF16 final output `[rows,D]`, FP32 final LSE and
int32 row status. A future binding must reject any non-FP32 LSE explicitly;
the kernel byte ABI cannot inspect a Torch dtype. Each core stages at most 16
rows. The finite path mirrors `merge_lse_out(splits=1)`: BF16→FP32 CANN Cast,
zero accumulator, Muls by one, Add, the same per-row `ReduceSum` finite check,
LSE plus CANN `Log(1)`, and FP32→BF16 RINT Cast. This preserves old `−0`
normalization and the old overflow status rule. `LSE=−inf` writes +0/-inf/
status0 without reading current values; NaN or +inf LSE writes +0/-inf/status2.

Buffer handoffs are MTE2→V/S after each batch load, V→S before checking each
ReduceSum, S→V before Log, V/S→MTE3 before publish, and MTE3→V/S before the
next batch reuses output/status UB. With 64-byte-aligned GM status/LSE base
pointers, the 16-row groups occupy separate 64-byte lines; a future binding
must enforce that alignment rather than infer it from CPU-debug. This avoids
the #145 reuse race and archived per-core cacheline collision. There is no history-sized temporary or
host readback. D.4's full-history dequant regression therefore does not recur
in this primitive.

The official CPU-debug baseline runs the actual production `merge_lse_kernel`
with `splits=1` between separate CANN BF16→FP32 and FP32→BF16 Cast adapters.
It compares every BF16 output bit, FP32 LSE bit, and int32 status bit for 17
rows at each of D64/D128/D256. Cases include BF16 NaN payload, signed zero,
±inf, BF16 finite values whose FP32 row sum overflows, LSE −inf/NaN/+inf,
extreme finite LSE, subnormals, and a 16+1 batch tail.

Real CANN `ascend910b4` device compilation and official CPU-debug passed.
Real NPU numerical completion, graph capture/replay, model quality and
performance have not run. This experiment is not a service optimization or
acceptance result.
