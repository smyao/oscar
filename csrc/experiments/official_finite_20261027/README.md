# Official finite certificate functional probe

Archive #126/#134/#145/#148/#164 and startup D.4 govern this probe. It compares the unchanged C16 row-sum predicate with the new sufficient certificate on the same score bytes; it does not benchmark either path.

The independent kernel uses an 8192-float scratch area with guards. The 1507 cases cover rows 0–32, NaN/±Inf/signaling NaN in the first and last element of every 64-element repeat in all 32 rows (including offset 8191), inactive rows, values immediately below/at/above 2^110, distributed sums near that threshold, maximum-finite overflow and six cancellation arrangements, signed zero and subnormals. It checks exact predicate equality, unchanged input/score bytes, GM/UB guards and untouched scratch beyond the live count.

Build using the official local CANN 9.1 installation:

```bash
cmake -S csrc/experiments/official_finite_20261027 -B build-finite-cpu \
  -DOSCAR_OFFICIAL_FINITE_CPU_DEBUG=ON \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
cmake --build build-finite-cpu -j4
LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/cann-9.1.0/tools/tikicpulib/lib \
  timeout --signal=TERM --kill-after=5s 300s build-finite-cpu/oscar_official_finite_cpu
```

Device compilation only:

```bash
export PATH=/usr/local/Ascend/cann-9.1.0/aarch64-linux/bin:/usr/local/Ascend/cann-9.1.0/aarch64-linux/ccec_compiler/bin:$PATH
cmake -S csrc/experiments/official_finite_20261027 -B build-finite-device \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
cmake --build build-finite-device -j4
```

Evidence is in `reports/long_context_20261027/finite/summary.json`. CPU-debug and compilation do not establish actual device timing, asynchronous behavior, graph replay or model acceptance.
