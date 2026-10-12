#!/usr/bin/env bash
# Archive #126/#145/#148/#154/#155: explicit CPU-only result, fixed deadlines,
# poison/guard/error-domain cases; never represent this as NPU/model/speed proof.
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u
cmake --build build/cpu -j4
mkdir -p reports/wide_fp32_simd_cpu_cases_20261006
run_case() {
  local case_name="$1" mode="$2" result_rc
  local result_log="reports/wide_fp32_simd_cpu_cases_20261006/${case_name}_${mode}.log"
  set +e
  timeout 120s build/cpu/oscar_wide_fp32_simd_cpu "artifacts/wide_fp32_cpu_cases_20261006/${case_name}" "$mode" 2>&1 | tee "$result_log"
  result_rc=${PIPESTATUS[0]}
  set -e
  printf 'WIDE_FP32_SIMD_CASE_RC case=%s mode=%s rc=%d\n' "$case_name" "$mode" "$result_rc" | tee -a "$result_log"
  if ((result_rc != 0)); then return "$result_rc"; fi
}
for b in 512 1024; do
  run_case "b${b}_normal" valid
  run_case "b${b}_cancel_high_value" valid
  for mode in faults grid2 empty_boundaries all_empty; do
    run_case "b${b}_normal" "$mode"
  done
done
