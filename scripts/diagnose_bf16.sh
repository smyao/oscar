#!/usr/bin/env bash
# 档案 #94/#95/#125/#169：实时输出、保留失败码，失败仍打包诊断附件。
# 只运行已有独立NPU探针，不启动服务、不修改配置或冻结算子。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "$TASK_ROOT"

if [[ "${1:-}" == "--help" ]]; then
  echo "用法：bash scripts/diagnose_bf16.sh [出错运行的日志目录]"
  echo "默认：logs/20261020T064430.498396Z；使用该次 effective-target.json"
  exit 0
fi
[[ $# -le 1 ]] || { echo "只接受一个日志目录参数"; exit 1; }
py="${OSCAR_PYTHON:-python3}"
run="${1:-logs/20261020T064430.498396Z}"
cfg="$run/effective-target.json"
[[ -f "$cfg" ]] || { echo "缺少配置：${cfg}；请传入出错运行的日志目录。"; exit 1; }
mkdir -p logs
latest="logs/bf16_diag_latest"
if [[ -e "$latest" && ! -L "$latest" ]]; then
  echo "$latest 已存在且不是软链接，停止以避免覆盖。"
  exit 1
fi
out=$(mktemp -d logs/bf16_diag_XXXXXXXX)
ln -sfn "${out#logs/}" "$latest"
git rev-parse HEAD > "$out/git-head.txt"
git diff --stat > "$out/local-diff.txt"
echo "只运行独立算子诊断，不启动服务；请先正常停止自己的模型服务。"
echo "实时日志：tail -f logs/bf16_diag_latest/bf16-probe.log"
echo "结果目录：$out"

rc=0
"$py" -u -m tools.phase \
  --name bf16-probe --log-dir "$out" --timeout 900 -- \
  "$py" -u -m tools.probe_bf16_history \
  --config "$cfg" --output "$out/result.json" || rc=$?
printf '%s\n' "$rc" > "$out/exit-code.txt"

files=("$out" "$cfg")
for file in "$run/serve.log" "$run/status.json" "$run/serve.json" \
            build/ascendc/build_manifest.json; do
  if [[ -f "$file" ]]; then
    files+=("$file")
  else
    printf '缺少可选附件：%s\n' "$file" >> "$out/missing-files.txt"
  fi
done
if tar -czf "$out.tar.gz" "${files[@]}"; then
  echo "请上传：$(pwd)/$out.tar.gz"
else
  echo "打包失败，请上传目录：$(pwd)/$out"
  [[ "$rc" -ne 0 ]] || rc=1
fi
printf 'probe_exit_code=%s\nscript_exit_code=%s\narchive=%s\n' \
  "$(cat "$out/exit-code.txt")" "$rc" "$out.tar.gz" > "$out/summary.txt"
# Summary is outside the already-created archive; exit-code.txt and phase JSON
# inside it retain the original probe outcome, including nonzero exits.
echo "查看结果：cat logs/bf16_diag_latest/summary.txt"
cat "$out/summary.txt"
exit "$rc"
