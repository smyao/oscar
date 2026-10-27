# Official C16 功能配对复现

档案 #126/#145/#148/#164、D.4：固定原始 control，验证完整有界融合算子的数值、错误域和所有权；这里不执行 NPU 或性能测试。

Control 固定为 `232b563b69e01495046a9b0025ac9150bc11b9d7` 中的 official C16，原始 SHA256 为 `df843b991b0112e97b524f9b2d10f53f8805a650c2187ae47345a545b0f0582f`。`stage.py --base` 会先将 ref 解析成 commit，再检查这些原始字节；即使当前 HEAD 已推进，也不会把候选当作 control。候选取当前工作树，双方文件哈希写入新 manifest。

在仓库根目录准备独立目录，避免覆盖已保存的历史证据：

```bash
.venv/bin/python csrc/experiments/official_c16_20261027/stage.py --base 232b563b --output /tmp/oscar-c16-replay-package
```

将生成的 `source.tar` 放到已有的本地 CANN 9.1.0 AArch64 SDK 环境，解压至新的空目录。在该目录执行以下命令；Python 环境须已有 torch、项目功能验证依赖和 SDK CPU-debug 库。`run.py` 使用 `/usr/local/Ascend/cann-9.1.0`，会设置 SDK 库路径、构建并运行两臂完整算子和冻结 v4 oracle。输出目录必须不存在；这里无需 NPU。

```bash
tar -xf /tmp/oscar-c16-replay-package/source.tar
OSCAR_C16_REPORT="$PWD/reports/c16-replay" python3 -m csrc.experiments.official_c16_20261027.run
```

可在同一 SDK 环境单独复现设备代码编译；该命令编译 actual official q4/q1/C4/C16 reader，不执行设备：

```bash
cmake -S csrc/experiments/official_math_20261021 -B build/c16-device-replay -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 -DSOC_VERSION=ascend910b4
cmake --build build/c16-device-replay --parallel 2
```

既有失败没有删除：初次运行缺少 `libunified_dlog.so`，prepare rc127；后续 Q337/context577 大例的 control 消耗了过多 SDK 检查 trace 资源，该例没有完成，不算 passed。证据分别保存在 `reports/long_context_20261027/c16/cpu_evidence.tar.gz` 和 `first_cpu_attempt.tar.gz`。没有延长 deadline、放宽 oracle 或把未完成输出计入结果。

最终使用六个互补 shape：Q337/context320/recent1 覆盖实际 C16 聚簇和 current；Q22/context577 覆盖独立 leader 的多 KV；Q22/context320/Hkv2/recent256 覆盖完整 window 和多头；Q337/context96/mode1 覆盖旧 reader mode；另外保留 metadata fault 与全遮罩 history。比较 tasks/partial/LSE/status/cluster stats，健康例还比较合并输出，并保持 v4 `0.005` oracle。六例通过不能替代原大例完成，也不证明真实 NPU、图回放、模型质量或端到端速度。

保存的 `source_manifest.json` 记录实际历史运行的输入；本次修订仅固定未来重放的 control 选择，未回写那份 manifest。复现应使用新的输出路径保存新 manifest。
