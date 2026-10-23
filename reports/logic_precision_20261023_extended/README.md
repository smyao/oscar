# R256 边界与反量化精度的真实 CPU 证据

本轮没有改动或重编生产 kernel。使用现存源码绑定的 q4/C16 CPU executor；隔离 C16 时只编译新 harness，链接同一组生产 `.o`。完整 NPU、图、模型质量和性能均未运行。

## 可直接证明的数值差异

`r7` 是两个输入字节完全相同的极小病例，实际执行 writer、prepare、reader 和 FP32 LSE merge。context=4、sink=0、recent=1、qLen=1，另有一行 NaN padding。

K 前 128 维为 1、后 128 维为 1.0078125；Q 对应为 +1/-1。四个 history 的 V=+1，current V=-1。真实 writer 的精确字节检查确认 K 的 scale 为 FP32 `1/384`、normalized zero=-384、code 为 0/3；默认 .96/.92 clipping 不裁掉这些值。

| reader 模式 | history K 的后半维 | history/current logit | 实际 FP32 merge 输出 | 对未量化解析值 0.6 的绝对误差 |
| --- | --- | --- | ---: | ---: |
| mode0：FP32 metadata/减乘后 BF16 发布 | 1.0078125 | -0.0625 / -0.0625 | 0.6000000238418579 | 2.38419e-8 |
| mode1：metadata 与减乘中间值 BF16 舍入 | 1.015625 | -0.125 / -0.0625 | 0.5796245336532593 | 0.02037546635 |

每个非空 source 内的 K/V 完全相同，P=1，因而没有 P 舍入噪声。预期由恒同 K/V 和标量 logsum 公式直接计算，不调用旧 oracle。两模式所有 phase 都 rc=0，writer 字节门、status、输入不变和 GM guard 均通过。

**表中是最终逆旋转之前的 FP32 merge 输出，不是最终 BF16 输出，也不是模型准确率。** 它证明在这一合法窄动态范围 K 输入上，mode0 消除了 mode1 额外引入的偏移；不能外推为模型损失已降至 1 点以内。

## R256 边界：分开记录完整用例与隔离结果

| run | 范围 | 真实结果 |
| --- | --- | --- |
| r3 | C16，context944/recent256/splits1，完整 history+window+current | CV 在固定 180s 时限内未完成，rc124；没有 partial 输出。原日志保留，未延时重跑完整用例。 |
| r4 | q4，同 context/recent，splits4 | 全部通过。source0 最后 split `[724,944)` 完全被 mask，实际输出精确 +0/-inf；真实 merge 正确忽略空 split。输出最大误差 0.0003051758，LSE 最大误差 1.90735e-6。 |
| r5 | C16 history-only，保留原 leader/follower，只把 source1/2 的 KV range 设为空 | 实际导出 partial/LSE/status/cluster stats 后 harness rc1；随后明确观察到 VM ENOSPC。数据文件完整，日志末行截断，无法还原该次最后失败的具体写盘调用。 |
| r6 | 对 r5 已保存的真实输出继续后置检查及真实 merge；不重跑 C16 | 输入复制逐项 SHA 相同，原核 merge 和独立解析检查通过。1 个成熟簇/16 members/1 tail，2028 个 status 全零，空 source1/2 精确 +0/-inf。 |

隔离 C16 的 history 候选为 880 行、可见 625 行；四个 KV256 tile 分别可见 256/256/113/0。非负查询系数的历史行 P=1，最大误差 1.19209e-7；所有 history partial/merge 最大误差 0.0005533099，LSE 对解析 FP32 逐位相等。因此已经验证该隔离范围内的全 masked 尾 tile 不破坏 alpha/acc/sum/LSE，**但 r3 完整 R256 C16 仍未完成，不能用 r6 替代完整用例或全部逻辑验收**。

## 空间事故与证据保留

本机 VM 的自动 `npuchk`/`cceprint` 日志累计占满磁盘：块使用率 100%，inode 使用率 4%。r5 中数值 evidence 约 21MiB，而自动 npuchk 约 3.8GiB；新建 r6 目录时明确返回 `No space left on device`。已把完成的 r2 SDK 日志完整归档至宿主临时路径、校验后仅清理其 guest 冗余目录；r3/r5 原失败目录未改。空间恢复后才继续 post-only 检查。收尾无 owned CPU 进程，锁可重新获取，剩余空间 5,593,894,912B。

原始小日志、summary、fixture 定义、解析结果和 r6 harness 回执收在 `cpu_evidence.tar.gz`；总览见 `summary.json`。GB 级自动日志、fixture 和输出 bin 不纳入 Git。实际产物仍保存在各 `/tmp/oscar-logic-precision-20261023-rN` guest 目录；r2 自动 SDK 日志归档信息见 `storage_incident.json`。

复现入口位于 `csrc/experiments/logic_precision_20261023/`：`run_extended_guest.py` 执行边界病例，`--family c16 --history-only` 运行隔离范围；`resume_history_check.py` 仅续已有结果的后置检查/merge；`run_dequant_guest.py` 运行 tiny 两模式真实核对照。所有脚本仍绑定现存 kernel 源码，不改冻结门、timer 或生产算子。

档案 #13-20 要求同时核查输出和 LSE；#126/#145 限制 CPU 对异步硬件行为的证明范围；#148 禁止以局部数值通过冒充模型质量；#164 的 BF16 转换形式保持。D.4 中被否决的全历史生产恢复没有重引入。**模型质量目标仍未完成。**
