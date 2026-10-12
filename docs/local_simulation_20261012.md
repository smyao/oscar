# 本地连续模拟、MTP 最大长度修复与验证边界

本轮只使用当前工程和只读 `references/` 原生实现。相关档案为 #34、#37–49、#127、#128、#156、#157；已阅读 D.4 完整原始打点。没有修改 AscendC 数学、参考树、精度阈值、设备配置、GDN 状态算法或生产路由开关。没有新增真机错误档案。

## 用户回传后的分发修复

用户在Linux/Python3.12.13运行同一命令，contracts为31 failed / 122 errors / 114 passed / 16 skipped。回传栈明确为 `references/vllm-ascend/.../llm_base_proposer.py` 不存在；原先整棵 `references/` 被Git忽略，本机有文件，新clone没有。这是CPU测试依赖交付遗漏，不是这条日志证明NPU算子故障，也不是通过跳过测试能解决的问题。

现将CPU契约用到的25个完整原生文件及两份原始Apache-2.0许可证打包为 `references/native_cpu_contracts.zip`（314749字节），作为Git跟踪文件交付。每个文件先与记录commit的原始Git blob逐字节核对，manifest记录全部27个文件SHA256、压缩包SHA256及原有10个方法AST指纹。没有摘录或改写原生方法，没有写入/修改原来的两棵参考树。

`tools.native_contract_sources` 仅用于本地测试：存在reference checkout时仍验证实际commit/tree/status和内容；不存在时读取包内相同文件，明确记录 `bundled_snapshot`，不伪称存在Git checkout。压缩包不解压、不联网、不从机器安装的任意vLLM版本取代码。残缺checkout、错误版本、包损坏、逐文件哈希不符均报错。新入口在输入缺失时只打印明确错误并保存`failure_stage=native_source_inputs`/rc2，阻止后续重复报错；部署/直启没有增加前置审计。

验证使用完整的临时Git克隆，确认 `references/vllm` 和 `references/vllm-ascend` 均不存在，再加入本轮交付改动。contracts 267 passed / 16 skipped，三轮回放各66 passed。本机及无参考树的完整Git克隆全量回归均1206 passed / 160 skipped / 6 subtests passed；执行环境仍为本机macOS/Python3.14，未冒充已在用户Linux/Python3.12重跑。新增负例验证缺包、容器损坏、文件被改写及残缺checkout。首次开发回归暴露公共AST加载器还读取本工程文件，已分流修复；没有将它们塞入冻结原生快照。纯文件导出缺少Git历史导致的已有冻结基线测试失败，用完整克隆复验，不放宽对应生产检查。结果见 `reports/local_simulation_portability_20261012/report.json`。

此分发修复不改变模型数学或性能结论；下文原MTP复现和内存账仍保持原范围。用户回传的是CPU模拟入口，未写为新NPU档案。

## 确定复现与修复

执行实际 Ascend `AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm` 方法的 AST，保留它的长度更新、slot 计算和 padding 行为。最大长度4096/262144、物理页2304/2816、接受数0–3，共16个边界用例：修复前12失败、4通过。失败来自本地 checked prepare 的标量查表模型，**不是执行了 AscendC/NPU**。

原生方法先把乐观 verify 长度加1，超过 `max_model_len` 后重置为1；slot 则独立使用实际接受位置。部分接受/拒绝时，slot 仍合法，长度已变为1（下一步为2）。OSCAR 按这个长度限制页表查找，找不到合法 slot，得到 `prepare status=5`。

修复在 `metadata.prepare_context_lengths()`，由真实 `OscarAttentionImpl.forward()` 传给 checked prepare。后续 draft 步 d 的正常正长度至少为1+d，1..d 是原生重置哨兵；只对这些行将**查表上界**恢复到模型最大长度，实际位置仍从唯一的 slot/table 匹配恢复。超过模型长度的负 slot 保持屏蔽，零/负长度、丢失或重复页表项仍失败；不改原生 metadata、不用 mRoPE 坐标推导 token 位置。main 和首次 draft 仍返回原生长度 view。

增加的是后续 eager MTP 的小张量比较/选择，没有主模型图内新增计算、CPU 值回读、请求循环或 KV 复制。需要真实 NPU 短门确认这些操作的目标支持及实际开销；本轮不声称提速。

该错误发生在最大长度附近。档案 #156/#157 约24K故障的非零状态尚未知，**不能用这个边界错误解释它**。原 `OSCAR_STATUS_DIAGNOSTICS=1` 的首错诊断和 fail-closed Trap 保留，使用方法见 [诊断文档](status_guard_diagnostics.md)。

## 可重复的本地模拟

```bash
.venv/bin/python -m tools.local_simulation --rounds 3
```

默认写 `logs/local-simulation-<UTC>/`。每轮有实时终端输出、完整相位日志、JUnit、状态及真实退出码，最终为 `report.json`；任一子进程失败、超时、清理失败、报告缺失或必需回放被跳过，立即停止。重复目录运行先移除本轮旧JUnit，不能用旧通过文件证明新运行。`--rounds`范围1–10，不进行后台调度。这不是任何真机部署前置步骤。

模拟覆盖：

- 原生 FULL/Mamba 分组、分配、prefix hash/联合命中、不可变完整页共享、独立尾页、释放和物理页循环复用。
- 实际 TP4/MTP3 GDN conv形状 `(6,2560)`、SSM `(12,128,128)`、页817152字节；没有将旧通用fixture的小conv形状当作本模型。
- 4736/20K/32K/50K/262140长度，三种写入顺序、两个并存prefix分支、24轮不同接受数、后两轮draft及请求重排；源数据身份与精确窗口tag一同校验，加入同tag旧请求数据的负例。
- 独立原生 MTP 方法回放、最大长度/部分接受/拒绝/全接受、int32 slot、虚拟128表、padding；额外32个正常长度组合，包括24624。
- 正式 forward 的长度实参，以及共享 candidate 的直启配置与 q4/q1/C4/C16算子选择。使用 mock 记录算子调用的测试只证明路由，不证明设备执行。

有界窗口读写是标量CPU模型；压缩attention数值由现有独立oracle测试覆盖。模拟不执行完整模型/GDN算术、异步DMA、HCCL或ACL图，释放回放也不等于完整调度器的抢占恢复验收。

参考版本：vLLM `0fc695fc`，Ascend `5cb98caa`。本轮增加原生MTP方法指纹，现9个文件/10个方法；历史组合pin `19e43698` 未恢复，不能把测试结果记到该版本。

## 实际验证结果

| 项目 | 结果 |
|---|---|
| 修改前完整本地回归 | 1130 passed / 156 skipped / 6 subtests passed |
| 新增最大长度用例，修复前 | 12 failed / 4 passed；负例继续保留 |
| 修复后本地契约相位 | 267 passed / 16 skipped |
| 修复后连续回放 | 三轮分别66 passed / 0 skipped / 0 failed |
| 修复后完整回归 | 1201 passed / 160 skipped / 6 subtests passed，53.81s |
| 直启candidate只读plan | prefix=true、MTP3、FULL_DECODE_ONLY、预算32768、全部候选开关启用 |
| CANN编译 / NPU / ACL图捕获与回放 | 本轮未执行；新增4个真实NPU边界用例接入原CV门 |
| 完整模型质量 / 实测HBM / 相对原生性能 | 本轮未验收；#157设备首错未闭环 |

160 skipped：142项显式真NPU（38 CV、104 rotation）、16个重复大形状主机分支、1个Linux ELF、1个缺少本机C++工具链。跳过不记为通过。主机为Python3.14/torch2.14，目标torch2.10/NPU另行验证。HTTP监管测试需本机回环权限；首次沙箱禁止bind属测试环境限制，没有修改生产服务超时或绕过探针。

机器可读证据、原始本地相位日志及失败/成功JUnit存档见 [报告](../reports/local_simulation_20261012/report.json)。本轮修复默认进入原服务路径，固定交付仍是：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

本次交付分支为GitCode `oscar-new`，与本地工作分支一致。GitCode默认 `codex/oscar-ascend` 已有独立分叉，不在本轮合并范围；固定 `git pull --ff-only` 应在跟踪本次分支的部署checkout执行。

直启不提供验收证据。完整探针流程保持原样；无NPU不得用模拟替代。必要目标验证仅短算子/图/prefix服务probe，不再要求AISBench。

## 内存账与性能边界

每卡Hq6/Hkv1/D256，INT2 payload为136字节/token（原生BF16 K+V为1024）。考虑GDN共用pool及对齐后，FULL物理页容量实际为768→2304，不能按payload比例声称整模型节省86.7%。页817152字节保持原生规格；精确snapshot333336字节位于393216字节padding内，没有再分配一份完整BF16历史。

candidate预算32768、20 Cube核、C16共用工作区为 **1259537664字节/卡（1.173 GiB）**。它不随历史长度线性增长，但不能漏算。下面只计算无prefix共享的稳态live block需求：17个共享tensor、每请求3个GDN组×4状态、1个null block、FULL长度加3个speculative保留。排除可回收cached checkpoint、瞬时额外页、权重、activation、图池和原生工作区，**不是实测HBM，也不是完整内存上界**。

| 请求形状 | native / OSCAR live blocks | pool需求减少/卡 | 减去OSCAR固定workspace后的差额 |
|---|---:|---:|---:|
| 1×20K | 40 / 22 | 0.233 GiB | -0.940 GiB |
| 1×32K | 56 / 28 | 0.362 GiB | -0.811 GiB |
| 32×20K | 1249 / 673 | 7.452 GiB | 6.279 GiB |

固定预算分配器可能仍占用相近HBM，收益表现为可容纳更多上下文，而非设备占用自动下降。小请求不能保证净节省；prefix共享和GDN checkpoint会改变账目。保持预算及候选策略不变，避免为了表面显存数字损害此前长prefill性能。

本轮没有可用的目标NPU或模型权重，也没有新增完整吞吐证据。q4/q1/C4原候选优化继续启用，未改变数学或放宽冻结验收；现有14分钟对8分钟等用户日志仍说明整轮目标未达成，局部CAModel/CPU数据不能用来宣布已接近原生。
