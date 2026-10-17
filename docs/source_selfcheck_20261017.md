# 源码自检、修复与剩余边界（2026-10-17）

母版本 `21efbb7`，分支 `oscar-new-3`。延续 PD 混合调度审查，按可执行源码及实际测试判定，不以注释、README 或主机通过替代设备证据。关联档案 #34/#36/#69/#126/#140/#142/#145/#148/#151/#154/#157–165；复读 D.4 原始打点，没有新真机事件，不追加档案编号。

**本轮关闭三处可复现缺口，重点回归 883 通过、1 跳过；全仓仍有 32 个失败节点。不能声明 OSCAR 已无 bug，也不能声明 32K Trap 或既有 BF16 数值失败已解决。**

## 修复与影响

| 项目 | 源码问题与修复 | 运行影响及验证 |
|---|---|---|
| CPU 元数据镜像 | `metadata.from_common` 命中累计长度缓存后未再次验证 CPU mirror；current-only 分支可直接对非法设备镜像 `tolist()`。新增 `cpu_metadata` 唯一转换点，设备、int32/int64、维度、长度检查先于读取；current/mixed 分流共用 | 生产 host 边界加固，默认 candidate 自动生效。正常原生 producer 提供 CPU 镜像，不能称为已知 NPU Trap 根因。新增检查读取 tensor 属性，无设备回读/同步、无 KV 数值变化；未测服务耗时。用母版真实函数 AST 复现非法镜像被接受/尝试读取，修复后零数值读取即报错 |
| 显式 APC 探针 | reader 选择漏传 BF16，workspace 仍是旧 FP32 MDL 大小；源签名未绑定实际 BF16 helper | 先按生产配置归一化，选择 BF16 MDL，D256 每 core 从 1,146,880 B 修正为 1,720,320 B，绑定实际源码。CPU 实际分配/路由/负例通过；真实 NPU probe 未运行。默认 APC=false 不变；正式服务原 allocator 已正确，不能把探针问题解释为服务 Trap |
| 离线显存账本 | 构造 WorkspaceGeometry 时漏传 BF16/C4 pipeline | 20 Cube、C16 的 CV arena 正确计为 149,913,600 B，旧公式为 99,942,400 B；补实际公式回归。只修离线计量，不改变服务分配 |

同步更新过期主机测试，使其验证当前显式 16384 预算、原生 scheduler、APC=false、实际 target 设备/端口、BF16 q4/q1/C4 reader，以及旧 FP32 生产入口拒绝。探针编排测试隔离设备执行，但仍保留失败禁止启动服务的断言；插件 readiness 测试显式清除 runtime，避免把本机 CANN 缺失与 provider 缺失混为一谈。

修改没有引入新开关；直启和 observe 仍共用 `tools/serving_variants.py`。两入口配置传递和实际 q4/q1/C4 选择已测试。未修改 CANN 核、FE0、原生参考、INT2 格式、旋转/softmax/合并或冻结精度阈值，未重跑 AISBench。

## 本地测试与证据

| 检查 | 结果 | 限制 |
|---|---|---|
| 最终重点回归 | **883 passed / 1 skipped / 0 failed**，18.71 秒 | 前轮生产/PD/MTP 套件加本轮所有修改和新增测试；重复 large FULL shape 跳过一项。与全仓结果重叠，不能相加 |
| 全仓最终回归 | **1819 passed / 18 failed / 14 errors / 134 skipped**；另有 29 subtests passed | 真实退出码 1；没有隐藏失败或因本轮审查新增 skip |
| 六路 BF16 地址/控制流 | 12 项通过，包含 C4 的 5 种 tile 数 × 16 个固定随机交错 | 真实提取 Init/workspace/Cube 函数与 C4 Vector 控制流，检查区间、member/round、flag、spill→atomic→restore；alias/缺失 PReady 负控拒绝。算术、DMA 和完成是 stub，mode2 双 AIV token 为明确假设；不是 CANN、官方 CPU-debug 或 NPU 验证 |
| 冻结/派生检查 | FE0 7/7 哈希一致；5 份派生 C++ 对应 6 路算子生成检查通过；42 个 checkpoint ID 不变 | q1 与 q4 共用 MDL 实现，不能把 6 路误写成 6 份生成源 |

[完整回执和失败节点](../reports/source_selfcheck_summary_20261017.json)、[全仓 XML](../reports/source_selfcheck_final_20261017.xml)、[重点 XML](../reports/source_selfcheck_scoped_20261017.xml)、[镜像问题前后对拍](../reports/cpu_metadata_mirror_witness_20261017.json)。初始全仓失败、沙箱回环端口限制及 Clang/SDK 环境失败分别保留在 `reports/source_selfcheck_*`，未覆盖为成功。正确 Xcode/SDK 配置后相关 C++ 用例通过；本机 loopback 测试获准绑定端口后 70 项通过。

全仓 32 个失败节点已逐项归类：

- **17 个缺历史文件**：8 个 64K 日志分析 setup、7 个 CAModel 完成日志分析、2 个原生服务日志分析；不伪造缺失日志。
- **13 个旧研究布局用例**：生产 BF16 正确拒绝 request-window/不完整依赖组合；这些测试尚未迁移到独立研究组件。不能绕过生产校验构造 provider 后宣称生命周期通过。
- **2 个旧实验静态规范门**：实验文件存在空异常处理和档案头缺失。检查保留，未扩大白名单。它们不是本轮新增生产算术缺陷，但全仓验收仍未通过。

复跑本轮重点套件（仅当前已配置的 Mac 开发机，不是真机部署前置步骤）：

```bash
.venv/bin/python - <<'PY'
import json, os, subprocess
with open('reports/source_selfcheck_scoped_command_20261017.json') as f:
    command = json.load(f)
raise SystemExit(subprocess.run(command['argv'], env={**os.environ, **command['env']}).returncode)
PY
```

## 尚未闭环

1. 冻结数值集仍为 **BF16 7/11 失败，其中 2 例最终输出失败**；旧 FP32 comparator 全过。BF16 输入舍入带来的差异不能靠改测试阈值消除。本轮没有数值算法改动，未重复已有筛查或将它记为通过。
2. 32K 故障 #165 缺当前 `site/family/code`；异步 GDN/event 栈不能定位首错，本轮不是该故障的确认修复。
3. Matmul `End/Reset/re-Init` 的目标 SDK 资源生命周期未核实；所需绑定 SDK 归档本机缺失，主机 stub 不足以判定合法。
4. 本机无 CANN/NPU/torch_npu/Lima，设备完成、图捕获/回放、实际服务、模型质量和性能均未运行。本轮不申领性能提升或设备资格。

本轮可证实的源码缺口已修复并回归；剩余证据缺口仍在，因此“源码阅读层面没有任何 bug”不是可给出的严谨结论。
