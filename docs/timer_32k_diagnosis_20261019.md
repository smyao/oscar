# 32K 小批量计时与瓶颈诊断

用户报告当前 OSCAR 可容纳约31条、原生约15条，但 OSCAR 总耗时仍更长。
本轮停止性能改造，先取得可对照的阶段与硬件证据。按用户随后要求，默认使用
**native BS4、OSCAR BS4、OSCAR BS8**，每条32768个输入token、256个输出token。
这些小批量结果用于定位，不能代替BS31或300请求的端到端结论。

## 一键运行

```bash
git pull --ff-only && bash scripts/test_timer_32k.sh
```

默认使用 `configs/target.json` 的原版 Qwen3.8-27B、0–3卡/8989。
保持 BF16、MTP3、APC关闭、16384预算、原生调度与graph配置；不改变可用KV比例或
`max_num_seqs` 来强制制造batch。BS4/8指客户端并发，实际每步调度B另行记录。

```bash
# 仅查看计划，不安装、不加载模型、不连接NPU
bash scripts/test_timer_32k.sh --plan

# 已授权的后四卡模式：4–7卡/7878
bash scripts/test_timer_32k.sh --rear-cards
```

脚本自动安装/复用编译产物、检查并复用或生成匹配的PT，然后依次管理原生与OSCAR
两个服务。服务启动后各做一个128→16短warmup；每个长请求组先跑采样关闭的control，
再跑同输入的profile重复，共32条长请求。相同BS4两侧精确token IDs相同，BS8前四条
也相同；使用不同请求盐。每条必须收到完整DONE、正确usage与256个生成token IDs。
默认每请求上限900秒，可显式设置 `--request-timeout`；不重试失败负载。
每个profile批次之后另发一个关闭采样的128→1短请求，让各rank落盘最终摘要；
这三次短请求独立保存，不计入配对批次耗时或指标。

正式服务和旧一键入口不会自动开启这套计时。诊断过程中原始编译/服务/profiler日志
全部保存，终端只显示简短进度、每60秒一行心跳、最终摘要。失败时显示最多60行错误
尾部，保留真实退出码。Ctrl-C只关闭本次拥有的服务进程组并检查资源释放。

## 需要复制什么

只复制下面两个标记之间的完整文本（包含标记）即可：

```text
=== OSCAR_TIMER_SUMMARY_BEGIN ===
...
=== OSCAR_TIMER_SUMMARY_END ===
```

摘要最多200行、50KiB；完整证据在终端第一行显示的 `logs/timer32k-.../`：
`run.json`、`timer-summary.json`、`timer-summary.txt`、每批请求/metrics和每rank的
step/window/profile产物。证据缺失时仍打印摘要并返回非零，不用手工翻设备trace。

## 断点与采样设计

每个rank、每次profile重复最多四个单步窗口：两次有至少16384历史的续prefill/mixed，
两次达到目标BS4/8的纯decode，分别采集 `PipeUtilization` 和 `Memory`。
冷的第一16K不耗尽续prefill配额；未达到目标batch或CPU上下文元数据缺失会显式报告。
每个窗口都有实际请求数、prompt/decode/draft token数、上下文范围、rank和图回放次数。

| 断点 | 解释与范围 |
|---|---|
| runner execute→sample | 主机提交包络，不冒充设备完成时间；profiler初始化/导出单列 |
| host update/prepare/metadata/bookkeeping | 状态更新、输入准备、metadata、采样记账；不读取设备张量数值 |
| target forward / graph replay / MTP proposal与forward | 保留原生graph与MTP；父子包络不重复相加 |
| native FULL / GDN | 原生路径分别标记，实际设备内核另由trace识别 |
| prepare / q_rotation / query_validation | OSCAR任务准备、BF16 Q旋转、查询校验 |
| fia / current_native_fia / merge | 压缩attention、当前块原生FIA、LSE合并 |
| kv_rotation / int2_quantize_store / guard | BF16 K/V旋转、量化打包与exact窗口写入、状态检查 |
| device GEMM / HCCL / copy / unknown top kernels | 记录模型公共成本与通信；没有关联证据的MatMul不强称MLP |

图回放绕过Python内层断点，因此通过真实设备trace识别图内kernel，不把图外host计时
假造为图内耗时。fused CV内的解包、QK/PV、softmax和窗口部分没有被拆成新算子；
只能报告该融合kernel整体耗时及其硬件计数器，不能虚构每个内部子阶段的毫秒数。

原档案#133曾记录HTTP start/stop profiler跨线程和RECORD状态停止造成崩溃，本轮不使用
HTTP profiler。worker在实际runner同一线程创建/start/step/export/stop，单步schedule
走 `RECORD_AND_SAVE → step → NONE → stop`。不向捕获图中添加Event或相位同步；
profiler自身的导出会扰动时序，故只用于归因。

## 如何分析结果

- **native4→OSCAR4**：同输入、同并发的control比较TTFT/TPOT/总耗时；再看对应采样中
  OSCAR旋转/写入/压缩attention/merge的增加，以及GDN、通用GEMM、HCCL和host开销变化。
- **OSCAR4→OSCAR8**：看每秒有效输出token与实际调度B、MTP接受和preemption。
  两组请求数不同，不能直接把批次wall比值当提速倍率，也不能外推31条时的收益。
- **计算或搬运候选**：Pipe组读取Cube/MAC、Vector、MTE2/MTE3比例；Memory组保留实际
  main-memory/L2/L1/UB带宽字段及原单位。`MTE2/max(Cube,Vector)`只表示相对流水线占比，
  不等于已证明HBM瓶颈。Vector很忙也可能是解包、量化、softmax等算术，不能只看Cube。
- **证据配对**：Pipe和Memory来自不同窗口，只有bucket、token/context几何、rank相符
  才进行阶段对照；缺字段、未知核、PMU不支持、缺rank或未匹配均明确保留。
  每rank分别计算设备busy区间并集，绝不把四rank之和当服务wall；task相加存在重叠，
  task-share也不是端到端占比。

control关闭profile/timer采样，但保留少量相同的空观察hook与每runner步控制检查；
它是本诊断内的配对对照，不是零插桩AISBench验收。profile wall不参与正常吞吐比较。
若计数器不足以区分HBM、片内搬运、算术或同步，摘要给出 `inconclusive` 与缺失依据，
待回传后分析，不能仅凭136/144B压缩比或BF16标签断言memory-bound。

计数器口径依据官方CANN的 [op_summary字段说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/devaids/Profiling/atlasprofiling_16_0067.html)
和 [AiCore采样指标](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC3alpha003/devaids/auxiliarydevtool/atlasprofiling_16_0033.html)。
采用真实 `Task Duration`；`aicore_time` 等理论统计不作为服务wall。

## 本地验证边界

功能回归覆盖采样开关、原生worker只观察注册、warm-history/满批筛选、profiler状态机、
缺产物/错误、配对输入/完成校验、quiet日志/中断清理、CSV/trace去重、单位、父子与rank
计账和摘要限长。执行回执见 `reports/timer_32k_20261019/summary.json`。
本轮没有连接真机，没有实测NPU计时、硬件计数器或性能结果；没有修改CANN、冻结guard、
精度容差、KV格式或算法。只交付 `codex/oscar-bf16-attention`，main/stable不变。

125项主机回归已通过；官方torch_npu源码的同步导出参数与schedule接口也已只读核对。
当前目标CANN/torch_npu是否能完整导出这四类窗口，仍以用户运行产物为准；若native
profiler失败，保留首错和不完整窗口，脚本有界退出并清理，不能冒充采样成功。
