# 默认小预算下的完整 C4 partition：当前分母不变

档案 #129/#132–135/#145/#148/#154/#155，启动文档 D.4。仅宿主结构与条件预算；没有新CANN修改、VM运行、原子加速收益或模型总分钟预测。

## 1. 范围与真实元数据校验

假定引擎实际预算为2048。纯单请求prefill给Q2048；若先分离31个q4，长后缀给Q1924。`mixed_decode_split.py:55–68,84–88`把后缀qstarts重基到0，实际query/task长度为1924；仅`cv_shape_tokens=2048`保留原split选择。`runtime.py:splits_for_tokens`对此GQA6/20core得到S1。**预算2048本身仍须以引擎实际日志确认，不由本报告设置。**

`full_partition_budget.py`从当前`attention_tasks.cpp`及两遍C4调度公式推导全部3Q任务；公式输出已与实际Q336/C769的1008×16个int64任务**逐字节相同**。Q1924/Q2048的新shape未运行真实prepare，当前是源码公式证据。

source2仍按生产规则仅清空健康任务的KV区间，全部任务、follower状态、window及输出位置保留。报告不包含另外的31个q4 kernel、native current FIA、rotate/store/merge或其他模型层。

## 2. 当前代码实际的工作归属

| Q | query组 | 前沿C1 | 尾query组C1 | 成熟C4 | 原history预算关键核 |
|---|---:|---:|---:|---:|---|
| 2048 | 97个21行组＋尾11 | 12 | 2 | 21 | core3：1 C1＋2 C4 |
| 1924 | 91个21行组＋尾13 | 12 | 4 | 19 | core8–11：2 C1＋1 C4 |

前沿C1起点0–231；成熟C4首簇起点252。Q2048 C4 bucket3–23经`bucket%20`使core3分到两簇；尾组2016/2037原属core16/17。Q1924尾组1848/1869/1890/1911原属core8/9/10/11，叠上各核前沿C1与一个C4，成为新瓶颈。

当前已测full256单位成本：C1第二个完整KV周期131,472；C4稳态周期390,694。跨成员流水加有界next-KV预取的**条件预算**为216,547/C4共享KV。C1未获得任何假设加速。

last query组M66/M78没有对应计时，表中的数值模型暂用M126作代价代理。这不是已证明的上界。尤其Q1924/context65100的proxy会选core11；已知完整M126工作最重的是core8–10，实际关键核仍在此组附近待设备确认。

## 3. 修正比较分母，分别列三种方案

所有方案都以**当前未重分配代码**的每核最大history工作量为分母；不会给baseline预先均衡。

| 形状 | 只做C4流水/预取 | 只重分配，C4不变 | 两者组合 |
|---|---:|---:|---:|
| Q2048，context20000 | 1.617× | 1.168× | **2.108×** |
| Q2048，context65100 | 1.617× | 1.168× | **2.107×** |
| Q1924，context20000 | 1.363× | 1.252× | **1.878×** |
| Q1924，context65100 | 1.363× | 1.252× | **1.878×** |

这是所有核的完整full-history工作量比较，不是只报一个C4簇。partial KV与固定成本暂显式留在未知项，故不是完整kernel实测加速。

重分配列有明确host LPT job assignment witness：同一history leader集合，无丢失/重复。Q2048把C1从“双C4核”迁开，组合关键核只承担两个C4；Q1924消除“两个C1与一个C4挤同核”，组合最大history负荷约为一个C1加一个C4。某个无C4核可能分到两个C1，但总权重低于关键核。

先前`(1C1+1C4_old)/(1C1+1C4_new)=1.5004`只能衡量**双方都已经均衡以后**的增量流水收益，不能用于否定联合方案。联合Q1924正确比较为`(2C1+1C4_old)/(1C1+1C4_new)=1.8782`。

## 4. 固定、尾部与planner成本不能归零

完整kernel的条件公式是：

`old = max_i(A_i + F_old_i + T_old_i)`

`new = max_i(B_i + F_new_i + T_new_i + P_i)`

- `A/B`：逐核full256历史工作，见JSON完整job清单。
- `F`：query读取、两遍metadata/follower扫描、精确window、空current任务、初始state、最终旋转/归一化/写回、启动排空。重分配会改变各核固定任务数，不能假定仍一样。
- `T`：每个partial KV单元及M66/M78尾query代价差异。context20000成熟tail为224行；context65100为12行，不能套用已测193行tail比例。
- `P`：GPU work-plan创建、压缩、分配及额外metadata/计划读取/同步成本。尚未实现或测量，不能写零。

仅在“各核统一共同F（含尾部代理），候选额外planner P”场景，至少1.5×需要 **`F+3P ≤ 2A_max−3B_max`**：

| 组合方案 | F+3P允许的周期上限 |
|---|---:|
| Q2048/C20000 | 40,535,726 |
| Q2048/C65100 | 133,452,308 |
| Q1924/C20000 | 20,267,863 |
| Q1924/C65100 | 66,857,626 |

这些裕量支持继续验证联合方案；不证明实际F/T/P落在范围内，也不证明短history测得的单位服务周期在20K/65K、不同GM争用下保持不变。只有pipeline-only时，Q1924连F/T/P设零的proxy也不到1.5×。

## 5. 通用GPU work-plan必须先保住的语义

当前assignment witness只证明健康R1、连续正slot形状的history leader集合完整，**没有完成通用非法任务/图生命周期证明**。实现前至少需要：

1. 从设备上真实3Q任务构造计划，完整沿用C4同request/head/split/context/KV范围、完整qcount和错误谓词。
2. 每个原history leader恰好属于一个C1或某个C4成员；计划只变执行owner，不变输出row/task id。
3. 全部qcount=0 follower仍按原逻辑写两条status；非法、前沿、尾组仍执行原验证及PublishEmpty/PublishStatus，不能被压缩掉。
4. source1及已抑制source2仍有完整输出/LSE/status责任。诊断clusterStats的owner期望必须更新，数值容差不能改。
5. 计划固定容量、同stream发布/消费、graph重放地址与代际正确；无host `.item()`读回，无原生vLLM预算或GDN改动。
6. 实际测量planner P及metadata变化，组合候选再做完整同名kernel验证；不把数学独立性或host LPT witness冒充已实现GPU调度。

证据文件：`reports/c4_phase_profile_q336_20261009/full_partition_conditional_budget.json`。所有核原/新job分配、full/tail量、固定成本条件和源hash均保留；整marker门仍失败，不被本预算放行。
