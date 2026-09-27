# fe0 基准上的端到端诊断与 C4 历史复用

依据：档案 #126/#129/#140–#145/#148、启动文档 D.4、当前项目 fe0 源码及用户回传。fe0已由用户确认精度正确；本轮不改变这一基准结论。

## 为什么22分钟不能靠此前的小改动追到8分钟

按用户给的22:22与约8分钟作量级预算，需要消掉约862秒、64.2%的总耗时。K4、单算子和同步诊断不能直接换算成127题AISBench的占比；输出长度、MTP轮数、实际批形状、图decode和排队都必须计入。

即便被优化部分恰好加速4倍，按Amdahl定律，也须它原本占整轮至少85.6%，单靠这一项才可能到8分钟。C4只减少合格历史组的加载/展开，不加速全部CV，更不加速q4 decode；现有证据未证明85.6%这个条件。因此这是一项能由真机立即证伪的结构性候选，不能把“4次变1次”当作已经给出完整追平保证。

fe0的长prefill每21个query一组，每组扫描相同历史。每256KV/D256工作单元读取34KiB INT2，却反复展开并写512KiB FP32 K/V供Cube读取。GM逻辑通信不等于实测HBM流量，但重复展开次数真实存在。此前的行检查、metadata和小DMA优化没有消掉这个规模，端到端收益因此可能很小。

## 本轮实验改变什么

新`attention_cv_cluster4_out`与原`attention_cv_out`是两个独立算子。旧kernel文件不改，新文件从本项目fe0复制其数学函数，保持Unpack、metadata校验、Softmax、QK/PV和最终旋转的原有计算方式。

只将同一request、kvhead、source0、split、context、kvbegin、kvend的4个完整query组组成簇，且要求kvend=context。每个256KV历史块解包一次，给4组各自的QK→softmax→PV使用。每组独立保存FP32 accumulator/max/sum，并保持该组原KV遍历顺序；窗口/current源不共享。

不同区间不能取union后只靠mask：fe0在mask之前检查整行有限性，padding域变化会改变溢出和错误语义。frontier、不足4组、非法或padding任务按原INT2单组执行，绝无原生BF16替代路径。

两遍只读调度保持唯一输出owner：原schedule处理followers和非共享任务，第二遍把连续84-token bucket分配给各Cube，寻找request-relative簇anchor。避免anchor stride4与20Cube产生公因子而只使用5个核心。

### 两本账

- 每个合格簇展开次数4→1，最多节省该簇75%的历史解包调用；这不是75%的端到端提速。
- 新增FP32状态读写。C4每簇每KVtile的粗略逻辑GM净节约约630KiB，未扣m/l与索引成本，也不等于HBM测量。
- D256 workspace每Cube从917504B变为1839104B，显式候选模式在原生KV预算前分配；不随历史长度分配完整恢复张量。
- QK/PV浮点工作总量不变，spill/L2/任务扫描与并行损失可能抵消收益。因此同输入速度门严格执行，达不到就不启动候选服务。

## 精度与收益门

`tools/probe_history_reuse.py`在同一签名扩展中运行fe0两次，再运行C4：

1. fe0自身同输入必须可逐位重复；不满足时标缺证据，不放宽容差。
2. 全部partial、LSE、status及合并输出逐位相等，普通有效样本再过独立冻结oracle。
3. 成熟长历史必须真的产生簇；frontier、请求边界、不同split不得误共享。未对齐请求起点3的多head/split病例还将terminal split改为非terminal做反证。
4. 共享KV非法metadata、仅一组QR NaN、poison及同地址改输入的独立NPU图回放均覆盖。
5. 正常2次预热、5次NPU Event测速，交替AB/BA顺序；原速度门不放宽。
6. 候选服务仅接受本轮签名/产物/源码匹配且图和精度、速度门通过的报告。默认仍fe0；显式候选只用于用户评测，不等于生产质量或全场景速度验收。

本地CANN编译与官方CPU-debug最终51项通过，包含C4实际参与时与fe0的逐字节对照。同一过重形状的两个CPU模拟器配置达到120s限制，已保留记录；以D64长多tile、D128双head、D256大维度的互补病例验证，未提高时限。相关主机回归95 passed、28 NPU skipped、4 subtests passed。完整记录见`reports/history_reuse_cpu_validation.json`。目标NPU、完整模型图、用户模型质量及8分钟性能仍需真机裁决。

## 如何找出端到端剩余时间

`install_observe_serve.sh`不替用户发AISBench请求。默认baseline为fe0，candidate显式选择C4，native禁用OSCAR数学路径；启动前锁定fe0原内核哈希，候选也保留这些原内核，并记录实验文件及构建签名。

worker在非dummy、非capture阶段按prefill/decode/mixed配额稀疏采样，异步Event仅在完成后读时间；pending满停止新采样，保留未完成handles，不强制同步或在热路径销毁。没有每相位同步，也不使用旧HTTP profiler。

记录模型step、target forward、MTP forward/proposal、整图replay及FULL路径各相位。原生通过其attention backend的只读wrapper测`native_attention`；对同一rank/step/stream求区间并集后再计算其余时间。OSCAR路径时间包含被替换的attention工作，不能全部称为“额外开销”。

图回放无法由Python拆出图内OSCAR/GDN，故只保留已完成的step/graph耗时、细项标missing。跨stream/线程配对失败仅保留可证局部包络；不能跨rank、跨轮相减。/metrics另记录Running、Waiting、KV、preemptions与MTP增量；这些是一秒采样，不伪作逐step事实。

用户只需回传`PERF_HISTORY_REUSE_*`与`OBSERVE_WINDOW_DONE`，停止后另有`OBSERVE_STOPPED`（失败为`OBSERVE_FAILED`）。完整JSONL保存每个样本、覆盖和缺失原因；控制台保持少量行。

## 下一步裁决

若C4展开次数显著下降而整个算子没有净收益，停止该候选；若算子有益但端到端仍远离8分钟，按实际同负载的attention、其余模型路径、MTP轮数、图decode与排队数据继续定位。精确整数展开或更细任务分配只在数据证明其预算足够后实施。本轮不宣称C4已追平，也不以小样本代替用户的最终质量验收。
