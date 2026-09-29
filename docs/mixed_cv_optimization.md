# 2026-09-29：64K与混合长历史的 balanced/C16 候选

依据：档案#126/#129/#143–#152、D.4完整原始打点；原文`reports/target_64k_mixed_20260929.txt`，计算口径`reports/performance_budget_20260929.json`。120×65100输入、每条512输出均成功；OSCAR3331.550s、原生1371.969s，墙钟差1959.581s。OSCAR平均TTFT快60.889s，TPOT却为1382.8ms vs212.1ms；TPOT包括被别的prefill阻塞的时间，不能称decode kernel慢6.52倍。容量31/12是用户观察，容量优势不能直接换算吞吐倍数。

### 已确认的开销和优化选择

1. **问题存在**：同31q4逻辑输入，N128/S3 CV11.328ms→N16384/S1 CV34.972ms；cold mixed CV35.656ms；warm mixed CV125.104ms、current FIA7.015ms。N和S同时变化，不能只归因split。CV是这些单层形状中的大项；它们不是完整服务各相位占比。
2. **所有者分配**：旧调度把连续21token作为遍历单元，31个q4 leader位于0,4,…120，在N16384/S1仅落到6/20核。独立`attention_cv_fast_balanced_out`把遍历单元改4；数学query组仍是128/GQA，prepare/task字节、所有KV范围及每query浮点顺序不变。全49152个task依然各访问一次，history leader变20核每核1–2个。旧N128/S3为18核/max6，新遍历可20核/max5，但固定循环成本增加；当前纯N128仍用已测fast_out。
3. **历史复用**：独立`attention_cv_fast_cluster16_out`从已测fast C4扩到16组完全相同request/head/split/context/KV范围、且kvend=context的query组。每组的FP32在线acc/max/sum、QK/PV/softmax、mask前有限性检查、错误覆盖和KV遍历次序不变；只改变组间执行次序和有界状态存放。first pass仍处理所有followers/非法/非共享组，但按4token遍历；second pass以16×queryTile的bucket分配唯一簇owner。
4. **两本账**：符合条件的组，C4→C16可把解包/packed读取次数再降到1/4；QK/PV、softmax及逐组spill/restore不会降到1/4。每核workspace D256为4,997,120B，20核99,942,400B；较C4每卡增加63,160,320B，在原生KV预算前预留且跨FULL层复用。UB形状与fast C4相同，无全历史恢复缓存。前沿/尾部/短组仍完整执行同一INT2 C1数学，绝无BF16历史替换。
5. **为什么不盲扩所有形状**：只有已知max_query_len≥8192才选C16（目标GQA6下有足够簇供20核）；较短或未知长query保留C4。later-MTP q1保留已验证fast_q1。原base路由只有N>128才选balanced。`experimental_mixed_cv`必须依赖history+fast；默认原配置/显式baseline仍保持既有fe0语义。candidate统一预设自动启用新两核，不把优化仅留在probe。
6. **收益的反证条件**：冷mixed可能无C16簇，却仍有扫描成本；C16增加状态与判定开销，balanced增加outer workitems，均可能抵消收益。旧C4已有27%左右局部提升，继续扩大复用不能假定再快4倍；也没有证据证明这两项足以消掉整程58.82%耗时。短NPU门必须逐例对比当前fast基线和新算子，实际不同算子的CV及整个attention包络均不允许回退；若失败即停。只分析与验证这些有证据的结构改动，不调整模型/GDN/MTP语义、不降低FP32精度，不改用户完整AISBench负载。

### D.4四问及验收边界

- 相位是融合history/window/current的CV，不是把INT2恢复到HBM后调用FIA。
- D.4的6.5秒恢复、约725ms host准备、209ms store均是要避免的结构；本轮不增加host逐请求循环/读回，不做整段BF16历史。
- 新核保持KV256有界展开和全部旧数学/同步，C16共享的是完全一致的读域；NPU图地址在建图前分配，不动态创建历史缓冲。
- 工作量上界是eligible C4块的读/解包再减75%、q4拥有者6→20；额外spill、扫描和未改计算须实测，不能把这些比例当作端到端速度。

短验证覆盖cold/warm混合、同cohort两种padding，以及65100历史的decode和48840历史加16260新token的长prefill；另测第一轮MTP draft保留完整source2 CV的路径，共8项；2次预热/5次ABBA NPU Event，逐位partial/LSE/status/merge、冻结oracle、active C16与balanced同地址改Q的独立图，非法shared metadata与单组QR NaN错误传播。`install_observe_serve.sh --variant candidate --probe-only --diagnose-mixed`自动包含新门，不加载模型、不重跑AISBench；一形状一条`PERF_MIXED_OPT`加最终结果。此为算子/单层attention实验，不是完整TP4模型图或8分钟/22.9分钟性能验收。

### 本轮本地验证与CPU超时修复

CANN ascend910b4编译通过，生产AscendC/ABI源码指纹与已编译版本一致。主机关键回归135 passed、36 skipped（28真NPU、1 Linux ELF、7不适用host组合）。官方CPU-debug最终9个完整用例通过：balanced q4的S1/S3、C16 D64有效簇/poison/尾部边界/非对齐请求、D256有效簇与小形状、非法shared metadata。数值是partial/LSE/status逐位对拍加独立冻结oracle。证据`reports/mixed_cv_local_validation.json`和`reports/mixed_cv_cpu_debug_20260929.json`。

原combined harness将fe0、C16、merge/oracle放入同一个120s预算，D256q336与非对齐q3+q336各出现过rc124。没有删除原例、缩小输入、放宽误差或调高120s。新`tools.run_mixed_cv_cpu_debug`先真实运行fe0并过oracle，写新建目录中的原始参考bytes，再独立运行C16逐位比/同oracle；每次执行仍限120s，核对golden及参考bytes SHA/尺寸且失败立即停止。相位计时证明D256 fe0 kernel约50.007s、C16约70.576s，两者相加已超过120s；含各自校验的两进程分别57.280/77.735s全部通过。非对齐完整例分别62.565/78.836s通过。旧超时日志与最终通过记录同时保存，不将模拟器耗时当NPU速度。该工具仅用于维护者本地CANN VM，不是真机部署前置命令。

新balanced/C16的真NPU精度、独立图、速度仍未在本机验证；新增目标短门须过关。已有旧fast/C4/q1真NPU通过不会代替新核证据，更不代表已经追平8分钟或本次1371.969秒。
