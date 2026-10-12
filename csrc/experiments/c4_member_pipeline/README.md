# C4 跨成员流水：独立研究原型

档案 #13–22/#126/#140/#145/#148/#151/#154–156，启动文档 §0.7/D.4。只使用当前工程生产 C4 的已验证 H32 可逆派生源；不读取失败工程源码，不修改生产、references、容差或服务配置。

## 资格边界

当前仅完成源码生成/逆向恢复、host Python/C++ driver 语法和离线 oracle 导出。CPU、CANN、CAModel、真 NPU、graph 与整轮速度**均未在本原型执行**。atomic v1 完成 K 后的位级语义已有独立证据；**v2 的 AIV 即时回读以及 atomic→普通写回接缝必须由 root 先核实**，不能拿源码检查代替。

已有 H32 helper 为2.346×，但完整 Q336/C113 C4 只有1.006×。本原型检验跨成员流水/atomic 的增量，不重复包装 H32 helper 倍数。baseline 和 candidate 都从已验证 H32 完整源生成，双臂同库、同全部任务；与生产 H16 的最终资格仍单独报告。

## 唯一改动范围

- baseline：H32 C4 原算法，仅研究符号/include。
- candidate：仅扩大 GM stride、增加第二个 KV256 和 score/P 槽、替换成熟 C4 的 `CubeCluster/VectorCluster`，新增槽选择、KV加载、atomic Matmul 三个 helper。
- `Process/IsClusterMember`、所有 C1/precise 数学和握手、Softmax、Matmul类型/config/K128顺序、HF32关闭、state spill/restore、输出/status发布均逐字保留。task/core归属、全部1008任务不变。
- 每请求仍按原 KV256 顺序更新 FP32 max/sum/acc；没有提前合并、原子多生产者竞争或数学精度变更。H32保留旧 alternating16-row lane错误域。

`generate.py --check` 先调用原 H32 生成器，再核对文件身份。新改动可逆恢复到该源；所有未改方法和 Matmul类型/config 整段逐字对照。`generate_entrypoints.py --check` 对已有完整 C4 的 driver 做明确有限变换并逆向恢复，保留 oracle/guards/status/完成/profile/比较门。

## buffer 与 flag 生命周期

原每 core 1839104B GM；新增 KV512行×D256×FP32=524288B、score128×256×FP32=131072B，共655360B，candidate每core2494464B，20核47.578MiB。两臂driver都分配这个容量，baseline仍用原stride。UB仍157760B/AIV，无新缓冲或 Softmax容量变化。

| 存储 | 生产者→消费者 | 覆写条件 |
| --- | --- | --- |
| KV槽0/1 | 两AIV有界H32解包→Cube | 写下一槽时Cube仅读当前槽；旧槽在上一RoundDone后已无人读 |
| score/P槽0/1 | Cube QK→两AIV Softmax→Cube PV | 同槽前一个PV的 `IterateAll<true>`＋`End`完成后才下一QK |
| member accState | AIV Spill(alphaAcc)→Cube完整K的FP32 atomic | PReady前Spill的MTE3完成；下一轮Restore前RoundDone |
| member stats | AIV Softmax→Spill→下轮Restore | 各成员独立slice，RoundDone前不恢复本轮atomic中的acc |
| UB packed/natural/dequant | 当前成员Softmax / 下一KV解包分时使用 | P3与所有alphaState发布完成后才预取下一KV |

每轮Cube顺序：`QK0,QK1,PV0atomic,QK2,PV1atomic,QK3,PV2atomic,PV3atomic`。AIV顺序：每成员 `WaitScore→Restore→原Softmax→Spill(alphaAcc/stats)→PReady`；P3后可预取下一KV，最终 `WaitRoundDone`。atomic只写真实M126，不写原来未参与Add的padding行。

| flag | 每AIV每round | AIC每round |
| --- | --- | --- |
| KV8 | set1（首轮预载；后续在前轮hasNext） | wait1，沿原mode2聚合两lane |
| score9/10 | 各wait2 | 各set2，PIPE_FIX |
| P11/12 | 各set2，Spill完成后的PIPE_MTE3 | 各wait2 |
| Round13 | wait1 | set1，PV3后的PIPE_FIX |

两AIV都完整执行上述协议。末轮不发下一KV8，最后Round13已消费，score/P全部消费后才复用原8/9作各成员norm/Rv/writeback。每个新cluster重新选择slot0；C1不调用atomic。

原KV错误顺序是 `memberErrors→本轮kvError→Softmax error`。下一KV预取用独立 `kvError`，不写任何当前memberErrors；只在下轮按原顺序覆盖。数值错误和无效metadata不跳过。atomic→normal是否彻底恢复普通写模式是前置核实项，不能因为默认参数为0就宣称设备已验证。

## 最小实验，不扩大目标

- Q336/R1/Hq6/Hkv1/D256/S1/20core。C113：成熟history[64,113)，一轮；C337：[64,337)，full256＋tail17两轮。两者保留真实prepare的全部1008task及前沿C1、精确窗口；只按生产prefill规则将合法source2 end置begin。
- C113 valid用于末轮drain/atomic→Rv；已有bad80/bad96是原lane1/0错误域。C337 valid用于双buffer；bad336用于下一轮预取错误域。只按前项成功后再推进，不自动重试。
- 离线独立oracle覆盖所有336query的source0+1。source2/native FIA不是本次时间范围。无效metadata域对照所有status与partial/LSE位，不把它宣称有效数值oracle。
- CPU每臂120秒；CAModel每臂900秒，不延长、不重跑Q1344。使用VM本地有界supervisor和同一flock；本子代理不自行启动VM。
- C113是功能最小点，C337是首/末轮最小点，都不代表Q2048/K32或64K性能。完整核≥1.5须另包含所有C1/窗口/固定成本，不能只报成熟簇倍率。

已导出的host-only fixture：

```text
/private/tmp/oscar-c4-member-pipeline-c113-valid-20261010
/private/tmp/oscar-c4-member-pipeline-c337-valid-20261010
```

## 入口（root串行安排）

```bash
python3 csrc/experiments/c4_member_pipeline/generate.py --check
python3 csrc/experiments/c4_member_pipeline/generate_entrypoints.py --check
```

CMake source目录是本目录，设 `ASCEND_HOME_PATH` 与 `SOC_VERSION=ascend910b4`。CPU额外 `-DOSCAR_C4_PIPELINE_CPU_DEBUG=ON`，得到 `oscar_c4_member_pipeline_cpu FIXTURE NEW_OUTPUT baseline|candidate`。CANN得到同库 `liboscar_c4_member_pipeline.so`。两臂可分别完成、各自立即保存真实prepare checkpoint与完整输出，失败不能采用局部ticks。

```bash
python3 csrc/experiments/c4_member_pipeline/cv_replay.py --variant baseline --fixture FIXTURE --prepared CPU_BASELINE_OUTPUT --library LIBRARY --output NEW_OUTPUT
python3 csrc/experiments/c4_member_pipeline/cv_compare.py --old BASELINE --new CANDIDATE --old-control BASELINE_CONTROL --new-control CANDIDATE_CONTROL --output COMPARISON
```

## 收益依据与 D.4 四问

已有C4完整闭合full256循环390694cycles；此前跨member＋nextKV条件预算216547cycles（约1.804×成熟循环），不是本原型测量。atomic可改变GM流量与等待，不当作零成本；H32倍率不相乘。原任务归属不变，C1会保留完整核关键路径，因此要以完整kernel配对决定继续或淘汰，不宣称整轮2–3倍。

对应D.4 fused history dequant/FIA；全历史恢复6499.8–6655.1ms的失败原因在本方案由固定两KV256/两score槽避免。FP32、真实task/domain/status保持，新增640KiB/core不随history增大。目标是可证的≥1.5长history完整算子条件收益，未将CAModel cycles换算成NPU毫秒或模型分钟。
