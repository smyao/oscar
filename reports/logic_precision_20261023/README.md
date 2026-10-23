# 本地 attention 数学链独立核验

基线为 `a2d9b42f` 的 CANN 源码。本轮未修改生产核、冻结 guard、timer 或原 oracle；后续启动策略提交不改变这些核。**q4、C4、C16 三个有界 AscendC CPU-debug 用例完成，但尚未完成“小于 1 点”的模型质量目标，不能据此写全部逻辑无错。**

## 补上的覆盖

旧 v4 C4/C16 fixture 使用 context=96、sink=64，成熟簇只处理 32 个历史 token，没有覆盖跨第二个 KV256 tile 的 `alpha * old_acc + PV`。本次三个 family 使用同一数学构造：

- D=256、GQA=6、context=704、sink=64、recent=16、splits=1；可见 history=625，三个 tile 分别可见 256/256/113 个 token。
- 每行查询系数轮换 `0, 1/8, 1/4, 1/2, 1, -1/8`，历史三个 tile 的基础 logit 为 `0, 6, 12`。它同时检验 alpha 的逐行布局、最大值更新和旧累积值缩放。
- V 是四组 64 维的分段 one-hot（非零值为 3）。K/V 使用可精确表示的 INT2 code 0/3、零点 -0 和二进制 scale；排除该构造中的 codec 不确定性。
- current 的基础 logit 轮换 `0, 6, 12, 24`，较大的 future logit 使因果 mask 错误可直接被输出发现；最后一行 padding 输入为 NaN。
- q4=4 个有效 query；C4=85，含一个成熟四成员簇及一个独立尾任务；C16=337，含一个成熟十六成员簇及一个独立尾任务。

预期值由新脚本 `closed_form` 按四个 V 类别直接计算 FP64 稠密 softmax，**不导入生产 attention、task/codec oracle 或旧 golden，不使用实际 CANN task/output 生成预期值**。既有 CPU harness 的 `official_expected_*` 文件名仅是复用接口；内容来自新解析构造。

## 实际结果

| family | history 无 P 舍入行最大误差 | 合并输出对 FP64 最大误差 | 最大 LSE 误差 |
| --- | ---: | ---: | ---: |
| q4 | 1.19209e-7 | 0.000474274 | 9.53674e-7 |
| C4 | 1.19209e-7 | 0.000475883 | 1.90735e-6 |
| C16 | 1.19209e-7 | 0.000641227 | 1.90735e-6 |

非负查询系数的 history 每个 tile 内分数相同、最大值不下降，故 P=1 精确表示，额外检查误差 <2e-5，直接隔离 alpha/acc 链。其余输出保持原冻结逐元素门 `0.005 + 0.005*abs(expected)`，没有修改或放宽旧门。padding 精确 +0/-inf、future mask、所有 status、GM 边界、只读输入、成熟簇/独立尾任务计数均通过。12 个 phase 全部 rc=0，无 timeout。

最大 partial 差异为 0.0031080246，位于 q4 token0/head1/source1。该 exact-window 的 V 恒为 3，64 个 sink logit=0，15 个 recent logit=1.5。FP32 denominator 使用 `exp(-1.5)`，而 PV 使用它的 BF16 值 0.22265625，因此解析结果为：

```text
3 * (64 * 0.22265625 + 15) / (64 * exp(-1.5) + 15)
= 2.9968924275
CANN = 2.9968919754；FP64 dense = 3
```

这证明现有 BF16 P/FP32 denominator 合同不严格保持常量 V，但差异符合舍入顺序，**不是本轮发现的新实现 bug**。不能据此擅自更改归一化或声称它解释模型的若干准确率点。

## 源码与 SDK 核实

- q4：`attention_cv_official_decode.cpp:475` 仅一次 QK scale；570/578 为 SoftmaxFlashV2 与 alpha 广播；649/773 归一化；792 计算 `max+log(sum)`。
- C4：`attention_cv_official_cluster4.cpp:791/868/876/1028/1048` 对应相同链；490 的 FP32 atomic PV 累加由本次成熟簇跨 tile 用例实际覆盖。
- C16：`attention_cv_official_cluster16.cpp:821/913/921/1073/1093` 对应相同链；498 的 atomic 累加同样实际覆盖。
- `merge_lse.cpp:66-83` 以 FP32 LSE 权重合并；本次独立 dense 预期同时覆盖 source0/1/2。
- 本机 CANN 9.1.0 的 `softmax_flashv2_base_impl.h:177` 将 WITHOUT_BRC 分派到 M1。`membase/v220/softmax_flashv2_impl.h:624-626` 对 `expMaxTensor[offset2]` 写出，并令 `offset2 += tiling.reduceM`；alpha 是每行一个连续值，现有 Brcb 读取方式与此一致。本次不同查询系数的实际跨 tile 结果也排除了该用例中的 alpha 布局疑点。

复用了此前完成构建的两个 CPU executor，没有重编无关扩展。25 个实际编译依赖与本地当前源码哈希一致，binary SHA 和完整列表见 `evidence/summary.json`。SDK 哈希及完成后锁检查见 `sdk_and_cleanup.json`。Lima 任务使用 `/tmp/oscar-cann-validation.lock` 独占；完成后重新获取锁成功。

首次 r1 在执行任何 CPU case 前因新 runner 没有规范化构建依赖中的 `../` 路径而拒绝，见 `preflight_r1.json`；只修 runner 路径解析，在新的 r2 目录运行，未改核或覆写失败记录。

## 复现与限制

新脚本位于 `csrc/experiments/logic_precision_20261023/`。将其放入新的本机 Lima 临时目录，使用已配置 CPU 环境执行 `python -u run_guest.py`；它严格绑定现存 `/tmp/oscar-official-math-20261021/build/official-cpu/` 的源码依赖，不匹配即失败，每个 CV phase 的超时固定为 180 秒。小日志和解析 JSON 已保存在本目录，不提交大 fixture/output 二进制。

本次是 R16 的通用多 tile 数学检查，**未覆盖 R256/context944 的额外全 masked 尾 tile，也不是完整 NPU、异步硬件竞争、图捕获/回放、真实请求、模型质量或性能验收**。依据档案 #126/#145，CPU 通过不能排除目标硬件上的异步所有权问题；依据 #13-20/#148，LSE/算子通过不能代替模型输出质量。D.4 旧全历史恢复 6499.8–6655.1ms 的结构仍禁止，本次只执行有界 reader/softmax/PV/merge 功能，不产生速度结论。
