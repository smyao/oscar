# 保留FP32量化元数据的读端精度

本轮基于48be7610继续处理用户“小于1点”的目标。默认推理与e2a1a97a不再等价：candidate新增固定`oscar_dequant_policy=fp32_metadata`，prefill、decode、MTP和图捕获统一选择现有reader的mode0。不是恢复旧提交，也不是再优化裁剪代理。

## 实际可消除的误差

用当前量化公式编码D256 BF16行，前128维1.0、后128维1.0078125；.96/.92裁剪均不改变该行。144B INT2得到codes 0/3，FP32 scale=0.0026041667442768812、normalized zero=-384。

- mode0按保存的FP32参数重建，最后转BF16：逐位恢复1.0/1.0078125。
- 旧decode mode1先将metadata转BF16、减法结果再转BF16后乘scale：恢复1.0/1.015625，行内差值翻倍。

100/100.5以及负值有同类反例，独立两key attention检查也能观察到由此产生的附加误差。它是码值和元数据本可精确表示时仍出现的额外舍入，不是INT2不可避免的表示误差。mode0更忠实于保存的量化等级，但不能保证对任意原始K/V或最终任务分数的误差单调下降。

另用真实AscendC CPU writer→reader→merge验证：四个历史K和一个current K相同，历史V=+1、current V=-1，Q在两半维度符号相反。输入字节与writer精确字节门相同、各非空source内部P恒1；未量化解析结果0.6，mode0的FP32合并输出0.6000000238418579，mode1为0.5796245336532593。此例隔离并证实读端metadata/减法舍入引入约0.020375的输出偏移；不是最终模型分数或1点目标的证明。

官方CUDA decode也采用mode1。这一变化明确是Ascend候选的精度优先政策，不称为修复官方源码bug，也不称为继续逐步复制其decode浮点顺序。用户已确认原生基线MTP=3及LongBench输入、thinking/采样参数相同。

## 实现边界

生产仅在统一mode选择处传入固定配置，四类reader、mixed分段和后续MTP共用它。capture同样选择0；保留原生阶段校验，未知状态/非法策略报错。不是仅修改eager、probe或名称。原native_phase模式仍保留，baseline/native清除candidate专属策略。

底层mode0原已实现：FP32减乘后发布BF16。CANN源码、INT2 writer、144B格式、旋转/PT recipe3、key centering、K=.96/V=.92、QK/PV操作数、FP32累加/softmax/LSE/merge及最终逆旋转不变。native GDN、MTP、prefix、窗口、预算、设备、端口及argv均不改；不启用recipe4裁剪实验。

PT不必重拟合：校准收集的是未启用OSCAR的原生模型输出，本轮只改变已有KV读取方式。测试验证有效recipe3 profile/hash路径完全相同、原PT字节不变且直接复用。原生图/编译缓存身份包括完整有效配置和插件源码，新的policy不会复用旧mode1图缓存。

## 合同与验证

新增`configs/acceptance_fp32_dequant_bf16.json`运行策略overlay，继承v4同一容差和协议门，不修改旧v4 oracle/合同。算子probe保留mode0/1能力检查，mode0包含全部reader的graph用例；额外绑定实际运行策略与overlay SHA，旧策略回执不能放行新candidate。current-only后续decode和APC探针的reader及golden同步选择mode，prepare-only结果不充当数值证明。

真实CPU边界检查和失败/资源事件见[扩展核验报告](../reports/logic_precision_20261023_extended/README.md)，此前三tile核验保留于reports/logic_precision_20261023。所有目标NPU图回放和LongBench质量仍未验收，不能把“小于1点”标成通过。

本轮主机回归283项通过；当前R256的q4/S4全masked history split输出精确0/-inf并完成真实merge。C16完整R256用例保留180秒超时状态；后续隔离history的真实输出数值通过，因VM磁盘耗尽未完成的后置检查在保留原输出并核对SHA后单独完成真实merge，不能把它写成原完整用例成功。

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

摘要应显示`dequant_policy=fp32_metadata`、`dequant_arithmetic=float32_then_bfloat16`、recipe3。无需额外参数、手工删PT或用户诊断命令。
