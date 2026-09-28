# q4 历史 attention 独立诊断

依据：档案 #126/#129/#143/#148–#150、启动文档D.4、当前fe0、原生`attention_v1.py:806–986/1174–1224`及`attention_mask.py:55–78`。q1在a13af95已实测获得3.27/6.71倍算子速度，本轮保留其实现。

## 测量对象

32请求、每请求4个query、20/23/27/30K历史各8条，TP4单卡几何Hq6/Hkv1/D256。OSCAR使用生产N128/S3；原生使用实际decode路径的BF16 paged FIA TND `.out`，128-token虚拟页、sparse_mode3、2048方形int8压缩因果mask。MTP配置下原生不走另一条paged-attention API，因此不能用current-only FIA或错误算子替代这个对照。

同一离线随机输入回调分别填充原生BF16 cache和原有OSCAR INT2 fixture，物理页各自独立。原生页表用满足该诊断的最小合法列数，明确标为`minimal_legal_synthetic_width_not_service_stride`，不伪称原生hybrid服务的物理页/表宽完全复刻。

- 原生输出对BF16因果dense oracle；默认不产出LSE，单元素LSE占位不作精度证据。
- OSCAR输出对独立INT2/精确窗口oracle；两种缓存数学不同，不要求native与OSCAR彼此逐位相等。
- mask、workspace、输出、缓存、量化与任务全部在测速前准备。2次预热、5次交替Event，分别记录native FIA、OSCAR CV、CV+merge。每次测量完成后检查输出/状态，验证不计入事件区间。
- 原生主模型权重、GDN、旋转/store准备及全模型图均不在这些区间内。测得差距用于定位，不表示服务速度门已通过，不能直接乘16层变成实测图耗时。

## 独立内部打点

`attention_cv_profile_out`是fe0数学的独立诊断副本，不进入生产路由。profile tensor为int64 `[Cube数,3引擎,4来源,24字段]`，64B对齐且各引擎独占cacheline；来源为history/window/current/total。记录query准备、packed读取、unpack、精确源读取、GM发布、QK/PV计算与等待、归一化/旋转/输出等。

时钟为NPU原始SYS_CNT。诊断额外使用局部MTE2_S/V_S/MTE3_S/FIX_S完成栅栏，改变流水重叠；每个字段只能解释为插桩下该核/引擎的局部跨度。不能把跨核最大值相加成wall。unpack的bits/meta字段是子项，不能再与unpack_total相加；AIV rotation_wait属于finalize，AIC该等待是独立阶段。field23从该引擎Init开始到Process末、排除计数发布，不是完整kernel Event。

新symbol先单独预热2次，再测一次profile Event，明确报告其相对正常CV中位数的扰动。profile及其预热都必须与未插桩fe0的全部partial/LSE/status逐位一致。正常CV和native FIA时间来自未插桩路径。

每种来源分别保留一个关键AIC和一个关键AIV的完整计数向量，不把不同核字段拼接成一行“最坏耗时”。全部引擎必须报告完成跨度；缺失计数、数值差异或资源清理失败都停止。

## 一键与证据

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate --probe-only
```

先跑已有算子、C4/q1门，再跑q4诊断，随后退出。不加载模型、不发推理请求、不运行AISBench。终端新增`PERF_Q4_COMPARE`一行和`PERF_Q4_PROFILE`三行，完整数据写本轮`q4-hotpath.json`。

本地CANN ascend910b4编译、D64/q4/context65和D256/q4/context511的官方CPU-debug通过，含fe0逐字节输出与计数覆盖；证据`reports/q4_profile_local_validation.json`。原生paged FIA调用及profile的目标NPU结果尚待本轮短probe，局部或全服务追平结果不预先判定。

相关主机回归66 passed、28真NPU skipped、4 subtests passed；跳过项不记为目标设备通过。
