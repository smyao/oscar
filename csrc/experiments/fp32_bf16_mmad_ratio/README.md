# 同形 FP32/BF16 操作数吞吐比

研究专用；不改生产、不放宽oracle、不恢复已否决的单cast或多分量方案。档案G3–G13/#21/#81–85/#126/#140/#145/#148/#154–156、完整D.4。只测试两例四臂：QK M24/N256/K256（B转置），PV M126/N256/K256（B不转置），grid1纯AIC。

两臂都用GetNormalConfig、basic64×64×128、singleCore128×256×256和FP32输出/累加，HF32关闭。区别只有A/B操作数类型；输入来自同一整数母数据÷64，均在[-1,1]。FP32低16位必须全零且逐元素等于BF16；任意256项部分和的整数分子绝对值≤2^20、分母4096，均可精确表示成FP32。因此用整数点积生成严格逐位golden，不用浮点Matmul作参考。padding、输入与guards亦严格。

```bash
python3 csrc/experiments/fp32_bf16_mmad_ratio/fixture.py --output NEW_FIXTURES

cmake -S csrc/experiments/fp32_bf16_mmad_ratio -B NEW_BUILD/cpu -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DOSCAR_MMAD_RATIO_CPU_DEBUG=ON
cmake --build NEW_BUILD/cpu -j1
NEW_BUILD/cpu/oscar_mmad_ratio_cpu NEW_FIXTURES/qk_m24 NEW_CPU_QK
NEW_BUILD/cpu/oscar_mmad_ratio_cpu NEW_FIXTURES/pv_m126 NEW_CPU_PV

cmake -S csrc/experiments/fp32_bf16_mmad_ratio -B NEW_BUILD/device -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH"
cmake --build NEW_BUILD/device -j1
```

root排他运行CAModel，使用既有durable supervisor，**每臂180秒，无重试或加时**。沿原SIM环境与cache配置，不增服务参数：

```bash
python3 csrc/experiments/fp32_bf16_mmad_ratio/replay.py --arm fp32 --fixture NEW_FIXTURES/qk_m24 --library NEW_BUILD/device/lib/liboscar_fp32_bf16_mmad_ratio.so --binding NEW_BUILD/device/source_binding.json --output RUN/qk_m24/fp32
```

另执行同case的bf16臂、pv_m126两臂；control位于`RUN/<case>/<arm>-control.json`。全部GM先保存，再核exact oracle和实际完成状态；SDK task/AIC完整记录、core0 completion/popped trace都必须保留，16MiB/file命中即失败，gzip后回收本次owned raw。

```bash
python3 csrc/experiments/fp32_bf16_mmad_ratio/compare.py --directory RUN --output NEW_COMPARISON.json
```

比较要求两形状各自同库、同SIM、同来源母数据、同grid、相同FP32输出全位及完整180秒control/profile。分别报告R24_QK和R126_PV，不跨形状计算比率。R是**含SDK控制和搬运的完整Matmul任务比**，不是纯Cube峰值；未来分量拆分/重组与控制复用都未测。R较小可以筛掉直接多次完整Matmul拼接，不能伪证所有未知融合算法不可能；R较大也不构成任意FP32→BF16或多分量精度资格。

当前仅host源码/输入检查。CANN、官方CPU、CAModel、NPU、graph和完整服务均待对应执行证据。
