# 完整 C4 Kt/Vt-NZ：一次有界发布链裁决

独立研究，不注册生产。档案#126/#140/#145/#148/#154及完整D.4：禁止全历史恢复，保留精度、错误域和跨pipe所有权；工作限定当前KV256。

## 同库两臂与真实成本

- **baseline**：已测原ND H32＋C4 member pipeline，母源`da59a923…`，自然Rv；不是已回退的B-only/Vt版。
- **candidate**：从已完成Vt版`6ea0ad44…`可逆派生；K/V均物理`[KV/8,D,8]`，QK读Kt时B=false，PV读Vt时B=true。A/P仍ND，原FP32/K128、task/core、lane16、padding/status/error与flags保持。
- Rv只做原逻辑矩阵的一次NZ位重排，B=true，不二次转置；保留原自然矩阵用于独立oracle。初始化pack和256KiB/独立矩阵额外存储单列，K/V没有host免费预pack。
- 完整KV256的两side各16次连续16KiB写，共**32个MTE3 bursts、512KiB payload**；仍有32次Gather、**64次索引Cast**及原有/新增fence。小burst减少不是净速度结论，QK改走SDK非转置Load3D也须计入。
- 同一个8KiB压缩索引服务K/V，去掉16KiB K索引；候选显式UB **165952B/AIV**，GM **2494464B/core**，原Softmax缓冲/view未缩。

## 有界验证与身份

固定Q336/C337/R1/Hq6/Hkv1/D256/S1、20 Cube、**全部1008 tasks**；合法source2沿生产prefill规则抑制，native current FIA不计在本核时间内。两臂输入逻辑矩阵完全相同，只有候选Rv物理位布局不同，必须逆向逐位核实；其他输入hash、全部输出/状态/guard门不放宽。

- **CAModel每臂600秒**，无自动重试/延长；完成后才可采用cycles。全部60actors、partial/LSE/status/clusterStats、独立oracle与同库配对均需通过。
- **publisher CPU门60秒**：提取候选四方法逐字绑定，history旧lane16与precise16各检K/V完整131072 words及双guards，覆盖±0、NaN/Inf、subnormal位搬运。它不是完整C4 CPU资格。
- 本Kt/Vt完整CPU **未跑**；原B-only完整CPU **120秒超时**保留，不能改成通过。旧同C337 baseline CPU确已完成时可引用其prepare、数值、输入和control证据，不伪造新的CPU执行。
- 回放链严格可逆继承`c4_b_nz_vt/generate_replay.py`；只改新generate/publisher/library及明确CPU状态字符串。符号由新source identity提供，CMake输入绑定与实际library hash均保留。

## 入口

```bash
python3 csrc/experiments/c4_kvt_nz/generate.py --check
python3 csrc/experiments/c4_kvt_nz/publisher_probe.py --check
python3 csrc/experiments/c4_kvt_nz/generate_replay.py --check
python3 csrc/experiments/c4_kvt_nz/cv_replay.py --variant baseline --fixture FIXTURE --prepared COMPLETED_BASELINE_CPU --cpu-control BASELINE_CONTROL --publisher-cpu NEW_PUBLISHER_CPU --publisher-control PUBLISHER_CONTROL --library LIBRARY --output NEW_OUTPUT
```

候选只改`--variant candidate`并使用新输出目录；两次回放均由已有有界supervisor限定600秒。CPU target为`oscar_c4_kvt_nz_publisher_cpu`，库为`liboscar_c4_kvt_nz.so`。最终去留依据包含真实publisher与consumer的完整核净收益；未通过前不进入组合，也不推算整轮分钟数。
