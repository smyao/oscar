# 本机 Ascend910B4 CAModel：striped CV 完整算子对照

依据档案 #126/#148/#151/#154 与启动文档 D.4。此处是本机 Lima `oscar` ARM64 VM 的 CANN 9.1 `dav_2201` **CAModel 周期模拟**，不是目标 NPU 毫秒、TP4 模型质量或 AISBench 验收。模型初始化显示 24 个 AIC；三臂均以 `grid=20` 启动。只使用已有编译产物及项目确定性合成算子输入，不使用用户数据集。

| 合成 q4、D256、Hq6/Hkv1、S1 | 旧 fast task ticks | 基础 striped | SIMD striped | 旧 fast / SIMD |
| --- | ---: | ---: | ---: | ---: |
| 旧历史 4,095 token | 1,674,968 | 1,059,571 | 844,724 | 1.983× |
| 旧历史 8,191 token | 3,195,883 | 2,021,108 | 1,525,119 | 2.095× |

4K 与 8K 的旧 fast / 基础 striped / SIMD striped `partial`、`LSE`、`status` 分别逐字节一致，全部 status 为零。8K 合并输出对独立冻结 oracle 最大绝对误差 `2.1242e-7`，LSE `1.1493e-6`，冻结 `atol=rtol=0.005` 均无超差元素。输入和编译库 SHA256、全部分核周期及日志摘要见 [`../simulator_b4_performance_20260930.json`](../simulator_b4_performance_20260930.json)；两个 fixture 清单在 [`fixtures/`](fixtures/)。

同一 source0 AIV/core0 历史路径在 4K→8K 增加 16 个 KV256 tile 时，每增加一 tile 的模拟周期分别为旧 fast `95,057`、基础 striped `60,096`、SIMD striped `45,108` ticks；旧/SIMD 增量比约 `2.107×`。4K SIMD 整核临界已移至 source1/window，因此不能用整核 4K→8K 增量冒充同一历史 actor 斜率。

**模型方向校准。** 同一 4K fixture 的 fe0 与 fast 输出逐字节一致；CAModel `2,465,188→1,674,968` ticks（fast/fe0 `0.679`）。已回传的真 NPU 是另一形状 q4 N128/S3、20–30K 历史，`15.29794→11.28152ms`（`0.737`）。方向相同，比例并不相等，所以本报告不把 CAModel ticks 换算为真机毫秒或整轮收益。

**最小重放。** 已存在的本机 VM fixture 与库路径写在 [`replay_ctx8191_triplet.py`](replay_ctx8191_triplet.py)；[输入清单](fixtures/synthetic_q4_ctx8191.json) 绑定 `make_fixture` 固定种子、`raw_to_striped` 的逐位排列及各文件 SHA256。以下命令从当前工程根运行，调用本机 Lima 127.0.0.1，不连目标机器：

```bash
limactl shell --workdir /home/sunao2000.linux/oscar-simulator-check/ctx8191_triplet oscar -- bash -lc '
set -e
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export LD_LIBRARY_PATH=/home/sunao2000.linux/oscar-simulator-check/camodel:$LD_LIBRARY_PATH
export CAMODEL_LOG_PATH=/home/sunao2000.linux/oscar-simulator-check/ctx8191_triplet/log_ca
export STARS_LOG_PATH=/home/sunao2000.linux/oscar-simulator-check/ctx8191_triplet/log_ca
export CORE_ENABLE_MASK=0x1
timeout --signal=TERM --kill-after=5s 3600s python3 -' \
  < reports/simulator_b4_evidence_20260930/replay_ctx8191_triplet.py
```

设备 Event 在 CAModel 返回零时间戳，比较采用每个 task 的 `profile_task_log0.toml` 周期及对应 AIC/AIV 日志。脚本仅将本轮 scratch 的大体积 `*.dump` 打开 FD 重定向至 `/dev/null`；在 helper 和完整 CV 两个固定输入上，该控制前后 task ticks 完全相同，数值不变。小型 profile 与输出张量均保存在此目录。曾因混用两个同名 fixture 目录而产生的无效对照及一次磁盘满的模拟已排除，不进入上表。

**边界。** 单请求 S1 的 4K/8K 合成算子不代表实际 32 并发 N128/S3、20–30K/65K、MTP、GDN、图或队列；实际长期性能及模型质量仍须用户侧 NPU 门裁决。不能从 `2.095×` 算子模拟比推定 LongBenchv2 八分钟或 64K 四百四十秒。

模拟使用的 VM 编译库 SHA256 是本次精确产物身份。[源码等价记录](source_equivalence.json)证实四个 striped decode 核/解包头中，两个头逐字节相同、两个 `.cpp` 仅注释变化；去 C++ 注释后的 token 序列与 SHA256 均相同，所测数学执行体与当前文件一致。最终接线、绑定和重新构建的实际产物仍需精度、图与真 NPU 速度门，不能把 CAModel 周期自动转签为服务性能。
