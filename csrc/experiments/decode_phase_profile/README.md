# 当前SIMD / history-pipeline相位计数（2026-10-06）

诊断克隆，未接生产。依据档案#140/#144/#148/#154/#155和启动文档D.4；不采用旧fe0 profile数学体或其相位占比。`generate.py`固定当前SIMD与pipeline源SHA，断言每次替换数量，逆向删除全部诊断改动后必须逐字节还原原source；独立记录数学helper body hash。只插入时钟、局部counter和最后的独立GM flush，**无新增完成Fence、PIPE_ALL或数学替换**。

## ABI

原15个GM指针后（workspace之后）插入第16个`profile`指针，再原17个int64属性和float scale。4个kernel：

- `oscar_decode_simd_profile_q4_kernel`
- `oscar_decode_simd_profile_q1_kernel`
- `oscar_history_pipeline_profile_q4_kernel`
- `oscar_history_pipeline_profile_q1_kernel`

`profile`为int64 `[cube_cores,3,4,24]`，每core2304B、每actor768B，base必须64B对齐且与所有数学输入/输出不重叠。actor依次AIC/AIV0/AIV1；source依次history/window/current/total。所有96slot/actor都会覆盖；可先poison检查完整写出。field23只在total行有值，其余source为0。

字段以`manifest.json`的schema2为准：0–19、22/23延续原布局，**20=task_metadata、21=final_writeback**；禁止用旧bits/meta标签解读20/21。当前共享SIMD unpack helper保持原样，未测其中bit/meta子项。

## 周期语义

- 0/1/2是有效task、KV256迭代、观察KV行数计数；其余是`GetSystemCycle`差值。
- 20包含每次LoadTask（包括空task槽），在LoadTask完成后按加载出的source归属；3仅覆盖合法task body，另含4/5/6/7/8/9–19/21/22等嵌套相位。
- 4=LoadQueries；5=packed load（pipeline测LoadPackedInto，避免与wrapper重复计数）；6=Unpack/PipelineUnpack；7=LoadPrecise，source1为window、source2为current；8=PublishKv/PipelinePublishKv。
- 14/16/18是Matmul helper的Scalar提交与SDK内部原有同步区间；**没有额外FIX_S保证完成**。4/5/6/7/8/10/12/21同样只保留原本的同步点，返回后异步引擎可能仍未完成，不能冒充独占设备计算耗时。
- 9/11/13/15/17/22包住原CrossCoreWaitFlag；这是该actor程序在对应依赖处观察的等待区间。pipeline没有原AIC acc wait，因此17可为0。
- 19覆盖归一化、RvT交接/等待及最后写回，包含21/22；3又包含19。**不可相加这些嵌套字段，不可相加不同actor最大值当wall。**
- 23覆盖该actor Init+Process和计数器开销，排除profile flush；统计开销会扰动调度，不能用带计数版本代替无计数性能门。
- 每actor只flush自己的12个64B cacheline。数学buffer布局、既有所有权屏障、KV顺序、有限性及输出逻辑均未替换。

## 本地准备

```bash
python3 csrc/experiments/decode_phase_profile/generate.py
python3 csrc/experiments/decode_phase_profile/generate.py --check
```

CMake configure自动执行`--check`。Lima staging时必须保留本工程相对结构，并带上两个pin源文件及其公共header，因为生成校验会重新读取它们。设备构建：

```bash
cmake -S csrc/experiments/decode_phase_profile -B build/decode-phase-profile -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 -DSOC_VERSION=ascend910b4
cmake --build build/decode-phase-profile -j1
```

当前仅本地生成/逆向校验完成；CANN、CPU、CAModel、NPU和实际相位数据尚未执行。短样本先验证数学输出与无计数同源baseline和独立oracle；计数有值不等于数学/图/整轮验收。
