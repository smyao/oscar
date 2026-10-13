# MDL narrow N/K 与输出 stride 源码审计

日期：2026-10-13。范围：当前 `attention_cv_mdl_decode.cpp`；只读实际归档 CANN 9.1 SDK 合同，没有执行新的 VM 或 NPU。当前没有证据将本次 Trap 归因于 narrow shape。

## 实际调用

- QK：`SetOrgShape(M,256,256)`、`SetSingleShape(M,16,256)`；物理 Q/K/score 行跨度均为 256。
- PV：`SetOrgShape(M,256,256)`、`SetSingleShape(M,256,16)`；物理 P/V/PV 行跨度均为 256。
- `GetMDLConfig` 保留 `enableSetTail=true`、`enableStaticPadZeros=false`。所以静态基本块 64×64×128 不代表窄 K 操作会读取 128 个 GM K 元素。

## SDK 路径

1. `artifacts/cann_sdk_matmul_tiling_20261012/include/adv_api/matmul/tiling.h` 的 GetMDLConfig 使能 SetTail。
2. 同归档 `impl/adv_api/detail/matmul/utils/matmul_config_utils.h`：NoTailM/N/K 均先要求 `!enableSetTail`。当前三个条件均 false；IsBasicK 也因 singleCoreK256/basicK128=2 为 false。
3. `scheduler/iterator/k_loop/k_loop_mdl_base.h`：singleK16 导出 kIter1、tailK16、tailStepKa/Kb16、tileShapeA/B16、innerTailK16。没有强行保持 128。
4. `scheduler/base/scheduler_mdl_base.h::CopyIn` 将实际 KLoop/NLoop/MLoop 的 tile shape 传入 CopyCubeIn；`copy_tile_to_cube_common.h::CopyTileToCubeFromGM` 将实际 height/width 与 OrgWidth stride 分开传给 ND2NZ。
5. `data_copy_wrapper_nd.h::CopyND2NZ` 的 `nValue/dValue` 是实际 height/width；`srcDValue` 是 OrgWidth。当前静态补零分支未启用，不能用该分支作为本候选安全论据。
6. `artifacts/cann_sdk_matmul_move_20261010/stage/copy_cube_out/copy_cube_out_fixpipe.h`：非 sequential 写通过 GetDstStride→GetOrgWidth→GetOrgN，故 QK N16 输出 stride 是 256。`IterateAll<true>` 的模板参数是 sync，enSequentialWrite 实参未给出、默认 false。
7. precise producer 每个有效 64 行 chunk 完整初始化，其中实际有效列以外填零；TailColumns 向上对齐 16，读取范围包含在已经初始化的 64 行 chunk 内。history 独立 HistoryVectorTask 始终发布全部 256 行，包括尾零，不走 precise 的 skip 分支。

## 验证覆盖边界

- 公共 fixture 在 `tools/probe_history_reuse.py` 使用 `torch.randn` 生成 Q，再生成非零 QR；不是零 Q 验证。
- 旧 CPU harness 的 GM workspace 初值 `0x85858585` 是有限极小浮点数，不是 NaN。这能检查字节边界，但可能掩盖意外读入未初始化数值的影响。
- NPU probe 的公共 `poison_outputs` 将 workspace 填 NaN；其真实设备完成状态必须单独核验。
- 最短进一步行动：在**研究 harness** 对 workspace 单独使用 quiet-NaN 初值，运行已有 q4/context0 与短 prefix、q1 tail 的真实 CPU 算子；保留随机 Q、所有 source/tasks、独立 oracle 和错误门。未经该反例确认，不修改生产 shape，也不声称排除所有 DMA/同步问题。
