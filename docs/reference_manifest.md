# 参考清单与当前环境

本任务只读 `references/` 和当前工作区的指令/故障档案；未访问其他失败项目源码。原生树不作任何修改。

| 参考 | 固定版本 | 本轮核实 |
|---|---|---|
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | 本地 git HEAD 一致、status 空 |
| vLLM Ascend | `19e436985102f4ed3aad36c137a6481653688a6c` | 本地 git HEAD 一致、status 空；v0.23.0 + PR12607 |
| OSCAR vLLM PR46774 | `57286d5d2cb08c3dcd8c17bb59e132d6985e6796` | 本地11文件快照，无独立git元数据；pin来自PROVENANCE，文件逐个hash记录 |
| OSCAR paper | `41ebcdba3db5f0ce1339c3727caea80df575d437` | 本地快照无独立git元数据；pin来自PROVENANCE |

原始溯源：`references/PROVENANCE.md` / `PROVENANCE.json`。文件指纹在本地 `reports/reference_fingerprints.json`（不入Git）；新的验收必须重新检查实际代码，而不是仅对版本字符串作相等判断（档案 #74–76）。

本地开发环境是 macOS arm64、Python3.12.6。独立 `.venv` 使用 torch2.14.0 做CPU oracle测试，**不代表目标torch2.10.0**。macOS本体没有CANN/NPU；本轮已发现并使用Lima oscar内的CANN9.1、bisheng15、aarch64 Linux与官方CPU调试库。VM独立环境为Torch2.12.0+cpu/torch_npu2.12.0/pybind11 3.1.0，与目标Torch2.10版本不同，不能代替目标运行。详见 reports/vm/validation.json 及 `reports/environment.json`、`local_environment.json`。

目标模型位置、TP4、图、MTP等保留于 `configs/target.json`。真机连接方式未提供；模型config与权重不在本机，FULL层数/head_dim/GDN实际dtype形状必须由目标环境核实。设备与编译目标按本次启动文档附录 F 固定为物理卡0–3和ascend910b4，端口按附录 A 保留8989；其他项目记忆不构成本任务配置来源。用户回传的 node93 默认设备空值错误见档案 #123；最新复跑已通过 `probe-ops` 的75项 store/merge 原语 NPU 探针，越过 #125 的依赖库加载错误；随后首个 CV 数值用例失败，见 #126；更新后的日志进入TP4/MTP模型加载，因prefix关闭时将GDN请求跨度误作FULL页对齐而失败，见 #127。新日志未附前序CV/旋转探针完整结果，不能补造通过项。

## 权威语义入口

- `references/oscar-vllm-pr46774/vllm/v1/attention/ops/triton_oscar_store.py:41`：FP32量化、先FP16 metadata、LSB-first编码。
- 同快照 `vllm/v1/attention/backends/oscar_attn.py:219`：旋转、裁剪；`:259`起窗口；`:745`起输出合并。PR工程限制见 `docs/semantics.md`。
- `references/oscar-paper/rotation/compute_kv_rotation.py`：真实数据二阶矩与data-free Hadamard两种明确方法。
- `docs/native_integration.md`：本轮核实的后端/spec/allocator/metadata/MTP/graph完整接缝。
- `docs/ascendc_design.md`：AscendC接口先例、D.4四问、官方A2数据通路与已实现CV的验证边界。

## 已纠正的历史假设

1. `P=801792` **不能证明 SSM 是 FP32**。原生 `vllm_ascend/patch/platform/patch_mamba_config.py:94–119` 将单K页对齐SSM，然后给K/V预留两份，故 native padded P 可为 `conv+2*SSM`：例如 BF16 SSM=393216、conv=15360 时仍是801792。实际状态形状/dtype与 `MambaSpec.page_size_bytes` 分开读取；尾padding计入容量账。这个源码证据修正启动文档中“conv+FP32 SSM”的推断。
2. 原生GDN是SoA；FULL按AoS页首寻址可能覆盖其他GDN页。
3. `136 B` 是D256/head的PR实际slot大小；`160 B`并非必需语义。
4. `vllm-ascend-v023-seam-map.md` 的强制eager结论过时；当前native代码保留FULL_DECODE_ONLY。
5. 档案实际为G1–G34及#1–#127，合计161条。标题里的120条以及部分旧索引行号已过时，按标题定位。
6. #122修正#113/#118/#119的OPP路径推断：自定义路径每项应是vendor目录。本工程采用原生已有的direct-launch单路径，不覆盖ASCEND_OPP_PATH。

## 仍缺少的输入/证据

真机连接及完整执行日志（用户已回传配置失败、开发命令误用、加载失败及其后75项原语通过/首个CV数值失败片段）；同环境原生基线；实际模型config与权重指纹；质量评估数据和预先冻结的logits/任务指标/MTP容差；完整目标设备/图/profiler结果；H18复杂度冲突的需求结论。VM CANN交叉编译结果已经存在，不属于缺失项。
