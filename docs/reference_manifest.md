# 参考清单与当前环境

本任务只读 `references/` 和当前工作区的指令/故障档案；未访问其他失败项目源码。原生树不作任何修改。

## 2026-10-12 实际可读参考与历史 pin（#74–78/#156）

下表区分**本轮真实测试所读版本**与**尚未恢复的历史版本**。不能把基础tag的CPU契约结果记到旧组合pin或目标部署版本名下。

| 参考 | 本轮实际版本/历史版本 | 本轮核实 |
|---|---|---|
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | 本地 git HEAD 一致、status 空 |
| vLLM Ascend 实际参考 | `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd` | 本地 git HEAD 一致、status 空；当前CPU契约测试读取这个版本 |
| vLLM Ascend 历史组合 pin | `19e436985102f4ed3aad36c137a6481653688a6c` | 未恢复/未验证；不能声称等同于当前5cb98caa，亦未更改目标版本要求 |
| OSCAR vLLM PR46774 历史快照 | `57286d5d2cb08c3dcd8c17bb59e132d6985e6796` | 本轮快照目录不存在，不能续用“本地11文件已核实”的旧声明 |
| OSCAR paper 历史快照 | `41ebcdba3db5f0ce1339c3727caea80df575d437` | 本轮快照目录不存在，历史pin待恢复后重新核实 |

历史 `references/PROVENANCE.md` / `PROVENANCE.json` 本轮亦不存在。遵守参考树只读要求，本轮没有checkout、补写PROVENANCE或把历史pin改成“已恢复”。新增入Git的 `tests/native_contract_manifest.json` 记录本轮两棵实际树的完整commit、Git tree、27个文件SHA256（25个完整原生源码及2份许可证）及10个原生方法AST指纹（含后续MTP长度/slot更新）。`tests/test_native_source_contracts.py` 检查真实源码与记录一致，变更需重新review；它仅属于本地开发回归，不加入直启/readiness，不扫描目标机器原生安装树。

2026-10-12实读本地 `.venv`：macOS arm64、Python3.14.6、torch2.14.1、pytest9.1.1，**不代表目标torch2.10.0**。项目 `pyproject.toml` 的test依赖要求torch>=2.10；review所用torch2.1不在这一测试依赖范围，不能通过放宽断言或改原生参考源码让它适配。macOS本体没有CANN/NPU；以下VM记录是历史独立证据，本轮没有重跑：Lima/CANN9.1、bisheng15、aarch64 Linux、官方CPU调试库，VM Torch2.12.0+cpu/torch_npu2.12.0/pybind11 3.1.0，见历史 reports/vm/validation.json 等。

目标模型位置、TP4、图、MTP等以 `configs/target.json` 为准。本轮实读设备4–7、端口7878、SOC ascend910b4，未修改配置；旧附录中的0–3/8989不是当前配置的描述。真机连接方式未提供，模型config与权重不在本机。历史node93 #123–#128日志见档案；2026-10-10 #156的prefix图回放故障及其修订仍缺新设备复验。编译/本地CPU通过不等于NPU捕图、回放、精度或吞吐通过。

## 语义入口索引（缺失快照为历史路径，不是本轮实读证据）

- `references/oscar-vllm-pr46774/vllm/v1/attention/ops/triton_oscar_store.py:41`：FP32量化、先FP16 metadata、LSB-first编码。
- 同快照 `vllm/v1/attention/backends/oscar_attn.py:219`：旋转、裁剪；`:259`起窗口；`:745`起输出合并。PR工程限制见 `docs/semantics.md`。
- `references/oscar-paper/rotation/compute_kv_rotation.py`：真实数据二阶矩与data-free Hadamard两种明确方法。
- `docs/native_integration.md`：后端/spec/allocator/metadata/MTP/graph接缝记录；本轮实际参考版本与复核范围以上表和测试指纹为准。
- `docs/ascendc_design.md`：AscendC接口先例、D.4四问、官方A2数据通路与已实现CV的验证边界。

## 已纠正的历史假设

1. `P=801792` **不能证明 SSM 是 FP32**。原生 `vllm_ascend/patch/platform/patch_mamba_config.py:94–119` 将单K页对齐SSM，然后给K/V预留两份，故 native padded P 可为 `conv+2*SSM`：例如 BF16 SSM=393216、conv=15360 时仍是801792。实际状态形状/dtype与 `MambaSpec.page_size_bytes` 分开读取；尾padding计入容量账。这个源码证据修正启动文档中“conv+FP32 SSM”的推断。
2. 原生GDN是SoA；FULL按AoS页首寻址可能覆盖其他GDN页。
3. `136 B` 是D256/head的PR实际slot大小；`160 B`并非必需语义。
4. `vllm-ascend-v023-seam-map.md` 的强制eager结论过时；当前native代码保留FULL_DECODE_ONLY。
5. 当前档案已经追加至#156；标题条数与旧索引行号有漂移，按条目标题定位。
6. #122修正#113/#118/#119的OPP路径推断：自定义路径每项应是vendor目录。本工程采用原生已有的direct-launch单路径，不覆盖ASCEND_OPP_PATH。

## 仍缺少的输入/证据

真机连接及完整执行日志（用户已回传配置失败、开发命令误用、加载失败及其后75项原语通过/首个CV数值失败片段）；同环境原生基线；实际模型config与权重指纹；质量评估数据和预先冻结的logits/任务指标/MTP容差；完整目标设备/图/profiler结果；H18复杂度冲突的需求结论。VM CANN交叉编译结果已经存在，不属于缺失项。

2026-10-12用户回传CPU入口缺少reference文件后，新增Git跟踪的 `references/native_cpu_contracts.zip`，逐字节来自上表实际本地commit。`tools.native_contract_sources` 在没有原生checkout的干净克隆里读取验证后的快照并明确报告 bundled_snapshot；存在checkout时仍验证原有commit/tree/status。压缩包不是完整Git仓库、安装的runtime或历史19e43698组合pin；不联网、不解压、不修改已有参考树、不加入部署审计。27个文件及整个包的SHA256均入manifest，两个上游Apache-2.0 LICENSE保持原文。
