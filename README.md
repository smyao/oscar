# OSCAR Ascend 外部适配工程

**当前是本地开发版本，尚不能部署完整服务。** 已有PR数值参考、旋转artifact、缓存布局、外部插件接缝、AscendC store/merge候选源码和诊断工具。真实fused CV attention、融合旋转/裁剪、MTP/prefix窗口事务与生产runtime仍未完成；AscendC未在CANN编译，NPU/图/精度/性能未验收。启用不完整运行时会明确失败。

所有内容保存在当前目录，本地Git分支为 `codex/oscar-ascend`，未配置远端。`references/` 保持只读，不纳入新工程Git内容；两个原始任务文档保留。

本地测试：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

当前 `.venv` 已就绪。CPU测试只是数学/布局/导入/工具契约验证，不能替代NPU测试。

查看目标流程与原始启动参数（不访问NPU）：

```bash
bash scripts/install_probe_serve.sh --plan
python3 -m tools.target_cli --print-command
```

目标入口为 `scripts/install_probe_serve.sh`（`scripts/serve.sh` 同入口）。它有独立阶段日志、超时、所属进程组清理、退出码和原生文件完整性检查。当前完整流程会在缺失能力处退出，**不能将这条命令理解为已经实现一键部署**。真机物理设备号必须先写入 `configs/target.json`；不可继承其他任务的设备选择。

可以单独运行环境记录和真实算子构建/探针。后两项需要目标CANN、torch_npu、编译依赖与已确定的NPU设备：

```bash
python3 -m tools.deploy --only environment
python3 -m tools.deploy --only build-ops
python3 -m tools.deploy --only probe-ops
python3 -m tools.deploy --only runtime-readiness
```

AscendC采用独立 `ascendc_library` + 当前NPU stream直接调用，以目标原生工程为先例；不是替换原生源码。构建目录由源码/工具链/SOC签名约束，变化则干净重建，编译失败只干净重试一次。`build_manifest.json`、`.so`存在、实际加载和设备完成是四个独立判据。

默认旋转来源为论文明确支持的data-free Hadamard，在NPU生成并绑定模型/PR指纹；它不等于样本校准。样本统计与artifact API已实现，真实模型采样和NPU eigensolver尚未接通。缺层不能静默使用identity。

重要文档：

- [设计与硬约束冲突](docs/design.md)
- [完成状态与证据](docs/checklist.md)
- [参考清单](docs/reference_manifest.md)
- [原生接缝](docs/native_integration.md)
- [PR数值语义](docs/semantics.md)
- [算子清单](docs/operator_inventory.md)
- [AscendC设计与D.4对照](docs/ascendc_design.md)
- [性能对照协议](benchmarks/README.md)

`reports/` 保存本轮实测和明确未运行的状态。附录D与旧档案的数据仅用于设计，绝不充当本轮性能结果。运行报错时先查 `issue1_full_record.md` 对应症状，真机新错误才追加新真机条目；本地测试问题不伪造为真机记录。
