# 一键启动校准数据证书错误修复（2026-10-19）

后续用户日志已确认数据准备通过；校准引擎的另一处线程池启动错误及修复见
[后续说明](rotation_process_startup_fix_20261019.md)。

用户执行 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`，
`build-ops` 返回 0；随后 `prepare-rotations` 的 GPQA 下载报
`SSLCertVerificationError: self-signed certificate in certificate chain`。
失败来自新增自动校准流程对在线下载的依赖，发生在 tokenizer 和 NPU 模型加载前。
原始日志见 [target_failure.txt](../reports/rotation_data_fix_20261019/target_failure.txt)。

默认流程现在使用随 Python 包发布的官方 GPQA Diamond 压缩数据，保留下载文件原字节，
在模型加载前解压并验证固定 SHA256。没有关闭 TLS 验证，没有修改系统证书，也不再依赖
服务器的外网或旧 `HF_HOME` 数据缓存。数据来源、CC BY 4.0 署名和 SHA 见
[数据说明](../tools/data/README.md)。官方 seed0 选项排列、chat template、完整 prompt
选取、30000 token 预算及 NPU 采样/离线 FP64 拟合保持原实现。

## 现在怎样检查与生成 PT

直启和 observe 两个入口都在正式服务前运行同一个 `prepare-rotations`：

1. 根据模型/权重身份、tokenizer、量化/RoPE 配置、官方配方、数据 SHA 和预算定位缓存。
2. 匹配的 K/V/combined PT 必须通过配对代际、完成标记、文件哈希和矩阵检查才会复用，日志包含 `reused=true`。
3. 缺少匹配 PT，或者自动发布曾被中断，则自动 NPU 采样并生成，日志先显示 `rotation_prepare=generating`，完成后包含 `reused=false`。
4. 生成或校验失败会阻断正式服务。损坏的完成缓存、显式指定但不匹配的 PT 会报错，不能冒充最新产物使用，也不会悄悄覆盖用户显式文件。

本次新增数据 SHA 绑定会启用新的默认缓存目录，旧版只记录数据源名称的缓存不会直接复用。
新 PT 完成后，模型和上述校准输入不变的下次启动会复用；**不是每次启动都重新校准**。
用户这次失败没有生成完成 PT，重新拉取后同一命令会继续自动生成，无须手工下载数据或删除文件。

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

## 验证与边界

- 55 项针对性主机回归通过：默认离线数据、损坏拒绝、自定义输入、数据变更使缓存失效、原生采样接口替身的成功/失败清理、PT 生成/复用、固定启动编排。
- 实际 wheel 构建成功；在与仓库隔离的安装目录中禁用网络与子进程后，数据解压与 SHA 校验通过，证明安装包包含数据。
- 证据见 [summary.json](../reports/rotation_data_fix_20261019/summary.json)。本次没有内核修改，不重复 CANN 构建；用户日志中的构建已成功。
- 没有连接真机、运行 NPU 校准或性能测试；上述证据不表示真实模型采样、图或模型质量验收已完成。

仅交付 `codex/oscar-bf16-attention`；main/stable、默认原版 Qwen3.8、BF16、MTP、APC 与设备端口设置保持。
