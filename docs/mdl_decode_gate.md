# MDL q4 候选的设备门

本节服从 `skills/oscar-quant-optimization/SKILL.md` 总 MD，属于量化/反量化后有界 CV 数据交付的组合项。档案依据：#126 的尾部/任务完整性，#129 的真实多请求工作，#140 的客户端与设备时间区别，#145/#154 的 CPU 无法证明异步设备数值，#148/#150 的完整路径分母与图输入，#151 的相位报告覆盖，#155 的 NaN 字节比较，#156 的局部与整轮证据边界。D.4 的 6499.8–6655.1ms 全历史恢复与约215ms写路径不能重现：本候选只改变当前有界 KV256 Cube 装载策略，数学、原三源、FP32累加及合并不变。

`experimental_mdl_decode` 的 q4 设备门为 `tools/probe_mdl_decode.py`；baseline 是当前 `attention_cv_bundle_decode_out`，candidate 是 `attention_cv_mdl_decode_out`。生产 q1 保持当前实现。源文件、构建签名和产物 SHA 绑定本轮报告；旧报告不放行。

## 门的范围

- 短 q4 R2、1341/1373历史、N12/S1，保留4行padding、KV尾部和三源。覆盖 signed zero、BF16最小正规数、live metadata零scale、live QR NaN和未来未读取尾部损坏。
- 有效输入要求全部 partial/LSE/status、原merge和BF16输出逐位一致，再与独立原空间oracle按冻结 `atol=rtol=0.005` 比较每个实际query。非法输入要求精确错误码/任务/lane、有限结果位与非有限分类一致。
- 短图捕获与回放覆盖原输入、同址改变Q/QR、同址改变context/INT2/metadata/精确窗口/物理页表/slots/tasks；每态都与eager逐位及独立oracle比较。此为独立算子图，不能称TP4全模型图已验收。
- 速度形状为明确的合成 **R4、q4、20K/23K/27K/30K、N32/S3**。两臂均完整执行原三源CV、原merge、BF16输出copy；2次预热、5次交替顺序NPU Event样本。计时外的CPU构造/oracle不进入设备性能分母。合格门为 `median(base)/median(candidate) >= 1.15`，不四舍五入、不调整冻结阈值。

## 一键行为

`install_observe_serve.sh --variant candidate` 在启动模型前运行本门，任何数值、图、速度、签名或资源释放失败都停止；数据报告 `mdl-decode-report.json` 与相位状态 `mdl-decode-npu.json` 独立。终端只增加 `PERF_MDL_DECODE` 和 `PERF_MDL_RESULT` 关键行，完整失败栈仍保留。

`install_serve.sh --variant candidate` 安装后直接启用同一候选，按既有约定不运行probe。两入口继续默认开启prefix caching、省略 `--max-num-batched-tokens`；本设备门不冒充prefix共享/恢复协议验收，后者按独立兼容门和文档裁决。

当前本机编排/报告测试只证明失败会关闭门、真实形状与签名约束保留。NPU、graph及完整服务性能必须由目标执行产生；无NPU时明确失败，不用CPU替代，不要求AISBench。
