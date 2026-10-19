# 校准引擎线程池启动错误修复（2026-10-19）

用户最新日志确认上次离线数据修复已生效：`calibration_data_ready` 的 SHA 正确，
99 条 prompt / 29867 tokens 已准备好。随后在 `LLM(**options)` 创建引擎时，
父进程和 EngineCore 均出现 Python `popen_fork.py` 多线程 fork 警告；TP worker
在 `torch::autograd::Engine::thread_init → set_num_threads` 报
`ParallelOpenMP.cpp:64 Invalid thread pool!`。正式服务尚未启动，也未开始采集校准样本。
原文见 [target_failure.txt](../reports/rotation_process_fix_20261019/target_failure.txt)。

## 修复

新增校准入口直接调用离线 `vllm.LLM`，遗漏了官方 Ascend 离线示例采用的
`VLLM_WORKER_MULTIPROC_METHOD=spawn`。vLLM 此处默认 `fork`；原先只在父进程调用
`torch.set_num_threads()` 的措施未消除本次两层 fork 后线程池故障。

现在准备模块在创建校准 Python 子进程时就传入独立环境，强制 `spawn`；直接运行
`tools.calibrate_model_rotations` 时也在 tokenizer/Torch/vLLM 导入前设置。
原生 EngineCore 和 Ascend TP worker 均经 `get_mp_context()` 读取这个变量，
分别启动新解释器，避免继承上层已启动的线程池状态。没有用全局
`multiprocessing.set_start_method()` 冒充原生的显式 context 选择。

此设置仅限独立校准进程。正式服务配置、TP4、MTP、APC、BF16、设备端口和
CPU binding 保持；原生源码未修改。校准仍真实加载 NPU 模型，不替换成随机样本、
固定 Hadamard 或跳过 PT。校准失败仍阻止正式服务，也不会发布完成 PT。

原生依据均来自本仓库只读 references：

- `vllm-ascend/examples/offline_inference_npu_tp2.py`：导入 vLLM 前设置 spawn。
- `vllm/vllm/utils/system_utils.py::get_mp_context`：按环境变量选择显式 context。
- `vllm/vllm/v1/engine/utils.py::CoreEngineProcManager`：EngineCore 使用该 context。
- `vllm-ascend/vllm_ascend/patch/platform/patch_multiproc_executor.py::AscendWorkerProc.make_worker_process`：TP worker 使用同一选择。

## 验证与使用

60 项针对性主机测试通过，其中包含真实两层 spawn：父进程预热 Torch 后，创建
EngineCore 替身及两个 worker，逐层运行线程池与 autograd、校验隔离和完整退出。
原生上下文选择函数取自只读源码，平台探测使用主机替身；未加载 NPU 引擎。结果记录在
[summary.json](../reports/rotation_process_fix_20261019/summary.json)。这些是主机与
原生源码契约证据，没有连接真机、运行 NPU 校准/图/质量验收或性能测试。
本次不改内核，用户日志已经确认 `build-ops` 返回 0。

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

无需手动设置环境变量或删除失败目录。启动会检查 PT；这次失败未发布完成 PT，
会自动重新校准。新日志先打印 `rotation_capture=process_environment`、
`worker_multiproc_method=spawn`，PT 生成后仍按模型与校准身份严格复用。
