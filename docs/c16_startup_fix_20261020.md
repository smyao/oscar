# C16实验启动失败：图缓存参数错配与错误后继续等待

> 后续用户反馈：run `2ae27c282db34edfbb383587023dcb63`已完成32K→16请求和四rank共68次C16配对计时，摘要issues=0。本实验的启动与请求链本次已跑通；不是所有模型/图/质量场景验收。[结果分析](c16_breakdown_analysis_20261020.md)。

**这次不是模型正常加载得慢。worker在00:19:49已经报错，脚本却继续等待health，日志一直到1142秒。**

原文见[用户日志](../reports/c16_startup_fix_20261020/target_failure.txt)。错误发生在
`determine_available_memory → profile_run → _dummy_run → AOT/piecewise → npugraph_ex`，
尚未开始128-token预热请求或32K计时请求：

```text
compiler_interface.py:363, in compiled_fn
    result = _inner_fn(*args, **kwargs)
npu_fx_compiler.py:511, in __call__
    gm_result = self.run_kernel(*args, **kwargs)
<string>:113, in kernel
ValueError: too many values to unpack (expected 20)
```

## 为什么是参数错配

只读原生 `vllm_ascend/compilation/compiler_interface.py:350–369` 中，这个`compiled_fn`
属于 **磁盘缓存load路径**。它加载保存的Python图代码，随后直接调用。
本机安装的torch_npu代码生成器在`kernel(*args, **kwargs)`里生成固定数量的
`input0, input1, ... = args`；本次“expected 20”是这个输入解包，不是attention返回20个结果。

这里有一个必须处理的契约：原生保存的是AOT后的内层代码，加载入口拿到的是外层图参数。
AOT允许合并别名输入，因此两者数量可以不同。真实CPU AOT回归已重现：外层3个输入中两个
是同一个被原地更新的Tensor，内层只保留2个；裸内层缓存接收3个输入就报同类解包错误，
保留完整AOT包装时结果和原地更新均正确。

**不能简单归咎于用户“旧缓存没清”。新生成的内层缓存也可能有这种数量差异。**
现场未提供那份生成代码及实参，所以不声称已经找到了实际20个输入的具体别名来源。
本轮修复针对已定位的输入数量契约缺口，并在主机重现和验证。

## 已做的修复

1. **加载前检查图缓存输入契约。** 外部插件检查缓存顶层kernel/forward的固定输入解包数，
   与本次`example_inputs`数量比较。数量不符、格式无法确认或代码损坏时，走原生缓存未命中的
   重新编译路径，保留完整AOT包装。相符的缓存沿原路径加载。不会截断参数，也不在运算报错后重试。
2. **外层AOT和内层缓存一起隔离。** 在导入vLLM之前，根据有效配置、实际插件源码、扩展构建身份、
   编译器版本、native/OSCAR及C16模式选择稳定缓存子目录。相同身份复用，变化自动分区，
   不删除用户已有缓存。此隔离不能替代第1项检查；不兼容的AOT内层缓存以后仍需重新编译。
3. **worker失败立即可见。** quiet启动识别明确的WorkerProc/EngineCore失败，打印有界错误片段，
   随即退出并清理拥有的进程。普通WARNING、可恢复ERROR或后台线程traceback不单独触发误杀。
   等待期间每分钟最多两行，显示实际worker阶段和最近一条服务日志。

原生启动argv、模型、BF16、MTP3、APC关闭、16384预算、graph、GDN及attention算子保持原设置。
普通candidate直启与C16实验共用缓存修复；main/stable不变。

## 验证与运行

验证包括真实CPU AOT别名/原地修改反例、执行只读原生`AscendCompiler.load`方法的cache-miss分支、
插件安装与恢复、稳定缓存身份，以及父进程保持存活时的真实子进程异常注入和清理。
完整通过数量与日志见[验证回执](../reports/c16_startup_fix_20261020/validation.json)。
没有连接真机，没有运行NPU、模型性能或重新编译未改动的CANN核；这些主机证据不能代替目标启动验证。

如果旧实验进程仍在等待，先在原终端按一次Ctrl-C，让脚本执行清理，然后运行：

```bash
git pull --ff-only && bash scripts/test_c16_breakdown.sh
```

启动会打印`COMPILE_CACHE`。遇到不兼容缓存时会打印`COMPILE_CACHE_REBUILD`，随后仍由原生编译器
生成图；服务若再有worker错误，会直接显示`STARTUP_FAILED`及原因，不再只刷waiting。
