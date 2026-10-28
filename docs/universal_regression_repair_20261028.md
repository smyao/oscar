# 同时保护短输入与长输入的阶段组合修复

2026-10-28。限定 `codex/oscar-bf16-attention`。

用户实测：`34bc2b75` 的64K已经达标，但32K及以下慢30%。短输入保护基底是 `232b563b69e01495046a9b0025ac9150bc11b9d7`。这说明上一轮组合失败；源码发射次数减少和数值逐位一致不能代替组合性能。用户原话节选保存在[现场反馈](../reports/universal_regression_repair_20261028/user_feedback.txt)。

这次只撤销一项：**q4/q1三个发布点的批量BF16转换**。所有输入长度统一使用232b的同一份decode源码。C16及其finite helper继续使用34bc的同一份源码，C4/current-only和其它执行路径保持不变。没有新增任何按输入、context或历史长度选择版本的条件。

## 为什么拆开decode与prefill

两提交的线上差分只有两份reader及一个新helper。整个`oscar_ascend/`、`configs/`、`scripts/`、CMake生产列表和binding没有变化；没有新增PC、调度、图、KV容量或模型参数逻辑。新experiments会改变一次性构建指纹，但不在生产CMake内，也不进入逐步forward。

实际调用链先处理无历史current-only suffix，其FIA/旋转/store全未改；其余短warm query走原C4。只有原来就进入C16的warm query才受C16优化影响。另一方面，target q4和后续MTP q1共用decode源，它们在所有上下文长度都被上轮批量Cast改变。[真实分流](../oscar_ascend/ops/cv_dispatch.py)、[current-only提前返回](../oscar_ascend/integration/impl.py)。

本地主机APC splitter回放的冷请求形状是：4K为3072/1024，8K为4608/2304/1280，16K为13824/1536/1024，32K为16128/13824/1536/1280，64K为16128/16128/16128/13824/2304/1024。这些是固定预算下的结构样本，不能冒充现场batch轨迹。在4K、8K、16K样本中，首块current-only之后只有未改的C4；这些路径无法执行新C16检查，却会执行新decode。因此，上一轮将短输入回归风险主要归到finite证书，遗漏了共用decode的独立回归面。

修复选择由上述执行覆盖决定：恢复已达标基底的整个decode，同时保持长prefill实现不动。不通过修改`experimental_c16_tuned`开关做替代，因为official-v4并不按该开关切换旧C16。也不拿两个端对端数值充当三项改动的消融测量。

**32K仍会执行一次warm C16；不能把恢复decode说成32K所有阶段都已恢复232b。** 新组合中这一次C16与64K使用完全相同的实现，遵守普世通用要求。

还对旧/新decode做了相同O3/架构/auto-sync参数的官方编译审查。[证据](../reports/universal_regression_repair_20261028/codegen/audit.json)表明：16384元素的旧IR仍是repeat=1循环，新IR是255+1两次转换，新增helper的固定发布while已常量折叠；没有ABI变化。故不能编造“旧循环已自动合并”或“新while必然增加循环开销”作为根因。AIV代码增加596B、两版固定stack记录32KiB，也不足以证明spill或长指令阻塞。此次处置根据真实回归和执行覆盖撤回decode项，不把未取得的微架构数据包装成结论。

## 固定的源码身份

| 路径 | 本次唯一实现来源 | 适用范围 |
|---|---|---|
| q4/q1 official decode | 232b，完整文件SHA `817a5cbb7738e9ad572e39796838cd915a902c98301c452e21caefe6ceaebc22` | 所有输入/context长度 |
| C16 official reader | 34bc，完整文件SHA `1bcf9ccfe0181f3103b255f67589845a9d83b4e232baa9aa11561b61326d8264` | 所有原本路由到C16的任务 |
| C16 finite helper | 34bc，完整文件SHA `776e507213aba7e7089dcc5c4706944859f4916929e9ad1688a22b90d1abd695` | 同上 |
| C4、current-only、native-aware prepare、writer、merge | 两基底相同原实现 | 原有全部任务 |
| 原生调度、PC/GDN/MTP、16K预算、PT、timer、argv | 两基底相同 | 原有全部请求 |

`generate_official_attention.decode()`默认恢复原实现，`outputs()`显式固定`optimize=False`，因此生产生成链不会再自动加回批量Cast。原变换只作为显式研究函数保留。部署仍编译实际生产reader，不靠probe中的特殊分流取得结果。

新增回归门固定上述文件身份，并核candidate实际算子映射、预算和APC传递；原dispatch和运行时文件本身按旧SHA保护，任何新增长度判断都会破坏身份门。后续不能只改测试期待值来容纳另一项未审清的共用路径修改。

## 验证与交付

本轮复用的数值证据具有明确源绑定：恢复的decode正是上一轮10组新旧配对中的旧臂；保留的C16正是6组配对中的新臂，finite helper对应1507例边界检查。未修改这些内核的任何其它语句。另对新的同库组合执行fresh CANN构建和主机路由/身份回归。实际回执见[验证记录](../reports/universal_regression_repair_20261028/validation.json)。

最终主机回归 **107 passed**。新组合fresh CANN configure/compile均rc0，18份必要源/头在工作区、暂存树、VM编译后哈希一致；同库导出q4/q1/C4/C16完整。原25个冻结配置、timer和母源哈希全部保持。

D.4四问：作用于有界fused FIA操作数发布；继续禁止旧全历史恢复6.5秒结构；撤回decode转换组织，不动其预取、DMA/cross-core协议、FP32状态和INT2数学；无新增生产条件、UB/GM或算子。档案#145/#164约束所有权与RNE，#172记录此次跨阶段组合回归。

固定入口保持：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

两个目标作为同一验收条件：短输入以232b为保护基底，64K保住已取得的收益；任一退化都不能把这一组合判为完成。
