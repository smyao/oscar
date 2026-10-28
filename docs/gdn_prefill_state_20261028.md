# PC 开启时的 prefill 状态修复

本轮修复一个已复现的状态协议错误：**真实 prefill 的短续块被按 MTP decode 执行，读取了未由该前缀提交的 GDN 状态。** 此前档案 #163 只修复了 PC 关闭的 none 模式，错误地把一次固定32K切分测试外推为 align 模式不会出现该问题。本轮将同一正确性规则扩展到 none/align，与输入总长度、数据集无关。

## 触发链与数值后果

使用当前 `split_prefix_checkpoint` 和原生 scheduler，5000-token prompt在合法16K总预算下可分成 `[2304,1536,1156,4]`；实际命中2304 prefix后也可继续 `[1536,1156,4]`。原生只在缓存边界前对齐，末尾区域允许短续块。

1. 前一轮真正prefill把SSM最终状态写到第0列，conv有效历史写到前3项。
2. align预处理在换页时选择已提交的SSM列复制到新页第0列，并将accepted重置为1。conv复制的是原状态段，末尾scratch并不会因此变成已提交历史。
3. GDN builder的fold把这4个prompt token误分类成spec，把accepted从1改成4。recurrent初态按`accepted-1`读第3列，conv历史也偏移3。
4. 即使只把accepted改回1，spec路径的最终状态仍写到第3列，下一轮accepted1会读第0列的中间状态。因此必须同时保留prefill分类、最终提交和正确图选择。

[真实调度/预处理/地址/下一轮复现](../reports/gdn_prefill_state_20261028/align_repro.py)执行固定版本native AST、实际copy-spec及当前插件。有限合成状态`.25`与旧scratch`-42`下，原生GDN golden首输出分别为正确的`0.05517578125`和错误的`-1.8125`；没有INT2参与。完整prefill的末态为`.953125`，只修accepted会留下`.625`，下一decode继续错误。正确结果还与独立闭式递推相符。[数值与来源SHA](../reports/gdn_prefill_state_20261028/numeric_repro.json)。

这说明错误可直接改变模型隐藏状态，且结果依赖旧页内容；不是量化舍入代理的改善。客户端日志没有该分支命中次数，不能把本地反例当成全部LongBench分差或耗时的已闭合归因。

## 默认执行改动

`prefill_state_contract.py`按MTP协议和真实prefill阶段处理none/align，不再以PC开关排除align：保留原始prefill mask和accepted counts。原来被误判为uniform decode图的warm续块走原生prefill执行；下一轮真正decode仍使用原图。MTP宽度从原配置读取，测试覆盖1/2/3/7个draft；没有按prompt/context长度切换版本。

修改仅位于已有外部插件两处接缝。原生文件、GDN算术、OSCAR attention、INT2 codec、PT/clip、窗口、页分配/引用/驱逐、hash/lookup、16K预算、MTP深度及timer均未改。上一轮native_exact继续退役；有代价的APC cohort实验未启用。

## 计算与性能保护

- q4/q1/C4/C16及全部生产CANN源码与性能基底614c完全一致，原32K decode和64K C16组合保持。
- 正常decode保留原FULL图、descriptor和算子；dummy/显式capture选择保留。普通prefill与mixed C4/C16的OSCAR选路不变。
- 23份合成原生切分trace覆盖4K至64K、16K预算及0/1/31条前置q4 decode占用。普通固定长度trace不触发改动；变长20740请求可形成末4续块。若32K/64K也真正形成这种prefill，按同一规则纠错，不保留错误状态读取。
- 新增的是align模式下原有CPU镜像的O(batch)阶段判断；同时不再执行无效fold的CPU检查与clone。没有新增KV页、修复用状态补拷或历史恢复。**被纠正的那一步改用原生prefill已有的gather/chunk/store及eager执行，不能把它写成该步指令与耗时完全不变。** 这是正确阶段的选择，不是异常后的回退或关闭正常decode图。

工作量/路由证据见[保护报告](../reports/prefill_state_protection_20261028/report.json)。这些是源码和主机合同，不替代用户性能数据集的整轮结果。

## 验证与交付

207项相关主机回归通过，覆盖真实prefix命中、同块/跨块预处理、有限旧状态污染、最终提交/下一decode、混合C4/C16、正常图保留、可逆插件注册、固定入口参数及全部生产CANN身份。[运行日志](../reports/gdn_prefill_state_20261028/host_tests.log)、[验证清单](../reports/gdn_prefill_state_20261028/validation.json)。没有新CANN修改或设备实验；四项质量的十轮top3验收口径不变，不填造通过分数。

固定candidate入口自动启用，无新参数：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

仅实际纠正误分类步骤时打印一次`PREFILL_STATE_FIX cache_mode=align route=native_prefill mtp=enabled decode_graph=retained`。该行表示分支发生，不是质量验收标志。
