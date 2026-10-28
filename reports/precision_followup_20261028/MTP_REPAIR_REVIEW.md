# MTP 精度 cut 修复反审

2026-10-28。档案 #37–49/#126/#145/#157/#170–173；启动 D.4 完整原始日志已阅读。本次只增加审计脚本和报告，没有修改生产源、窗口、启动参数或 timer。

结论：**当前 fixed-per-forward cut 确实使 q4 与 serial q1 的近似目标数学不同，但不能把改为 query-relative cut 当作已证明的质量优化。该修改会把原本正确的 q4 结果改错。** 仅修 decode 源也不能覆盖所有真实 mixed 路由。

复现：`.venv/bin/python reports/precision_followup_20261028/review_mtp_cut.py`。当前源码 SHA、真实输出及路由位于 `mtp_cut_review.json`。

## 1. query-relative cut 会降低已有精度

当前 `attention_cv_official_decode.cpp:500–534`、C4 同族 Softmax、`attention_cv_official_cluster16.cpp:857–912` 使用固定 `max(min(sink,C), C+1-R)`；当前 task 的 source1 和 `LogicalPosition` 也使用同一固定起点。

现有反例 C=768、逻辑 query=770、被关注旧 token=514、R=256。使用当前 144B codec、mode0、K=.96/V=.92、D256/Hq6/Hkv1：

| 算法 | cut | 输出第 0 维 | 同一固定两类 head argmax |
|---|---:|---:|---:|
| 当前 q4 | 513 | 0.2001953125 | 0 |
| 改 query-relative | 515 | 0.333984375 | 1 |
| 全 BF16 | 不量化 | 0.2001953125 | 0 |

head 为 `[.25-x, x-.25]`。当前 q4 与全 BF16 输出逐位相同。统一 q1/q4 会消除此例的分组依赖，但这是把更精确的一边降为较差的一边，并非恢复 native 数学。一般而言，q4 第 j 行会把原来的 j 个 exact 缓存 token 降为 INT2。若对所有 prefill 行应用同一改法，最多会把全部原 recent 尾段降为 INT2；current chunk 仍保持 exact，因此也没有消除一般 chunk-dependent 模型轨迹。

## 2. 现有 259 环不是免费精度余量

`lifecycle.py:39–42` 明确规定 ring=recent+spec，额外 3 行用于容忍未接受 draft 写入。

实际 writer `store_aligned_int2.cpp:150–184` 在 verify 结束后写入所有合法 slot；快照按 `sink + inPage % ring` 放置。构造 q4 写入768..771，仅已知 token768提交，下一 context=769：

- 当前 R256、cut514：所需尾段 tags 全部正确。
- 若把 extra spec 也作为精度窗口、cut511：511/512 对应 tag 已变为770/771，reader 将报 error4。

所以“不增加存储、直接读满259行”的方案不成立。已有测试 `tests/test_lifecycle.py` 按所有接受长度0..3和跨物理页边界证明原 R256 窗口保留；它没有保证扩大的有效精度窗。此审计不建议增加窗口或更换 writer 提交规则。

## 3. decode-only 改动会遗漏 mixed 路由

脚本直接调用当前 `variant_config`、`plan_mixed_decode` 和 `select_cv_op`：

| 每请求 query 数 | 原 mixed split | q4 实际 reader |
|---|---|---|
| `[4,768]` | 分割 | official decode |
| `[768,4]` | 不分割 | official C4 的 C1 路径 |
| `[4,8192]` | 分割 | official decode |
| `[8192,4]` | 不分割 | official C16 的 C1 路径 |
| `33×4 + 8192` | 不分割，short prefix 超过128 | official C16 的 C1 路径 |

现有 `attn_state` 是 batch 级信息，prefill batch 内可含 q4 请求；`is_draft` 仅区分目标/草稿层，不区分目标 batch 内各请求的 decode/prefill。按现有 phase 标记只改 decode 不能覆盖以上情况。按 qcount 另加数值政策又会引入新的 query 长度分界；本次不采用。

若未来独立定义 query-relative 新数学合同，最小低层变化应只改 Softmax 的每行可见集合，保持 task 枚举/LogicalPosition 的最早候选起点；C16 fullyVisibleHistory 必须按最早行 cut，fullyVisibleWindow 必须按最后行 cut 检查中间洞。不能全局替换 `context+1-R`，否则 candidate 坐标和历史簇资格改变。现有所有 source 的 KV256 迭代数可保持，但窗口 mask/有限性慢路径会增加，不能称为零成本。

## 4. 未发现新的未提交 token 发布缺陷

本次继续检查 native 发布和 writer，发现的事实不支持把未接受 draft 当成已证实污染：

- `kv_cache_manager.py:425–434` 将 publication cap 限为 `min(total_computed+new, request.num_tokens)`，明确排除尚未接受的 draft。脚本执行该真实 AST 方法，覆盖 context767/768/2301/2302/2303/2304/4607；q4+3 lookahead 的 cache end 均只为 context+1，未把跨2304边界的未提交draft发布出去。
- `scheduler.py:1415–1431` 在输出更新中按拒绝数回退 computed/placeholders；`request.py:220–236` 只在真实追加输出 token 后推进 block hashes。
- FULL 页内 slot 是 native 分配的请求可写区间。writer 在 attention 读取后发布，拒绝位置之后会被下一次合法写覆盖；原 259 环保留已经提交的 R256 exact 尾段。完整 prefix 页按物理页分别持有快照，不会被后续物理页的环写覆盖。
- 已执行 `tests/test_lifecycle.py tests/test_mtp_logical_slots.py`：71 passed。它们是 CPU/真实原生 AST 和所有权测试，不是 NPU、完整模型或性能验收。

这里没有证明所有异步现场都不存在缺陷，但尚无新的可复现 writer/hash 错误可作为默认改动依据。前述 APC 不同计算轨迹的联合状态拼接是独立问题；不能把这份 MTP 数学反例冒充它的直接设备因果证明。
