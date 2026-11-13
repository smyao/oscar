# 卸载路径批量扩展审查与多抢占计数读取

审查基底 `de0c85b`，2026-11-13。本项仅修改 `oscar_ascend/offload/connector.py::handle_preemptions`，在同一 metadata 包含多个私有抢占请求时复用现有 `WorkerCheckpointTracker.resolve_many`。没有修改 CANN、模型数值、调度、PC 语义、启动参数或 SDK 协议，没有连接真机或测速。

## 已实现的有限改动

原路径对 N 个抢占请求逐条调用 `resolve`。即使记录来自同一个 `_CounterSnapshot`，每条仍单独等待同一 ready event，并用 `.item()` 读取 base、raw accepted、以及 speculative 时的 next accepted。MTP3 的 N 条请求产生 N 次 event 等待和 3N 次标量读取。

现在只有 N>1 使用现有 lazy batch resolver。它按 `(CounterSnapshot identity, ready_event identity)` 缓存本次调用内的主机整数列，每个快照只等一次，且只读取该快照的有效行。不同 step/stream 仍各自等待；没有跨调用缓存设备计数。单请求继续走 scalar 路径，空列表不创建 counter mirror。

每条仍严格按照 resolve → synchronous PUT → ack → forget 顺序执行。后项的 generation、缺失记录、accepted、producer 或 SDK 失败不阻止前项保存/确认；后项失败不能确认或释放其记录。每次私有保存的 current-stream synchronize、完整 FULL/GDN 字节集合及失败 rollback 均保留。只共享 host counter 读取，不合并请求载荷，不引入后台保存或提前释放源页。

## 本地主机功能计数

用实际 `handle_preemptions`、真实 CPU tensor 指针及 strict transport 的 SDK 形状内存存储，对照旧 scalar resolver，得到以下调用次数。它们是功能工作量，不是 NPU 延迟或吞吐测量。

| 同快照 MTP3 抢占数 | 旧 ready-event 等待 | 新等待 | 旧 scalar item | 新 none / align 主机列读取 | 每条保存 stream barrier |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 3 | 仍 3 个 scalar item | 1 → 1 |
| 16 | 16 | 1 | 48 | 2 / 3 | 16 → 16 |
| 28 | 28 | 1 | 84 | 2 / 3 | 28 → 28 |

每次列读取为该快照有效请求数，未读取 full-capacity 尾部。非 speculative 情况旧值为 2N 个 item，新值为两列；true prefill 的 canonical accepted=1 不变。测试逐字节比较 FULL packed/windows/tags、canonical conv/SSM 和全部 receipts；none/align、spec/non-spec、1/16/28 请求均与旧路径相同。

回归命令：

```text
.venv/bin/python -m pytest -q tests/test_offload_preemption_batch_counters.py tests/test_offload_cleanup.py tests/test_offload_native_bridge.py tests/test_offload_prefix_batch_counters.py --junitxml=/private/tmp/oscar-20261113-preemption.xml
```

最终代码实际结果：**71 passed in 1.39s**。新增 18 项覆盖实际私有保存入口；其余覆盖原生抢占/恢复桥、TP 清理和既有 prefix batch 行为。CPU 存储、受控 event 与失败注入不等同于真实 SDK、NPU、graph 或端到端性能验收。

## 现有行为与归因边界

- `connector.py::_load_payload/start_load_kv` 已把对象规划、producer event 等待及同步 SDK get 移到后台线程；一帧 restores 共用一个 producer event，async 每 rank 只校验并恢复自己需要的载荷，all-TP 完成后才准入。不能重复记为本轮新优化。
- `connector.py::build_connector_meta/observe_and_save` 已有同 step 单 writer、完整 manifest 判定、prefix batch counter 及首次保存一次 stream barrier。本项只补私有抢占支路。
- 无 prefix 保存的正常 decode 仍 `record_step` 保留小型计数快照，但在 `observe_and_save` 中直接返回，不执行 prefix producer 等待或 PUT。`state.py::record_step` 每步有 2/3 个设备计数 copy、一个 event 记录和 O(B) 记录维护；这是当前可抢占所有权机制，不能无证据删除。
- `metadata.py::MetadataClient._rpc` 为同步请求；rank-zero worker 的 `MetadataServer` 与 payload SDK 使用同 transport 锁。prefix 查询及发布可能等当前 SDK batch，但不能从现有 aggregate profile 定量出等待占比。取消序列化、增加 SDK 并发或跳过完整性检查需要另行证明 SDK 和所有权协议，本项未做。
- `integration/prefix_stats.py` 每初次准入请求只打印一次；`step_stats.py` 只读原生 host counters，完成主模型 step 每 100 次/排空时输出统计，没有 NPU 回读。没有证据说明这些统计是主要瓶颈。
- `offload/worker.py::CheckpointWorker` 在工程中只有测试引用；实际服务使用 connector 的保存/恢复路径。不能把该辅助类的独立 sync 调用当作生产重复同步。

用户已有 BS16/28 结果说明相同数据集、PC 命中率相同而吞吐未提升；Running 峰值不等于各 step 的实际 batch。916/1200=76.33% 纯 decode 是次数比例，不是时间比例，也没有本轮私有抢占次数。因此本改动消除一个确定的按抢占数放大的冗余，但**不能声称它已解释或解决现场 BS28 吞吐停滞**。

已有 BS8 两侧为同批 11 请求、32K 输入、50 输出、MTP3。归档算子累计中 C4+C16 为 13.782876 秒，占 OSCAR 表总累计 31.437%、全部 CV 累计 90.613%；decode 命名家族仅为 1.427912 秒、全表 3.257%。这支持继续优先检查 prefill/混合阶段的 C4/C16，不支持仅按纯 decode step 比例忽略它们。两侧算子总累计 43.842644/29.188942 秒不是 wall 比；调用计数与 shape/phase 对应未展开，部署提交未绑定，不能把 BS8 聚合结果直接外推成当前 BS28 的 HBM bound、L2 命中率、通信等待或卸载瓶颈。来源：`reports/bs8_profile_20261112/comparison.json`。
