# OSCAR PD 分离（MemCache）

从 `oscar-new-7-offload` / `b18f7e0` 增加显式 P/D 角色、完整状态交接和 OpenAI HTTP 代理。不传 `--pd-role` 时保持原来的融合服务和卸载策略。只改外部插件及入口；原生参考树、CANN、PT/clip/旋转、q4/q1/C4/C16、MTP3、图模式、16384 预算与 timer 不改。

档案依据：#27/#77/#78（外部接缝与轻量加载）；#34/#36/#37–49（容量、页所有权与提交状态）；#94/#125（错误可见）；#127/#128（none/align 物理页）；#157/#163（真实 prefill 与 MTP 已提交 GDN 游标）；#171（共享页生命周期）；#172/#174（保护计算实现）；#182/#183（外部 prefix 语义和缺失索引查询）。本地测试不是新真机记录。

只读原生依据：`references/vllm-ascend/.../kv_p2p/mooncake_connector.py` 的 `_state_prefill_token_count` / `_truncate_request_for_prefill` / `request_finished`；`references/vllm/vllm/v1/core/sched/scheduler.py` 的 `_free_request` / `has_finished_requests` / `_update_from_kv_xfer_finished`；Ascend `model_runner_v1.py` 的 `handle_preemptions`、postprocess 与空步执行顺序。

## 交接流程

1. 代理将原始请求发给 P，指定 `do_remote_decode=true`、`max_tokens=1`，禁用 P 的 streaming/stop/EOS 提前终止；用户采样参数留给 D。
2. 外部 `Scheduler.add_request` hook 在原生 prefix 查找之前把 P 的 tokenized prompt 截为 N−1，同时截断 token 列表和完整 block hashes。原生 chunked prefill/APC/MTP 继续工作。P 产生的一个采样结果丢弃。
3. P 完成后返回含 nonce、原始完整 prompt 摘要和 namespace 的 `kv_transfer_params`，并通过原生 `delay_free_blocks=True` 保留源页。下一 worker 步（包括最后空步）在请求移除/页复用前，按 worker 确认的提交游标保存 packed/metadata、window K/V/tags、canonical GDN conv/SSM。末尾半页保存完整物理页，读取范围受已提交 token 约束。
4. 每 rank 保存成功后报告 `finished_sending` 和交接回执。原生聚合器收齐全部 TP；scheduler 验证完整 payload、发布 manifest，再释放源 HBM 页。缺 rank、提交位置错误或 SDK 失败不发布。
5. 代理把凭据随**未改变的原始请求**交给 D。D 验证完整 prompt tokenization/cache salt 及模型/PT/math/layout/TP namespace。manifest 尚未发布时限频等待；超过 `kv_offload.timeout_seconds` 明确报错，不转为普通 prefix miss 重算长 prompt。
6. D 分配物理页后进入原生 `WAITING_FOR_REMOTE_KVS`，复用异步恢复线程和新页清零 event。所有 TP 完成前不得 forward；实际准入时先重置 GDN/MTP 游标，再计算原始最后一个 prompt token，生成全部用户输出。

这是原生混合模型的 h(N−1) → h(N) 边界，避免 GDN 重复推进。N=1 显式声明 0 个可复用 token，不传输不存在的状态，D 正常计算该 token；未增加按长度选择内核/数学版本的策略。

PD 使用独立索引，不要求对齐 prefix 命中；none/align 均支持。若本地命中已覆盖 N−1，可使用原生已验证状态。D 后续抢占继续使用原有私有 swap。

## 生命周期和成本

- P 的源页保留至全部保存回执；D 取消时目标页仍由原生持有至全 TP receive completion，不提前回收 DMA 地址。请求、等待和匹配记录在完成/取消时清理。
- P 失败/取消不返回有效交接；HTTP 客户端断开会关闭上游连接；P 失败或缺凭据时代理不调用 D。
- 交接 payload 是独立、不可变的 MemCache 缓存对象，由存储水位驱逐回收，不在 D 读完后删除，也没有长期 lease。客户端不再提交 D、进程崩溃或取消后，已写缓存可能驻留到驱逐。已发布 payload 被驱逐会明确失败。
- 每次交接有一次完整压缩状态保存和恢复，存在网络、DRAM、同步和 host manifest 成本；没有全历史 BF16 物化，不承诺吞吐提升。
- N−1 边界可能改变与融合运行的分块舍入轨迹；字节恢复正确不等于模型输出逐位等价或质量通过。

## 部署

两实例必须连接同一可达 MemCache，使用相同模型/PT、OSCAR 数值配置、TP 数及 none/align 布局。`KV_OFFLOAD_READY namespace=...` 应一致。沿用已配置 SDK；SDK 本机服务/地址配置须能同时容纳所选实例。项目不启动或修改中央服务。

已配置 node93 环境时，同机前后各四卡示例，在三个终端依次启动，先完成 P 的安装/编译：

```bash
# P：0–3 卡，7878，沿用 node93 的 APC 开启默认值
bash scripts/start_node93_oscar.sh --pd-role prefill --port 7878

# D：4–7 卡，显式使用不同 HTTP 端口
bash scripts/start_node93_oscar.sh --pd-role decode --rear-cards --port 7880

# 标准库代理；客户端访问代理地址
python3 -m tools.pd_proxy --prefill-url http://127.0.0.1:7878 \
  --decode-url http://127.0.0.1:7880 --host 0.0.0.0 --port 8989
```

跨机器部署时各自可保留前四卡/7878，代理填写实际 P/D 地址。两侧复用同一 PT，并满足已有 namespace 的模型文件身份要求，不能分别生成不同旋转后强行兼容。

固定直启入口已接通：`bash scripts/install_serve.sh --variant candidate --pd-role prefill` 或 `--pd-role decode`；同机 D 叠加 `--rear-cards --port 7880`，需要 APC 时两侧都传 `--enable-prefix-caching`。不指定角色的固定 candidate 命令保持融合运行。`install_probe_serve.sh` 也接受 `--pd-role`，现有 probe 验证本实例算子/卸载/服务，不冒充跨实例 PD 验收。

只读计划：

```bash
python3 -m tools.install_serve --variant candidate --pd-role prefill --plan
python3 -m tools.install_serve --variant candidate --pd-role decode --rear-cards --port 7880 --plan
```

`tools/serving_variants.py` 统一处理角色和端口。服务实际收到专用 connector、`kv_producer`/`kv_consumer`、对应 `pd_role` 和 `load_async=true`。baseline/native variant、显式关闭卸载、非法角色/端口与 PD 冲突时拒绝。

代理支持文本 `/v1/completions`、`/v1/chat/completions` 和 SSE，n=best_of=1；`/health` 检查两侧，`/v1/models` 转发 D。完整 prompt logprobs、批量 prompt、MM/embedding、LoRA、可追加输入的 streaming session 暂不支持且明确报错；HTTP streaming 输出正常支持。高级工具调用/chat 模板未取得现场验证。代理是一对 P/D 的路由器，没有多实例负载均衡或生产网关的认证管理。

## 验证

新增测试覆盖独立实例、TP1/TP4、none/align、完整/半页 FULL/GDN 字节、原生最后空步/延迟释放/异步完成/游标重置、全 TP 屏障、错误提交位置/ticket/提示词、损坏或被驱逐 payload、超时、取消、N=1、启动配置、真实本机 HTTP/SSE/断连。

最终命令、结果和文件指纹见 `reports/pd_separation_20261119/host_validation.json`。既有卸载、prefix、插件、真实 prefill 状态和固定入口回归一并执行，没有放宽断言或冻结 hash 门。

未连接真机、未运行性能实验；真实 SDK/NPU 跨实例传输、模型 PD 生成、图回放、质量和吞吐仍待设备验收。本机 HTTP stub 与 CPU tensor 证据不升级这些状态。
