# node93：接续已完成的基线部署

依据档案 #27/#29/#74/#75/#94/#125/#175/#176/#177，以及用户提供的成功基线教程和 `vllm-ascend.0.23.0.patch`。用户确认已完成教程 2.2，且成功基线使用物理卡 0–3 和 `device_rdma`。后续复用这套环境，不重新安装 SDK、不重复应用原生补丁。

## 已核实的对应关系

| 项目 | 基线证据 | OSCAR 接续方式 |
| --- | --- | --- |
| SDK | 教程安装 MemFabric 1.2.1 与 MemCache 1.2.0 | node93 配置指定 `offload_sdk_source: installed`；部署只补 pyzmq，不再让 pip 解析两个通用 SDK 包 |
| 环境 | CANN → ATB → MemFabric → MemCache 四个 `set_env.sh` | 复用 `env_node93_memcache.sh`，在同一 Bash 设置并继承到正式服务 |
| 设备与端口 | 用户本轮指定 0–3、7878 | `benchmark_node93.json` 固定该组合；不用 `--rear-cards` |
| 中央服务 | Meta 5000、Config Store 6000 | 已运行时经 SDK 元数据查询确认后复用；未运行时 `MetaService.main()` 前台启动；模型 HTTP 端口独立 |
| 内存及存储 | 每 LocalService 40GB、`device_rdma`、70/60 水位、DirStore | 既有配置准备只更新这些已给定值；不补造缺失的 `dir_store.conf` |
| 初始化 | 原生 `MemcacheBackend` → `DistributedObjectStore.init(local_rank, init_bm=True)` | 保留同一调用线程和参数；scheduler 使用原生 `init_bm=False` 客户端 |
| KV 状态 | 基线存原生 KV | 外部 connector 保存 OSCAR 的 packed KV、窗口、tags 以及原生 GDN 提交状态；不能把通用 connector 名称直接替换进 OSCAR 布局 |

`installed` 是安装来源选择，不是版本检测或兼容性保证；缺少 SDK 仍由原有导入/初始化错误明确失败。它不证明旧 pip 命令曾替换现场包，也不被当成此次 HCCL 错误的根因。node93 不需要另行执行会解析通用 SDK 依赖的 `pip install '.[offload]'`。

## 私有补丁的实际作用

用户已确认补丁在现场集成。它只修改以下接口，不涉及 HCCL 初始化、协议、卡号或地址：

- `Backend.exists(keys, flags=0)` 和 `batch_is_exist(keys, flags=0)` 传递标志。
- `MemcacheBackend.exists` 调用 SDK 的 `batch_is_exist(keys, flags)`。
- `KVTransferThread.lookup` 默认 `flag=1`，两个发送线程的写前去重查询显式传 `1`。

OSCAR 直接调用 SDK，因而需要在自身 transport 接口透传该第二参数。现有查询用于读命中、恢复前校验和联合发布校验，保留默认 `0`。OSCAR 的 payload 使用独立 generation 的不可变 key，当前没有原生发送线程的写前去重调用点；不为套用补丁新增查询或把所有读查询改成 `1`。补丁没有提供 SDK 标志枚举，不能凭名称将 `1` 解释成“只查 DRAM”或“强制 SSD 预取”。[官方公开 Python API 文档](https://github.com/Ascend/memcache/blob/master/doc/memcache_python_api.md)也未解释该标志，现场补丁是本轮参数透传的依据。

## 启动参数的边界

附件要求 `--disable-log-stats`，它在当前原生版本同时关闭 EngineCore stats 和默认 Prometheus logger，不只是隐藏终端吞吐日志。显式服务探针必须独立验证实际 MTP/抢占结果，不能通过重新开启 stats、跳过验收或补造 Prometheus 指标来配合这个开关。正常服务不启用探针观察代码。

附件的 4096 token budget、APC 开启、YaRN×4 与 OSCAR 既有的 16384、APC 默认关闭、模型自身 RoPE 不同。它们改变调度或数学，不是 MemCache 接口补丁；本轮保留已经约定的 OSCAR 参数，显式 `--enable-prefix-caching` 仍可开启原生 align 路径。MTP 的 `qwen3_5_mtp` 拼写由当前原生配置规范化为 `mtp`，两侧均为 3 draft tokens、draft eager，无需更换方法。

教程的双实例、负载均衡和第二组卡不属于本轮前四卡单实例服务。客户端应连接实际服务器地址的 7878 端口。

## 从教程 2.2 后接续

在已准备好的同一容器内、更新后的 OSCAR 工程目录运行。中央服务入口可以重复执行：已有服务通过元数据查询后直接复用并返回，不再重复初始化 MetaService。

终端一：

```bash
bash scripts/start_node93_memcache.sh --prepare-host
```

入口先设置 SDK 环境并只读检查控制服务状态，再决定是否准备主机。已运行的服务通过 `init(0, init_bm=False)` 与单键 `batch_is_exist(..., 0)` 查询后，打印 `reusing existing MemCache MetaService` 并返回 0；即使带 `--prepare-host`，也不会修改 hugepages、重挂载 `/dev/shm`、写 SDK 配置或再次启动中央服务。查询在独立子进程中执行，最多 15 秒并关闭自己的客户端，不请求 LocalService、NPU 缓冲注册或数据传输；不能把这个结果当成 SSD/NPU 验收或现有服务运行参数核实。

仅在控制端点均未监听、SDK 进程锁未持有且本入口获得启动互斥锁时，`--prepare-host` 才应用教程给出的 `nr_hugepages=200000`、`/dev/shm` remount 为 8G，并执行 `mkdir -p /home/kvcache`；随后准备既有 MMC 配置并前台启动 MetaService。主机准备已做过时可省略该参数。任一步失败保留错误和退出码，不继续启动，不自动清空已有缓存。只查看这三个主机操作可用 `python3 -m tools.node93_host --plan`。

只有一个端点监听、SDK 元数据查询失败/超时、已有启动尚未就绪或锁状态无法读取时，入口在主机/配置变更之前明确停止。SDK 锁检查依据[官方 `CheckIsRunning` 实现](https://github.com/Ascend/memcache/blob/master/src/memcache/csrc/meta_service/mmc_meta_service_process.cpp)：只读 `F_GETLK` 查询 `/tmp/mmc_meta_service.lock`，残留文件本身不算运行，绝不删除或接管 SDK 锁。OSCAR 自己的非阻塞锁覆盖检查到前台服务退出，防止本入口并发准备；绕过入口的外部并发启动仍由 SDK 自己的最终进程锁裁决，其真实非零返回不会被吞掉。公开源码并不是现场私有 SDK 的逐位版本证明。

这次幂等启动修正的本地主机回归为 **121 passed**，包含真实子进程/文件锁与超时回收，但 SDK 本体由模拟模块替代；未连接 node93 或运行 NPU/SSD 测试。详见[专项验证](../reports/memcache_idempotent_startup_20261110/validation.json)。

终端二（中央服务成功启动后）：

```bash
bash scripts/start_node93_oscar.sh
```

该入口在当前 Bash 依次 source 四个 SDK，再执行：

```bash
bash scripts/install_serve.sh --config configs/benchmark_node93.json --variant candidate
```

因此仍完成 OSCAR 安装、当前算子编译、旋转文件准备及正式服务启动，默认开启卸载，前四卡、7878、统计关闭。`--enable-prefix-caching` 等已有显式参数可以继续传给该入口；它不会默认重跑 benchmark。直启本身不是验收证据。`--plan` 原样传给安装入口，但仍先 source 现场 SDK；开发机只查看安装计划用上面 `install_serve.sh` 命令追加 `--plan`，不运行 node93 的环境/主机准备脚本。

需要完整功能探针时，沿用同一配置：

```bash
source scripts/env_node93_memcache.sh && \
bash scripts/install_probe_serve.sh --config configs/benchmark_node93.json
```

统计关闭时，探针仅在显式观测窗口记录原生 scheduler 已完成的 MTP 验证调用与抢占结果，配合已有全 TP 保存/恢复、prefix/chunked prefill 和 token 一致性检查；不重新启用原生 stats。记录的 scheduled draft slots 包含原生 padding/invalid 槽位，不冒充 Prometheus 的有效 draft 分母或接受率。普通正式服务没有这个观测窗口。

当前 candidate 完整部署走既有 BF16 correctness-only 流程和专用卸载探针。依赖原生 Running/Waiting/token counters 的旧合成性能诊断不适用于这个统计关闭配置，其原有证据门保持；本轮不改 timer 或降低旧性能门。用户 HTTP benchmark 可以继续测客户端真实时延、吞吐与输出；关闭的服务器统计记为未采集，不将客户端结果换算成未观测的抢占、有效 draft 或 SSD I/O 证据。

完整探针通过后复用进程进入正式服务时，统计关闭的配置不再启动依赖这些指标的被动观察器，报告明确标记为 `unavailable / disabled_by_config`。仍检查服务存活并保留退出/清理；窗口结束后停止探针记录及持续文件轮询。

## 故障证据与验收边界

用户最新 `hccn_tool` 输出确认查询当时物理卡 3 link DOWN、health Receive timeout；光模块存在，所列光功率在给出的阈值内，LOS/LOL 标志为零。这些信息不能单独定位为光模块、线缆或交换机故障，也不能否定用户在同卡同协议下成功运行基线的事实。

已有失败 plog 明确在 MemFabric 建通信域时报告 RoCE link down；用户补丁未修改这一阶段。保留两组事实，不能声称透传查询参数或关闭统计已修复初始化故障。接续代码的本地主机验证，与 TP4 服务启动、SSD 实际写入/回读、prefix/chunked prefill/MTP 图执行、模型质量和 benchmark 分开记账。agent 未连接真机或运行性能测试。

最终组合主机回归 **390 passed**，涵盖 SDK 旗标/严格错误、真实本机 IPC、原生方法 AST 的异步输出/MTP/抢占、none/align 卸载流程、启动脚本、部署参数、hook 组合与异常清理。正式服务交接使用本机模拟 HTTP 服务测试；没有真实 SDK/NPU 传输。候选校准身份与 PT 路径对照保持相同，shell 语法和 diff 检查通过。具体范围见 [验证记录](../reports/node93_baseline_alignment_20261110/validation.json)。
