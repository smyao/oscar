# 双段 packed STORE 与 request sidecar 实验

档案依据：G26–G34、#13–22 保留独立 PR 数值与完整状态；#37–49 保留 MTP 回滚和身份；#69 只在设备侧消费元数据；#126/#145/#154 用明确流水事件保护 UB；#127/#128 区分双段物理页与 metadata128；#155 按字节比较 BF16 NaN poison。已亲读启动文档 D.4 完整原始打点。这里只是独立原型，未接生产路由。

## 接口

约定生产绑定名为 `rotate_clip_store_request_window_out`；本实验提供的 C++ 两核入口为 `oscar_ascend::rotate_clip_store_request_window_launch`，完整声明见 `request_window_launch.h`。

张量顺序为 `key,value,rk,rv,slots,positions,qstarts,window_map,packed,wk,wv,tags,packed_status,window_status`。标量沿用旧几何，并增加尾部 `windowCapacity,positionBits`。launch ABI 还显式传 `tokens,requests,heads,dim,dtype`，由绑定从张量与 metadata 推导。

- `slots`、`positions`：int64 `[N]`，物理 FULL 槽与完整逻辑位置分别解释。
- `qstarts`：设备 int32 `[requests+1]`；允许末项小于 N，剩余位置必须是负 slot padding。
- `window_map`：设备 int64 `[map_rows,2]`，允许 `requests<=map_rows<=windowCapacity`，kernel 只读前 requests 行。fixture 特意使用 `[requests,2]`。各行按请求顺序存 `(lease,epoch)`；lease 是稳定所有权，不是 batch row。
- Native FULL capture 允许非空 qstarts 配全部负slot：先扫描成功live行，全部padding时不读取/要求该request的map，返回0且不发布。provider将未发布或padding request的map写成`(-1,0)`，从而不参与其他live请求的lease重复判断；不能用其他活跃request的正lease标记padding。
- `packed_status`：设备 int32 `[N,Hkv]`；保留旧状态优先级：负 slot=0、越界 slot=1、负 position=4、K/V 数值或 metadata=2/3。
- `window_status`：固定预分配设备 int32 `[windowCapacity]`。每次 kernel 都清零非请求尾行，可直接交外部 `status_guard` 检查。

| window status | 含义 |
|---|---|
| 0 | 成功或无写入 |
| 1 | qstarts 范围非法 |
| 2 | lease 或 epoch 越界/非法 |
| 3 | 两个非空请求同时使用相同 lease |
| 4 | 一个成功 packed token 的逻辑 position 越域或非严格递增 |
| 5 | heads/window/positionBits/请求容量几何非法 |
| 6 | qstarts[0]不等于0；所有请求拒绝窗口发布 |
| 7 | qstarts末项之后出现正slot；仅req0扫描一次并报错 |

任一请求元数据错误时，该请求不发布任何 sidecar K/V/tag。末尾正slot的全局错误仅由req0在O(N-tail)扫描中检测：req0不发布，其他请求可能已发布自身合法窗口；这不是跨请求事务回滚。packed 阶段也可能已经写入有效 packed 行。**调用方必须在后续 reader 消费前对两份 status 都执行 fail-closed 检查**。输入 buffer 尺寸、段容量、stride、请求数量与最大128请求资格必须先由绑定检查；实验不能处理任意错误 tensor 指针。

## 写入与竞争边界

`generate.py` 原样提取 `rotate_clip_store_striped.cpp` 的 RotationEngine、ClipQuantize，记录原文与 helper SHA。只改变 packed 目的地址：

```text
page,inPage = divmod(slot,2*segmentTokens)
segment,within = divmod(inPage,segmentTokens)
address = offset + page*pageStride + segment*blocks*pageStride
        + (within*Hkv + head)*(D/2+8)
```

不再写旧的每物理页精确快照。原生 GDN 所属 id 和 guard/gap 字节不在本核写集合中；上游必须确保本次合法物理 slots 唯一。

第二核在同 stream 上启动，所以它读取的是第一核所有 head 的完成状态。每 request 的唯一 block 以64行批量 DMA 扫一次新 token 元数据，UB 保存323个 ring 的最后成功 writer；只有所有 Hkv 的 packed status 都为0才选择该 token。它不逐 token 扫 later，不扫描历史。按完整逻辑位置选择 `p<S ? p : S+p%(R+spec)`；原 BF16 K/V 写完后才发布 `(epoch<<positionBits)|p`。

每个 request/ring 只有一个 writer。重复 lease 明确失败；head 在该 owner 内顺序处理。请求数不超过128时另有独立的 O(requests²) lease 检查，比较的是一次 DMA 到 UB 的短 metadata，不是 tokens。总工作为 O(N×Hkv + requests² + requests×323×Hkv×D)。短 q1 使用 N128、32个 live、Hkv2、128状态容量 fixture。

sink64 + recent256 + speculative3 共323行。保留最后259个非 sink position 可支持拒绝0–3后的完整256 recent窗口；reader 必须同时验证 epoch、完整 position 及当前位置可见性。epoch 只隔离旧内容，不能抵消异步旧 writer：请求释放/复用、两个 STORE launch 和 reader 必须保持同 stream 顺序。

## 数值与验证

CPU probe 对 packed/status 做三方比较：独立 PR golden、原 striped 核、实验 packed 核。原核通过等价 slot 映射把双段映为 `2*blocks` 个旧段，数值数学和最终地址一致。新窗口另与逻辑位置字典 oracle 的完整 BF16/tag 字节比较；包括未写 NaN payload、signed zero、prefix/gap/suffix guard。

主机 fixture 共19例：小请求、非128对齐 segment、物理页跨界、请求重排、negative holes、rollback0–3、epoch复用、all-padding poison、多 head 数值/地址故障、D256 clipping+Hadamard、N128/32live/Hkv2、非法 epoch、重复 lease、position越域、非法 qstarts、正qstarts起点、正slot尾部、全部capture padding未发布map、128个非空request行中仅32行live且其余map未发布。

```bash
python3 csrc/experiments/request_windows_store/generate.py
.venv/bin/python csrc/experiments/request_windows_store/fixture.py --output /private/tmp/oscar-request-windows-store-fixtures-20261009
.venv/bin/python -m pytest -q csrc/experiments/request_windows_store/test_fixture.py
```

取得 root 的 VM 排他时段后，再配置独立 CPU build（不得将以下步骤当作已经执行）：

```bash
cmake -S csrc/experiments/request_windows_store -B build/request-windows-store-cpu -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DOSCAR_REQUEST_WINDOWS_CPU_DEBUG=ON
cmake --build build/request-windows-store-cpu -j2
timeout --signal=TERM --kill-after=5s 120s build/request-windows-store-cpu/oscar_request_windows_store_cpu /path/to/fixtures/small_bf16
```

D.4 四问的设备目标写在两份 kernel 头部：phase1_stores，历史208.6–216.0ms/16K；禁止6499.8–6655.1ms全历史恢复与约725ms host准备。

执行记录在 `reports/request_windows_store_20261009/validation.json`：主机18项通过；独立CANN ascend910b4及CPU-debug构建通过。首个`small_bf16`已实际通过官方CPU-debug三方packed/status与sidecar全字节门，rc0。其余18个CPU case尚未由本记录确认通过；随后suite启动的自动审批服务多次deadline超时，没有启动进程，已把排他时段交给reader代理，允许其在reader结束后顺序接跑。无NPU、graph或性能结果。

构建前须加载官方环境：`source /usr/local/Ascend/cann-9.1.0/set_env.sh`，并明确`CMAKE_BUILD_TYPE=Release`；本次初始构建分别因`llvm-objdump`不在PATH、SDK merge参数缺少build-type失败，原日志保留，没有改量化数学。
