# 长历史 striped INT2 候选（2026-09-30）

依据：档案 #126/#129/#145/#148–154、启动文档 D.4、当前工程源码、用户已提供的真实日志。用户允许明显的 1.5 倍优化点；此文区分算子收益与整轮收益。

## 已确认的瓶颈与本次取舍

旧候选真实 NPU q4（32 请求、20–30K、N128/S3）CV 约 11.28ms；同期 native BF16 FIA 约 1.45ms。64K 完整已提供结果为相同 120 请求、7,812,000 输入及 61,440 输出 token，OSCAR 3331.55s、native 1371.97s；OSCAR 平均 TTFT 较低，但首 token 后明显更慢。当前代码每个历史 KV256 块仍分八个小批展开、重排两次大向量并逐行读取元数据。故此次重点覆盖每轮 target q4 和后续 MTP q1，而非仅优化少量成熟 prefill 簇。

之前 B4 的设备精度失败版本已完整回退到 `3615f04`。新候选不包含 B4/延迟 alpha 数学变化。原生 GDN 未修改。

## 数据与执行变化

D256 每槽仍为 136B、每元素仍是原来两个 bit、四个 FP16 元数据逐字节保留。旧槽 `[K64,Kmeta4,V64,Vmeta4]` 改为 `[K64,V64,Kmeta4,Vmeta4]`；每个 16-bit code word 中的八个量化码重排为维度 `i+32*b`。写入时直接生成新排列，读取时 Shift/And 直达自然维序，免去原大 Gather/plane staging。没有全历史 BF16 展开或额外历史缓存。

短 query 使用 M32 有界工作区（实际 q4/GQA6=24 行），释放 UB 后每 lane 解包 64KV，KV256 的批次由 8 变 2；元数据有效性用 FP16 原始位的精确向量比较，保持 NaN/Inf/正 scale 检查。FP32 QK、softmax、PV、旋转和 Mul→Add 顺序保留。PV 读回明确限制在各 lane 的 16 行范围。prefill 的 generic/balanced/C4/C16 采用相同格式的 reader，仍保留已有调度与数学。

`LayerState.cache_format` 在创建时固定；配置改变、错误格式、非 D256 均显式报错。全部服务读者和 `rotate_clip_store_striped_out` 成对选择。GQA>8 使用同格式 M128 reader，不进入 M32。旧 fe0 内核与默认配置不变。当前只读原生 `model_runner_v1.py:3399` 的 uniform decode 使用 `uniform_decode_query_len`（本配置 MTP3 即4），外部 metadata 原样传递 `max_query_len`；因此该 q4 选择同时作用于图捕获，而不只作用于 eager probe。

D.4 四问：本次改变融合 FIA 的有界历史解包；避免历史失败的 6.5s/card 全量恢复；通过物理码排列和较大有界批次减少重复展开/握手；以下周期证据仅验证该工作量变化，不能当作真机毫秒或模型结论。

## 周期模拟证据及整轮预算

本机 Lima `oscar`、CANN 9.1 Ascend910B4 CAModel，grid20，同一完整输入逐字节绑定。旧 fast 与候选全输出 partial/LSE/status 逐字节一致，独立 PR oracle 最大误差远小于冻结 0.005。

| q4/GQA6/D256 合成历史 | 旧 fast task ticks | 新 striped+SIMD ticks | 旧/新 |
| --- | ---: | ---: | ---: |
| 4095 | 1,674,968 | 844,724 | 1.983 |
| 8191 | 3,195,883 | 1,525,119 | 2.096 |

同一 source0 AIV/core0 的 4K→8K 历史增量为 1,520,914 / 721,723 ticks，约 2.107 倍。4K 时 window 接替成为整 task 临界路径，不能把整 task 差分称作历史斜率。CAModel Event 返回零，表中用实际 `profile_task` 周期，未换算 NPU ms；模型 topology、缓存及多请求负载与真机仍有差别。

若可加速部分占总时长比例为 f、其真机加速达到 2.096，则整轮加速约 `1 / (1-f+f/2.096)`。整轮 1.5 倍需 f≥约64%，整轮 2 倍需 f≥约96%。现有片段不能证明当前整轮 f；因此不承诺 16→8 分钟，更不把算子模拟的两倍写作全服务两倍。新方案的优势是同时进入每轮 q4/q1 热路径，且已由完整设备指令周期模拟排除了“只减少源码操作数却没减少运行周期”的情况。

## 本地交付验证

完整 CANN 9.1/ascend910b4 内核库及 PyTorch 扩展已在本机 Lima 编译、链接成功；首次缺 `llvm-objdump` 的本机构建失败保留，补齐工具路径后在干净目录完成，未将失败中间产物当作通过。关键主机回归为118 passed、8 skipped（大 FULL 选路的重复组合）、4 subtests passed；冻结阈值未修改。官方 CPU 的 q1 历史、S3 frontier、活跃簇和新格式字节对拍另列到`reports/striped_delivery_vm_20260930.json`。D256 `(3,385)` / context641,1025 / S3 的较大 C4 参考 CPU 模拟达到120s上限（rc124），未产生数值通过证据；采用小而实际成簇的用例验证局部改动，原较大非对齐 NPU 用例仍保留。上述不替代真 NPU/图验收。

## 一键入口与验收边界

候选开关由 `tools/serving_variants.py` 唯一维护，两入口均实际启用新 reader/writer：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

直启安装、编译、启动，保持无测试/probe；启动行打印 `STRIPED=on`。本轮默认设备为 4,5,6,7、端口7878；兼容参数 `--rear-cards` 选择相同设备。

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate
```

observe 保留原有真实算子、CV/旋转及候选门，并增加 `striped-cache`：同输入新旧 writer 原始字节/窗口对照，q4/q1 的 20–30K 与65K、padding、split、非对齐、C4/C16、非法 metadata、同地址改变输入的图回放；速度为2预热+5次NPU Event交替测量。report 与相位账本分开命名；任何精度/图/速度或释放失败停止服务，控制台只打印关键 `PERF_STRIPED` 行及错误。

只需短算子门时可在 observe 命令附加 `--probe-only`，不加载模型、不生成 AISBench 请求。此项供必要的真机最终认证，不能用 CPU 替代 NPU。真实 NPU 数值、图、完整服务性能和模型质量在新候选上仍待认证；原已确认 fe0 精度保持其原证据范围。
