# oscar-new-6 的 OPTR 最小迁移

目标是把用户指定的 `oscar-new-4-back` 最终 OPTR 策略接到当前分支，不合并源分支的线上实现。迁移基底为 `oscar-new-6@ca46f57e0267f15aadba08d6913cef81b46e21b1`，源为 `oscar-new-4-back@e7dc8a92000e0d9b49800eadda6d1d16f978436f`。

## 策略与接入范围

- 保留源 `optr_oscar_shared_candidates_ascend_v9`：train-only 共享零均值 seed（recipe6），每层共享 Rk/Rv，`Rbase @ exp(A-Aᵀ)`，先 K 后 V 各80步。FP32 上游代理目标、80步/0.02学习率、已有检查点的确定性筛选均不改。
- 整层部署回放执行逐请求、frame、query、terminal 非劣门，覆盖 FP32、BF16 最终输出、BF16 rank/最终输出三视图，以及单分片和捕获的真实分片。主候选失败才检查最多6个已有 V 候选，K 固定，不新增优化步，不使用 test 重选。
- 保留源实验性发布政策：validation 必须满足门控；独立 test 退化如实记录并显示，仍可发布实验 PT。`independent_test_blocks_publication=false` 不表示模型质量合格。
- 最终 recipe5 PT 继续使用既有逐头形状 ABI；每头展开同一共享旋转、K 均值为零。**非劣比较对象是新的共享 seed，不是迁移前 recipe3 的逐头中心化 PT。**
- 新增19个离线模块，既有生产文件仅修改11处准备、配置、产物加载、日志和审计接缝。离线捕获、spawn 拟合、原子发布、完整性校验、断点续拟合及有界准备超时都是这条策略实际可用的依赖。
- PT 审计工具同步识别 `optr_<profile digest>/refined.pt`。recipe3/4、旧 OPTR policy、损坏/未完成产物及显式冲突路径不会冒充当前 PT。

源文件中的档案编号保留来源含义；源分支的 #173–176 与当前分支同号条目不能混用。本轮依据当前 #86、#165/#166、#170、#172、#174保护校准边界、spawn、质量声明、性能来源及退役政策，未追加任何真机故障记录。

## 线上保护

当前所有 CANN 源、生成器、dispatcher、runtime、plugin/GDN、PC/MTP、量化 codec、KV 布局、窗口、图、timer 和配置均保持迁移前字节。没有引入源分支的旧 GDN prefill 行为、旧 C16、设备租约、正式服务监管或进程组重试。

q4/q1 保持232b源，C4保持原源，C16及 finite helper 保持34bc源；未新增长度阈值或路由。`oscar_current_only_policy` 仍在所有 variant 中清除。native/baseline 清除 OPTR 设置。

对32组 variant/PC/预算/配方组合与迁移前配置、完整 argv 对拍；两个入口 plan 对拍仅允许新增 OPTR 信息。安装、部署、观察入口剥离准备阶段 timeout 接缝后，完整执行 AST 与基底相同。默认物理卡 `0,1,2,3`、端口 `7878`、预算16384、原生调度/MTP及PC开关不变。

这些是源码与操作数量的保护证据，不是端到端速度测量。新旋转会改变数值，可能间接影响 MTP 接受率等；不能由字节保护推导真机吞吐必然无变化。首次离线采样/拟合需要额外准备时间；匹配 PT 后复用。

## 本地验证与已知边界

联合回归 **1217 passed、1 skipped、0 failed**；跳过项为需显式指定外部目录的可选源码审计，随测试保存的独立固定数学快照对拍已通过。858个既有生产文件保持基底字节，19个新增生产模块与源分支原字节一致。本轮验证回执、完整命令和日志位于 [reports/optr_migration_20261031](../reports/optr_migration_20261031/validation.json)。覆盖上游独立数学对拍、部署数值、查询/投影退化反例、候选集合、train/validation/test 隔离、PT 身份/损坏/中断恢复、四 worker 分配、安装入口，以及现有 GDN/PC/MTP/性能来源保护。

联合运行还修复了两处测试问题，生产逻辑不变：源 NPU 选择测试直接写环境导致后续测试继承卡号，现确保撤销；当前 GDN 状态地址反例传入了 cache 布局而非原生 chunk 输出布局，现按原生 transpose 合同构造输入。后者已在迁移前基底复现。旧端口和 recipe 断言也按已经授权的实际配置更新，未放宽内核 hash 或数值容差。

另有4项 `test_request_window_serving_config.py` 旧研究模式测试，在原始基底同样失败（BF16 与研究窗口组合冲突、现场模型路径不存在）。不属于此次 OPTR 回归，保留原测试与生产逻辑，单独记入回执；不能声称全仓 pytest 全绿。

隔离构建 wheel 并核对源码/离线数据字节；从解包目录导入全部新增模块、计算拟合实现身份并离线读取 GPQA。未连接真机，未运行 NPU、图回放、AISBench、模型质量或性能测试，也未把主机结果写成这些验收通过。

## 使用

当前本地代码使用原入口，无需额外 OPTR 开关：

```bash
bash scripts/install_serve.sh --variant candidate
```

部署到远端后的固定入口仍为 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`。本轮仅修改本地分支，不替用户启动正式服务。启动摘要应显示 `rotation_refinement=optr`、recipe5、v9 policy，以及独立 test 的实际状态。
