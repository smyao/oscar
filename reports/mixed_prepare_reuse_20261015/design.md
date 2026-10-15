# Mixed eager prepare 复用实验设计

档案依据：#34/#36 固定容量和图地址；#37–49 物理页所有权；#129/D.4 不恢复历史、不转回host组装历史；#142 排除dummy/capture；#157/#161 仅native显式discard，活行错误不清洗。实现已在本地主分支 B1 基底 b6f8666 上完成；未运行VM/NPU。

## 当前真实重复

默认mixed路径每FULL层依次调用short-prefix与long-suffix；二者当前都从同一tasks/positions/slots arena零偏移写入，并在mixed调用前后失效旧prepare租约。已有单段prepare复用因此完全覆盖不到mixed续prefill。

31个q4请求占124token、一个历史prefill占16260token的16K父batch中，两段任务表合计6MiB（Hkv1/S1/每task16×int64）。16个同metadata组FULL层重复生成96MiB任务表。两个独立region只各生成一次可降至6MiB，prepare启动32→2；这是结构性工作账，不是时间比例。

## 零新增HBM的region布局

不新增Tensor分配。在父batch现有arena中使用不相交的token跨度：短段[0,124)、长段[124,16384)。tasks跨度必须乘Hkv×3×splits，positions与转换slots仍按token跨度；不得把task条数误作token数。

各段仍调用同一冻结prepare算子，输出局部query/request编号，CV继续读本段local Q/K/V与local block-table视图。不同之处仅为输出Tensor在已有arena中的storage offset。任务表入口始终128B对齐，positions仍遵守算子本来逐token8B写入的对齐契约。

QKV拷贝、partial/status/acc workspace、CV、current FIA、merge、store、boundary/slot/status guards全部按原路径逐层执行。只复用prepare及其原source2 suppress结果，不复用数值输出、status或物理KV。

## 显式生命周期

每个缓存region须同时满足以下条件：

- 真实eager main prefill；非draft/dummy/capture。
- 当前native ForwardContext的attn_metadata[layer_name]确实注册同一个父OscarMetadata对象。
- 同一ForwardContext对象、父metadata对象、stream及几何。
- 子段携带的父token跨度、请求跨度与父batch容量完整一致；所有region独立查找，不依赖调用顺序。
- 分区计划与CPU cumulative从builder已经冻结的current_cumulative tuple推导。不会缓存或依赖可异步复写的query_start_loc_cpu Tensor值保持不变。

任何prepare writer若覆盖已有region范围必须先使该region失效。whole/current-only/draft/capture等其他路径清空全部租约；新context或owner也清空。重复layer属于保守失效检查，不能被当作识别模型调用边界的唯一依据。租约仅保存至多两个region，不添加跨请求、跨step或跨graph状态。

保留native生命周期前提：同一个ForwardContext注册的设备metadata由native input-prep发布，模型层只读；scope切换必须被显式识别。不会以GPU地址复用推断同一步，也不会对任何live slot做新的discard推断。

## 计划验证

Host验证覆盖：16K典型混合步、不同split数/Hkv、int32/int64 slots；区域地址/容量与不相交性；完整descriptor/positions/输出等价；prepare32→2且其他逐层调用不减少；层与region顺序变换；相同地址的新ForwardContext；注册owner变化；CPU镜像在创建tuple后变化；whole/current-only等writer后的失效；非法长度与原guard保留。既有mixed/APC页域及discard回归保留，未改物理页语义。

`tests/test_mixed_prepare_reuse.py` 执行只读reference的完整ForwardContext类及get/override代码，16K metadata链使用CPU算子替身；实际 `OscarAttentionImpl.forward` + mixed dispatch 的512长段完成数值对拍，16K长段用shape-only QKV和真实CPU任务表检查默认selector每层进入hybrid与B1。前者不是NPU精度，后者不是模型数学验证。

`tools/probe_mixed_prepare_reuse.py` 是独立可执行NPU门：同一固定地址16384token/31q4+16260long/16FULL层，off每层两次prepare，on每区域一次；真实native ForwardContext每step新建，layer名跨step不重叠，故刷新不能依靠重复首层碰巧命中。依次32K→64K→非法long length，逐位比较全部16列descriptor与positions，独立检查非法long域error1保留。使用冻结prepare/source2抑制/边界guard，不执行CV、模型或AISBench。

计时预声明为32K有效case的完整metadata链：native scope、分区计划、device边界guard、子metadata切片、prepare、source2抑制、每层每段同样的consumer copy。两侧均2warmup+5sample、交错顺序、device events、median、non-regression1.0；所有回读与哈希在计时外，两侧同样收完整输出。消费审计buffer约98MiB仅存在于probe，不计为生产容量；生产新增HBM为0。64K/非法case只查本改动的生命周期和错误风险，不扩大CV矩阵。

共享candidate已启用prepare_reuse。observe必须先通过此门（source/artifact绑定、独立phase/report、资源释放与真实rc）才能进入服务；direct仍是同flag的直接启动，无probe证据。缺NPU或设备门失败即失败。当前仅本机host实现和接缝证据，设备正确性/时间尚待用户最终NPU验证；不把32→2当时间比例，也不把这项metadata门当32K/64K模型E2E>30%。
