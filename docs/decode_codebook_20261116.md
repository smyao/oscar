# Decode 历史四级码本查表

基底：`oscar-new-7` / `1f44be7`。依据：档案 #126/#145/#150/#154/#155/#164/#172、启动文档 D.4 完整原始日志；本轮只有本地源码与功能验证，没有新增真机错误档案。

## 实际改动和入口

主模型 q4 与 MTP q1 的 source0 历史流水使用独立 `attention_codebook_unpack.h`。每次处理的64行、每行D256中，先严格按原公式计算4个BF16等级，再通过uint16 Gather选取原始BF16位。初始两块以及后续K/V预取均使用该路径；原 `PipelinePublishKv` 直接发布其结果，不再把整块FP32转换为BF16。

`decode_codebook_transform` 仅接受完整前一版active源码SHA `c286404e59a9242541dfa047b50ac698a808f53d45bdc2e0419f3c37b6efe7ba`，逆变换恢复原源。它叠加在原active-row与physical-page变换之后。完整母源、旧aligned helper、原CastRne helper和所有其他内核保持；generic `Unpack/PublishKv`、current/window、QK/PV、FP32 softmax/LSE、任务调度、错误提交顺序与跨核flags均不变。

现有CMake实际编译同一个active decode源，两种公开q4/q1 ABI不变。生成清单、全csrc构建指纹和aligned/official/prefix功能probe绑定覆盖新helper和变换。candidate直启、observe以及node93脚本所调用的同一算子实际启用本实现，无新增参数或长度阈值；`git pull --ff-only && bash scripts/install_serve.sh --variant candidate` 仍为固定入口。构建与实际设备功能验收不是一回事。

## 数值和错误域

mode0逐级计算 `BF16_RNE(FP32_Mul(FP32_Sub(float(code), zero), scale))`。保留FP32 metadata，不改写成`code*scale-zero*scale`或FMA。mode1同时保留metadata和差值的BF16舍入、metadata转换前后两次检查。实际索引只取0到3；码字布局和bit顺序保持。

原live行的 `scale>0`、scale/zero有限性检查及`error=3`保持。非法mode仍按原顺序赋`error=1`。死行不新增metadata检查。四级表可能包含从未使用的Inf/NaN，不能据此新增拒绝：例如scale=FLT_MAX/2、zero=0、实际codes全0时，code3虽溢出，原有效值仍全0。uint16 Gather不进行浮点计算，不把BF16位当FP16解释。

PC/MemCache仍存相同的 `striped_rotated_fp32_v3` 144B行。writer、PT、clip/recipe、精确sink/recent、原生GDN、MTP accepted/discard及缓存命名空间均不变。临时表在每次读取时重新建立，没有跨请求、跨层、跨step持久状态，也就没有新增页回收/恢复失效协议。

## 缓冲生命周期

| 现有缓冲 | 本次用途 | 字节范围 |
|---|---|---|
| natural，34816B | 原INT16码→行地址广播→BF16结果，三个不重叠活跃期 | 最后写32768B |
| dequant，65536B | 码的FP32表示→精确字节偏移→INT32索引 | 65536B |
| scratch，2048B | FP32 metadata | 0–511 |
| scratch | 四个平面的FP32等级，随后前256B复用为行号 | 512–1535 |
| scratch | 四级BF16表 | 1536–2047 |
| metadataHalf，256B | 原mode1 metadata舍入 | 256B |

Gather索引为`code*128 + row*2`，是[0,510]内的偶数，FP32精确表达。先通过已存在的INT16→FP32路径，再就地FP32→INT32，避免假设未核实的INT16→INT32接口。表源、索引、输出互不重叠。natural被改作输出前，PIPE_V完成全部行地址读取；原MTE2/V、V/MTE3、MTE3/V所有权及预取错误提交全部保留。AIV显式UB仍158656B，HBM/workspace和graph地址分配不增加。

API先例：本项目 `attention_hybrid_l1_unpack.h` 已使用raw uint16 Gather、ArithProgression<int32_t>、INT32→FP32 Cast；原生只读参考 `references/vllm-ascend/csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_helper.h:182` 使用同址FP32→INT32 CAST_RINT。原解码器已使用相同的Shift/And/strided Cast/Brcb。当前没有本地SDK，先例不能冒充新代码的CANN编译通过。

## 成本与取舍

每次HalfKv64、单K或V，等级Sub/Mul从16384个元素变为256个；最后BF16转换的normal-mask单repeat调用从256次变为4次。mode1的差值BF16转换也同样缩小，原single-repeat转换形式保持，未重启#172的大块BF16 Cast实验。

同时新增全元素索引乘128、加行地址、FP32→INT32，以及8次2048元素raw Gather。前两项会补回大量浮点算术，不能说总算量减少64倍。按实际显式Vector API调用（不计barrier、标量guard和mask设置），mode0为303→76、mode1为563→84；这不是硬件指令退休数、周期或吞吐倍率。收益假设是减少转换发射和重复等级重建，其净效果还取决于Gather、UB bank及流水时序。

GM交接本轮保持：每完整KV256读36KiB压缩数据，发布256KiB BF16 K/V，Cube再读取256KiB；合计548KiB逻辑GM流量，部分可能命中L2，不能当HBM实测。项目既有SDK审计记录指出A2公开UB→L1接口本身通过GM软件通道（`quant_dequant_next_decision_20261011.md`）；此次引用的归档SDK头已不在checkout，不伪称新SDK实证。直接替换接口没有消除这笔搬运的依据，因此不加入未经证明的跨核通道。

持久码本也不在本次同格式方案中：用16B BF16表替换16B FP32 metadata虽容量相同，却无法保持两mode和原metadata错误域；追加表则改变物理行跨度、容量和卸载格式。这需要独立codec/namespace合同，不能悄悄混用旧MemCache数据。本次选择tile内四级复用，不增加历史镜像。

## D.4 四问与验证

1. 对应融合FIA内部历史解包和BF16发布。
2. 历史失败为全历史物化单相位6499.8–6655.1ms；该数字不作当前占比。
3. 本实现只在固定H64 tile构造四级表，原KV256流水/GQA与q4共享保留，无历史长度大小的额外恢复缓存。
4. 资源固定、操作账如上；未测不能承诺优于原生或端到端收益。

验证分三层：实际新旧C++ helper在有界主机指令模型中比较输出位、错误、padding和UB访问；生成逆变换/构建缓存/生产route回归；真实NPU功能probe增加source0发布workspace逐位和固定地址图回放门，由真实设备执行。数学oracle、主机模型、CANN、NPU/graph、模型质量、性能分别记录于 `reports/decode_codebook_20261116/validation.json`。本机缺CANN、Lima及NPU，未运行后四类验证，不把主机执行冒充官方CPU-debug。

本轮最终本地主机回归为488项通过：201项生成/路由/构建/卸载合同回归，287项解包/参考/probe控制与生命周期回归。解包模型包含全部65,536种packed uint16码字、mode0/1及非法mode、尾行、特殊浮点、未使用等级溢出和越界/别名/同步负对照。新增真实设备发布门覆盖q4/q1×mode0/1、13种尾长、两个交接槽、固定地址更换codes/页映射及三个数值边界；该设备门仅完成主机控制验证，尚未在NPU执行。生成清单一致性和`git diff --check`通过。
