# 有界FP32 KV直接发布NZ：布局/生命周期审计

2026-10-10，仅源码与host坐标推导；未写新CANN、未运行VM。档案#13–22/#70–73/#126/#145/#148/#154–156与D.4。问题、方案和证据分别列出。

## 问题：当前确有ND→NZ搬运，但没有重复GM读的证据

当前C4/H64 decode将解包后的FP32 K/V以自然`[KV256,D256]`发布GM，Matmul的B为`GM/ND/float`。当前SDK `copy_tile_to_cube_common.h:459–487` 对ND调用CopyND2NZ，对NZ调用CopyNZ2NZ。真实M126 PV trace未发现重复GM输入读取；候选只消除格式转换并改变DMA切片布局，**不把必要数据读取算作可删除工作**。

## 方案：B=GM NZ，A/score/P/PV仍ND

SDK只读证据目录`artifacts/cann_sdk_matmul_move_20261010/`：`utils/mx_matmul_utils.h:102–105`及`matmul_config_utils.h:41`确认FP32 C0=8；`data_copy_wrapper_utils.h:291–334`的CopyNZ2NZ使用`srcOffset=row*8+col*align(gRow,16)`，按8列一块复制。故全KV矩阵的确切float地址为：

```text
NZ(c0,row,lane) = c0*256*8 + row*8 + lane
natural(row,d)  -> NZ(d//8,row,d%8)
等价连续形状 [D/8, KV256, 8]，或 [D/8, KV256/16, 16, 8]
```

QK转置B物理为`[KV,D]`，PV非转置B物理也是`[KV,D]`，同一NZ物理公式适用；SDK通过transpose参数交换逻辑轴。**SetOrgShape保留完整物理256跨度**，短N/K由SetSingleShape表示。仅改B类型后，全部source0/1/2的K/V发布都必须匹配NZ；不能只改history而让同一Matmul实例继续把window/current的自然B解释为NZ。无需第二套Matmul/TPipe。

## H32：保留旧lane16两段，零额外常驻UB

现H32 C4显式UB157760B，直接加32KiB会达到190528B，超过184KiB的188416B。可复用：dead natural17408B放4096个uint32 Gather索引（16384B）；dead score32768B拆成16KiB构造scratch和16KiB输出；dequant32768B保持输入。acc64KiB、stats、packed、永久metadataIndex/masks不动。

H32的UB行不是连续32个逻辑行：`logical=subtile*64 + lane*16 + (j//16)*32 + j%16`。优先每次处理一个旧16行半段、所有32个C0块，输出`[32,16,8]`，两次Gather即可覆盖H32：

- 半段h=0/1：Gather源为`dequant[h*16*D]`，索引`byte=4*(row*D+c0*8+lane8)`；输出地址`(c0*16+row)*8+lane8`。
- GM起点float为`(subtile*64+lane*16+h*32)*8`。
- 一条DataCopy2D：`blockCount=32, blockLen=16, srcStride=0, dstStride=240`，后三项使用32B单位；每burst为16×8×4=512B。
- 两半段分别发布；**禁止把H32当成连续32行一次落GM**。每H32 side是2 Gather＋2次2D调用、64个burst，数据量仍32KiB。原自然H32发布只有一条2D调用/两段burst，新增发射和burst成本不能隐藏。

## H64：四个8-C0块

decode的natural34816B可拆16KiB索引＋16KiB构造scratch；score16384B做输出，dequant65536B保留输入，显式UB仍158656B。

- 每次取8个C0块：输出`[8,64,8]`共16KiB；c0Base=0/8/16/24。
- 同一索引`4*(row*D+c0Local*8+lane8)`可复用，Gather的字节base分别0/256/512/768。
- GM起点float为`c0Base*256*8+(subtile*128+lane*64)*8`。
- DataCopy2D：`blockCount=8, blockLen=64, srcStride=0, dstStride=192`；每burst64×8×4=2048B。四次Gather/发布覆盖64行D256，32个burst、总64KiB。

Host仅坐标枚举已验证：上述H32/H64全subtile×两lane各使65536个KV矩阵float地址恰好写一次，无越界/重叠，临时输出最大索引4095。它不是设备数值或性能验证。

## 生命周期和必须增加/保留的依赖

- Unpack最终FP32 Mul→Add/metadata检查完成后，natural内容才死；V侧会从packed重新Gather metadata，因此可以暂借natural。不能借packed/packedNext，它们跨K预取到V预取仍活着。
- **H64 PrefetchValues的scoreBuf尚可能被上一Softmax的P写出读取。** 原Softmax结尾只有MTE3_MTE2，保护的是下一次MTE2回填；新Gather直接Vector覆写score时必须先加MTE3_V，不能误用#145旧屏障。
- C4串行及研究pipeline的SpillState已含MTE3_V，P3后预取具备相应release链，但仍需按实际调用点验证；不据“逻辑上PReady已set”推断P的UB源已可覆写。
- 每块Gather结束后V_MTE3再发布；输出scratch下一次Gather覆写前必须等该次MTE3_V。H32两块/H64四块比原发布增加同步次数，必须计入净收益。
- 索引位于会被下一次Unpack覆盖的natural，须每次重新构造或重新载入；不是一次Init永久可用。构造也必须向量化并实测，不能重新加入4096次scalar SetValue。
- 尾行的原零初始化和每lane错误归属保持。History仍为KV256整块，不能因live较短漏写原本Cube会读到的padding；原metadata/finite/status检查不能省略。

## Rv与收益边界

同一B=NZ实例执行最终Rv时也需要NZ矩阵。可在模型初始化、KV profiling前按`R.view(256,32,8).permute(1,0,2).contiguous()`做一次纯位重排，并按原矩阵身份缓存；每个独立Rv增加256KiB，必须列显存账。现有binding要求二维`[D,D]`，新NZ专用reader可对这份packed存储再取二维view，必须明确其NZ含义；不能把三维tensor直接塞入旧binding。原自然Rv/转置仍可能供writer或旧reader使用，不可把自然指针传给NZ reader，也不可偷改其含义。先用uint32视图验证所有位（含±0），不做任何浮点转换。

先测host预pack B且同Matmul配置的潜力上限；即使有收益，还要扣除真实Gather/索引构造/更多2D burst/屏障以及source1/current转换。C4一次发布供4成员复用，有摊销机会；q4/q1每tile仅一次QK/PV，必须另算，不套C4倍率。D.4禁止全历史恢复：这里只重排当前KV256，不增加历史长度相关缓存。当前没有新的完整dequant、NPU、graph或整轮2倍通过证据。
