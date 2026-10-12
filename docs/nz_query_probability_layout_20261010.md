# A/P-NZ布局与生命周期：只读可行性

2026-10-10；不改核/ABI/精度，不运行VM，等待同配置AB-NZ微测。档案#126/#140/#145/#148/#154–156、startup D.4。A/B保持FP32、HF32关闭，C/score/PV输出仍ND。

## 1. 物理行跨度不能当成真实M

SDK `data_copy_wrapper_utils.h:302–308` 的NZ输入按`align(GetOrgHeight(),16)`跨C0块。固定Q/P slab高度H=32（decode）或128（C4每member）时，地址为`(col//8)*H*8+row*8+col%8`。

- 必须`SetOrgShape(H,256,256)`配`SetSingleShape(realM,n,k)`；后者仍保留M6/M24/M126和真实N/K尾。当前ND代码的OrgM=realM不能原样搬到固定pitch的A-NZ。
- q1 M6/H32的第二C0块正确float偏移256，若OrgM6则SDK读128；C4 M66/H128正确1024，若OrgM66则读640。M24/H32和M126/H128恰好对齐，单测这两例会漏错。
- C4四个member分别拥有独立H128 slab，偏移仍`member*128*D`，不能当作一个H512 NZ矩阵。

## 2. GM空间足够，但P必须独立

当前`pOffset=scoreOffset`只对ND原地逐row变P成立；改NZ后会覆盖另一block/lane尚未读取的ND score，必须拆开。

| 核路径 | 独立P_NZ增加量/core | 新需求/core | 现C16 arena剩余/core |
| --- | ---: | ---: | ---: |
| decode bundle | 32KiB | 1179648B | 3817472B |
| C4串行 | 128KiB | 1970176B | 3026944B |
| C4研究双score pipeline | 两槽共256KiB | 2756608B | 2240512B |

当前candidate的`runtime.py:112–118`按4997120B/core预留C16 arena，足够上述三种需求；全部actor必须同步更新**per-core stride**，不能在旧stride后追加而撞下一core。旧C4-only arena不够，需显式容量门；如果C16核本身也加P_NZ，其原4997120B还需再增加，本结论不涵盖它。复用原Q区域，不额外分配Q；RV-NZ单独一次pack按上一页另计。

## 3. Q初始化：从原输入直接发布NZ

`LoadQueries`仍按原query row ownership读qr（history）或BF16 q并Cast（精确source），原逐row ReduceSum/错误2检查不变。最小方案每个D256行用DataCopy2D `{32,1,0,H−1}`，GM起点`qOffset+row*8`；不先写ND再原地转NZ。

原padding范围`[validEnd,laneEnd)`仍置+0，但改成NZ行段：每个C0块写`padRows*8`个float，目标stride为H。可在dead dequant中生成有限C0批次的全零块，按其容量分批2D写，不需新UB；全部MTE3读完成后才复用dequant。活跃qFloatBuf仍用原MTE3_MTE2保护下一行MTE2覆盖。不能省掉#126的MTE3→V padding依赖。该发布增加burst，Q仅在每个history task开始一次，不能称免费。

## 4. Softmax→P_NZ

保留全部原mask、finite、max/sum、alpha与acc Mul；只改最后P发布。`dequant`在V2 scratch执行及其V_S完成后才死；`natural/alphaBlocks`在acc Mul后的PIPE_V完成后才可借作索引。**scoreBuf此时是活P输入，不能拿来做索引scratch。**

- decode：natural34KiB可放16KiB索引＋16KiB构造scratch，dequant64KiB的一段作16KiB Gather输出。
- C4 H32/列分工研究布局：natural≥17KiB放16KiB索引；dequant32KiB拆16KiB构造scratch＋16KiB输出。每次Gather16个自然P行的全部256列，输出`[32,16,8]`，DataCopy2D `{32,16,0,H−16}`；原32row Softmax块分两次。
- 若仍用生产H16的8KiB natural/16KiB dequant，只能相应按8行分块，不能假设已有H32容量。所有view必须保持原Softmax可见scratch大小，不能通过缩小GetSize改变V2 tiling。

SDK NZ copy会把输入height向上对齐16。当前GetNormalConfig实际M尾下，每个活跃P块至少发布到`align(activeRows,16)`，将active之外的这些P行显式置+0；全部无活跃行的block仍不改变status/flag次数。不能依赖旧P_NZ poison恰好不影响M6。若后续trace发现某配置加载更大的静态height，必须先扩大明确padding范围，不放行未初始化读取。

原`ClearPaddedScores`仍清**ND score**未由QK尾N写出的列，保证原pre-mask ReduceSum读到相同+0；不能改清P_NZ来替代它。Softmax之后再把P转换成NZ。

Gather完成后V_MTE3再发布P_NZ；每次复用Gather输出前必须MTE3_V。旧Softmax末尾MTE3_MTE2继续保留，且decode后续V预取会直接Vector写dequant，**仅旧MTE3_MTE2不足，必须显式MTE3_V**。索引若由MTE2载入natural，还需V_MTE2保护先前alpha读；纯Vector构造则沿原PIPE_V顺序。

## 5. P槽消费与最终Rv

PReady只能在完整P_NZ写出后发布。decode沿原Cube完成PV后才允许下一P；C4 pipeline的slot0/1沿原“PV0完成才QK2/Score2”关系保护复用。虽然score与P已分离，也**不能提前发QK2后让P2覆盖仍被PV0读的P0**；本方案不新增这种重排。

最后normalized acc也必须发布A-NZ到原Q slab，不能再用ND `DataCopy(work[qOffset+laneRows*D],acc,...)`。按16行Gather/2D分块即可复用dead natural/score/dequant，保留原每行归一化顺序与错误；全部写完才NormReady。Rv参数使用独立NZ版，结果`rotOffset`仍ND，之后原回读/PublishRows不变。

## 判定

空间与布局方案可行，仍须验证M6/H32、C1短M/H128、P尾行、poison、全部错误/flag和固定地址改输入。B-only的QK1.463/PV1.655是免费producer条件上限；A/P实际Gather、索引、padding、burst和新增fence成本未测。没有完整算子或整轮速度承诺。
