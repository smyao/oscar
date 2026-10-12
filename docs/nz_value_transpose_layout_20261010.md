# Vᵀ-NZ备选：合法布局与双向成本

2026-10-10，只读审计/host坐标验证；未改核、未运行VM。限D256/KV256。依据档案#143–145/#148/#154–156、startup D.4及当前只读CANN SDK。

## 与旧失败方向的区别

实际移除旧V转置的是档案#144：旧实现将每16行Gather成`[D,16]`，再以D个64B小块写**ND `[D,KV]`**，形成大量GM散写。不能恢复那个发布方式。

本方案仍从当前自然FP32解包结果出发，仅把当前KV256的V发布成**NZ的物理Vᵀ `[D,KV]`**。8个连续KV行组成一个完整NZ C0块，因而可连续写8/16KiB；不增加历史长度相关物化，也不新增一份GM V矩阵。

## 格式与PV语义

SDK C0=float8。对自然V[kv,d]：

```text
V_NZ   = [D/8, KV256, 8]；地址(d//8)*256*8 + kv*8 + d%8
Vt_NZ  = [KV256/8, D256, 8]；地址(kv//8)*256*8 + d*8 + kv%8
```

Cube PV的B从非转置改为转置，物理B=Vᵀ，逻辑仍是P@V。`SetOrgShape`的物理N/K仍256，`SetSingleShape`保留真实m和K尾；A若NZ继续服从Q32/Q128物理pitch，不能借方阵尺寸掩盖M6错误。QK和Rv原本Btrans=true，R_v保持原矩阵的NZ pack，**不可再把R_v额外转置一次**。其他D必须重新检查OrgN/OrgK，不能直接套当前依赖D=KV=256的布尔分支。

当前SDK `load_to_l0b_loadInstr.h:91–145` 的Btrans=true走float LoadData2d；false的`:177–196`走LoadData3DV2/FMatrixB。真实现PV trace已证明后者。切换布局后只有在L0B数学值、MMAD m/n/k/顺序和结果逐位相同的微测通过，才算方向成立。

## 原行ownership发布，不需要虚拟error lane

每16个**相对KV tile内**连续行，Vt_NZ覆盖两个连续C0块，目标GM区间恰为`[rowBase*D,(rowBase+16)*D)`。Gather输出索引t对应：

`source_row=(t//(D*8))*8+t%8`，`source_dim=(t%(D*8))//8`。

- H64：每lane的64连续行分4次16行Gather，每次16KiB输出后一次连续GM copy。
- H32：原lane16两个不连续段各做一次16行Gather/连续copy，目标rowBase仍`subtile*64+lane*16+h*32`。不改变LoadPacked、liveKvRows、metadata guard/error顺序。
- 精确source仍保持旧row ownership和全D finite检查。C4 H32布局下的H16 precise可用同一个16行publisher；生产H16若scratch只有8KiB，须按8行小块单独计成本，不能假定H32资源已经启用。

Host坐标枚举H32/H64两种旧ownership均覆盖65536个float恰一次，并证每个16行组连续；没有设备数值/速度含义。

## 不增加UB，但依赖不变

H64沿前页natural34KiB拆索引/scratch，score16KiB作输出，dequant64KiB为输入。H32沿natural17KiB索引、score32KiB拆scratch/output，dequant32KiB为输入。索引可由有限位重排构造，仍要计构造/Gather成本及UB bank行为，不能因DMA连续就声称Gather更快。

每次输出覆盖前MTE3_V，Gather后V_MTE3；decode V预取可能仍有P写出，原MTE3_MTE2不能代替这个MTE3_V。AB-NZ若P发布改读dequant，也必须先完成P源消费再由V Unpack覆写dequant。packed/packedNext、acc、stats均不借用，metadata与±0/NaN位模式保持。

History始终将完整KV256的尾行按原规则置零。precise只跳过Cube不会读取的块，最后live64/H16块继续填零到原范围；PV K尾为16倍数，所需Vt C0块必须已初始化。绝对逻辑位置可以不对齐8，出版使用tile内relativeRow，不据全局position除8。

## 收益必须两面核算

| 当前有界KV256发布 | V_NZ | Vt_NZ | 含义 |
| --- | ---: | ---: | --- |
| H64 V publisher bursts | 128 | 16 | Gather/API次数相同，仅burst连续性改善 |
| H32 V publisher bursts | 512 | 16 | 同样未减少Gather次数或FP32 payload |

但SDK每个逻辑B tile的GM→L1代价相反变化：

| B tile（K,N） | V_NZ bursts×每块 | Vt_NZ bursts×每块 |
| --- | ---: | ---: |
| 128×64 | 8×4KiB | **16×2KiB** |
| 16×64尾 | 8×512B | 2×2KiB |

因此完整K128时MTE2 burst数会翻倍，虽不再走L0B Load3D，也不能直接断言净利。支持做同库、同逻辑P/V的**完整Matmul**预pack微测，保持非对称随机V和真实MMAD序列，分别观察M126/M24与尾K；不只测Load2D或数publisher burst。微测仅是免费producer上限，最终还需包含完整解包/重排的净周期。没有整核或整轮速度承诺。
