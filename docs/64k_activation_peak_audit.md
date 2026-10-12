# whole输入32768→66048的原生激活峰值资格

只读本工程指定references及当前源码；未改GDN数学、生产代码或原生文件，未运行VM/CPU-debug/CAModel/NPU。本报告补充 `64k_latency_budget.md`。档案#69/#127–129/#137/#140及D.4要求实际shape、dtype、同时存活区间和设备证据分开，不能把模型profile、ready或源码字节账当真机峰值验收。

**结论：新增native峰值不能预设为0，K32尚未获显存资格。** 当前参考GDN CANN scratch是有界的，并非随完整T增长的大FP32状态池；但QKV、卷积、归一化和MLP激活会随输入增大。下列可见tensor的增量已达GiB量级，未知的FIA/量化/HCCL workspace、编译器复用和allocator保留会决定是否越过约2.17–2.53GiB的预算余量。该余量是重新规划KV池可容忍的扣减，不代表启动后空闲KV块能被临时借作空闲HBM。

## 1. 必须区分实际GDN路径与旧Triton路径

指定 `references/vllm-ascend/vllm_ascend/ops/gdn.py:399–435` 将q/k归一化后调用 `torch.ops._C_ascend.npu_chunk_gated_delta_rule`。C++adapter在 `csrc/attention/chunk_gated_delta_rule/chunk_gated_delta_rule_torch_adpt.h:39–70` 分配BF16 out `(T,Nv,Dv)` 和原dtype final_state `(B,Nv,Dv,Dk)`，随后EXEC_NPU_CMD。

该算子的tiling（同目录`op_host/chunk_gated_delta_rule_tiling.cpp:118–153,249–254`）固定 `chunk=64`、`s=2*AIC*64`，interWorkspace按s计费，**不是按完整T计费**。条件AIC20、Nv12、Dk=Dv128、BF16 state、B1时：

```text
s = 2560
workspace = 16MiB
  + 4*Nv*s + 2*Nv*s*(3*Dk+2*Dv+64)
  + 4*B*Nv*Dv*Dk + 4*64*64*AIC*4
  + 2*64*(2*64+3*Dk+Dv)*4*AIC
  = 68,804,608 B = 65.617 MiB
```

同B、核数和dtype下，T32768→66048不会使这块workspace翻倍。B变化另按state项计算，不能从旧Triton文件的中间tensor推断当前CANN路径要分配整段FP32历史。

## 2. 同时存活的GDN tensor

条件几何沿用当前报告中的GDN全局K/V heads16/48、Dk/Dv128、TP4，故本rank Nk4/Nv12。表内均为一次prefill **CANN chunk调用点** 的不同storage；reshape/split/squeeze别名不重复计费。

| storage | shape或保留原因 | B/token |
|---|---|---:|
| mixed_qkvz | `[T,4096]` BF16；z随后仍用于norm，保留其底层projection | 8192 |
| conv output | `[T,2560]` BF16；`gdn.py:268–283` | 5120 |
| contiguous fused QKV | `[T*2560]` BF16；`qwen_gdn_linear_attn.py:825–840`的cat | 5120 |
| normalized q与k | 各`[T,4,128]` BF16；`gdn.py:412–413`，l2norm创建empty_like | 2048 |
| caller core_attn_out | `[T,12,128]` BF16；`gdn.py:118–130` | 3072 |
| chunk op out | 同上，C++adapter另分配 | 3072 |
| a、b + g、beta | a/b BF16各12；g FP32×12、beta BF16×12 | 120 |
| 可选仍活着的原ba | `[T,24]` BF16；编译器可在形成a/b后释放 | 0–48 |
| 合计（不含H维模型状态） | | **26744–26792** |

`ΔT=66048−32768=33280`，上述同一调用点的可见增量约 **0.829–0.830GiB/rank**。初态/终态只随prefill请求数B变化，未按T放大。mixed的spec/non-spec选择、merged output会增加其它phase的可见tensor，但不能把在更早conv/cat或更晚merge时已释放的临时tensor全部加到此调用点。

外层decoder还要保留residual和attention destination，normalized hidden是否被编译器提前释放决定额外约`4H–6H B/token`；`qwen3_next.py:410–430`。**H=实际hidden_size未知**。若只作H5120的条件示例，这个GDN调用点的总可见增量约 **1.464–1.783GiB**，不是整模型峰值保证或绝对上下界。

## 3. MLP的phase增量范围，不能与GDN直接相加

Qwen3.5 dense层使用 `Qwen2MoeMLP`（`qwen3_5.py:81,162–169`），按顺序 gate_up→SiluAndMul→down_proj（`qwen2_moe.py:114–122`）。令J=实际intermediate_size/TP：

- gate_up为BF16 `[T,2J]`；swiglu输出BF16 `[T,J]`，`vllm_ascend/ops/activation.py:37–43`调用npu_swiglu。显式out-of-place路径在此调用点至少有这两份激活，加仍需保留的residual，visible集合约`6J+2H B/token`。
- 在eager/未提前释放的保守调用路径，down matmul期间还可能保留MLP输入、residual、旧attention输出、gate_up、BF16 activation、INT8 quantized activation及BF16 matmul输出；对应约`7J+8H B/token`。W8A8输出dtype和量化输入见`w8a8_static.py:110–139` / `w8a8_dynamic.py:82–121`。
- 这不是同一tensor重复计算，也没有把gate projection时已释放的量化输入加到down阶段。当前指定NPUCommunicator未override all_reduce，其基类 `base_device_communicator.py:180–182`原地all_reduce并返回input，因此**不再人为叠加一份H维all-reduce输出**。HCCL内部workspace仍未知。

以**尚未由本轮目标config证实的**H5120、I17408、TP4（J4352）仅作代入示例，MLP这些phase的增量包络约 **1.127–2.214GiB/rank**。上沿已经高于2.17GiB预算区间低端；距离2.53GiB也只余约0.32GiB，不能在FIA/HCCL/quant内部workspace未知时宣称K32必然可用。实际H/I、具体W8A8分支、compile buffer复用需要现场配置/峰值确认；这些参数未提供，不能把示例当模型实测值。

GDN与MLP顺序执行，不能将1.783和2.214相加成约4GiB。正确的阶段账是取同一执行图各时刻的最大live集合，加共同持久分配；输入增长前后的**总峰值差**也不一定等于某一phase的上述增量，因为最大值所在phase可能改变。

## 4. profile覆盖缺口与仍未知的上界

`gdn.py:159–161`在metadata=None的V1 profile中直接返回core；普通Ascend `_should_build_dummy_attn_metadata`只在force_attention或FULL图模式返回True（`model_runner_v1.py:3352–3364`）。因此标准 `profile_run` 的大N可以覆盖projection/MLP等外层分配，却**未必执行真实GDN conv/rearrange/l2norm/chunk workspace**。不能用profile成功证明它们已被完整计入。

这也不意味着必须把全部GDN真实峰值再加一次：如果被profile的MLP阶段已覆盖更高峰值，则GDN额外部分可能被其预留吸收。需要比较相同rank上真实phase峰值，不能将所有算子workspace按顺序累加。

已有合法native current FIA的显式返回output `[T,6,256]` BF16及LSE `[T,6,1]` FP16/BF16/FP32（本工程`current_attention.py:145–178`）单独会增加约 **0.096GiB/rank**（按FP32 LSE上界）。这是一个FULL phase的临时返回，与外部预分配GraphWorkspace不同；不能在它已通过profile计入后重复扣费。FIA内部workspace没有本轮实际大小，亦不应按`[T,T]`想象一个全attention矩阵。

目前无法给出全模型峰值的可靠有限上界：还缺target H/I及实际operator分支、compile生命周期、FIA/quant/HCCL内部workspace、graph/caching allocator峰值。输入增加也可能改变原生算法tiling与workspace选择。**结论为未知且需要资格验证，不是已证明OOM，也不是已证明+peak0。**

最小后续只需一次真实whole65100（含首MTP）短probe，在原生profile预算之后记录实际peak allocated/reserved与NPU内存、实际max batch/live shape、GDN/MLP/current路径是否执行，并与同栈32768形状对照；不需要重跑AISBench。若真实额外峰值使新预算少于K32所需物理池，必须如实降低容量验收结果或继续优化OSCAR固定/临时分配，不能修改GDN语义或藏掉失败。
