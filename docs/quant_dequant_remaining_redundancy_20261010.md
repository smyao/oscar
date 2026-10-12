# 当前重复量化/反量化工作：只读筛选

档案 #13–22/#70–73/#145/#148/#154–156、startup D.4。原生预算保持；没有VM新运行、生产改动或新整轮倍率。只报告当前源码仍存在的项。

| 项 | 当前源码与语义 | 最大收益边界 / 决策 |
| --- | --- | --- |
| target 与 draft “共享历史” | `runtime.py:706–741`分别绑定各层`raw[name]`与LayerState；指定native `qwen3_5_mtp.py:89–105,135–149`有独立fc、FULL层和输入hidden。target与draft相同位置不是同一K/V | 不允许跨层共用解包值；可消除比例为0，除非逐值证明，当前没有这种条件 |
| 三轮draft重复读取稳定历史 | `llm_base_proposer.py:1087–1097,1208–1289`后续Q依赖上一轮完整model/sample；`impl.py:178–194`每次独立调用CV。当前bundle的next-KV缓存仅在单次task内部 | 三次相同历史的dequant理想最多3→1（仅该阶段），但下一Q尚不存在。跨调用保留全decoded历史违反固定有界缓存约束；固定256token持久缓存覆盖65100的不足0.4%，顺序扫描LRU还可能零命中。不能据此提出完整decode3× |
| 同位置draft重写K/V | 首draft来自新的target_hidden_states（native proposer:769–776），后续来自前一draft hidden/input token；writer位置相同不能证明输入相同 | 不能按已有tag跳过量化，尤其拒绝/重算后。先算量化再比较bytes不能省该量化成本 |
| C4簇之间重读相同历史 | `attention_cv_window_range_cluster4.cpp:328–353,901–910`每个成熟同域cluster重新解包；簇内4成员复用已存在，不能再次计收益。默认Q2048/S1有21簇，跨簇确有重复 | 理想去重最多把这些簇的解包从21份变1份，但QK/PV/softmax不减少；跨core消费者需有界生产/消费同步。已存闭合full256轮53896cycles KV准备/390694总轮，假定其他不变且KV准备全部免费也只1.160×该轮；不是长history全核上界。新pipeline已有nextKV重叠，不预记第二份收益。当前不值得单独做复杂全core共享 |
| draft identity Q/K/V旋转 | `runtime.py:668–681`显式identity且hadamard=False；canonical `rotate_clip_store.cpp:103–113`与striped `113–125`仍逐256列Mul/ReduceSum；cache_storage_ops无identity路由 | 数学冗余确实活跃，已有研究非生产。旧完整q1 writer8.563×不能套当前整步；历史835fc22片段两个q1 store仅1.138ms/428.101ms，即全消失仅1.00267×该步。保留为已知未接入项，不重跑小收益单点 |
| target Hadamard元数据与逐row屏障 | `impl.py:161–163`仍canonical rotate_out；canonical Init仍4096scalar写。striped writer只改了Init/pack，Transform仍逐row8stage | 紧凑stage-major原型是新增候选：保持每row算术、indices/sign16KiB，work增7KiB；满8行调用200→137、屏障200→25。完整writer当前96566ticks需净省32189才1.5，尚无内部Transform占比。详见`csrc/experiments/hadamard_batch8/README.md`；不拿次数当速度 |
| 每history split的最后Rv旋转 | `attention_cv_decode_bundle.cpp:695–698`每split执行密集Cube旋转。指定PR `triton_oscar_decode.py:5–13,254–270,336`是先合split，caller再逆旋转 | 实数线性可移到split合并后，但FP32顺序与现有partial/merge契约变化；不能假定逐位等价。S3最多省2/3该固定旋转，长history相对占比下降，没有大收益证据，暂不改 |

## 已有优化明确剔除

q4/q1已是HalfKv64 padded unpack、单task next-KV预取；C4内四member已共享一次解包；H32完整短C4只有1.006×，不能重复报helper2.346×；writerSIMD pack和向量Init已进入本地candidate；first-draft current-only和native current FIA已存在。later-MTP compact在`compact_later_mtp.py:26–29,130–158`仅N≥1024且N≥8B时启用，不能再拿旧16K×2后续draft全padding当新收益；N128短路径仍有padding，但旧active-only writer样本仅1.001×，不据此追加新核。

## 精度边界

dequant缓存必须绑定真实layer、request/epoch、physical page与写代际，保持FP16 metadata舍入→FP32 Mul/Add；不同请求/不同层不共享值。C4跨簇仅同KV域可共享，不能把宽域非法metadata错误传播给窄域C1。batch8仅改独立row调度，跨管道所有权屏障保留；identity若未来单独处理，finite/NaN/Inf/±0/subnormal和padding/status必须完整资格，不能把hadamard=False推断为identity。

以上均无“FULL字节比→吞吐”的乘数；没有用旧片段推断当前全程占比。
