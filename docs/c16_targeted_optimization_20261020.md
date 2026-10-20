# 基于timer的C16定向优化

本轮只改两项现有timer指向的开销：**softmax有限性检查中的多余结果回写/同步**，以及
**BF16转换逐个64元素repeat发射的开销**。现有timer没有修改；本轮也没有运行性能测试。

依据是用户run `2ae27c282db34edfbb383587023dcb63`：
64/64个目标调用主AIV的首位具体阶段是softmax处理链，四个代表样本23.4–23.8ms；
BF16转换7.5–7.6ms。原始证据和范围见[计时分析](c16_breakdown_analysis_20261020.md)。
finite检查只是softmax大阶段的一部分，**没有把整个23.8ms都算作本轮可消除成本**。

## 1. softmax：保留同一归约，去掉无用结果发布

完整可见的历史tile仍对每个256列score行检查finite。只读CANN 9.1 A2 SDK显示，
`ReduceSum<float>(...,count)`内部依次执行：

1. 同一count-mode `vcadd(...,1)`完成归约；
2. `V_S`等待后读取`ACC_VAL`；
3. 将标量结果写回UB；
4. 用`S_V`及`S_MTE3`发布该标量写入。

原C16随后只是把这些UB标量读回来判断是否有限。新实现直接用第2步的标量作相同判断，
省去第3、4步；256元素计数mask移到行循环外设置，结束后恢复normal mask。
每行仍执行相同`vcadd`、相同归约次序、相同`get_acc_val`及必要的`V_S`，所有行都检查。

这不是改用另一种归约树，也没有跳过NaN/Inf/有限数溢出检查。窗口、尾tile、不可见列的mask分支、
`SoftmaxFlashV2`本体、FP32 max/sum/alpha以及score/P复用屏障保持原逻辑。

实现：[有限性检查helper](../csrc/kernels/attention_c16_softmax_helpers.h)、
[严格锚点变换](../tools/c16_softmax_transform.py)。

## 2. BF16：把独立repeat合成一次发射

旧helper每64个FP32元素发射一次normal-mask转换，8192个元素需要128次发射。
新C16专用helper在相同`{1,1,4,8}`stride和RNE下，一次发射多个repeat：
8192元素用一条repeat=128，超过255 repeats则分批，最后不足64元素的尾部独立mask。

每个repeat仍读取同一段256字节并写出同一段128字节；没有重叠、跳过或访问count以外的元素。
**这只是减少发射指令数量，不是减少转换元素数量，也不是128倍提速。**
不会重引入档案#164出错的大count-overload。所有调用者原有DMA fence保留。

共享`attention_bf16_cast.h`不改，q4/q1/C4也不改。C16文件在所有原头文件解析完毕后，
把本文件内六处发布调用绑定到新helper，末尾解除宏；原unpack头内的转换不受影响。
这也保留原有timer识别的调用文本与阶段边界。

实现：[转换helper](../csrc/kernels/attention_c16_cast.h)、
[转换边界及SDK依据](../reports/c16_targeted_opt_20261020/cast_design.md)。

## 3. 保留的内容与实际启用

- 144B INT2存储、量化器、Q/Rv旋转、BF16矩阵操作数和FP32累加/softmax/LSE全部保持。
- 不新增生产UB或GM workspace，不改16成员调度、KV256 tile或跨核flag。
- 模型、MTP、APC、token预算、设备/端口、原生调度及graph配置保持。
- 原核以`attention_cv_aligned_cluster16_control.cpp`保存；除独立符号名外，源码恢复后与此前原核逐字一致。
  显式关闭`experimental_c16_tuned`选择control，不是运行出错后的自动切换。
- `tools/serving_variants.py`统一开启candidate的`c16_tuned`，直启与observe共用；
  原生产operator名和ABI保留，现有timer无需改动。

## 4. 验证规则及当前证据

验证不是检查几组随机输出就放行：

- softmax helper：289个官方CPU-debug病例，覆盖rows0–32、每行位置的NaN/Inf、有限溢出/抵消、
  signed-zero/subnormal、inactive行、输入和guard。验证循环外mask设置后每一行仍实际执行。
- 转换helper：66个官方CPU-debug病例，覆盖64元素及255-repeat边界、halfway和尾部；
  317020个活跃元素旧新逐位一致，280205个元素另经独立RNE参考，全部inactive guard与输入不变。
- 转换独立参考首轮在FP32→BF16溢出边界失败，而旧新SDK结果相同。只读SDK确认FLOAT饱和由CTRL[48]控制；
  修正了参考对该控制位的假设，保留全部失败输入及原日志，且验证旧新调用不改变控制位。
  没有修改生产helper来迎合参考，也没有放宽模型精度容差。
- 完整C16：同一份prepare任务和输入，先原核后候选；候选重新填充不同输出poison和NaN workspace，
  检查partial/LSE/status/统计逐字节一致，再检查独立v3 oracle、错误域、padding及所有内存边界。
  **12个病例全部通过**，包含16成员成熟簇、多KV tile、尾tile、空历史、Hkv=2/S=3、
  当前/窗口混合、最后成员NaN/Inf、完整tile非法输入及不可见非法metadata。
- 完整CANN库与Torch绑定fresh构建通过，包含55项能力；1116项源码签名与构建时副本一致。
  主机验证实际candidate路由，q4/q1/C4仍选择原实现。构建签名为`8efb9fb74791976c`。
- 主机两组检查合计148项通过、3项已有静态检查失败、1项Linux专用检查在Mac跳过。
  3项失败分别位于旧`replay_shared_fp32.py`、`metadata.py`和`export_shared_fp32.py`；
  已逐字核对它们与基底`b388590f`相同，保留失败日志，本轮不扩大修改范围。
- 尝试核对最终设备汇编：官方编译器不支持device侧`-S`，`save-temps`未生成可读汇编，
  官方objdump也无法解码。因此没有声称最终机器指令/同步条数已实证；源码与CPU功能证据单列。

证据入口：[总回执](../reports/c16_targeted_opt_20261020/validation.json)、
[完整C16主矩阵](../reports/c16_targeted_opt_20261020/evidence/cpu_matrix.json)、
[完整tile错误域](../reports/c16_targeted_opt_20261020/evidence/cpu_full_tile_errors.json)、
[构建清单](../reports/c16_targeted_opt_20261020/evidence/fullbuild_manifest.json)。
有限性和转换的首错、修正后结果、主机日志、源哈希均在同目录保留。

这些是功能与构建证据。CPU模拟不能证明真实异步DMA时序、NPU图回放、完整模型质量或速度收益。
本轮没有新增这几项通过结论；尤其不会用源码发射数量变化宣称端到端超过30%。

## 5. 启动命令

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

正常启动配置摘要应包含`optimizations.c16_tuned=true`。
main/stable保持不变。本轮没有新增timer入口，也没有修改现有timer输出。
