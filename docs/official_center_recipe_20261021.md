# 一键 candidate 的逐头旋转与 key centering

用户要求直接修复，不再运行诊断实验；现场操作仍只有：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

candidate 默认使用 `rotation_variant=perhead_center`、校准 recipe 3、PT schema 2。每个全局 KV 头分别拟合 K/V 矩阵，K 矩阵来自该头的 `unit_mean_eig(M_q)`，V 矩阵来自该头的能量加权 `S_v`；二者均为官方 `U·H·Pbr`。每头保存 post-RoPE K 均值。公式来源为官方固定版本38680852的 `rotation/tools/fit_oscar2_variants.py` 的 center 分支，不再先跨头平均成共享矩阵。

当前服务目标Qwen27B的4个KV头、TP4按原生TP rank取对应矩阵和均值，避免把某头参数广播到其它头。此版本的center运行路径明确要求每rank一个KV头，其他TP布局拒绝启动，不悄悄退回共享V1。旧V1路径仍保留。

生产K在预分配缓冲执行BF16均值相减，然后BF16旋转；Q只旋转。相同的中心化K供给历史INT2写入、精确窗口和current，禁止只中心化历史。最终输出仍沿用v4的FP32合并后一次BF16逆旋转。MTP独立使用identity和零均值，不使用目标层统计。新增当前步scratch在KV profiling前分配，纳入字节账；不修改CANN、INT2 codec、timer、原生GDN、graph或启动argv。

## PT自动升级

- 新recipe对应新的自动缓存目录；旧V1 PT不能命中新recipe。
- 如果唯一对应旧recipe2缓存的原始校准rank文件仍完整，验证模型、采样profile、rank身份和SHA后直接离线重拟合新PT；原始manifest与旧PT保留，不修改来源标签。
- 缺少可复用样本时，一键脚本自动进行原有NPU模型校准并生成新PT。
- 新PT完整且匹配后自动复用；用户无需删PT、手工指定路径或运行诊断命令。显式不匹配或损坏的PT仍报错。

现场启动摘要会显示perhead_center；首次升级的prepare日志区分从旧样本重新拟合、重新采样和复用新PT。二者都属于正式生成流程，不是额外模型评测。

## 验证边界

仅做必要主机功能验证：官方公式、分头TP身份、均值与pair完整性、旧样本升级/缺失重采/新缓存复用、current-only与writer只中心化一次、固定地址scratch和MTP零均值。CANN源码未改，沿用已完成的v4构建证据；不连接真机、不测速，不要求用户重跑诊断实验。

本轮没有取得新的LongBench分数，不能声明5点已恢复。此修复实现官方center公式；未取得官方Qwen3.8发布PT及其逐头K clip表，现有标量clip .96/.92保留，不声称发布产物逐值一致。原生物理窗口/MTP生命周期的已声明边界同v4。
