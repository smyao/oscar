# LongBenchv2 附件定量核对（2026-10-28）

仅解析用户提供的客户端日志；未重跑模型、未连接真机、未改生产。每组10个完整轮次；每轮127/127、FAIL=0。原始指标仅从evaluation字典抽取，重复展示的两位小数表不重复计数。

## 完整十轮

| 组 | 轮 | overall | easy | hard | short | 推断正确数 overall/easy/hard | 进度条时间 | Api infer task(s) |
|---|---:|---:|---:|---:|---:|---|---|---:|
| baseline | 1 | 47.24409449 | 51.16279070 | 45.23809524 | 47.24409449 | 60/22/38 | 14:02 | 869.06 |
| baseline | 2 | 42.51968504 | 44.18604651 | 41.66666667 | 42.51968504 | 54/19/35 | 12:01 | 745.44 |
| baseline | 3 | 38.58267717 | 39.53488372 | 38.09523810 | 38.58267717 | 49/17/32 | 13:09 | 814.07 |
| baseline | 4 | 40.15748031 | 41.86046512 | 39.28571429 | 40.15748031 | 51/18/33 | 11:45 | 729.86 |
| baseline | 5 | 47.24409449 | 53.48837209 | 44.04761905 | 47.24409449 | 60/23/37 | 11:32 | 716.26 |
| baseline | 6 | 43.30708661 | 51.16279070 | 39.28571429 | 43.30708661 | 55/22/33 | 12:15 | 761.57 |
| baseline | 7 | 43.30708661 | 48.83720930 | 40.47619048 | 43.30708661 | 55/21/34 | 12:24 | 771.43 |
| baseline | 8 | 47.24409449 | 51.16279070 | 45.23809524 | 47.24409449 | 60/22/38 | 11:42 | 726.94 |
| baseline | 9 | 40.15748031 | 41.86046512 | 39.28571429 | 40.15748031 | 51/18/33 | 12:36 | 785.21 |
| baseline | 10 | 43.30708661 | 46.51162791 | 41.66666667 | 43.30708661 | 55/20/35 | 13:07 | 813.21 |
| candidate | 1 | 40.15748031 | 41.86046512 | 39.28571429 | 40.15748031 | 51/18/33 | 18:42 | 1149.45 |
| candidate | 2 | 43.30708661 | 46.51162791 | 41.66666667 | 43.30708661 | 55/20/35 | 18:39 | 1143.76 |
| candidate | 3 | 41.73228346 | 46.51162791 | 39.28571429 | 41.73228346 | 53/20/33 | 18:43 | 1147.58 |
| candidate | 4 | 40.94488189 | 48.83720930 | 36.90476190 | 40.94488189 | 52/21/31 | 18:36 | 1141.40 |
| candidate | 5 | 41.73228346 | 48.83720930 | 38.09523810 | 41.73228346 | 53/21/32 | 17:32 | 1077.39 |
| candidate | 6 | 39.37007874 | 46.51162791 | 35.71428571 | 39.37007874 | 50/20/30 | 17:56 | 1100.65 |
| candidate | 7 | 40.15748031 | 44.18604651 | 38.09523810 | 40.15748031 | 51/19/32 | 18:53 | 1158.46 |
| candidate | 8 | 43.30708661 | 51.16279070 | 39.28571429 | 43.30708661 | 55/22/33 | 19:52 | 1218.20 |
| candidate | 9 | 41.73228346 | 46.51162791 | 39.28571429 | 41.73228346 | 53/20/33 | 16:57 | 1045.65 |
| candidate | 10 | 40.15748031 | 44.18604651 | 38.09523810 | 40.15748031 | 51/19/32 | 18:32 | 1138.35 |

## 两种top3口径

| 指标 | baseline独立top3 | candidate独立top3 | 差(pp) | baseline按overall选轮 | candidate按overall选轮 | 差(pp) |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 47.24409449 | 42.78215223 | 4.46194226 | 47.24409449 | 42.78215223 | 4.46194226 |
| accuracy_easy | 51.93798450 | 49.61240310 | 2.32558140 | 51.93798450 | 48.06201550 | 3.87596899 |
| accuracy_hard | 44.84126984 | 40.07936508 | 4.76190476 | 44.84126984 | 40.07936508 | 4.76190476 |
| accuracy_short | 47.24409449 | 42.78215223 | 4.46194226 | 47.24409449 | 42.78215223 | 4.46194226 |

按overall统一选轮：baseline=[1,5,8]；candidate显示=[2,8,3]，其中第三名3/5/9轮并列；选择3或9时同表，选5时easy/hard均值分别48.83720930/39.68253968。overall与short均值不变。不能将任一tie-break当成用户已指定。

## 全部轮次统计

| 指标 | baseline均值 | candidate均值 | 差(pp) | baseline范围 | candidate范围 | baseline样本标准差 | candidate样本标准差 |
|---|---:|---:|---:|---|---|---:|---:|
| accuracy | 43.30708661 | 41.25984252 | 2.04724409 | 38.58267717–47.24409449 | 39.37007874–43.30708661 | 3.14960630 | 1.34858085 |
| accuracy_easy | 46.97674419 | 46.51162791 | 0.46511628 | 39.53488372–53.48837209 | 41.86046512–51.16279070 | 4.87818069 | 2.68535009 |
| accuracy_hard | 41.42857143 | 38.57142857 | 2.85714286 | 38.09523810–45.23809524 | 35.71428571–41.66666667 | 2.62024984 | 1.60702038 |
| accuracy_short | 43.30708661 | 41.25984252 | 2.04724409 | 38.58267717–47.24409449 | 39.37007874–43.30708661 | 3.14960630 | 1.34858085 |

这两组附件中，candidate总体均值和最高轮次低于baseline；baseline十轮overall范围与标准差反而更大。因此这些日志不支持“原生更稳定”的全十轮统计结论，不能只挑最高三轮来描述稳定性。

## 时间统计

| 口径 | baseline十轮均值(s) | candidate十轮均值(s) | candidate/base |
|---|---:|---:|---:|
| progress_duration_seconds | 747.300 | 1106.200 | 1.48026228 |
| api_infer_task_seconds | 773.305 | 1132.089 | 1.46396183 |

首轮进度条是14:02和18:42；十轮进度条均值为12:27.3和18:26.2，不能把首轮时间当成全部轮次均值。Api infer task为日志另一独立口径，不和进度条时间混算。

## 证据边界

两组均为dataset version324293、并发32、burstiness1.0、127题。baseline为10/22、10 RPS，文本首行cat baseline.log；candidate为10/28、0 RPS并明确同时提交。终端标签分别node205/node93，仅标识客户端终端，不能据此推断推理服务器或NPU机器不同。附件没有完整服务配置；不能据RPS差异归咎精度。

在普通correct/count计分且easy/hard互斥覆盖全集假设下，全部20个原始百分比唯一对应overall127、easy43、hard84、short127；各轮easy正确数+hard正确数=overall正确数。它是分母推断，不是附件直接打印的分组count。

- Only10 aggregate evaluations per source; no127 per-item answers, predictions, IDs or per-item scores are present.
- No input/output token counts, finish_reason, empty-output count, truncation count or answer-parser error count are printed. FAIL=0 is request completion, not proof all answers parsed or were untruncated.
- No temperature/top_p/top_k/seed/max_tokens values are printed; cannot establish identical sampling.
- RPS differs(10 vs0), dates differ(10/22 vs10/28). Shell labels node205/node93 identify client terminals, not necessarily serving hosts. Full engine/model launch configurations are absent. Do not assign accuracy differences to RPS or infer different NPU machines.
- Native/candidate roles follow user identification plus date/experiment/header evidence; these client logs do not contain loaded OSCAR/commit/PT identities.
- Averages of per-metric independent top3 can combine different rounds. Unified overall top3 retains cutoff ties, with chronological tie break only for display.
- Top3 summaries are selected best runs and do not replace the full10-run variability summary.

## 来源

- candidate: `/Users/sunao2000/gpt_new_oscar_kimi/reports/precision_stability_20261028/candidate_client.txt`；65737 bytes；SHA256 `e23c4c8db2620954cccb6574bc5c2731b855cd2d7b35cf27be7270c01283aaaf`。各轮具体原文行号在log_analysis.json中。
- baseline: `/Users/sunao2000/gpt_new_oscar_kimi/reports/precision_stability_20261028/baseline_client.txt`；50304 bytes；SHA256 `36c8c0f5e7d28da1e6830b566b9a94b2ae05c330f2dd1c64ba9db9d77f4bbb58`。各轮具体原文行号在log_analysis.json中。
