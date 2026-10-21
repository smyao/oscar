# 2026-10-21 官方数值链审计与修复证据

修复前基线为7786910f。summary.json、attention/codec/calibration审计及其反例属于修复前证据；修复结果以repair_validation.json、kernel_validation.json和docs/official_math_v4_20261021.md为准。

- repair_validation.json：最终主机回归命令、714 passed / 9 skipped / 6 subtests passed、当前数学源绑定和旧内核/timer冻结检查。
- repair_host_tests.log：最终主机结果；repair_host_environment_failure.log.gz原样压缩保留此前测试进程环境泄漏的真实失败。修复只隔离模拟CLI测试的环境，不改生产readiness。
- kernel_validation.json：本地Lima CANN完整构建、11个官方AscendC CPU-debug病例、生产CANN源绑定、失败尝试与验收边界。
- vm_evidence.tar.gz：真实构建/CPU日志和摘要，保留最初staging与harness失败。大张量只保存SHA，不纳入Git；通过csrc/experiments/official_math_20261021/fixture.py重新生成。
- stage_local_vm.py、collect_vm_evidence.py、source_stage.json：本轮本地VM输入和证据收集。source_stage记录首次打包时的源，最终生产CANN身份以kernel_validation和包内build.json为准。

CPU执行包含writer、query-copy、prepare、reader和FP32 merge；最终BF16矩阵逆旋转由主机数值测试与设备probe覆盖，未执行真实NPU。NPU完成、图捕获/回放、LongBench、性能均未验收。

在本目录执行 `tar -xzf vm_evidence.tar.gz -C vm_evidence` 可查看包内日志，需先创建vm_evidence目录。包内路径为logs/、reports/、build/；所有通过与失败阶段均保留。

## 修复前反例的上游源

upstream_manifest.json固定官方commit、URL和SHA256。下载的上游树不纳入本工程生产代码或Git；需要重跑attention_repro.py或codec_rounding_probe.py时，在仓库根目录用以下命令恢复只读审计输入（需要网络）：

```bash
python3 - <<'PY'
import hashlib, json, urllib.request
from pathlib import Path
root = Path('reports/official_semantics_audit_20261021')
manifest = json.loads((root / 'upstream_manifest.json').read_text())
for name, item in manifest['files'].items():
    if 'sha256' not in item:
        continue
    data = urllib.request.urlopen(item['url'], timeout=60).read()
    assert hashlib.sha256(data).hexdigest() == item['sha256'], name
    target = root / 'upstream' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
PY
```

这些反例人工构造相同输入，证明数学差异存在；它们不是用户LongBench下降幅度的归因或修复后恢复分数。
