"""Compare full offline fitter against the fixed pre-fix commit, without timing."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from oscar_ascend import optr_calibration as current

torch.set_num_threads(1)
commit = subprocess.check_output(["git", "rev-parse", "e5423c2"], cwd=ROOT, text=True).strip()
source = subprocess.check_output(["git", "show", commit + ":oscar_ascend/optr_calibration.py"], cwd=ROOT)
baseline = types.ModuleType("pre_fix_optr_calibration")
exec(compile(source, "pre_fix_optr_calibration.py", "exec"), baseline.__dict__)
trace = runpy.run_path(str(ROOT / "tests/test_optr_calibration.py"))["trace"]
rows = []
for dimension, splits in ((8, 1), (64, 20), (256, 20)):
    train = trace(0, "train", dim=dimension, length=48, queries=5, group=6)
    validation = [trace(10, "validation-history", dim=dimension, length=541, queries=9, group=6),
                  trace(11, "validation-current", dim=dimension, length=29, queries=5, group=6)]
    validation[1]["contexts"].zero_()
    for sample in (train, *validation):
        sample["history_splits"] = torch.full_like(sample["contexts"], splits)
    policy = {"sink_tokens": 2, "recent_tokens": 4, "steps": 2, "eval_every": 1,
              "query_batch_size": 2, "lr": .005, "replay_partition": "captured_runtime_frames"}
    rotation = torch.eye(dimension)
    mean = torch.linspace(-.4, .4, dimension)
    before = baseline.fit_head(rotation, rotation, mean, [train], validation, policy=policy)
    after = current.fit_head(rotation, rotation, mean, [train], validation, policy=policy)
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
    assert before[2] == after[2]
    rows.append({"head_dim": dimension, "gqa": 6, "history_splits": splits,
                 "history_tokens": 541, "rk_rv_bitwise_equal": True,
                 "complete_decision_equal": True, "accepted": after[2]["accepted"]})
result = {"baseline_commit": commit, "baseline_source_sha256": hashlib.sha256(source).hexdigest(),
          "current_source_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
          "cases": rows, "performance": "not_measured", "scope": "synthetic_cpu_math_only"}
Path(__file__).with_name("fitter_equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
