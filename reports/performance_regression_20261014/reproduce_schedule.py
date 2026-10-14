"""Archive #34/#36/#130/#156/#157: host-only schedule counts, never timings.

Run from repository root: python3 reports/performance_regression_20261014/reproduce_schedule.py
Uses the pinned native function body, without importing a vLLM runtime.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oscar_ascend.integration.prefix_cache import aligned_chunk_tokens

native_path = ROOT / "references/vllm/vllm/v1/core/sched/scheduler.py"
tree = ast.parse(native_path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_mamba_block_aligned_split")
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
namespace = {}
exec(compile(ast.fix_missing_locations(module), str(native_path), "exec"), namespace)


def schedule(prompt, budget, oscar):
    end, endpoints, chunks = 0, [], []
    while end < prompt:
        proposed = min(budget, prompt - end)
        if oscar:
            n = aligned_chunk_tokens(computed=end, proposed=proposed, prompt_tokens=prompt,
                full_tokens=2304, native_tokens=768, hash_tokens=768, guarded=True, mtp=True)
        else:
            n = namespace[fn.name](SimpleNamespace(cache_config=SimpleNamespace(block_size=768), use_eagle=True),
                SimpleNamespace(num_computed_tokens=end, num_prompt_tokens=prompt, num_tokens=prompt), proposed)
        if not 0 < n <= proposed:
            raise RuntimeError(f"non-convergent/over-budget schedule at {end}, proposed={proposed}, result={n}")
        end += n
        endpoints.append(end)
        chunks.append(n)
    return {"steps": len(endpoints), "endpoints": endpoints, "chunks": chunks}


if __name__ == "__main__":
    rows = []
    for prompt in (32768, 65100):
        for budget in (2048, 2304, 8192, 16384):
            native, oscar = schedule(prompt, budget, False), schedule(prompt, budget, True)
            rows.append({"prompt": prompt, "budget": budget, "native_mamba": native,
                         "oscar": oscar, "step_count_ratio": oscar["steps"] / native["steps"]})
    print(json.dumps({"scope": "single cold request, full budget available at each step, no mixed decode/preemption",
        "assumed_geometry": {"full":2304, "native_mamba":768, "hash":768, "mtp":True},
        "not_claimed": ["current service resolved budget", "old ten-minute baseline configuration",
                        "wall time ratio", "NPU correctness/performance", "complete mixed scheduler simulation"],
        "source_sha256": {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            native_path, ROOT / "oscar_ascend/integration/prefix_cache.py")}, "rows":rows}, indent=2))
