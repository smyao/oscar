"""Reproduce logical lookup work, never device timing or physical HBM bytes."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import active_decode_transform as decode
from tools import cluster4_l1_transform as l1
from tools import cluster_score_ring_transform as ring
from tools import cluster_state_transform as state
from tools import cluster_visibility_transform as visible
from tools import precise_finite_transform as finite
from tools import precise_zero_fill_transform as zero
from tools import physical_page_cache_transform as cache


def source(family):
    text = (ROOT / f"csrc/kernels/attention_cv_official_{family}.cpp").read_text()
    if family == "decode":
        return decode.transform(text)
    base = state.transform(zero.transform(finite.transform(visible.transform(text) if family == "cluster4" else text)))
    return l1.transform(base) if family == "cluster4" else ring.transform(base)


def calls(begin, end, lane, family):
    """Scalar positions in the exact LoadPackedInto/LoadPacked traversal order."""
    half = 64 if family == "decode" else 16
    subtiles = 2 if family == "decode" else 4
    for tile in range(begin, end, 256):
        for sub in range(subtiles):
            start = tile + sub * (128 if family == "decode" else 64)
            first = start + lane * half
            live = max(0, min(half, end - first))
            if family != "decode":
                live += max(0, min(half, end - first - 2 * half))
            j = 0
            while j < live:
                candidate = first + j if family == "decode" else start + (j // half) * (2 * half) + lane * half + j % half
                rows = min(live - j, 128 - candidate % 128)
                if family != "decode":
                    rows = min(rows, half - j % half)
                yield candidate
                j += rows


def count_interval(begin, end, family):
    calls_before = misses = 0
    for lane in (0, 1):
        last = None
        for position in calls(begin, end, lane, family):
            calls_before += 1
            page = position // 128
            if page != last:
                misses += 1
                last = page
    return calls_before, misses


def measure():
    sources = {name: source(name) for name in ("decode", "cluster4", "cluster16")}
    for name, text in sources.items():
        assert cache.family(text) == name
        assert state.method(text, "Physical", "bool") == cache.OLD_PHYSICAL
    # These full-source pins constrain the loop formulas mirrored above.
    assert state.method(sources["cluster4"], "LoadPacked") == state.method(sources["cluster16"], "LoadPacked")
    old = json.loads((ROOT / "reports/batch_scaling_20261112/workload.json").read_text())
    cases = []
    for item in old["cases"]:
        if item["phase"] != "main_q4":
            continue
        B, C, S = item["batch_requests"], item["context_tokens"], item["source_splits"]
        width = (C - 64 + S - 1) // S
        before = after = 0
        per_split = []
        for split in range(S):
            begin, end = min(C, 64 + split * width), min(C, 64 + (split + 1) * width)
            a, b = count_interval(begin, end, "decode")
            before += a
            after += b
            per_split.append({"begin_mod128": begin % 128, "lookups": a, "dma_misses": b})
        cases.append({"context_tokens": C, "batch_requests": B, "source_splits": S,
                      "phase": "q4_history; same history traversal in later q1",
                      "lookups_and_fences_before_and_after": B * before,
                      "table_dma_before": B * before, "table_dma_after": B * after,
                      "table_scalar_reads_before": B * before, "table_scalar_reads_after": B * after,
                      "table_dma_payload_before": 4 * B * before, "table_dma_payload_after": 4 * B * after,
                      "hit_pct": round(100 * (1 - after / before), 3), "per_split_two_aiv": per_split})
    isolated = []
    for family in ("decode", "cluster4", "cluster16"):
        for begin, end in ((0, 256), (64, 32768), (77, 32768)):
            before, after = count_interval(begin, end, family)
            isolated.append({"family": family, "begin": begin, "end": end,
                             "lookups_and_fences_before_and_after": before,
                             "table_dma_before": before, "table_dma_after": after,
                             "hit_pct": round(100 * (1 - after / before), 3)})
    return {"kind": "static_logical_work_count_not_measurement", "baseline_commit": "de0c85b",
            "source_sha256": {name: sha256(text.encode()).hexdigest() for name, text in sources.items()},
            "derived_sha256": {name: sha256(cache.transform(text).encode()).hexdigest() for name, text in sources.items()},
            "decode_cases": cases, "isolated_history_intervals": isolated,
            "assumptions": ["Valid nonempty source0 tasks and physical table mappings; sink64; two AIV lanes.",
                            "Decode B16/B28 split counts inherited from the recorded Hq6/Hkv1,20-Cube scenario.",
                            "One cold scalar cache at each task start. Production cache is per kernel instance; extra cross-task same-key hits are excluded.",
                            "C4/C16 isolated intervals represent one shared history stream, not multiplied by member count.",
                            "Full source hashes bind the unchanged loader traversal used by the counting formulas."],
            "costs_and_limits": ["Every Physical call retains its old MTE2_S fence, including a cache hit.",
                                 "Each valid-position call adds two scalar key comparisons, one branch, and a cached scalar access; a miss also updates three scalar fields.",
                                 "No extra UB/GM; three int64 scalar fields. Compiler register/spill effects are not measured.",
                                 "Aligned decode intervals can have zero hits: extra comparisons then have no DMA savings.",
                                 "Removed payload is only 4 bytes per hit. Submission reduction is not HBM traffic, elapsed wait reduction or CV/throughput speedup.",
                                 "No change to INT2 dequant, KV staging, Cube/softmax work, split/owner schedule, native graph/MTP, PT, precision, argv or timer."],
            "reproduce": ".venv/bin/python reports/bs_scaling_20261113/physical_page_cache_workload.py"}


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
