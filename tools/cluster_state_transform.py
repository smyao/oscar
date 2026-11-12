"""Archive #145/#150/#151/#154/#164/#172: exact C4/C16 state traffic.

D.4(1): bounded clustered FIA online-state transfers, not attention math.
D.4(2): never materialize full history; keep all KV256 and Cube/Vector flags.
D.4(3): defer the identical initial state until its first consumer, and merge
only the two strided max/sum DMA submissions; retain their addresses/fences.
D.4(4): delete one state roundtrip/member, no extra allocation or speed claim.

Complete source pinning belongs to the composing owner generator. This local
transform has exact anchors and an inverse for whole-source identity checking.
"""
from __future__ import annotations

MARKER = "// CLUSTER_STATE_FIRST_USE_V1"
HEADER = """// CLUSTER_STATE_FIRST_USE_V1
// Archive #145/#150/#151/#154/#164/#172; full startup D.4 reviewed.
// D.4(1): bounded clustered FP32 online-state traffic inside C4/C16 FIA.
// D.4(2): keep KV256 and prohibit the old6.5s full-history restoration.
// D.4(3): identical state initialization at first use; same max/sum addresses
// through two-block DMA; all math, live-transfer fences and cross flags stay.
// D.4(4): no added memory; remove one state roundtrip per cluster member.
// Address-byte savings do not establish HBM savings or measured NPU speed.
"""
INITIALIZE = """      Duplicate(acc,0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
      Duplicate(stats,kEmptyMax,kLaneRows);
      Duplicate(stats[kLaneRows],0.0F,kLaneRows);
      Fence<HardEvent::V_S>();
"""
OLD_QUERY = INITIALIZE + """      LoadQueries();
      memberErrors[member]=error;
      SpillState(member);
"""
NEW_QUERY = """      // LoadQueries never reads online state and finishes its own publication.
      // The first Softmax initializes state directly; no zero-state GM trip.
      LoadQueries();
      memberErrors[member]=error;
"""
OLD_RESTORE = """        RestoreState(member);
        for(int32_t block=0;block<kRowBlocks;++block)
"""
NEW_RESTORE = """        if(start==first) {
          // IsClusterMember proves first<last. Cube first consumes state only
          // after the unchanged SpillState/PReady below, never before this.
          Duplicate(acc,0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
          Duplicate(stats,kEmptyMax,kLaneRows);
          Duplicate(stats[kLaneRows],0.0F,kLaneRows);
          Fence<HardEvent::V_S>();
        } else {
          RestoreState(member);
        }
        for(int32_t block=0;block<kRowBlocks;++block)
"""
OLD_SPILL = """    DataCopy(work[statBase],stats,kLaneRows);
    DataCopy(work[statBase+kQueryRows],stats[kLaneRows],kLaneRows);
"""
NEW_SPILL = """    static_assert(kQueryRows==2*kLaneRows && kLaneRows*sizeof(float)%32==0,
        "cluster max/sum DMA requires two aligned lane blocks");
    constexpr uint16_t statBlocks=kLaneRows*sizeof(float)/32;
    // UB max/sum are adjacent; GM has the other AIV's lane between them.
    DataCopyParams statCopy{2,statBlocks,0,statBlocks};
    DataCopy(work[statBase],stats,statCopy);
"""
OLD_RELOAD = """    DataCopy(stats,work[statBase],kLaneRows);
    DataCopy(stats[kLaneRows],work[statBase+kQueryRows],kLaneRows);
"""
NEW_RELOAD = """    static_assert(kQueryRows==2*kLaneRows && kLaneRows*sizeof(float)%32==0,
        "cluster max/sum DMA requires two aligned lane blocks");
    constexpr uint16_t statBlocks=kLaneRows*sizeof(float)/32;
    DataCopyParams statCopy{2,statBlocks,statBlocks,0};
    DataCopy(stats,work[statBase],statCopy);
"""
REPLACEMENTS = ((OLD_QUERY, NEW_QUERY), (OLD_RESTORE, NEW_RESTORE),
                (OLD_SPILL, NEW_SPILL), (OLD_RELOAD, NEW_RELOAD))


def method(text: str, name: str, result: str = "void") -> str:
    signature = f"  __aicore__ {result} {name}("
    if text.count(signature) != 1:
        raise ValueError(f"cluster state {name} method anchor drift")
    start = text.index(signature)
    first = text.index("{", start)
    depth = 0
    for pos in range(first, len(text)):
        depth += (text[pos] == "{") - (text[pos] == "}")
        if depth == 0:
            return text[start:pos + 1]
    raise ValueError(f"cluster state {name} is unclosed")


def _check(text: str) -> None:
    if sum(s in text for s in ("oscar_attention_cv_official_cluster4_kernel",
                              "oscar_attention_cv_official_cluster16_kernel")) != 1:
        raise ValueError("cluster state requires complete C4/C16 source")
    cluster = method(text, "VectorCluster")
    qualification = method(text, "IsClusterMember", "bool")
    queries = method(text, "LoadQueries")
    if ("kvbegin>=kvend" not in qualification or "kvend!=context" not in qualification or
            "kvbegin!=original.kvbegin || kvend!=original.kvend" not in qualification):
        raise ValueError("cluster state requires nonempty same-range cluster qualification")
    if "accBuf" in queries or "statsBuf" in queries or "Fence<HardEvent::MTE3_V>();" not in queries:
        raise ValueError("cluster state query staging ownership changed")
    if (OLD_QUERY not in cluster or OLD_RESTORE not in cluster or
            OLD_SPILL not in method(text, "SpillState") or
            OLD_RELOAD not in method(text, "RestoreState")):
        raise ValueError("cluster state method scope drift")
    for old, _ in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError("cluster state exact replacement anchor drift")
    for anchor in ("const int64_t first=members[0].kvbegin,last=members[0].kvend;",
                   "error=memberErrors[member];RestoreState(member);",
                   "        SpillState(member);\n        CrossCoreSetFlag<2,PIPE_MTE3>"):
        if anchor not in cluster:
            raise ValueError("cluster state final publication/atomic ownership drift")


def transform(text: str) -> str:
    if MARKER in text:
        raise ValueError("cluster state transform already applied")
    _check(text)
    for old, new in REPLACEMENTS:
        text = text.replace(old, new, 1)
    return HEADER + text


def restore(text: str) -> str:
    if not text.startswith(HEADER):
        raise ValueError("cluster state derived header drift")
    original = text.removeprefix(HEADER)
    for old, new in REPLACEMENTS:
        if original.count(new) != 1:
            raise ValueError("cluster state derived replacement drift")
        original = original.replace(new, old, 1)
    _check(original)
    if transform(original) != text:
        raise ValueError("cluster state inverse is not exact")
    return original
