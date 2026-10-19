"""Host execution of actual BF16 reader address/control-flow expressions.

Archive #126/#145/#148/#151/#154/#164-165 and startup D.4: bounded scratch
ownership matters independently of numerical precision. These are NOT CANN
execution tests: Matmul/vector arithmetic and pipeline completion are stubs.
The event check assumes the mode-2 one-Cube/two-Vector token contract; it
cannot validate SDK Reset/re-Init, FIX completion, numerical or NPU behavior.
"""
from collections import defaultdict
import os
from pathlib import Path
import random
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "csrc/kernels"
FAMILIES = (
    ("range", "attention_bf16_history_window_range.cpp", 0),
    ("balanced", "attention_bf16_history_window_range_balanced.cpp", 0),
    ("cluster16", "attention_bf16_history_window_range_cluster16.cpp", 16),
    ("cluster4", "attention_bf16_history_member_pipeline_cluster4.cpp", 4),
    ("mdl_q4", "attention_bf16_history_mdl.cpp", 0),
    ("mdl_q1", "attention_bf16_history_mdl.cpp", 0),
)


def _without_comments(source):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)


def _function(source, name):
    source = _without_comments(source)
    match = re.search(r"(?:__aicore__\s+)?(?:constexpr\s+)?(?:void|int64_t|int32_t)\s+"
                      + name + r"\s*\(", source)
    assert match, name
    opening = source.index("{", match.start())
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end].replace("__aicore__", "")


DECLARATIONS = r"""
using int64_t=long long; using int32_t=int;
using uint16_t=unsigned short;
extern "C" int printf(const char*,...);
extern "C" int atoi(const char*);
int actor=0;
int activeMember[3]={};
enum class HardEvent{V_S,V_MTE3,MTE2_V};
enum Pipe{PIPE_MTE3,PIPE_MTE2,PIPE_FIX,PIPE_V};
template<HardEvent>void Fence(){} template<Pipe>void PipeBarrier(){}
void CrossCoreWaitFlag(int flag){printf("F %d wait %d\n",actor,flag);}
template<int,Pipe>void CrossCoreSetFlag(int flag){printf("F %d set %d\n",actor,flag);}
struct Tensor{int64_t offset=0;bool gm=false;
  Tensor operator[](int64_t index)const{return {offset+index,gm};}
  float GetValue(int)const{return 1.0F;}};
struct Buffer{template<class T>Tensor Get(){return{};}};
template<class...A>void Duplicate(A...){} template<class...A>void Muls(A...){}
void DataCopy(Tensor dst,Tensor src,int elements){
  if(dst.gm)printf("E %d normwrite %d %lld %d\n",actor,activeMember[actor],dst.offset,elements);
  if(src.gm)printf("E %d rotread %d %lld %d\n",actor,activeMember[actor],src.offset,elements);}
bool Finite(float){return true;}
int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
"""


def _workspace_functions():
    source = _without_comments((ROOT / "csrc/include/oscar_attention_launch.h").read_text())
    constants = "\n".join(re.findall(
        r"constexpr int64_t kAttention(?:Query|Kv)Rows\s*=[^;]+;", source))
    header = _without_comments((ROOT / "csrc/include/oscar_bf16_history_launch.h").read_text())
    functions = header[header.index("struct Bf16AttentionWorkspace"):
                       header.index("#define OSCAR_BF16_HISTORY_DECLARE")]
    return "namespace oscar_ascend {\n" + constants + functions + "\n}\n"


def _model_source(filename, cluster_size, family):
    source = _without_comments((KERNELS / filename).read_text())
    constants = source[source.index("constexpr int32_t kQueryRows="):
                       source.index("constexpr SoftmaxConfig")]
    init = _function(source, "Init")
    sizes = re.search(r"constexpr oscar_ascend::Bf16AttentionWorkspace layout\{[^;]+;", init).group()
    offsets = init[init.index("qOffset=layout.Query();"):init.index("if ASCEND_IS_AIC")]
    body = _function(source, "CubeTask")
    if "mdl" in family:
        body += _function(source, "HistoryCubeTask") + _function(source, "TailColumns")
    if cluster_size:
        body += _function(source, "CubeCluster")
    if cluster_size == 4:
        body += _function(source, "SelectClusterSlots")
        body += _function(source, "VectorCluster")
    return constants + r"""
struct TaskState{int64_t id,kvbegin,kvend,qcount;};
struct Model {
  static constexpr int D=256;
  struct{int hq=6,hk=1;}g;
  int64_t qOffset=0,kOffset=0,vOffset=0,scoreOffset=0,pOffset=0,pvOffset=0,rotOffset=0;
  int64_t accStateOffset=0,statStateOffset=0,baseKOffset=0,baseScoreOffset=0;
  int64_t extraKvOffset=0,extraScoreOffset=0,basePOffset=0,extraPOffset=0,normOffset=0;
  int64_t counters[8]={},arena=0,operandArena=0,bytesPerCore=0;
  int64_t kvbegin=0,kvend=0,qcount=1;
  int lane=0,error=0,kind=0,current=0;
  Buffer accBuf,statsBuf;Tensor work{0,true};
  void InitOffsets(){
""" + sizes + offsets + r"""
    arena=layout.FloatElements();operandArena=layout.OperandElements();bytesPerCore=layout.Bytes();
  }
  void Layout(){printf("L %lld %d %d %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
      arena,kQueryRows,kKvRows,kOffset,vOffset,scoreOffset,pvOffset,
      accStateOffset,statStateOffset,extraKvOffset,extraScoreOffset,qOffset,
      operandArena,bytesPerCore,normOffset,pOffset,extraPOffset);}
  void UseTask(const TaskState&t){current=t.id;activeMember[actor]=current;}
  void LoadQueries(){printf("E %d query %d 0 0\n",actor,current);}
  void LoadClusterKv(int64_t start){printf("E %d kvwrite %lld %lld 0\n",actor,kOffset,start);}
  void SpillState(int member){printf("E %d spill %d 0 0\n",actor,member);}
  void RestoreState(int member){printf("E %d restore %d 0 0\n",actor,member);}
  void Softmax(int64_t start,int row){if(row==lane*kLaneRows)
      printf("E %d soft %d %lld %lld %lld\n",actor,current,start,scoreOffset,pOffset);}
  void PublishRows(int64_t id){printf("E %d publish %lld 0 0\n",actor,id);}
  void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,
      bool rotation=false,bool transposeB=true){
    if(rotation)printf("E %d rotation_query %lld 0 0\n",actor,qOffset/(kQueryRows*D));
    printf("M %d %lld %lld %lld %d %d %d %d %d 0\n",actor,a,b,c,m,n,k,rotation,transposeB);}
  void MatmulAtomic(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k){
    printf("M %d %lld %lld %lld %d %d %d 0 0 1\n",actor,a,b,c,m,n,k);}
""" + body + "\n};\n"


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    compiler = os.environ.get("OSCAR_TEST_CLANGXX") or shutil.which("clang++")
    if not compiler:
        pytest.skip("clang++ unavailable: host control-flow audit not executed")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=15)
    if version.returncode:
        pytest.skip("clang++ unavailable: " + version.stderr.strip())
    cpp = DECLARATIONS + _workspace_functions()
    for index, (family, filename, cluster_size) in enumerate(FAMILIES):
        cpp += f"namespace family{index} {{\n" + _model_source(filename, cluster_size, family) + "}\n"
    cpp += "int main(int argc,char**argv){int f=atoi(argv[1]),kind=atoi(argv[2]),length=atoi(argv[3]),count=atoi(argv[4]),cluster=atoi(argv[5]),group=atoi(argv[6]),protocol=atoi(argv[7]);\n"
    for index, (family, _, cluster_size) in enumerate(FAMILIES):
        cpp += f"if(f=={index}){{using namespace family{index};Model m;m.InitOffsets();m.Layout();m.kind=kind;m.kvend=length;m.qcount=count;m.g.hq=group;\n"
        if cluster_size:
            cpp += f"TaskState members[{cluster_size}];for(int i=0;i<{cluster_size};++i)members[i]={{i,0,length,count}};\n"
            if cluster_size == 4:
                cpp += "if(protocol){for(actor=0;actor<3;++actor){Model v;v.InitOffsets();v.lane=actor-1;v.g.hq=group;if(actor==0)v.CubeCluster(members);else v.VectorCluster(members);}return 0;}\n"
            cpp += "if(cluster)m.CubeCluster(members);else "
        if "mdl" in family:
            cpp += "if(kind==0 && length)m.HistoryCubeTask();else m.CubeTask();"
        else:
            cpp += "m.CubeTask();"
        cpp += "return 0;}\n"
    cpp += "return 1;}\n"
    path = tmp_path_factory.mktemp("bf16_reader_protocol")
    source, binary = path / "control.cpp", path / "control"
    source.write_text(cpp)
    result = subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return binary


def _run(binary, family, source, length, queries, cluster=0, group=6, protocol=0):
    result = subprocess.run([str(binary), str(family), str(source), str(length),
                             str(queries), str(cluster), str(group), str(protocol)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    lines = [line.split() for line in result.stdout.splitlines()]
    layout = dict(zip(("arena", "rows", "kvrows", "k", "v", "score", "pv", "acc",
                       "stat", "extra_k", "extra_score", "q", "operand_arena", "bytes",
                       "norm", "p", "extra_p"), map(int, lines[0][1:])))
    return layout, lines[1:]


def _check_regions(layout, cluster_size, mdl):
    d, m, n = 256, layout["rows"], layout["kvrows"]
    members = cluster_size or 1
    mirror = [(0, members * m * d), (layout["k"], n * d),
              (layout["v"], n * d), (layout["p"], m * n)]
    results = [(layout["norm"], m * d), (layout["score"], m * n),
               (layout["pv"], m * d)]
    if mdl:
        mirror += [(layout["k"] + 2*n*d, n*d), (layout["v"] + 2*n*d, n*d)]
    if cluster_size:
        results += [(layout["acc"], members*m*d), (layout["stat"], members*2*m)]
    if cluster_size == 4:
        mirror += [(layout["extra_k"], n*d), (layout["extra_k"] + n*d, n*d),
                   (layout["extra_p"], m*n)]
        results += [(layout["extra_score"], m*n)]
    for ranges, limit in ((mirror, layout["operand_arena"]), (results, layout["arena"])):
        ordered = sorted(ranges)
        for start, size in ordered:
            assert 0 <= start < start + size <= limit, "arena overflow"
        assert all(a + size <= b for (a, size), (b, _) in zip(ordered, ordered[1:])), "live region alias"
    assert layout["bytes"] == layout["arena"]*4 + layout["operand_arena"]*2
    assert all(value % 32 == 0 for value in (layout["arena"]*4, layout["operand_arena"]*2, layout["bytes"]))


@pytest.mark.parametrize("family", range(len(FAMILIES)), ids=[f[0] for f in FAMILIES])
def test_six_readers_execute_real_offsets_and_cube_bounds(executable, family):
    name, _, cluster_size = FAMILIES[family]
    mdl = "mdl" in name
    for group in (6, 8):
        query_max = 1 if name == "mdl_q1" else 4 if mdl else 128 // group
        for source in (0, 1, 2):
            for length in (0, 1, 17, 255, 256, 257, 769):
                modes = (0, cluster_size) if source == 0 and length and cluster_size else (0,)
                for cluster in modes:
                    layout, trace = _run(executable, family, source, length, query_max, cluster, group)
                    _check_regions(layout, cluster_size, mdl)
                    calls = [list(map(int, event[2:])) for event in trace if event[0] == "M"]
                    qk = [call for call in calls if not call[6] and call[7]]
                    pv = [call for call in calls if not call[6] and not call[7]]
                    rot = [call for call in calls if call[6]]
                    members, tiles = cluster or 1, (length + 255) // 256
                    assert len(qk) == len(pv) == members * tiles
                    assert len(rot) == (members if source == 0 and length else 0)
                    for a, b, c, m, n, k, rotation, transposed, atomic in calls:
                        assert m == query_max * group
                        if rotation:
                            assert n == k == 256 and c == layout["pv"]
                            assert a == layout["norm"] and a + m*256 <= layout["score"]
                        elif transposed:
                            assert k == 256 and 0 < n <= 256
                            assert a + m*k <= layout["k"]
                            assert 0 <= b and b + n*k <= layout["operand_arena"]
                            assert c in (layout["score"], layout["extra_score"])
                        else:
                            assert n == 256 and 0 < k <= 256
                            assert a in (layout["p"], layout["extra_p"])
                            assert 0 <= b and b + k*n <= layout["operand_arena"]
                            assert c == layout["pv"] or atomic
                        # Scores/P retain physical stride256 even for MDL tails.
                        assert 0 <= c and c + m*(256 if rotation else n if not transposed else 256) <= layout["arena"]
                    if mdl and source != 0:
                        assert [c[4] for c in qk] == [min(256, (length-i*256+15)//16*16) for i in range(tiles)]
                        assert [c[5] for c in pv] == [c[4] for c in qk]


def _protocol_trace(lines):
    traces = [[], [], []]
    for event in lines:
        actor = int(event[1])
        if event[0] == "F":
            traces[actor].append((event[2], int(event[3])))
        elif event[0] == "E":
            traces[actor].append((event[2], *map(int, event[3:])))
        else:
            traces[actor].append(("matmul", *map(int, event[2:])))
    return traces


def _replay(traces, layout, tiles, seed):
    positions, v_to_c, c_to_v = [0, 0, 0], defaultdict(int), defaultdict(int)
    kv, scores, probabilities, atomics, produced = {}, {}, {}, defaultdict(int), defaultdict(int)
    restored, spilled = defaultdict(int), defaultdict(int)
    normalized, rotated, rotation_query = {}, {}, None
    rng = random.Random(seed)
    while any(positions[a] < len(traces[a]) for a in range(3)):
        ready = []
        for actor in range(3):
            if positions[actor] == len(traces[actor]):
                continue
            event = traces[actor][positions[actor]]
            if event[0] == "wait":
                flag = event[1]
                if actor == 0 and not all(v_to_c[lane, flag] for lane in (1, 2)):
                    continue
                if actor and not c_to_v[actor, flag]:
                    continue
            ready.append(actor)
        assert ready, "deadlock"
        actor = rng.choice(ready)
        event = traces[actor][positions[actor]]
        positions[actor] += 1
        kind, *args = event
        if kind == "wait":
            if actor == 0:
                for lane in (1, 2):
                    v_to_c[lane, args[0]] -= 1
            else:
                c_to_v[actor, args[0]] -= 1
        elif kind == "set":
            if actor == 0:
                for lane in (1, 2):
                    c_to_v[lane, args[0]] += 1
            else:
                v_to_c[actor, args[0]] += 1
        elif kind == "kvwrite":
            offset, start, _ = args
            kv[actor, offset] = start // 256
        elif kind == "soft":
            member, start, score_offset, p_offset = args
            assert scores[score_offset] == (member, start//256), "score member/round mismatch"
            probabilities[actor, p_offset] = (member, start//256)
        elif kind == "restore":
            member = args[0]
            assert atomics[member] == restored[actor, member], "restore before previous atomic"
            restored[actor, member] += 1
        elif kind == "spill":
            spilled[actor, args[0]] += 1
        elif kind == "normwrite":
            member, offset, count = args
            assert offset == layout["norm"] + (actor-1)*layout["rows"]//2*256
            assert offset+count <= layout["score"], "normalized input overflow"
            normalized[actor] = member
        elif kind == "rotation_query":
            rotation_query = args[0]
        elif kind == "rotread":
            member, offset, count = args
            assert rotated.get(actor) == member, "stale shared rotation result"
            assert layout["pv"] <= offset < offset+count <= layout["pv"]+layout["rows"]*256
        elif kind == "matmul":
            a, b, c, _, _, _, rotation, transpose, atomic = args
            if rotation:
                member = rotation_query
                assert atomics[member] == tiles, "rotation before all atomics"
                assert all(normalized.get(lane) == member for lane in (1, 2)), "stale shared normalized input"
                assert a == layout["norm"]
                for lane in (1, 2):
                    rotated[lane] = member
            elif transpose:
                member = a // (layout["rows"]*256)
                round_ = produced[member]
                assert all(kv.get((lane, b)) == round_ for lane in (1, 2)), "stale K"
                scores[c] = (member, round_)
                produced[member] += 1
            else:
                assert atomic
                member = (c-layout["acc"]) // (layout["rows"]*256)
                round_ = atomics[member]
                assert all(kv.get((lane, b-256*256)) == round_ for lane in (1, 2)), "stale V"
                assert all(probabilities.get((lane, a)) == (member, round_) for lane in (1, 2)), "P member/round mismatch"
                assert all(spilled[lane, member] == round_+2 for lane in (1, 2)), "atomic before alpha spill"
                atomics[member] += 1
    assert not any(v_to_c.values()) and not any(c_to_v.values()), "undrained flags"
    assert dict(atomics) == {member: tiles for member in range(4)}
    for lane in (1, 2):
        assert sum(e[0] == "publish" for e in traces[lane]) == 4


@pytest.mark.parametrize("tiles", [1, 2, 3, 7, 16])
def test_c4_real_control_flow_preserves_member_and_buffer_ownership(executable, tiles):
    layout, lines = _run(executable, 3, 0, tiles*256, 21, cluster=4, protocol=1)
    traces = _protocol_trace(lines)
    for seed in range(16):
        _replay(traces, layout, tiles, seed)


def test_control_checks_reject_alias_and_missing_readiness(executable):
    layout, lines = _run(executable, 3, 0, 3*256, 21, cluster=4, protocol=1)
    aliased = dict(layout, extra_k=layout["k"])
    with pytest.raises(AssertionError, match="live region alias"):
        _check_regions(aliased, 4, False)
    traces = _protocol_trace(lines)
    # A producer that omits one actual P-ready event must not be accepted.
    index = next(i for i, e in enumerate(traces[1]) if e == ("set", 11))
    del traces[1][index]
    with pytest.raises(AssertionError):
        _replay(traces, layout, 3, 0)


@pytest.mark.parametrize("old_fp32_first", [False, True], ids=["bf16-first", "old-init-negative-control"])
def test_actual_matmul_lifetime_helper_starts_with_one_bf16_allocation(tmp_path, old_fp32_first):
    # This executes the real control helper, with resource API calls recorded.
    # It does not model the official allocator or prove SDK repeated-Init safety.
    compiler = os.environ.get("OSCAR_TEST_CLANGXX") or shutil.which("clang++")
    if not compiler:
        pytest.skip("clang++ unavailable")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=15)
    if version.returncode:
        pytest.skip("clang++ unavailable: " + version.stderr.strip())
    source = (KERNELS / "attention_bf16_history_operands.h").read_text()
    source = source[source.index("template<class Bf16Matmul>"):source.rindex("}  // namespace")]
    source = source.replace("__aicore__", "")
    declarations = r"""
extern "C" int printf(const char*,...);
struct TCubeTiling{};
struct TPipe {void Reset(){printf("reset\n");}};
enum Pipe {PIPE_ALL};
template<Pipe>void PipeBarrier(){printf("barrier\n");}
void SetHF32Mode(bool){printf("hf32_off\n");}
struct Matmul {const char* name;
  void Init(const TCubeTiling*,TPipe*){printf("init_%s\n",name);}
  void End(){printf("end_%s\n",name);}};
"""
    initial = ("fp32.Init(nullptr,&pipe);" if old_fp32_first else
               "InitMatmul(active,pipe,bf16);")
    main = "int main(){bool active=false;TPipe pipe;Matmul fp32{\"fp32\"},bf16{\"bf16\"};"
    main += initial + "SelectMatmul(true,active,pipe,fp32,bf16);printf(\"first_qk\\n\");"
    main += "SelectMatmul(true,active,pipe,fp32,bf16);SelectMatmul(false,active,pipe,fp32,bf16);"
    main += "SelectMatmul(true,active,pipe,fp32,bf16);return 0;}"
    cpp, binary = tmp_path / "lifetime.cpp", tmp_path / "lifetime"
    cpp.write_text(declarations + source + main)
    built = subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert ran.returncode == 0, ran.stderr
    before, after = ran.stdout.split("first_qk\n")
    if old_fp32_first:
        assert before.splitlines() == ["init_fp32", "end_fp32", "barrier", "reset", "init_bf16", "hf32_off"]
    else:
        assert before.splitlines() == ["init_bf16", "hf32_off"]
    assert after.splitlines() == ["end_bf16", "barrier", "reset", "init_fp32", "hf32_off",
                                 "end_fp32", "barrier", "reset", "init_bf16", "hf32_off"]
    for _, filename, _ in FAMILIES:
        init = _function((KERNELS / filename).read_text(), "Init")
        assert "oscar_bf16_history::InitMatmul(historyMatmulActive,pipe,historyMm);" in init
        assert "mm.Init(" not in init and "historyMm.Init(" not in init
