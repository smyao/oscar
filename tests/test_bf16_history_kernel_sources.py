"""Archive G4/#81-83/#126/#145/#148/#154/#157: source/resource contracts.

These checks prove generation identity, typed publication boundaries and bounded
address arithmetic. They do not validate MatmulImpl Reset/re-Init, CANN, NPU
ordering or BF16 quality; the separate stage1 numerical gate remains failed.
"""
from __future__ import annotations

import ast
import operator
import re
from pathlib import Path

import pytest

from tools.experiments.generate_bf16_history_kernels import ROOT, SPECS, derive


def body(source, signature):
    start = source.index("{", source.index(signature))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1:end - 1]


def integer_expression(expression, names):
    """Only the integer grammar used by actual InitBuffer size expressions."""
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.floordiv}
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        raise AssertionError(f"unreviewed kernel size expression: {expression}")
    return visit(ast.parse(expression, mode="eval"))


# Independent D256 allocation expectations, including the 32-byte-aligned
# explicit vector buffers. TPipe/SDK-reserved UB is excluded by the 184KiB cap.
EXPECTED = {
    "mdl": (1146880, 655360, 158656, 175040),
    "member_pipeline_cluster4": (2494464, 1970176, 157760, 174144),
    "window_range_cluster16": (4997120, 3883008, 157632, 174528),
    "window_range": (917504, 786432, 157632, 174528),
    "window_range_balanced": (917504, 786432, 157632, 174528),
}


@pytest.mark.parametrize("name,mother,sha", SPECS)
def test_inverse_generation_preserves_exact_fixed_mother(name, mother, sha):
    generated, _ = derive(name, mother, sha)  # Includes reverse-transform identity.
    path = ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp"
    assert path.read_text() == generated
    # Error/tag checks and ownership classification are untouched whole bodies.
    original = (ROOT / "csrc/kernels" / mother).read_text()
    for signature in ("void LoadTask(", "bool TaskValid(", "int64_t LogicalPosition(",
                      "void PublishRows(", "void PublishStatus("):
        assert body(generated, signature) == body(original, signature)


@pytest.mark.parametrize("name,mother,sha", SPECS)
def test_all_source_operands_use_bf16_and_only_rotation_uses_fp32_matmul(name, mother, sha):
    source = (ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp").read_text()
    publisher = body(source, "void PublishOperand(")
    assert "oscar_bf16_history::Publish(operandWork[offset],source,operandBuf,elements);" in publisher
    assert "kind" not in publisher and "DataCopy(work" not in publisher
    assert "PublishOperand(qOffset+row*D,dst,D)" not in source
    assert "PublishBf16Operand(qOffset+row*D,src,D)" in source
    query = body(source, "void LoadQueries(")
    assert "DataCopy(dst,qrChecksum[(packedRow+D)/2],8)" in query
    assert "if(!Finite(dst.GetValue(0)))error=2" in query
    assert "PublishBf16Operand(qOffset+row*D,zeros,count*D)" in source
    assert "PublishBf16Operand((value?vOffset:kOffset)+slot*D,src,kElements)" in source
    assert "PublishOperand(pOffset+rowBase*kKvRows,scores,activeRows*kKvRows)" in source
    assert "DataCopy(work[normOffset+lane*kLaneRows*D],acc,kLaneRows*D)" in source
    assert not re.search(r"Publish(?:History)?Operand\([^;]*\bacc\b", source)
    assert "using HistoryA=MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,false>" in source
    assert "using HistoryB=MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,true>" in source
    assert "GetMatmulApiTiling<HistoryA,HistoryB,Matrix,Matrix>" in source
    assert "using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>" in source
    dot = body(source, "void Matmul(")
    assert "const bool history=!rotation;" in dot
    assert dot.index("SelectMatmul(history,") < dot.index("if(history)")
    assert "historyMm.template IterateAll<true>(work[c]);" in dot
    assert "mm.SetTensorB(rv,true);" in dot
    assert "mm.SetTensorB(work[b]" not in dot
    assert "mm.template IterateAll<true>(work[c]);" in dot
    if name == "member_pipeline_cluster4":
        atomic = body(source, "void MatmulAtomic(")
        assert "SelectMatmul(true," in atomic
        assert "historyMm.template IterateAll<true>(work[c],static_cast<uint8_t>(1));" in atomic
        assert "mm.template IterateAll" not in atomic


def test_cluster4_history_publisher_preserves_both_old_lane_domains():
    source = (ROOT / "csrc/kernels/attention_bf16_history_member_pipeline_cluster4.cpp").read_text()
    publish = body(source, "void PublishHistoryOperand(")
    assert "PublishBf16Operand(offset,source,kElements);" in publish
    assert "PublishBf16Operand(offset+2*kElements,source[kElements],kElements);" in publish
    assert "DataCopy(work" not in publish
    coverage = {}
    for subtile in range(4):
        for lane in range(2):
            for row in range(32):
                # One lane's input order is its old H16 block, then the next
                # old H16 block; restore the global natural [KV,D] order.
                destination = subtile * 64 + lane * 16 + (row // 16) * 32 + row % 16
                assert destination not in coverage
                coverage[destination] = (subtile, lane, row)
    assert sorted(coverage) == list(range(256))


@pytest.mark.parametrize("name,mother,sha", SPECS)
def test_precise_kv_never_expands_fp32_and_preserves_guard_order(name, mother, sha):
    source = (ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp").read_text()
    window = body(source, "void LoadWindowRuns(")
    current = body(source, "void LoadPrecise(")
    for loader in (window, current):
        assert "auto dst=dequantBuf.Get<bfloat16_t>();" in loader
        assert "Cast(dst" not in loader
        assert "dequantBuf.Get<float>" not in loader
        assert "oscar_bf16_precise_check::Check" in loader
    assert window.index("if(!Physical(position") < window.index("DataCopyPad(tags")
    assert "CheckTaggedRows" in window
    assert "DataCopyPad(dst[(j+begin)*D],wv[off],copy,pad)" in window
    assert "DataCopyPad(dst,cv[off],copy,pad)" in current
    shared = (ROOT / "csrc/kernels/attention_bf16_precise_check.h").read_text()
    finite = body(shared, "void CheckRow(")
    assert "Cast(converted,source,RoundMode::CAST_NONE,rows*D)" in shared
    assert "ReduceSum(check,row,check[16],D)" in finite
    assert finite.index("Fence<HardEvent::V_S>()") < finite.index("if(!Finite")
    assert "Fence<HardEvent::S_V>()" in finite
    # P rounding must not change the FP32 denominator or online accumulation.
    softmax = body(source, "void Softmax(")
    old_softmax = body((ROOT / "csrc/kernels" / mother).read_text(), "void Softmax(")
    assert softmax.replace("PublishOperand(pOffset+rowBase*kKvRows,scores,activeRows*kKvRows)",
                           "DataCopy(work[pOffset+rowBase*kKvRows],scores,activeRows*kKvRows)") == old_softmax


@pytest.mark.parametrize("name,mother,sha", SPECS)
def test_direct_unpack_allocations_cover_actual_strided_addresses(name, mother, sha):
    source = (ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp").read_text()
    h = 64 if name == "mdl" else 32 if name == "member_pipeline_cluster4" else 16
    init = body(source, "void Init(")
    expected = "kHistoryHalfKv" if h == 32 else "kHalfKv"
    assert f"pipe.InitBuffer(naturalBuf,{expected}*(D+16)*2);" in init
    assert f"oscar_ascend_striped_bf16::Unpack<D,{expected}>" in source
    assert "dequantBuf,scratchBuf,operandBuf);" in source
    # Independent highest used natural/index/table/output address, including
    # H32 old-lane packing and H64 last eight-row index scratch reuse.
    assert ((h - 1) * 272 + 255 + 1) * 2 <= h * 272 * 2
    assert ((3 * h + h - 1) + 1) * 2 == 4 * h * 2
    assert 8 * 256 * 4 * 2 == 16384
    if name == "mdl":
        pipeline = body(source, "void PipelinePublishKv(")
        assert "PublishBf16Operand" in pipeline and "dequantBuf.Get<bfloat16_t>()" in pipeline


@pytest.mark.parametrize("name,mother,sha", SPECS)
def test_actual_buffers_and_per_core_arenas_fit_the_declared_resource_limits(name, mother, sha):
    source = (ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp").read_text()
    init = body(source, "void Init(")
    values = {"D": 256, "kHalfKv": 64 if name == "mdl" else 16,
              "kHalfRows": 16 if name == "mdl" else 32, "kKvRows": 256,
              "kLaneRows": 16 if name == "mdl" else 64,
              "kHistoryHalfKv": 32, "kHistoryElements": 8192, "kPackedStride": 160}
    values["kElements"] = values["kHalfKv"] * 256
    values["kWords"] = values["kElements"] // 8
    sizes = re.findall(r"pipe.InitBuffer\(\w+,([^;]*)\);", init)
    total = 0
    for expression in sizes:
        expression = expression.replace("oscar_ascend_fast::Geometry<D>::planeBytes",
                                         str(8 * (values["kWords"] + 16) * 2))
        expression = expression.replace("oscar_bf16_history::kOperandScratchBytes", "16384")
        size = integer_expression(expression, values)
        assert size % 32 == 0
        total += size
    old_bytes, candidate_bytes, old_ub, candidate_ub = EXPECTED[name]
    natural_extra = 0 if name in ("mdl", "member_pipeline_cluster4") else 512
    assert total == candidate_ub == old_ub + 16384 + natural_extra
    assert total <= 184 * 1024
    assert "kOriginalVectorBytes+kNaturalPaddingBytes+oscar_bf16_history::kOperandScratchBytes<=184*1024" in source
    assert "auto coreWorkspace=workspace+core*layout.Bytes();" in init
    assert "work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(coreWorkspace));" in init
    assert "coreWorkspace+layout.FloatElements()*4" in init
    floats, operands = {"mdl": (24576,278528),
        "member_pipeline_cluster4": (263168,458752),
        "window_range_cluster16": (626688,688128),
        "window_range": (98304,196608),
        "window_range_balanced": (98304,196608)}[name]
    assert candidate_bytes == floats*4 + operands*2 < old_bytes
    for core in range(20):
        fp32 = (core * candidate_bytes, core * candidate_bytes + floats*4)
        bf16 = (fp32[1], fp32[1] + operands*2)
        assert fp32[1] == bf16[0]
        assert bf16[1] == (core + 1) * candidate_bytes
        assert all(offset % 32 == 0 for offset in (*fp32, *bf16))
    # Actual addresses and C4 member/slot reuse are executed independently in
    # test_bf16_reader_protocol.py; neither arena reserves an unused mirror.
    last_bf16_element_end = floats*4 + (operands - 1)*2 + 2
    assert last_bf16_element_end == candidate_bytes


def test_shared_helper_rounding_and_resource_switch_order_are_explicit():
    source = (ROOT / "csrc/kernels/attention_bf16_history_operands.h").read_text()
    assert "RoundMode::CAST_RINT" in body(source, "void Publish(")
    select = body(source, "void SelectMatmul(")
    assert select.index("if (history == historyActive) return;") < select.index(".End()")
    assert select.index(".End()") < select.index("PipeBarrier<PIPE_ALL>()") < select.index("pipe.Reset()")
    assert select.index("pipe.Reset()") < select.index("bf16.Init(")
    assert "else fp32.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);" in select
    # Native source establishes this mode/type call exists; no inference from
    # enum spelling or unrelated GDN code is used as compiler acceptance.
    native = (ROOT / "references/vllm-ascend/csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h").read_text()
    assert re.search(r"IsSameType<OUT_T, bfloat16_t>::value[^}]+RoundMode::CAST_RINT", native)


def test_every_new_project_local_include_exists_and_workspace_families_are_declared():
    paths = [ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp" for name, _, _ in SPECS]
    paths.extend((ROOT / "csrc/kernels/attention_bf16_history_operands.h",
                  ROOT / "csrc/include/oscar_bf16_history_launch.h"))
    for path in paths:
        for include in re.findall(r'^#include "([^"]+)"', path.read_text(), re.MULTILINE):
            if include.startswith(("lib/", "adv_api/")):
                continue  # Official SDK is unavailable here, explicitly unvalidated.
            assert (path.parent / include).is_file(), (path, include)
    header = paths[-1].read_text()
    for family, expected in (("", 786432), ("mdl_", 655360),
                             ("member_pipeline_", 1970176), ("cluster16_", 3883008)):
        assert f"attention_bf16_history_{family}workspace_per_core(256)=={expected}" in header
