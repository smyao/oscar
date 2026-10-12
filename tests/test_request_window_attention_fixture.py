"""Archive #126-129/#145/#154/#155: host address/math preservation, not NPU proof."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import torch

from csrc.experiments.request_windows_attention import fixture, generate
from tools.striped_fixture import striped_slots_to_canonical

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cases():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return {case.name: fixture.make_case(torch, case) for case in fixture.case_plan()}


def test_independent_physical_layouts_and_logical_request_windows(cases):
    from oscar_ascend.ops.reference import encode_kv
    for name in ("q4_cross_segments_pages", "q1_cross_segments_pages", "q1_native2816_to5632"):
        f = cases[name]; data = f["inputs"]
        n,hq,hk,d,requests,columns,old_b,old_blocks,new_b,new_blocks,prefix,stride,capacity,bits,*_ = f["shape"]
        assert new_b == 2*old_b
        assert data["old_raw"].data_ptr() != data["new_raw"].data_ptr()
        assert data["old_wk"].data_ptr() != data["new_wk"].data_ptr()
        assert not torch.equal(data["old_table"], data["new_table"])
        active = len(f["case"].contexts)*(4 if f["case"].mode=="q4" else 1)
        assert bool(data["old_slots"][active:].eq(-1).all())
        assert bool(data["new_slots"][active:].eq(-1).all())
        rk = fixture.reuse._hadamard(torch, d)
        for request, (old_k, old_v) in f["logical"].items():
            packed = encode_kv(old_k.float()@rk, old_v.float()@data["rv"])
            slot, epoch = map(int, data["request_map"][request])
            assert slot != request
            for position in range(old_k.shape[0]):
                virtual_old = int(data["old_table"][request, position//128])
                old_page, old_quarter = divmod(virtual_old, old_b//128)
                old_row = old_quarter*128+position%128
                virtual_new = int(data["new_table"][request, position//128])
                new_page, new_quarter = divmod(virtual_new, new_b//128)
                new_row = new_quarter*128+position%128
                segment, row = divmod(new_row, old_b)
                old_offset = prefix+old_page*stride+old_row*136
                new_offset = prefix+new_page*stride+segment*new_blocks*stride+row*136
                # Byte comparison includes codes and both FP16 scale/zero pairs.
                assert torch.equal(data["old_raw"][old_offset:old_offset+136], data["new_raw"][new_offset:new_offset+136])
            sample = (0, min(old_k.shape[0]-1, old_b), old_k.shape[0]-1)
            for position in sample:
                page = f["new_assignments"][request][position//new_b]
                segment, row = divmod(position%new_b, old_b)
                off = prefix+page*stride+segment*new_blocks*stride+row*136
                assert torch.equal(striped_slots_to_canonical(data["new_raw"][off:off+136]), packed[position, 0])
            precise = set(range(min(64, old_k.shape[0]))) | set(range(max(0, old_k.shape[0]-256), old_k.shape[0]))
            for position in precise:
                row = position if position<64 else 64+position%259
                assert data["new_tags"][slot,row] == (epoch<<bits)|position
                assert torch.equal(data["new_wk"][slot,row].view(torch.int16), old_k[position].view(torch.int16))
                assert torch.equal(data["new_wv"][slot,row].view(torch.int16), old_v[position].view(torch.int16))
        if old_b == 2816:
            assert stride == 393216 and f["case"].contexts == (5901,)
            assert data["new_raw"].numel() == prefix+2*new_blocks*393216


@pytest.mark.parametrize("base,reordered,order,qlen", [
    ("q4_cross_segments_pages", "q4_reordered", (1,0), 4),
    ("q1_cross_segments_pages", "q1_reordered", (2,0,1), 1)])
def test_batch_reorder_preserves_sidecar_identity_and_physical_payload(cases, base, reordered, order, qlen):
    a,b = cases[base]["inputs"],cases[reordered]["inputs"]
    for name in ("old_raw", "new_raw", "old_tags", "new_tags"):
        assert torch.equal(a[name], b[name])
    for name in ("old_wk", "old_wv", "new_wk", "new_wv"):
        assert torch.equal(a[name].view(torch.int16), b[name].view(torch.int16))
    for dst,src in enumerate(order):
        assert torch.equal(b["request_map"][dst], a["request_map"][src])
        assert torch.equal(b["expected_output"][dst*qlen:(dst+1)*qlen], a["expected_output"][src*qlen:(src+1)*qlen])


def test_faults_have_explicit_independent_status_domains(cases):
    for name in ("q4_stale_epoch", "q1_poison_stale", "q1_bad_tag"):
        f = cases[name]
        assert f["flags"] == (1,1,4)
    for name in ("q1_invalid_slot", "q1_invalid_epoch"):
        assert cases[name]["flags"] == (1,0,1)
    overflow = cases["q1_position_overflow"]
    assert overflow["flags"] == (1,0,4)
    slot = int(overflow["inputs"]["request_map"][0,0])
    assert overflow["inputs"]["new_tags"][slot,64+501%259] == -1
    assert bool(torch.isnan(overflow["inputs"]["new_wv"][slot,64+501%259].float()).all())


def _function(text, name):
    start = text.index(" "+name+"(")
    begin = text.index("{", start); depth = 1; end = begin+1
    while depth:
        depth += (text[end] == "{")-(text[end] == "}"); end += 1
    return text[start:end]


def test_generator_preserves_actual_math_and_guards_map_and_negative_tag():
    root = ROOT/"csrc/experiments/request_windows_attention"
    for row in json.loads((root/"sources.json").read_text()):
        source = (ROOT/row["source"]).read_text()
        candidate = (root/row["candidate"]).read_text()
        assert candidate == generate.transform(source)
        assert hashlib.sha256(source.encode()).hexdigest() == row["source_sha256"]
        for name in ("Matmul", "Softmax", "PublishRows"):
            assert _function(source,name) == _function(candidate,name)
        assert "request>=g.windowCapacity" in candidate
        assert "return expected>=0 && tag==expected;" in candidate
        assert "g.blocks*g.pageStride" in candidate
    with pytest.raises(ValueError, match="source drift"):
        generate.transform("source no longer contains the expected ABI")


@pytest.mark.parametrize("source", ["cpu_probe.cpp", "c16_cpu_probe.cpp"])
def test_cpu_probe_host_cpp_syntax_and_declared_kernel_arity(tmp_path, source):
    compiler = shutil.which("clang++")
    if compiler is None: pytest.skip("host C++ compiler unavailable; no CANN claim")
    # This stub permits only C++ parsing/type checking of our host harness.
    # No kernel is compiled or executed, and it provides no SDK validation.
    (tmp_path/"tikicpulib.h").write_text('''#include <cstddef>
enum class KernelMode { AIV_MODE, MIX_MODE };
namespace AscendC { void* GmAlloc(std::size_t); void GmFree(void*); void SetKernelMode(KernelMode); }
#define ICPU_RUN_KF(kernel,cores,...) kernel(__VA_ARGS__)
''')
    result = subprocess.run([compiler, "-std=c++17", "-fsyntax-only", "-I", str(tmp_path),
        str(ROOT/"csrc/experiments/request_windows_attention"/source)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_c16_cpu_fixture_has_full_cluster_and_explicit_reduced_oracle():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ordinary,shifted=(fixture.make_case(torch,case) for case in fixture.cluster_case_plan())
    assert ordinary["shape"][0]==shifted["shape"][0]==336
    assert shifted["shape"][15]==1 and shifted["oracle_scope"]=="source0_and_source1_only"
    assert shifted["case"].suppress_current is True
    _slot,epoch=map(int,shifted["inputs"]["request_map"][0])
    assert epoch>0 and int(shifted["inputs"]["new_table"][0,0])%8==4
    for name in ("expected_output","expected_lse"):
        assert bool(torch.isfinite(shifted["inputs"][name]).all())
        assert torch.equal(ordinary["inputs"][name],shifted["inputs"][name])


def test_reader_binding_header_kernel_and_torch_argument_contracts():
    header = (ROOT/"csrc/include/oscar_request_window_launch.h").read_text()
    binding = (ROOT/"csrc/request_window_attention_bindings.cpp").read_text()
    sources = ROOT/"csrc/experiments/request_windows_attention"
    specs = (("", "window_range.cpp", False), ("_balanced", "window_range_balanced.cpp", False),
             ("_decode", "decode_bundle.cpp", False), ("_decode_q1", "decode_bundle.cpp", False),
             ("_cluster4", "window_range_cluster4.cpp", True), ("_cluster16", "window_range_cluster16.cpp", True))
    for suffix, filename, cluster in specs:
        name = "attention_request_window"+suffix
        match = re.search(r"void "+name+r"_launch\((.*?)\);", header, re.S)
        assert match is not None
        args = [item.strip() for item in match.group(1).split(",")]
        assert args == ["void*"]*(18 if cluster else 17)+["int64_t"]*17+["float", "int64_t", "int64_t", "uint32_t"]
        source = (sources/filename).read_text()
        macro = source.split("#define OSCAR_CV_ARGUMENTS ", 1)[1].split("\n#define ", 1)[0]
        assert macro.count("GM_ADDR ") == (17 if cluster else 16)
        assert macro.count("int64_t ") == 19 and macro.count("float scale") == 1
        assert "GM_ADDR tags, GM_ADDR requestMap" in macro
        assert "int64_t splits, float scale, \\\n    int64_t windowCapacity, int64_t positionBits" in macro
        # Kernel/launch symbols can be supplied by exact preprocessor aliases.
        for before, after in re.findall(r"^#define (\w+) (\w+)$", source, re.M):
            source = re.sub(r"\b"+before+r"\b", after, source)
        kernel = "oscar_"+name+"_kernel"
        assert kernel+"(" in source and name+"_launch" in source
        schema = re.search(r'm.def\("'+name+r'_out\((.*?)\) ->', binding).group(1)
        assert schema.count("Tensor") == (17 if cluster else 16)
        assert "Tensor window_tags, Tensor request_map, Tensor tasks" in schema
        assert schema.endswith("int cube_cores, int window_capacity, int position_bits")
        assert f'm.impl("{name}_out"' in binding
    assert binding.count("requestMap.size(0)==table.size(0)") == 2
    assert binding.count("requestMap.size(0)<=windowCapacity") == 2
    assert binding.count("blocks<=(raw.numel()-ssmOffset)/pageStride/2") == 2
    assert binding.count("attention_decode_bundle_workspace_per_core(256)") == 2
    assert "ClusterSize==16 ? oscar_ascend::attention_cluster16_workspace_per_core(d)" in binding
