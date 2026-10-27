"""Archive #13-20/#126/#145/#148/#164: official partial-coordinate contract.

D.4 fixed-tile buffers and stream ownership remain; this is source/ABI
qualification, not device compilation, NPU completion, or model acceptance.
"""
from pathlib import Path
import hashlib
import re

import pytest

from tools import generate_official_attention as generated
from oscar_ascend.ops.contracts import SOURCE_CAPABILITIES
from oscar_ascend.ops.cv_dispatch import OFFICIAL_CV_MAP, OFFICIAL_CV_OPS, CLUSTER_CV_OPS

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    start, end = generated.function_bounds(source, name)
    return source[start:end]


@pytest.mark.parametrize("family", ["decode", "cluster4", "cluster16"])
def test_generated_sources_keep_numerical_hot_path_and_old_mothers(family):
    original_name = f"attention_cv_aligned_{family}.cpp"
    original = ROOT / "csrc/kernels" / original_name
    assert hashlib.sha256(original.read_bytes()).hexdigest() == generated.PINS[original_name]
    path = ROOT / f"csrc/kernels/attention_cv_official_{family}.cpp"
    source = path.read_text()
    assert source == generated.outputs()[path]
    # Only the independently verified transforms below may differ. Reverse
    # their exact anchors before checking ALL numerical/ownership code.
    numerical_source = source
    if family == "decode":
        numerical_source = numerical_source.replace(
            "oscar_c16_cast::CastRne(rounded,scores,activeRows*kKvRows);",
            "oscar_bf16::CastRne(rounded,scores,activeRows*kKvRows);")
    if family == "cluster16":
        from tools import official_finite_transform as finite
        from tools import official_softmax_visibility_transform as visibility
        assert numerical_source.count(finite.NEW) == 1
        assert numerical_source.count(visibility.NEW) == 1
        numerical_source = numerical_source.replace(finite.NEW, finite.OLD)
        numerical_source = numerical_source.replace(visibility.NEW, visibility.OLD)
    for name in ("Softmax", "LoadQueries", "PublishRows", "PublishStatus", "PublishEmpty"):
        assert function(numerical_source, name) == function(generated.fixed_cut(original.read_text()), name)
    assert "Matmul(qOffset,0,rotOffset" not in source
    assert "MatmulResidentB(qOffset,rotOffset" not in source
    assert "LoadResidentB(0,true)" not in source
    assert "PublishNormalizedQueries" not in source
    if family == "decode":
        assert "kNormReady" not in source and "kRotReady" not in source
        cube = function(source, "HistoryCubeTask")
        vector = function(source, "HistoryVectorTask")
        assert cube.count("CrossCoreWaitFlag(kTaskDone)") == 1
        assert vector.count("CrossCoreSetFlag<2,PIPE_MTE3>(kTaskDone)") == 1
        assert vector.index("PublishRows(id)") < vector.index("CrossCoreSetFlag<2,PIPE_MTE3>(kTaskDone)")
    else:
        cube = function(source, "CubeCluster")
        vector = function(source, "VectorCluster")
        assert "CrossCoreWaitFlag(kVectorReady);\n      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);" in cube
        assert "PublishRows(members[member].id);\n      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);\n      CrossCoreWaitFlag(kCubeReady);" in vector
        assert "#define OSCAR_STRIPED_D256_ONLY 1" in source
        assert f"#define attention_cv_fast_cluster4_launch attention_cv_official_{family}_launch" in source


def test_official_abi_names_aliases_and_registration_match_aligned():
    old = (ROOT / "csrc/aligned_attention_bindings.cpp").read_text()
    new = (ROOT / "csrc/official_attention_bindings.cpp").read_text()
    for name in OFFICIAL_CV_OPS:
        oldname = name.replace("_official_", "_aligned_")
        def declaration(source, opname):
            begin = source.index('m.def("' + opname + '(')
            end = source.index(' -> ()");', begin) + len(' -> ()");')
            return source[begin:end]
        assert declaration(new, name).replace(name, oldname) == declaration(old, oldname)
        assert name in SOURCE_CAPABILITIES
        assert name in (ROOT / "csrc/torch_bindings.cpp").read_text()
        assert name in (ROOT / "csrc/CMakeLists.txt").read_text()
        assert name in (ROOT / "oscar_ascend/ops/meta.py").read_text()
    assert len(OFFICIAL_CV_OPS) == 4
    assert set(OFFICIAL_CV_MAP.values()) == OFFICIAL_CV_OPS
    assert {name for name in OFFICIAL_CV_OPS if "cluster" in name} <= CLUSTER_CV_OPS
    assert "official_cluster16_control" not in new


def test_c1_completion_token_is_paired_after_output_publication():
    for family in ("decode", "cluster4", "cluster16"):
        source = (ROOT / f"csrc/kernels/attention_cv_official_{family}.cpp").read_text()
        cube = function(source, "CubeTask")
        vector = function(source, "VectorTask")
        assert "if(kind==0 && kvbegin<kvend) CrossCoreWaitFlag(kVectorReady);" in cube
        assert "PublishRows(id);\n    if(kind==0 && kvbegin<kvend)\n      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);" in vector


def test_all_generated_output_files_are_current():
    for path, expected in generated.outputs().items():
        assert path.read_text() == expected


def test_fixed_forward_precision_cut_and_full_history_candidates_are_consistent():
    source = (ROOT / "csrc/kernels/attention_official_tasks.cpp").read_text()
    assert "const int64_t cut=Max(Min(sink,context),context+1-recent);" in source
    assert "if(kind==0) {first=Min(sink,context);last=context;}" in source
    assert "if(kind==1) {first=0;last=Min(sink,context)+context-cut;}" in source
    assert "if(kind==2) {first=context;last=firstPos+count;}" in source
    # MTP logical-slot validation and every source's ownership/error fields
    # remain unchanged; only precision membership is shared across queries.
    for anchor in ("matches!=1 || position<0 || position>=lens.GetValue(r)",
                   "row.SetValue(10,metadataError)", "leader=slot.GetValue(0)<0"):
        assert anchor in source
    for family in ("decode", "cluster4", "cluster16"):
        reader = (ROOT / f"csrc/kernels/attention_cv_official_{family}.cpp").read_text()
        assert "Max64(Min64(g.sink,context),context+1-g.recent)" in reader
        assert "firstPosition+1-g.recent" not in reader
        assert "position+1-g.recent" not in reader
        assert "earliestPosition+1-g.recent" not in reader
        assert "Min64(tileEnd,position+1)" in reader  # current causal bound remains


def test_official_prepare_has_exact_original_schema_and_optional_mtp_contract():
    original = (ROOT / "csrc/attention_bindings.cpp").read_text()
    official = (ROOT / "csrc/official_attention_bindings.cpp").read_text()
    def schema(source, name):
        begin = source.index('m.def("' + name + '(')
        end = source.index(' -> ()");', begin) + len(' -> ()");')
        return source[begin:end]
    assert schema(official, "prepare_official_attention_tasks_out").replace(
        "prepare_official_attention_tasks_out", "prepare_attention_tasks_out") == schema(original, "prepare_attention_tasks_out")
    assert "prepare_official_attention_tasks_out" in SOURCE_CAPABILITIES
