"""Archive #87/#103/#108/#125/#128/#153: packaging/ABI contracts only.

These tests read project sources and parse schemas. They do not load kernels,
run CANN, or establish NPU/graph/service/numerical acceptance.
"""
import ast
import hashlib
import json
from pathlib import Path
import re

import pytest
import torch

from oscar_ascend.ops.contracts import PRODUCTION_CAPABILITIES, SOURCE_CAPABILITIES
from tools import package_request_window_kernels as package

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={"rotate_clip_store_request_window_out","attention_request_window_decode_out",
          "attention_request_window_decode_q1_out","attention_request_window_out",
          "attention_request_window_balanced_out","attention_request_window_cluster4_out",
          "attention_request_window_cluster16_out"}
OLD_REQUIRED={"store_int2_out","merge_lse_out","rotate_out","rotate_clip_store_out",
              "prepare_attention_tasks_out","attention_cv_out","status_guard"}
PIPELINE_REQUEST_OP="attention_request_window_member_pipeline_cluster4_out"


def _runtime_caps():
    tree=ast.parse((ROOT/"oscar_ascend/runtime.py").read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="REQUEST_WINDOW_CAPABILITIES" for t in n.targets))
    return set(ast.literal_eval(node.value.args[0]))


def _schemas():
    result={}
    for path in ("csrc/request_window_attention_bindings.cpp","csrc/request_window_store_bindings.cpp"):
        text=(ROOT/path).read_text()
        for call in re.findall(r'm\.def\((.*?)\);',text,re.S):
            strings=re.findall(r'"(?:[^"\\]|\\.)*"',call)
            schema=''.join(ast.literal_eval(s) for s in strings)
            if "(" not in schema:continue
            parsed=torch._C.parse_schema(schema)
            assert parsed.name not in result
            result[parsed.name]=parsed
    return result


def _types(params):
    types=[]
    for parameter in params.split(','):
        match=re.match(r'\s*(void\s*\*|(?:int64_t|int32_t|uint32_t|float|bool)\b)',parameter)
        assert match,parameter
        types.append(re.sub(r'\s+','',match[1]))
    return types


def test_all_capability_surfaces_match_and_old_modes_do_not_require_new_ops():
    assert _runtime_caps()==EXPECTED
    assert EXPECTED<=SOURCE_CAPABILITIES
    assert {"attention_cv_member_pipeline_cluster4_out",PIPELINE_REQUEST_OP}<=SOURCE_CAPABILITIES
    from oscar_ascend.ops.contracts import BF16_HISTORY_CAPABILITIES
    # Archive #154/#167: every production BF16 reader consumes the packed
    # D+16 query ABI, so its producer is mandatory even when request-window
    # research operators are not enabled.
    assert PRODUCTION_CAPABILITIES==((OLD_REQUIRED-{"attention_cv_out"})|
        BF16_HISTORY_CAPABILITIES|{"rotate_bf16_query_out"})
    assert not EXPECTED.intersection(PRODUCTION_CAPABILITIES)
    text=(ROOT/"csrc/torch_bindings.cpp").read_text()
    body=re.search(r'std::vector<std::string>\{(.*?)\};\}\);',text,re.S)[1]
    cpp=re.findall(r'"([a-z0-9_]+)"',body)
    assert len(cpp)==len(set(cpp)) and set(cpp)==SOURCE_CAPABILITIES
    text=(ROOT/"csrc/CMakeLists.txt").read_text()
    body=re.search(r'\\"source_capabilities\\": \[(.*?)\]',text,re.S)[1]
    cmake=re.findall(r'\\"([a-z0-9_]+)\\"',body)
    extra=re.search(r'set\(OSCAR_BF16_CAPABILITIES_JSON ", (.*?)\)',text,re.S)[1]
    cmake+=re.findall(r'\\"([a-z0-9_]+)\\"',extra)
    assert len(cmake)==len(set(cmake)) and set(cmake)==SOURCE_CAPABILITIES
    text=(ROOT/"oscar_ascend/ops/meta.py").read_text()
    body=re.search(r'for optional in \((.*?)\):',text,re.S)[1]
    optional=re.findall(r'"([a-z0-9_]+)"',body)
    assert len(optional)==len(set(optional)) and set(optional)==SOURCE_CAPABILITIES-PRODUCTION_CAPABILITIES


def test_source_packaging_identity_and_local_headers_are_complete():
    from csrc.experiments.request_windows_store import generate as store_generate
    store_generate.generate(check=True)
    actual=package.package(check=True)
    # The earlier six-reader report remains historical evidence. Compare the
    # current source inventory with the regenerated 2026-10-16 source-only receipt.
    # The previously referenced 2026-10-11 file was absent from HEAD; no NPU evidence is recreated.
    recorded=json.loads((ROOT/"reports/request_window_member_pipeline_packaging_20261016.json").read_text())
    assert actual==recorded
    assert len(actual["mapping"])==8
    assert actual["reader_kernel_count"]==7 and actual["store_kernel_count"]==2
    assert len(actual["shared_helpers"]) == 2
    for helper in actual["shared_helpers"]:
        assert (ROOT/helper["packaged"]).read_text() == package.relocated_helper(
            (ROOT/helper["experiment"]).read_text())
        assert helper["reversible_source_identity"] is True
    for row in actual["mapping"]:
        source=(ROOT/row["experiment"]).read_text();target=(ROOT/row["packaged"]).read_text()
        assert package.without_includes(source)==package.without_includes(target)
        assert row["non_include_body_sha256"]==hashlib.sha256(package.without_includes(target).encode()).hexdigest()
        assert "../../kernels/" not in target and "../../include/" not in target
    assert actual["NPU"]=="not_run" and actual["performance"]=="not_run"


def test_packaging_rejects_unrecorded_math_change(tmp_path,monkeypatch):
    (tmp_path/"source.cpp").write_text("float square(float x) { return x*x; }\n")
    (tmp_path/"target.cpp").write_text("float square(float x) { return x+x; }\n")
    (tmp_path/"source.h").write_text("void square();\n");(tmp_path/"target.h").write_text("void square();\n")
    monkeypatch.setattr(package,"SOURCES",(("source.cpp","target.cpp"),))
    monkeypatch.setattr(package,"SHARED_HELPERS",())
    monkeypatch.setattr(package,"HEADER_SOURCE","source.h");monkeypatch.setattr(package,"HEADER_DEST","target.h")
    with pytest.raises(ValueError,match="differs from include-only mapping"):
        package.package(tmp_path,check=True)


def test_build_uses_formal_sources_and_both_new_binding_units():
    cmake=(ROOT/"csrc/CMakeLists.txt").read_text()
    kernels=re.search(r'ascendc_library\(oscar_ascend_kernels SHARED(.*?)\)',cmake,re.S)[1]
    bindings=re.search(r'pybind11_add_module\(_oscar_ascend_ops(.*?)\)',cmake,re.S)[1]
    for _,target in package.SOURCES:
        assert kernels.split().count(target.removeprefix("csrc/"))==1
    assert "experiments/" not in kernels
    assert bindings.split().count("request_window_attention_bindings.cpp")==1
    assert bindings.split().count("request_window_store_bindings.cpp")==1
    store=(ROOT/"csrc/request_window_store_bindings.cpp").read_text()
    assert '#include "include/oscar_request_window_store_launch.h"' in store
    assert "experiments/request_windows_store/request_window_launch.h" not in store


def test_schema_names_void_returns_mutable_outputs_and_identity_tail():
    schemas=_schemas();assert set(schemas)==EXPECTED|{PIPELINE_REQUEST_OP}
    for name,schema in schemas.items():
        assert len(schema.returns)==0
        assert [a.name for a in schema.arguments[-2:]]==["window_capacity","position_bits"]
        written={a.name for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write}
        if name=="rotate_clip_store_request_window_out":
            assert len(schema.arguments)==25
            assert [a.name for a in schema.arguments[6:8]]==["qstarts","request_map"]
            assert written=={"packed","window_key","window_value","window_tags","packed_status","window_status"}
        else:
            assert schema.arguments[10].name=="request_map" and schema.arguments[11].name=="tasks"
            expected={"partial","lse","status","workspace"}
            if "cluster" in name:expected.add("cluster_stats")
            assert written==expected
            assert len(schema.arguments)==(29 if "cluster" in name else 28)


def test_public_launch_declarations_match_packaged_definition_types():
    header=(ROOT/"csrc/include/oscar_request_window_launch.h").read_text()
    declarations={name:_types(params) for name,params in re.findall(r'void (attention_request_window\w*_launch)\((.*?)\);',header,re.S)}
    assert len(declarations)==6
    for _,target in package.SOURCES[:5]:
        text=(ROOT/target).read_text()
        aliases=dict(re.findall(r'^#define (attention_cv_fast\w*_launch) (attention_request_window\w*_launch)$',text,re.M))
        if target.endswith("_decode.cpp"):
            params=re.search(r'void name\((.*?)\) \{ \\',text,re.S)[1].replace('\\\n','')
            names=re.findall(r'OSCAR_STRIPED_LAUNCH\((attention_request_window\w*_launch),',text)
            assert len(names)==2
            for name in names:assert declarations[name]==_types(params)
        else:
            name,params=re.search(r'void (attention_cv_fast\w*_launch)\((.*?)\) \{',text,re.S).groups()
            assert declarations[aliases[name]]==_types(params)
    store_header=(ROOT/package.HEADER_DEST).read_text()
    declarations={name:_types(params) for name,params in re.findall(r'void (\w+_launch)\((.*?)\);',store_header,re.S)}
    for _,target in package.SOURCES[5:7]:
        text=(ROOT/target).read_text()
        name,params=re.search(r'void (\w+_launch)\((.*?)\) \{',text,re.S).groups()
        assert declarations[name]==_types(params)


def test_member_pipeline_reader_has_its_own_matching_launch_declaration():
    # #108/#153: the independent reader must not accidentally reuse the old
    # C4 declaration or omit its request identity arguments / cluster stats.
    header=(ROOT/"csrc/include/oscar_member_pipeline_launch.h").read_text()
    declarations={name:_types(params) for name,params in
                  re.findall(r'void (\w+_launch)\((.*?)\);',header,re.S)}
    expected="attention_request_window_member_pipeline_cluster4_launch"
    assert expected in declarations
    matches=[target for _,target in package.SOURCES if "member_pipeline" in target]
    assert len(matches)==1
    text=(ROOT/matches[0]).read_text()
    aliases=dict(re.findall(r'^#define (\w+_launch) (\w+_launch)$',text,re.M))
    name,params=re.search(r'void (\w+_launch)\((.*?)\) \{',text,re.S).groups()
    assert aliases.get(name,name)==expected
    assert declarations[expected]==_types(params)


def test_default_request_window_flag_is_still_false():
    target=json.loads((ROOT/"configs/target.json").read_text())
    assert target.get("experimental_request_windows",False) is False
