# Archive #27/#34/#50/#108/#153: real dispatcher/FakeTensor schemas, not NPU graph execution.
import json
from pathlib import Path
import re
import pytest
import torch
from oscar_ascend.ops.meta import register_meta
from oscar_ascend.ops.contracts import source_capabilities, BF16_HISTORY_CAPABILITIES


@pytest.mark.parametrize("bf16_history", [False, True])
def test_cpp_schemas_accept_out_only_abstract_dispatch(bf16_history):
    root=Path(__file__).resolve().parents[1]
    namespace = "oscar_meta_contract_research" if bf16_history else "oscar_meta_contract"
    library=torch.library.Library(namespace,"DEF")
    names=[]
    for path in (root/"csrc").glob("*bindings.cpp"):
        text=path.read_text()
        for match in re.finditer(r'm\.def\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)',text):
            schema="".join(json.loads(literal) for literal in re.findall(r'"(?:[^"\\]|\\.)*"',match.group(1)))
            if not bf16_history and schema.split("(",1)[0] in BF16_HISTORY_CAPABILITIES:
                continue
            library.define(schema);names.append(schema.split("(",1)[0])
    assert set(names)==set(source_capabilities(bf16_history=bf16_history))
    register_meta(namespace)
    for name in names:
        assert torch._C._dispatch_has_kernel_for_dispatch_key(f"{namespace}::{name}", "Meta")
    q=torch.empty((4,6,256),device="meta",dtype=torch.bfloat16)
    rot=torch.empty((256,256),device="meta")
    out=torch.empty((4,6,256),device="meta")
    status=torch.empty((4,6),device="meta",dtype=torch.int32)
    assert getattr(torch.ops, namespace).rotate_out(q,rot,out,status,False) is None
    assert getattr(torch.ops, namespace).status_guard(status,status,status,status) is None
