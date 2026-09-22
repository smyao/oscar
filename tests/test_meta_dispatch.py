# Archive #27/#34/#50/#108: real torch dispatcher/FakeTensor schemas, not claimed NPU graph execution.
import json
from pathlib import Path
import re
import torch
from oscar_ascend.ops.meta import register_meta
from oscar_ascend.ops.contracts import PRODUCTION_CAPABILITIES


def test_cpp_schemas_accept_out_only_abstract_dispatch():
    root=Path(__file__).resolve().parents[1]
    library=torch.library.Library("oscar_meta_contract","DEF")
    names=[]
    for path in (root/"csrc").glob("*bindings.cpp"):
        text=path.read_text()
        for match in re.finditer(r'm\.def\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)',text):
            schema="".join(json.loads(literal) for literal in re.findall(r'"(?:[^"\\]|\\.)*"',match.group(1)))
            library.define(schema);names.append(schema.split("(",1)[0])
    assert set(names)==set(PRODUCTION_CAPABILITIES)
    register_meta("oscar_meta_contract")
    q=torch.empty((4,6,256),device="meta",dtype=torch.bfloat16)
    rot=torch.empty((256,256),device="meta")
    out=torch.empty((4,6,256),device="meta")
    status=torch.empty((4,6),device="meta",dtype=torch.int32)
    assert torch.ops.oscar_meta_contract.rotate_out(q,rot,out,status,False) is None
    assert torch.ops.oscar_meta_contract.status_guard(status,status,status,status) is None
