"""#147/#171 local pipe dependency proof, not a CANN scheduling simulation."""
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parents[1]


def body(source, name):
    start = source.index(f"void {name}(")
    end = source.index("  __aicore__", start + 8)
    return re.sub(r"//[^\n]*", "", source[start:end])


def publication_precedes_overwrite(source):
    publish = body(source, "PipelinePublishKv").split("PublishBf16Operand", 1)[1]
    consume = body(source, "PipelineConsumePv").split("DataCopy(", 1)[0]
    # A Fence records Set on its source pipe and Wait on its destination.
    # CPU issuing a Wait on V does not serialize another MTE2 DMA.
    last = {"MTE3": "outgoing_v", "MTE2": "current_pv_ready", "V": "vector"}
    edges = {}
    for i, (src, dst) in enumerate(re.findall(r"Fence<HardEvent::(MTE[23]|V)_(MTE[23]|V)>", publish + consume)):
        node = f"wait_{i}"
        edges[node] = {last[src], last[dst]}
        last[dst] = node
    def depends(node, wanted):
        return node == wanted or any(depends(p, wanted) for p in edges.get(node, ()))
    return depends(last["MTE2"], "outgoing_v")


def test_next_v_dma_retires_before_shared_ub_is_overwritten_by_current_pv():
    source = (ROOT / "csrc/kernels/attention_bf16_history_mdl.cpp").read_text()
    assert publication_precedes_overwrite(source)
    # Mutation of this actual source reproduces the old missing edge.
    old = source.replace("    Fence<HardEvent::MTE3_MTE2>();\n    auto result=",
                         "    auto result=", 1)
    assert old != source
    assert not publication_precedes_overwrite(old)


def test_finite_fp32_bytes_can_poison_bf16_padding_but_not_fp32_padding():
    # Possible byte-overwrite payload, not captured target data.
    fp32 = torch.tensor([0x3F807FC0], dtype=torch.int32).view(torch.float32)
    bf16 = fp32.view(torch.bfloat16)
    assert torch.isfinite(fp32).all()
    assert torch.isnan(bf16[0]) and torch.isfinite(bf16[1])
    assert torch.isnan((bf16.float() * 0)[0])
    assert (fp32 * 0).item() == 0
