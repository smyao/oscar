"""Archive #108/#153: preserve original input validation and exact op schemas.

These source checks protect deliberate derivation from the qualified bindings;
they are not compilation, numerical or device-execution evidence.
"""
from pathlib import Path
import re

from oscar_ascend.ops.cv_dispatch import BF16_HISTORY_OP_MAP

ROOT = Path(__file__).resolve().parents[1]


def _schema(source, name):
    block = re.search(r'  m.def\("' + name + r'\(.*?;\n', source, re.S)
    assert block is not None, name
    return ''.join(re.findall(r'"([^"]*)"', block.group()))


def test_research_binding_validation_is_identical_to_qualified_parent():
    old = (ROOT/'csrc/striped_attention_bindings.cpp').read_text()
    new = (ROOT/'csrc/bf16_history_bindings.cpp').read_text()
    def templates(text):
        return text[text.index('namespace {'):text.index('TORCH_LIBRARY_FRAGMENT')]
    assert templates(old) == templates(new)


def test_all_six_schemas_preserve_aliases_argument_order_and_cluster_stats():
    old = (ROOT/'csrc/striped_attention_bindings.cpp').read_text()
    new = (ROOT/'csrc/bf16_history_bindings.cpp').read_text()
    assert len(BF16_HISTORY_OP_MAP) == 6
    for parent, candidate in BF16_HISTORY_OP_MAP.items():
        assert _schema(old, parent).replace(parent, candidate, 1) == _schema(new, candidate)
    assert set(re.findall(r'm.impl\("([^"]+)"', new)) == set(BF16_HISTORY_OP_MAP.values())


def test_binding_uses_explicit_enlarged_workspace_for_every_research_op():
    new = (ROOT/'csrc/bf16_history_bindings.cpp').read_text()
    impls = re.findall(r'm.impl\("[^"]+",\s*&(.*?);', new)
    assert len(impls) == 6
    for impl in impls:
        assert 'oscar_ascend::attention_bf16_history_' in impl
        assert '_workspace_per_core(256)' in impl
