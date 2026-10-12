"""Archive #144/#156, review R2: checked prepare has an explicit delta only.

This is source drift detection, not a substitute for NPU behavior/graph tests.
The frozen fe0 source is never edited to make the checked variant match.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


def compact(source):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", source))


GUARD = compact("""
      const int64_t logicalPosition=context+token-begin;
      if(!padding && !invalid && tableColumns>0 && !slotContext) {
        const int64_t column=logicalPosition/128;
        if(logicalPosition<0 || column>=tableColumns) {
          invalid=true;metadataError=1;
        } else {
          if(cachedRequest!=r || cachedColumn!=column) {
            auto tableLocal=tableBuf.Get<int32_t>();
            DataCopyExtParams tableCopy{1,4,0,0,0};
            Fence<HardEvent::S_MTE2>();
            DataCopyPad(tableLocal,tableGm[r*tableColumns+column],tableCopy,pad);
            Fence<HardEvent::MTE2_S>();
            cachedBlock=tableLocal.GetValue(0);
            cachedRequest=r;cachedColumn=column;
          }
          if(cachedBlock<0 || cachedBlock*128+logicalPosition%128!=physicalSlot) {
            invalid=true;metadataError=1;
          }
        }
      }
""")


def replace_once(source, old, new):
    assert source.count(old) == 1, f"checked prepare delta changed: {old[:100]}"
    return source.replace(old, new)


def verify_equivalence(original, checked):
    original, checked = compact(original), compact(checked)
    for new, old in (("CheckedAttentionTasks", "AttentionTasks"),
                     ("oscar_prepare_attention_tasks_checked_kernel", "oscar_prepare_attention_tasks_kernel"),
                     ("prepare_attention_tasks_checked_launch", "prepare_attention_tasks_launch")):
        checked = checked.replace(new, old)
    checked = replace_once(checked, "if(tableColumns>0)tableGm.SetGlobalBuffer",
                           "if(slotContext)tableGm.SetGlobalBuffer")
    checked = replace_once(checked, "int64_tcachedRequest=-1,cachedColumn=-1,cachedBlock=-1;", "")
    # The entire guard is frozen as the allowed addition, not erased with a
    # broad regex that would also hide a changed comparison or DMA ordering.
    checked = replace_once(checked, GUARD, "")
    checked = replace_once(checked, "row.SetValue(0,padding||invalid?-1:logicalPosition);",
                           "row.SetValue(0,padding||invalid?-1:context+token-begin);")
    for dma in ("DataCopyPad(slot,slotsGm[token],slotCopy,slotPad);",
                "DataCopyPad(slot,slotsGm[token-1],slotCopy,slotPad);",
                "DataCopyPad(tableLocal,tableGm[r*tableColumns+column],tableCopy,pad);",
                "DataCopyPad(groupSlots,slotsGm[token],groupCopy,slotPad);"):
        checked = replace_once(checked, "Fence<HardEvent::S_MTE2>();" + dma, dma)
    assert checked == original, "task semantics/ABI/geometry drifted outside the reviewed guard delta"


def sources():
    return tuple((ROOT / "csrc/kernels" / name).read_text() for name in
                 ("attention_tasks.cpp", "attention_tasks_checked.cpp"))


def test_checked_preparer_matches_frozen_algorithm_outside_reviewed_delta():
    verify_equivalence(*sources())


@pytest.mark.parametrize("old,new", [
    ("row.SetValue(8,context)", "row.SetValue(8,context+1)"),
    ("last=firstPos+count", "last=firstPos+count+1"),
    ("(last-first+splits-1)/splits", "(last-first)/splits"),
    ("pipe.InitBuffer(taskBuf, 128)", "pipe.InitBuffer(taskBuf, 256)"),
    ("cachedBlock*128+logicalPosition%128!=physicalSlot", "cachedBlock*128!=physicalSlot"),
    ("Fence<HardEvent::S_MTE2>();", ""),
    ("bool slotContext,uint32_t cores", "bool slotContext,int32_t cores"),
])
def test_unreviewed_preparer_mutations_are_detected(old, new):
    original, checked = sources()
    assert old in checked
    with pytest.raises(AssertionError):
        verify_equivalence(original, checked.replace(old, new, 1))
