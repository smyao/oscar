"""Archive #145/#150/#151/#154/#164/#172: skip unused task scalar reads.

This derived source preserves the protected decode mother source, all public
symbols and every method except LoadTask. Zero-count tasks already return from
Process after publishing zero status, without inspecting their other fields.
No floating-point operation, DMA, cache invalidation or fence is removed.
"""
from hashlib import sha256


BASELINE_SHA256 = "817a5cbb7738e9ad572e39796838cd915a902c98301c452e21caefe6ceaebc22"
DERIVED_HEADER = """// Derived by tools/active_decode_transform.py from protected 232b decode.
// Archive #145/#150/#151/#154/#164/#172; startup D.4 full raw log reviewed.
// D.4(1): fused q4/q1 CV task-descriptor scalar reads, not attention math.
// D.4(2): old full-history dequant6499.8-6655.1ms/card, stores209-215ms,
// FIA7-18.7ms are historical failure evidence, not this kernel's timings.
// D.4(3): inspect qcount first; existing zero-count status-only tasks need no
// other scalar fields. Retain full AIV DMA/MTE2_S and AIC cache invalidation;
// active/negative tasks load all11 fields. No history restore or new buffer.
// D.4(4): at most10 fewer scalar reads per zero task per actor; UB/GM capacity,
// split/order/math/errors unchanged. Host control proof is not CANN/NPU proof.
"""

_ORIGINAL_LOAD_TASK = """  __aicore__ void LoadTask(int64_t id) {
    int64_t data[11];
    if ASCEND_IS_AIC {for(int32_t i=0;i<11;++i)data[i]=taskGm.GetValue(id*16+i);}
    else {
      auto row=taskBuf.Get<int64_t>();DataCopy(row,taskGm[id*16],16);
      Fence<HardEvent::MTE2_S>();
      for(int32_t i=0;i<11;++i)data[i]=row.GetValue(i);
    }
    qbegin=data[0];qcount=data[1];kvhead=data[2];kvbegin=data[3];kvend=data[4];
    request=data[5];split=data[6];kind=data[7];context=data[8];requestBegin=data[9];
    taskError=static_cast<int32_t>(data[10]);
  }
"""

_ACTIVE_LOAD_TASK = """  __aicore__ void LoadTask(int64_t id) {
    int64_t data[11];
    if ASCEND_IS_AIC {
      data[1]=taskGm.GetValue(id*16+1);
      if(data[1]==0) {qcount=0;return;}
      for(int32_t i=0;i<11;++i)if(i!=1)data[i]=taskGm.GetValue(id*16+i);
    } else {
      auto row=taskBuf.Get<int64_t>();DataCopy(row,taskGm[id*16],16);
      Fence<HardEvent::MTE2_S>();
      data[1]=row.GetValue(1);
      if(data[1]==0) {qcount=0;return;}
      for(int32_t i=0;i<11;++i)if(i!=1)data[i]=row.GetValue(i);
    }
    qbegin=data[0];qcount=data[1];kvhead=data[2];kvbegin=data[3];kvend=data[4];
    request=data[5];split=data[6];kind=data[7];context=data[8];requestBegin=data[9];
    taskError=static_cast<int32_t>(data[10]);
  }
"""


def _require_baseline(text: str) -> None:
    if not isinstance(text, str) or sha256(text.encode()).hexdigest() != BASELINE_SHA256:
        raise ValueError("active decode transform requires the exact protected mother source")
    if text.count(_ORIGINAL_LOAD_TASK) != 1:
        raise ValueError("active decode mother LoadTask anchor changed")


def transform(text: str) -> str:
    """Create the narrowly derived kernel; reject all unaudited source drift."""
    _require_baseline(text)
    return DERIVED_HEADER + text.replace(_ORIGINAL_LOAD_TASK, _ACTIVE_LOAD_TASK, 1)


def restore(text: str) -> str:
    """Prove exact inverse scope, including every untouched numerical method."""
    if (not isinstance(text, str) or not text.startswith(DERIVED_HEADER)
            or text.count(_ACTIVE_LOAD_TASK) != 1):
        raise ValueError("active decode derived header or LoadTask changed")
    baseline = text[len(DERIVED_HEADER):].replace(_ACTIVE_LOAD_TASK, _ORIGINAL_LOAD_TASK, 1)
    _require_baseline(baseline)
    if transform(baseline) != text:
        raise ValueError("active decode derived source is not canonical")
    return baseline
