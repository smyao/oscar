"""Archive #126/#134/#145/#148/#164: certify unchanged row finite semantics.

D.4: bounded fused FIA, no full-history restoration; reuse dead KV256 UB.
Only official C16 is changed. Aligned mothers, old helpers and timers stay fixed.
"""

MARKER = '// OFFICIAL_FINITE_CERTIFICATE_V1'
OLD = 'if(!oscar_c16_softmax::FiniteRows256(finiteSums,scores,activeRows))error=2;'
NEW = '''// OFFICIAL_FINITE_CERTIFICATE_V1
      // KV staging is complete; this UB is dead until the following V2 call.
      static_assert(kHistoryElements>=kHalfRows*kKvRows,
          "finite certificate needs one complete score block of dead KV UB");
      auto finiteWork=dequantBuf.Get<float>(kHalfRows*kKvRows);
      if(!oscar_official_finite::FiniteRows256(
          finiteSums,scores,finiteWork,activeRows))error=2;'''


def transform(text: str) -> str:
    if MARKER in text:
        raise ValueError('official finite transform already applied')
    if text.count(OLD) != 1 or text.count('#include "attention_c16_softmax_helpers.h"') != 1:
        raise ValueError('official finite source anchor drift')
    return text.replace('#include "attention_c16_softmax_helpers.h"',
        '#include "attention_c16_softmax_helpers.h"\n#include "attention_official_finite.h"').replace(OLD, NEW)
