"""One CPU-only original-baseline error-order diagnostic; never timed.

No new math/barriers or production edits. Print scalar values at existing
V_S sites only, qbegin0/lane0/history. Preserve the original failed fixture.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_finite_20261015.generate import generate,method,sha

def make():
    original,_=generate('q4','baseline');text=original;changes={}
    def change(name,old,new):
        nonlocal text
        a,b=method(text,name);body=text[a:b]
        if body.count(old)!=1:raise ValueError('diagnostic anchor '+name)
        replacement=body.replace(old,new)
        if replacement.replace(new,old)!=body:raise ValueError('diagnostic inverse')
        changes[name]={'original':sha(body),'diagnostic':sha(replacement)}
        text=text[:a]+replacement+text[b:]
    change('LoadQueries','if(!Finite(check.GetValue(0)))error=2;',
      '''const float diagnosticSum=check.GetValue(0);
      if(!Finite(diagnosticSum))error=2;
      if(qbegin==0 && lane==0 && kind==0 && row==0)
        std::printf("ERROR_ORDER LoadQueries sum=%g error=%d\\n",diagnosticSum,error);''')
    change('Softmax','if(!Finite(tmp.GetValue(0)))error=2;',
      '''const float diagnosticSum=tmp.GetValue(0);
      if(!Finite(diagnosticSum))error=2;
      if(qbegin==0 && lane==0 && kind==0 && r==0)
        std::printf("ERROR_ORDER Softmax start=%lld sum=%g error=%d\\n",static_cast<long long>(start),diagnosticSum,error);''')
    change('CommitPrefetchErrors','for(int32_t sub=0;sub<kKvSubtiles;++sub) {',
      '''if(qbegin==0 && lane==0 && kind==0)
      std::printf("ERROR_ORDER Commit before=%d K=%d,%d V=%d,%d\\n",error,pendingKeyError[0],pendingKeyError[1],pendingValueError[0],pendingValueError[1]);
    for(int32_t sub=0;sub<kKvSubtiles;++sub) {''')
    change('CommitPrefetchErrors','if(pendingValueError[sub])error=pendingValueError[sub];',
      '''if(pendingValueError[sub])error=pendingValueError[sub];
      if(qbegin==0 && lane==0 && kind==0)
        std::printf("ERROR_ORDER Commit sub=%d after=%d\\n",sub,error);''')
    change('HistoryVectorTask','const float sum=stats.GetValue(kLaneRows+r);',
      '''const float sum=stats.GetValue(kLaneRows+r);
      if(qbegin==0 && lane==0 && r==0)
        std::printf("ERROR_ORDER Normalize sum=%g entering_error=%d\\n",sum,error);''')
    text='#include <cstdio>\n'+text.replace('oscar_decode_finite_q4_baseline','oscar_decode_finite_q4_diagnostic').replace('decode_finite_q4_baseline_launch','decode_finite_q4_diagnostic_launch')
    probe=(HERE/'probe.cpp').read_text().replace('oscar_decode_finite_q4_baseline_kernel','oscar_decode_finite_q4_diagnostic_kernel')
    (HERE/'q4_diagnostic.cpp').write_text(text);(HERE/'diagnostic_probe.cpp').write_text(probe)
    (HERE/'diagnostic.json').write_text(json.dumps({'original_baseline_sha256':sha(original),'diagnostic_sha256':sha(text),'checker_sha256':sha(probe),'changes':changes,'scope':'CPU-only scalar error-order diagnosis, never performance','new_math_or_barriers':False},indent=2)+'\n')
if __name__=='__main__':make()
