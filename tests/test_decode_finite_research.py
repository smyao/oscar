"""Host source/protocol checks only; actual CPU/CAModel evidence is separate."""
from pathlib import Path
import pytest
from csrc.experiments.decode_finite_20261015.generate import generate,apply_finite,method,SOURCES,ROOT,SOFTMAX_SHA,sha
from csrc.experiments.decode_finite_20261015.finite_probe import artifacts
from csrc.experiments.decode_finite_20261015.run import adapted

@pytest.mark.parametrize('family',('q4','q1'))
def test_only_softmax_changes_with_fixed_denominator(family):
    baseline,b=generate(family,'baseline');candidate,c=generate(family,'candidate')
    assert b['all_other_methods_unchanged']==c['all_other_methods_unchanged']
    assert sha(baseline[slice(*method(baseline,'Softmax'))])==SOFTMAX_SHA
    assert c['proof']['fallback_loop_byte_identical']
    assert c['crosscore_flags_unchanged'] and c['additional_GM_UB_L1_bytes']==0
    with pytest.raises(ValueError,match='Softmax drift'):
        apply_finite(baseline.replace('if(activeRows==0)', 'if(activeRows<=0)'))

def test_full_tile_predicate_implies_original_visibility_for_every_active_row():
    accepted=0
    for context in (64,65,319,320,321,337,511,641,1344,1376):
      for sink in (0,1,64):
       for recent in (0,1,256):
        for queries in (1,4):
         for group in range(1,9):
          for rowbase in (0,16):
           active=max(0,min(16,queries*group-rowbase))
           if not active:continue
           earliest=context+rowbase//group;cut=max(sink,earliest+1-recent)
           for start in (0,1,63,64,65,256,320,576,1088):
            end=start+256
            predicate=start>=sink and end<=min(context,cut,earliest+1)
            if not predicate:continue
            accepted+=1
            for row in range(rowbase,rowbase+active):
             position=context+row//group
             lo=max(start,sink);hi=min(end,context,max(sink,position+1-recent),position+1)
             assert (lo,hi)==(start,end)
    assert accepted>1000

def test_exact_method_uses_real_decode_capacities_and_no_fd_scan():
    files=artifacts()
    for family in ('q4','q1'):
      for arm in ('baseline','candidate'):
        text=files[f'{family}_{arm}.cpp']
        assert 'kHalfRows=16,kLaneRows=16,kElements=16384' in text
        assert 'InitBuffer(dequantBuf,65536)' in text
        assert 'InitBuffer(naturalBuf,34816)' in text
    text=adapted()
    assert 'os.dup2' not in text and '/proc/self/fd' not in text
    assert "'rv':('rv_nz.bin' if mode=='q4' else 'rv.bin')" in text
    assert "profile_{kind}_log0.toml" in text
