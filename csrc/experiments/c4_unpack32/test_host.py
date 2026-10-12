"""Archive #13-22/#126/#145/#154/#155: helper algebra and per-lane ownership.

These are host/source contracts; actual vector/CANN precision is not certified.
"""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_unpack32 import fixture,generate,generate_c4


class HostContracts(unittest.TestCase):
    def test_exhaustive_fp16_guard_and_tail_masks(self):
        self.assertEqual(fixture.check_guard()['all_fp16_patterns'],65536)

    def test_old_lane_rows_preserved_for_every_tail_and_unaligned_start(self):
        for begin in (0,1,63,64,127,128,255,256,511):
            for width in range(257):
                for lane in (0,1):
                    old=[begin+s*32+lane*16+j for s in range(8) for j in range(16)
                         if begin+s*32+lane*16+j<begin+width]
                    new=[]
                    for s in range(4):
                        start=begin+s*64;first=start+lane*16
                        live=max(0,min(16,begin+width-first))+max(0,min(16,begin+width-first-32))
                        new.extend(start+(j//16)*32+lane*16+j%16 for j in range(live))
                    self.assertEqual(new,old)

    def test_source_contract_does_not_spill_acc_or_change_softmax_capacity(self):
        text,info=generate_c4.generate(True)
        self.assertEqual(info['AIV_explicit_UB_bytes'],157760)
        self.assertLess(info['AIV_explicit_UB_bytes'],184*1024)
        self.assertIn('dequantBuf.Get<float>(kElements)',text)
        self.assertIn('scratchBuf.Get<float>(128)',text)
        self.assertIn('kHalfRows=32, kHalfKv=16',text)
        self.assertIn('DataCopyParams copy{2,blocks,0,blocks};',text)
        self.assertEqual(text,(generate_c4.HERE/'c4_candidate.cpp').read_text())
        self.assertEqual(generate.helper(),(generate.HERE/'unpack32.h').read_text())

if __name__=='__main__':unittest.main()
