"""Keep KV256 online boundaries but omit padded zero columns in Cube work.

Archive #126/#129/#140/#145/#148/#151/#154/#155, startup D.4. The original
range mask/finiteness checks still see a fully initialized 256-column score
row. Real K/V, visible domains, online iteration order and final RvT remain.
Only the final tile's padded zero QK/PV work and empty producer subtiles go.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_range_mask.generate import main as generate_mask
OUT=Path(__file__).resolve().parent


def main():
    generate_mask()
    text=(ROOT/'csrc/experiments/decode_range_mask/attention_range_mask.cpp').read_text()
    def edit(old,new,count=1):
        nonlocal text
        assert text.count(old)==count,(old[:70],text.count(old),count)
        text=text.replace(old,new)
    edit('    mm.SetOrgShape(m,n,k);mm.SetSingleShape(m,n,k);',
         '    // Physical Q/P/K/V strides remain the original bounded layout.\n'
         '    mm.SetOrgShape(m,rotation?D:(transposeB?kKvRows:D),\n'
         '        rotation?D:(transposeB?D:kKvRows));\n'
         '    mm.SetSingleShape(m,n,k);')
    edit('  __aicore__ void CubeTask() {',
         '''  __aicore__ int32_t TailColumns(int64_t start) {
    return static_cast<int32_t>((Min64(kKvRows,kvend-start)+15)/16*16);
  }
  __aicore__ void ClearPaddedScores() {
    // QK writes only TailColumns columns. All omitted scores are the same
    // +0 produced by finite Q multiplied by the old padded zero K rows.
    // Invalid Q is still checked by unchanged LoadQueries and status logic.
    auto zeros=scoreBuf.Get<float>();
    Fence<HardEvent::MTE3_V>();
    Duplicate(zeros,0.0F,kHalfRows*kKvRows);Fence<HardEvent::V_MTE3>();
    for(int32_t block=0;block<kRowBlocks;++block)
      DataCopy(work[scoreOffset+(lane*kLaneRows+block*kHalfRows)*kKvRows],
          zeros,kHalfRows*kKvRows);
    Fence<HardEvent::MTE3_V>();
  }
  __aicore__ void CubeTask() {''')
    edit('Matmul(qOffset,kOffset,scoreOffset,rows,kKvRows,D);',
         'Matmul(qOffset,kOffset,scoreOffset,rows,TailColumns(start),D);')
    edit('Matmul(pOffset,vOffset,pvOffset,rows,D,kKvRows,false,false);',
         'Matmul(pOffset,vOffset,pvOffset,rows,D,TailColumns(start),false,false);')
    edit('        const int64_t subStart=start+subtile*(2*kHalfKv);',
         '        const int64_t subStart=start+subtile*(2*kHalfKv);\n'
         '        // A live 64-row producer chunk initializes its entire tail.\n'
         '        // TailColumns rounds only to16, so skipped chunks are unread.\n'
         '        if(subStart+lane*kHalfKv>=kvend)continue;')
    edit('''      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      for(int32_t block=0;block<kRowBlocks;++block)''',
         '''      if(TailColumns(start)<kKvRows)ClearPaddedScores();
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      for(int32_t block=0;block<kRowBlocks;++block)''')
    for old,new in (('oscar_range_mask_q4_kernel','oscar_tail_range_q4_kernel'),
                    ('oscar_range_mask_q1_kernel','oscar_tail_range_q1_kernel'),
                    ('range_mask_q4_launch','tail_range_q4_launch'),
                    ('range_mask_q1_launch','tail_range_q1_launch')):
        edit(old,new,2 if old.startswith('oscar_') else 1)
    header='''// ISOLATED TAIL+RANGE DECODE EXPERIMENT, not serving.
// Archive #126/#129/#140/#145/#148/#151/#154/#155; full startup D.4 reviewed.
// 1. Complete q4/q1 history/window/current CV, targeting short precise/tail tiles.
// 2. No full history restoration or BF16 history substitution.
// 3. Same KV256 traversal and FP32 math, but Cube consumes round_up(live,16)
//    columns, skips only zero-padded producer chunks, and initializes omitted
//    score columns to +0 before the unchanged pre-mask finite check and V2.
//    Every skipped padded row had zero K/V; no visible row or query is dropped.
//    No extra allocation. Padded-zero accumulation parity must be verified.
// 4. Expected benefit includes tiny current/window tiles; actual full A/B and
//    frozen oracle determine validity, never an assumed whole-service ratio.
'''
    (OUT/'attention_tail_range.cpp').write_text(header+text)
    cmake=(ROOT/'csrc/experiments/decode_range_mask/CMakeLists.txt').read_text()
    cmake=cmake.replace('oscar_range_mask_full','oscar_tail_range').replace('OSCAR_RANGE_MASK_FULL','OSCAR_TAIL_RANGE')
    cmake=cmake.replace('attention_range_mask.cpp','attention_tail_range.cpp').replace('range_mask_full','tail_range')
    cmake=cmake.replace('oscar_range_mask_q4_kernel','oscar_tail_range_q4_kernel').replace('oscar_range_mask_q1_kernel','oscar_tail_range_q1_kernel')
    (OUT/'CMakeLists.txt').write_text(cmake)
    print('TAIL_RANGE_GENERATED unchanged KV256 boundaries; zero-column work omitted')


if __name__=='__main__':main()
