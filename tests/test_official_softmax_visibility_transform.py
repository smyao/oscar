# Archive #126/#129/#134/#145/#148/#164: visibility is proved against the
# original candidate-to-logical-position mask, including causal diagonals.
# This is a host proof, not CANN/NPU numerical or performance acceptance.
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.official_softmax_visibility_transform import HEADER, MARKER, NEW, OLD, transform
from tools import generate_official_attention

ROOT = Path(__file__).resolve().parents[1]


def source_before_transform():
    return generate_official_attention.cluster(16, optimize=False)


def test_only_complete_visibility_predicate_changes():
    source = source_before_transform()
    candidate = transform(source)
    assert candidate.removeprefix(HEADER).replace(NEW, OLD, 1) == source
    assert candidate.count("oscar_c16_softmax::FiniteRows256(") == source.count(
        "oscar_c16_softmax::FiniteRows256(")
    assert candidate.count("oscar_range_mask::MaskComplement(") == source.count(
        "oscar_range_mask::MaskComplement(")


def test_reject_wrong_family_moving_cut_and_duplicate_transform():
    source = source_before_transform()
    with pytest.raises(ValueError, match="already applied"):
        transform(transform(source))
    for old, new in (
        ("oscar_attention_cv_official_cluster16_kernel", "other_kernel"),
        ("const int64_t cut=Max64(Min64(g.sink,context),context+1-g.recent);",
         "const int64_t cut=Max64(g.sink,position+1-g.recent);"),
        ("const int64_t tailBegin=Max64(Min64(g.sink,context),context+1-g.recent);",
         "const int64_t tailBegin=Max64(g.sink,firstPosition+1-g.recent);"),
    ):
        with pytest.raises(ValueError, match="source contract"):
            transform(source.replace(old, new))
    with pytest.raises(ValueError, match="predicate source contract"):
        transform(source.replace("if(fullyVisibleHistory)", "if(false)"))


def test_compiled_predicate_against_independent_logical_position_enumeration(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler is unavailable")
    # Compile the exact emitted predicate. The oracle below independently
    # enumerates every candidate column and every active query/head row.
    predicates = NEW[:NEW.index("    if(fullyVisibleHistory ||")]
    program = r'''
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
using I = int64_t;
I Min64(I a,I b){return std::min(a,b);}
I Max64(I a,I b){return std::max(a,b);}
constexpr I kKvRows=256;
struct Geometry {I sink,recent;};
int proof(I kind,I start,I kvend,I context,I qbegin,I requestBegin,I rowBase,I group,Geometry g){
    const I jEnd=Min64(kKvRows,kvend-start),tileEnd=start+jEnd;
    const I sinkCount=Min64(g.sink,context);
    const I tailBegin=Max64(Min64(g.sink,context),context+1-g.recent);
    const I earliestCut=Max64(Min64(g.sink,context),context+1-g.recent);
    const I earliestPosition=context+qbegin+rowBase/group-requestBegin;
PREDICATES
    return fullyVisibleHistory?1:fullyVisibleWindow?2:fullyVisibleCurrent?4:0;
}
bool visible(I kind,I candidate,I position,I context,Geometry g){
    const I sink=std::min(g.sink,context);
    const I cut=std::max(sink,context+1-g.recent);
    if(kind==0)return candidate>=g.sink && candidate<context && candidate<cut && candidate<=position;
    if(kind==1){
        const I logical=candidate<sink?candidate:cut+candidate-sink;
        return candidate>=0 && candidate<sink+context-cut && logical<=position &&
            (logical<sink || (logical>=cut && logical<context));
    }
    return candidate>=context && candidate<=position;
}
int main(){
    I checked=0,newBlocks=0,maskedBlocks=0;
    for(I context: {I(0),I(1),I(63),I(64),I(65),I(255),I(256),I(257),I(512),I(16384),I(32768),I(49152)})
    for(I sink:{I(0),I(64),I(256)})for(I recent:{I(1),I(256),I(513)})
    for(I group:{I(1),I(6),I(16)})for(I queryOffset:{I(0),I(1),I(21),I(255),I(336)}){
        const I queryTile=128/group,requestBegin=37,qbegin=requestBegin+queryOffset;
        Geometry g{sink,recent};
        for(I qcount:{I(1),std::min(I(4),queryTile),queryTile})for(I kind=0;kind<3;++kind){
            I first=0,last=0;
            const I cut=std::max(std::min(sink,context),context+1-recent);
            if(kind==0){first=std::min(sink,context);last=context;}
            if(kind==1){first=0;last=std::min(sink,context)+context-cut;}
            if(kind==2){first=context;last=context+queryOffset+qcount;}
            for(I splits:{I(1),I(3),I(32)}){
                const I width=(last-first+splits-1)/splits;
                for(I split:{I(0),splits/2,splits-1}){
                    const I begin=std::min(last,first+split*width),end=std::min(last,begin+width);
                    std::vector<I> starts{begin,begin+256,end-256,end-1};
                    for(I start:starts){
                        if(start<begin || start>=end)continue;
                        for(I rowBase:{I(0),I(32),I(64),I(96)}){
                            I active=std::max(I(0),std::min(I(32),qcount*group-rowBase));
                            if(active==0)continue;
                            const int eligible=proof(kind,start,end,context,qbegin,requestBegin,rowBase,group,g);
                            bool complete=start+256<=end;
                            for(I row=0;row<active && complete;++row){
                                I position=context+queryOffset+(rowBase+row)/group;
                                for(I col=0;col<256;++col)
                                    if(!visible(kind,start+col,position,context,g)){complete=false;break;}
                            }
                            if(eligible && !complete){std::cerr<<"false full-tile proof";return 1;}
                            // Complete window/current tiles must enter the new
                            // branch; tails/diagonals must retain the old path.
                            if(kind!=0 && bool(eligible)!=complete){std::cerr<<"missed exact full tile";return 2;}
                            ++checked;newBlocks+=(eligible==2 || eligible==4);maskedBlocks+=!complete;
                        }
                    }
                }
            }
        }
    }
    if(checked<100000 || newBlocks<1000 || maskedBlocks<1000)return 3;
    std::cout<<checked<<" "<<newBlocks<<" "<<maskedBlocks<<"\n";
}
'''.replace("PREDICATES", predicates)
    source = tmp_path / "visibility.cpp"
    binary = tmp_path / "visibility"
    source.write_text(program)
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
    checked, new_blocks, masked_blocks = map(int, result.stdout.split())
    assert checked >= 100000 and new_blocks >= 1000 and masked_blocks >= 1000
