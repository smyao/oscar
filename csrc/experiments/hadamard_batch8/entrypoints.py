"""Generate bounded writer drivers from the previously qualified full harness."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.hadamard_batch8.generate import HERE
OLD=HERE.parent/'striped_store_simd_pack'


def sources():
    cpu=(OLD/'store_cpu.cpp').read_text()
    cpu=cpu.replace('oscar_rotate_clip_store_striped_simd_pack_init_kernel','oscar_rotate_clip_store_striped_batch8_kernel')
    cpu=cpu.replace('oscar_rotate_clip_store_striped_simd_pack_kernel','oscar_rotate_clip_store_striped_simd_pack_init_kernel')
    cpu=cpu.replace('if(argc!=3)throw std::runtime_error("usage: store_cpu CASE_DIRECTORY NEW_OUTPUT");',
        'if(argc!=3&&argc!=4)throw std::runtime_error("usage: store_cpu CASE_DIRECTORY NEW_OUTPUT [0|1|2|all]; default1,2");')
    cpu=cpu.replace('std::vector<std::vector<uint8_t>> baseline;', '''std::vector<std::vector<uint8_t>> baseline;
    int firstVariant=1,lastVariant=3;
    if(argc==4) {
      if(std::string(argv[3])=="all")firstVariant=0;
      else {firstVariant=std::stoi(argv[3]);lastVariant=firstVariant+1;}
      if(firstVariant<0||firstVariant>2)throw std::runtime_error("invalid arm");
    }''')
    cpu=cpu.replace('for(int variant=0;variant<3;++variant)', 'for(int variant=firstVariant;variant<lastVariant;++variant)')
    cpu=cpu.replace('if(variant==0)baseline.emplace_back', 'if(variant==firstVariant)baseline.emplace_back')
    replay=(OLD/'replay.py').read_text()
    replay=replay.replace('from csrc.experiments.striped_store_simd_pack import generate',
        'from csrc.experiments.hadamard_batch8 import generate')
    begin=replay.index('ARMS=');end=replay.index('\n\n\ndef run',begin)
    replay=replay[:begin]+'''ARMS={'old':('oscar_rotate_clip_store_striped_kernel','csrc/experiments/striped_store_simd_pack/writer_baseline.cpp'),
      'current':('oscar_rotate_clip_store_striped_simd_pack_init_kernel','csrc/experiments/striped_store_simd_pack/writer_pack_init.cpp'),
      'batch8':('oscar_rotate_clip_store_striped_batch8_kernel','csrc/experiments/hadamard_batch8/writer_batch.cpp')}'''+replay[end:]
    return {'store_cpu.cpp':cpu,'replay_writer.py':replay}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    for name,text in sources().items():
        if a.check:
            if (HERE/name).read_text()!=text:raise ValueError('entrypoint source drift '+name)
        else:(HERE/name).write_text(text)
    print('writer harness preserves oracle/bytes/guards/SDK completion; no execution')
