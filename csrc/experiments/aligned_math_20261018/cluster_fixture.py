"""One bounded H32 fixture: C16 mature16 members plus an independent C1 tail."""
from pathlib import Path
import argparse
import torch
from tools import probe_history_reuse as reuse
from tools.aligned_fixture import make_fixture
from .fixture import dump,golden

def make(folder,fault=False):
    torch.set_num_threads(1);folder.mkdir(parents=True,exist_ok=False)
    spec=reuse.Shape('aligned_h32_mature_tail',(337,),(96,),256,1,1,True,
                     recent_tokens=1,padded_tokens=338,corrupt_metadata=fault,corrupt_position=80)
    fixture=make_fixture(torch,spec);data=fixture['cpu']
    positions=torch.full((338,),-1,dtype=torch.int64);positions[:337]=torch.arange(96,433)
    checks=[(qb*3,1,3) for qb in range(0,337,21)] if fault else []
    data['expected_positions']=positions
    data['required_status_checks']=torch.tensor(checks,dtype=torch.int64).reshape(-1,3)
    torch.save(fixture,folder/'fixture.pt')
    for name,value in data.items():dump(folder/(name+'.bin'),value)
    dump(folder/'raw_striped.bin',data['raw'])
    shape=(338,6,1,256,96,64,1,3,reuse.BLOCK_TOKENS,fixture['blocks'],reuse.PREFIX,fixture['stride'],4,1,1,data['table'].shape[1])
    (folder/'shape.txt').write_text(' '.join(map(str,shape))+'\n')
    (folder/'case_flags.txt').write_text(str(int(fault))+'\n')
    # Current segment is provided by native FIA in the production long path.
    # This checker explicitly validates sources0/1 and empty source2 only.
    (folder/'cluster_flags.txt').write_text(f'1 1 {80 if fault else -1} -1\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['make','golden']);p.add_argument('folder',type=Path)
    p.add_argument('--fault',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.action=='make':make(a.folder,a.fault)
    else:golden(a.folder,a.output,1)
