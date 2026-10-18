"""Independent bounded CPU fixture for aligned writer/prepare/read/merge."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import torch
from tools.aligned_reference import pack_kv,rotate_bf16,fixture_tasks_reference

def dump(path,tensor):
    path.write_bytes(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()))

def make(folder):
    folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(20261018)
    d=256;active=4;n=5;hq=6;hk=1;context=70;total=75;b=128;prefix=32;stride=144*b
    h=torch.ones((1,1))
    while h.shape[0]<d:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    h=h/16
    original_k=torch.randn((total,hk,d)).bfloat16();original_v=torch.randn_like(original_k)
    rotated_k=torch.empty_like(original_k);rotated_v=torch.empty_like(original_v)
    rotated_k[:-1]=rotate_bf16(original_k[:-1],h);rotated_v[:-1]=rotate_bf16(original_v[:-1],h)
    for value in (original_k,original_v,rotated_k,rotated_v):value[-1]=float('nan')
    q=torch.randn((n,hq,d)).bfloat16();q[-1]=float('nan')
    qrbf=torch.empty_like(q);qrbf[:-1]=rotate_bf16(q[:-1],h);qrbf[-1]=float('nan')
    qr=qrbf.clone();qr[-1]=0
    raw=torch.full((prefix+stride,),0x85,dtype=torch.uint8)
    expected_raw=raw.clone();expected_raw[prefix:prefix+74*144]=pack_kv(rotated_k[:74],rotated_v[:74]).flatten()
    wk=torch.full((68*hk*d*2,),0x85,dtype=torch.uint8).view(torch.bfloat16).reshape(1,68,hk,d);wv=wk.clone()
    tags=torch.full((1,68),-1,dtype=torch.int64);expected_wk=wk.clone();expected_wv=wv.clone();expected_tags=tags.clone()
    for pos in range(74):
        row=pos if pos<64 else 64+pos%4
        expected_wk[0,row]=original_k[pos];expected_wv[0,row]=original_v[pos];expected_tags[0,row]=pos
    store_slots=torch.arange(total,dtype=torch.int64);store_slots[-1]=-1
    cpu={'q':q,'qr':qr,'ck':original_k[70:75].clone(),'cv':original_v[70:75].clone(),
         'rv':h.bfloat16(),'raw':expected_raw,'wk':expected_wk,'wv':expected_wv,'tags':expected_tags,
         'table':torch.tensor([[0]],dtype=torch.int32),'starts':torch.tensor([0,4],dtype=torch.int32),
         'lens':torch.tensor([74],dtype=torch.int32),'slots':torch.tensor([70,71,72,73,-1])}
    fixture={'cpu':cpu,'spec':SimpleNamespace(kv_heads=hk,splits=1,recent_tokens=1),
             'stride':stride,'blocks':1,'scale':d**-.5}
    torch.save(fixture,folder/'fixture.pt')
    files={**cpu,'raw_striped':raw,'wk':wk,'wv':wv,'tags':tags,
           'qr_bf16':qrbf,'store_rk':rotated_k,'store_rv':rotated_v,'store_k':original_k,'store_v':original_v,
           'store_slots':store_slots,'store_positions':torch.arange(total),
           'store_expected_raw':expected_raw,'store_expected_wk':expected_wk,'store_expected_wv':expected_wv,
           'store_expected_tags':expected_tags,'expected_positions':torch.tensor([70,71,72,73,-1]),
           'required_status_checks':torch.empty((0,3),dtype=torch.int64)}
    for name,tensor in files.items():dump(folder/(name+'.bin'),tensor)
    (folder/'shape.txt').write_text(f'{n} {hq} {hk} {d} {context} 64 1 3 {b} 1 {prefix} {stride} 1 1 1 1\n')
    (folder/'case_flags.txt').write_text('0\n');(folder/'store_tokens.txt').write_text(str(total)+'\n')

def golden(folder,output,mode):
    fixture=torch.load(folder/'fixture.pt',weights_only=False)
    tasks=torch.frombuffer(bytearray((output/'tasks.bin').read_bytes()),dtype=torch.int64).reshape(-1,16).clone()
    results=fixture_tasks_reference(fixture,tasks,mode)
    for key in ['partial','lse','output','output_lse']:dump(output/('bf16_expected_'+key+'.bin'),results[key])
    dump(output/'rv_bf16.bin',fixture['cpu']['rv'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['make','golden']);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path);p.add_argument('--mode',type=int)
    a=p.parse_args()
    if a.action=='make':make(a.folder)
    else:golden(a.folder,a.output,a.mode)
