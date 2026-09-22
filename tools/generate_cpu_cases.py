# Archive G26/G27/G30-G34/#13-22: independent PR oracle generates external goldens for the real AscendC CPU debugger.
"""Generate small CPU-debug test inputs; never invoked in production."""
import argparse
import json
from pathlib import Path
import torch
from oscar_ascend.ops.reference import encode_kv


def write(path, value):
    path.write_bytes(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    gen=torch.Generator().manual_seed(46774)
    cases=[]
    for d in (64,128,256):
        for slot64 in (False,True):
            n,h,b,nb=8,2,256,3
            offset=nb*256;stride=b*h*(d//2+8)+256
            path=root/f"store_{d}_{int(slot64)}";path.mkdir(exist_ok=True)
            k=torch.randn(n,h,d,generator=gen);v=torch.randn(n,h,d,generator=gen)
            slots=torch.tensor([-1,0,127,128,255,256,767,768],dtype=torch.int64 if slot64 else torch.int32)
            packed=encode_kv(k,v);raw=torch.full((offset+nb*stride,),173,dtype=torch.uint8)
            status=torch.zeros(n,h,dtype=torch.int32);status[-1]=1
            for i,s in enumerate(slots.tolist()):
                if 0<=s<b*nb:
                    p,t=divmod(s,b);start=offset+p*stride+t*h*(d//2+8)
                    raw[start:start+h*(d//2+8)]=packed[i].flatten()
            for name,tensor in (("key",k),("value",v),("slots",slots),("expected_raw",raw),("expected_status",status)):
                write(path/f"{name}.bin",tensor)
            (path/"shape.txt").write_text(f"{n} {h} {d} {b} {nb} {offset} {stride} {int(slot64)}\n")
            cases.append({"op":"store","path":str(path)})
        for splits in (1,3,128):
            rows=5;path=root/f"merge_{d}_{splits}";path.mkdir(exist_ok=True)
            partial=torch.randn(rows,splits,d,generator=gen);lse=torch.randn(rows,splits,generator=gen)*10
            lse[0]=-torch.inf;partial[0]=torch.nan
            if splits>1: lse[1,0]=-torch.inf;partial[1,0]=torch.nan
            out=torch.zeros(rows,d);expected_lse=torch.logsumexp(lse,1)
            safe=torch.where(torch.isneginf(lse[...,None]),0,partial)
            out[1:]=(safe[1:]*torch.softmax(lse[1:],1)[...,None]).sum(1)
            for name,tensor in (("partial",partial),("partial_lse",lse),("expected_output",out),("expected_lse",expected_lse),("expected_status",torch.zeros(rows,dtype=torch.int32))):
                write(path/f"{name}.bin",tensor)
            (path/"shape.txt").write_text(f"{rows} {splits} {d}\n")
            cases.append({"op":"merge","path":str(path)})
    for invalid in (False,True):
        path=root/("guard_error" if invalid else "guard_zero");path.mkdir(exist_ok=True)
        for name,size in zip(("a","b","c","d"),(2049,17,1,1025)):
            values=torch.zeros(size,dtype=torch.int32)
            if invalid and name=="d":values[-1]=3
            write(path/f"{name}.bin",values)
        (path/"shape.txt").write_text("2049 17 1 1025\n")
        cases.append({"op":"guard","path":str(path),"expect_kernel_error":invalid})
    (root/"cases.json").write_text(json.dumps(cases,indent=2)+"\n")
    return cases


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    print(json.dumps(generate(p.parse_args().output),indent=2))
