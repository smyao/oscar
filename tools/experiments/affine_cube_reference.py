"""Archive #13-22/#126/#154, D.4: CPU falsification of affine INT2 attention.

Research only: never imported by serving. No device performance is inferred
from CPU wall time. Original FP32 dequant/attention is the independent oracle.
"""
import json
from pathlib import Path
import torch
from oscar_ascend.ops.reference import quantize_int2, dequantize_int2, unpack_int2


def split_half_matmul(a,b,components=2):
    # Per-row power of two normalization: removes FP16 overflow without
    # changing the real-number product. float32 roundoff is still evaluated.
    maximum=a.abs().amax(-1,keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    exponent=torch.floor(torch.log2(maximum))
    scale=torch.pow(2.0,exponent)
    remainder=a/scale
    result=torch.zeros((a.shape[0],b.shape[1]))
    for _ in range(components):
        term=remainder.half().float()
        remainder=remainder-term
        result=result+term@b
    return result*scale


def run_case(n,dim,qscale,seed,components=2,biased=False):
    gen=torch.Generator().manual_seed(seed)
    q=torch.randn((24,dim),generator=gen)*qscale
    k=torch.randn((n,dim),generator=gen)
    v=torch.randn((n,dim),generator=gen)
    if biased:
        k=k+256
        # Signed cancelling query is not excluded by the frozen oracle.
        q[:,::2]=qscale
        q[:,1::2]=-qscale
    kq,vq=quantize_int2(k),quantize_int2(v)
    kd,vd=dequantize_int2(kq),dequantize_int2(vq)
    kc,vc=unpack_int2(kq.packed,dim).float(),unpack_int2(vq.packed,dim).float()
    logits=q@kd.T/dim**.5
    refp=torch.softmax(logits,dim=-1)
    ref=refp@vd
    qk=split_half_matmul(q,kc.T,components)*kq.scale.float().T
    qk=qk+q.sum(-1,keepdim=True)*kq.zero.float().T
    alt_logits=qk/dim**.5
    p=torch.softmax(alt_logits,dim=-1)
    out=split_half_matmul(p*vq.scale.float().T,vc,components)
    out=out+p@vq.zero.float()
    allowed=.005+.005*ref.abs();delta=(out-ref).abs()
    return {'n':n,'dim':dim,'query_scale':qscale,'biased':biased,'components':components,
            'max_abs':float(delta.max()),'failures':int((delta>allowed).sum()),
            'max_logit_difference':float((alt_logits-logits).abs().max()),
            'passed':bool((delta<=allowed).all())}


def main():
    torch.set_num_threads(min(4,torch.get_num_threads()))
    cases=[]
    for n in (511,20000,65536):
        for scale in (.01,1,10,100):
            row=run_case(n,256,scale,154+n)
            cases.append(row);print('AFFINE_CUBE_REFERENCE '+json.dumps(row),flush=True)
    for scale in (1,16,256,4096):
        for components in (2,3):
            row=run_case(511,256,scale,154,components,True)
            cases.append(row);print('AFFINE_CUBE_REFERENCE '+json.dumps(row),flush=True)
    report={'status':'passed' if all(r['passed'] for r in cases) else 'rejected',
            'scope':'CPU mathematical experiment; not NPU or model acceptance',
            'production_route':'unchanged','cases':cases}
    Path('reports/affine_cube_reference_20260930.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
