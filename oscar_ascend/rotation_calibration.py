"""Offline official OSCAR qqt/sst calibration; never a serving CPU fallback.

Archive G19/G20/G23/#23-25/#86/#96/#97/#111: one aligned observation,
complete FULL-layer coverage, finite FP64 statistics, separate draft identity.
#86/#97/#165/#166: imported/cached pairs must identify the actual model and
native sampling recipe, including both effective GDN state dtypes.
Implements the requested official FP64 offline fitting; serving stays NPU.
Model forwards remain NPU-only. No Q/K/V is replayed to estimate a second pass.
Primary: FutureMLS-Lab/OSCAR sglang-rebase oscar_calibration.py, observe,
local_covariance_sums, _energy_weighted_v_covariance and finalize.
Recipe3: official fit_oscar2_variants.py perhead/center keeps global KV-head
identity, normalizes each M_q by trace/D, and publishes the FP32 key mean.
"""
from __future__ import annotations
import fcntl, hashlib, importlib, json, math, os, tempfile, time, uuid
from pathlib import Path
from typing import Mapping

OFFICIAL_COMMIT="339368e8fb440c53d5e157a5974036f25f3841a0"
K_FILENAME="k_rotation_qqt_r_h_pbr.pt"
V_FILENAME="v_rotation_sst_r_h_pbr.pt"
GRAM_CHUNK=512
COVARIANCE_CHUNK=2048


def _torch():return importlib.import_module("torch")


def _validated_calibration_profile(profile, model_fingerprint):
    if not isinstance(profile, Mapping):
        raise ValueError("rotation calibration profile is required")
    variant = profile.get("rotation_variant", "shared_v1")
    if variant not in {"shared_v1", "perhead_center"}:
        raise ValueError("unsupported rotation calibration variant")
    recipe = 3 if variant == "perhead_center" else 2
    if (profile.get("recipe_version") != recipe or
            profile.get("official_commit") != OFFICIAL_COMMIT or
            profile.get("model_fingerprint") != model_fingerprint):
        raise ValueError("rotation calibration recipe/model identity differs")
    execution = profile.get("model_execution")
    if (not isinstance(execution, Mapping) or execution.get("dtype") != "bfloat16" or
            any(execution.get(name) != "bfloat16"
                for name in ("mamba_cache_dtype", "mamba_ssm_cache_dtype"))):
        raise ValueError("rotation calibration execution profile requires the current native BF16 state dtypes")
    # Keep a portable snapshot: changing the caller's config must not relabel
    # an already fitted pair, and no tensors/custom objects belong in identity.
    return json.loads(json.dumps(profile, sort_keys=True, allow_nan=False))


class OfflineLayerAccumulator:
    def __init__(self,*,head_dim,local_kv_heads,local_q_heads,token_budget,testing=False):
        if any(type(x) is not int or x<=0 for x in (head_dim,local_kv_heads,local_q_heads,token_budget)):
            raise ValueError("positive integer calibration geometry required")
        if local_q_heads%local_kv_heads or head_dim&(head_dim-1):raise ValueError("invalid calibration GQA/D")
        self.head_dim,self.local_kv_heads,self.local_q_heads=head_dim,local_kv_heads,local_q_heads
        self.token_budget,self.testing,self.count=token_budget,testing,0
        self.gqa_ratio=local_q_heads//local_kv_heads
        self.q_gram=_torch().zeros((local_kv_heads,head_dim,head_dim),dtype=_torch().float64,device='cpu')
        self.k=self.v=None;self.source_device=None

    def observe(self,q,k,v):
        torch=_torch()
        if self.count==self.token_budget:return
        if any(not torch.is_tensor(x) for x in (q,k,v)):raise ValueError("tensor samples required")
        if not (q.device==k.device==v.device):raise ValueError("aligned Q/K/V must share device")
        if q.device.type!="npu" and not (self.testing and q.device.type=="cpu"):
            raise RuntimeError("calibration model forward must use NPU; CPU samples require testing=True")
        if q.shape[0]!=k.shape[0] or k.shape[0]!=v.shape[0]:raise ValueError("aligned sample token counts differ")
        if any(x.dtype not in (torch.float32,torch.bfloat16,torch.float16) for x in (q,k,v)):
            raise ValueError("native sample dtype must be FP32/BF16/FP16")
        if q.numel()!=q.shape[0]*self.local_q_heads*self.head_dim or any(
                x.numel()!=x.shape[0]*self.local_kv_heads*self.head_dim for x in (k,v)):
            raise ValueError("sample head geometry changed")
        n=min(q.shape[0],self.token_budget-self.count)
        if not n:return
        # Explicit synchronous transfer: statistics and host rows must describe
        # this exact forward, and no pending DMA may outlive the native input.
        q=q[:n].detach().to("cpu").reshape(n,self.local_q_heads,self.head_dim)
        k=k[:n].detach().to("cpu").reshape(n,self.local_kv_heads,self.head_dim)
        v=v[:n].detach().to("cpu").reshape(n,self.local_kv_heads,self.head_dim)
        if any(not bool(torch.isfinite(x).all()) for x in (q,k,v)):
            raise ValueError("non-finite calibration sample")
        if self.k is None:
            self.k=torch.empty((self.token_budget,self.local_kv_heads,self.head_dim),dtype=k.dtype,device='cpu')
            self.v=torch.empty_like(self.k,dtype=v.dtype)
        if self.k.dtype!=k.dtype or self.v.dtype!=v.dtype:raise ValueError("native sample dtype changed")
        for start in range(0,n,GRAM_CHUNK):
            stop=min(start+GRAM_CHUNK,n)
            grouped=q[start:stop].reshape(stop-start,self.local_kv_heads,self.gqa_ratio,self.head_dim)
            grouped=grouped.permute(1,0,2,3).reshape(self.local_kv_heads,-1,self.head_dim).to(torch.float64)
            self.q_gram.add_(torch.bmm(grouped.transpose(1,2),grouped))
        self.k[self.count:self.count+n].copy_(k);self.v[self.count:self.count+n].copy_(v)
        self.source_device="cpu_test" if self.testing else "npu"
        self.count+=n

    def export_payload(self):
        if self.count!=self.token_budget:raise ValueError("calibration token budget not reached")
        return {"count":self.count,"token_budget":self.token_budget,"head_dim":self.head_dim,
            "local_kv_heads":self.local_kv_heads,"local_q_heads":self.local_q_heads,
            "gqa_ratio":self.gqa_ratio,"q_gram":self.q_gram,"k":self.k,"v":self.v,
            "source_device":self.source_device,"test_only":self.testing}


def local_head_covariances(payload,*,testing=False):
    torch=_torch();n=payload["count"];hk=payload["local_kv_heads"];d=payload["head_dim"]
    if n<=0 or n!=payload["token_budget"]:raise ValueError("incomplete calibration rows")
    if not testing and (payload.get("test_only") or payload.get("source_device")!="npu"):
        raise ValueError("production calibration needs native NPU samples")
    gram,k,v=payload["q_gram"],payload["k"],payload["v"]
    if gram.dtype!=torch.float64 or gram.shape!=(hk,d,d) or k.shape!=(n,hk,d) or v.shape!=k.shape:
        raise ValueError("calibration sufficient-statistic geometry/dtype differs")
    if any(x.device.type!="cpu" or not bool(torch.isfinite(x).all()) for x in (gram,k,v)):
        raise ValueError("offline statistics must be finite host tensors")
    if payload["local_q_heads"]!=hk*payload["gqa_ratio"]:raise ValueError("GQA ratio mismatch")
    q_cov=gram/(n*payload["gqa_ratio"])
    denominator=torch.zeros(hk,dtype=torch.float64,device='cpu')
    numerator=torch.zeros((hk,d,d),dtype=torch.float64,device='cpu')
    for start in range(0,n,COVARIANCE_CHUNK):
        kc=k[start:start+COVARIANCE_CHUNK].to(torch.float64)
        vc=v[start:start+COVARIANCE_CHUNK].to(torch.float64)
        energy=torch.einsum("thd,hde,the->th",kc,q_cov,kc)
        denominator.add_(energy.sum(dim=0))
        numerator.add_(torch.einsum("th,thd,the->hde",energy,vc,vc))
    # Same official denominator clamp; no sqrt, fitted epsilon or FP32 pass.
    v_cov=numerator/denominator.clamp_min(1e-12)[:,None,None]
    if not bool(torch.isfinite(q_cov).all() & torch.isfinite(v_cov).all()):
        raise ValueError("non-finite official covariance")
    return q_cov,v_cov


def local_covariance_sums(payload,*,testing=False):
    q_cov,v_cov=local_head_covariances(payload,testing=testing)
    return _torch().stack((q_cov.sum(dim=0),v_cov.sum(dim=0)))


def fit_checkpoint_pair(rank_payloads,*,layer_names,head_dim,model_path,model_fingerprint,
                        global_q_heads,global_kv_heads,tp_size,prompt_sha256,max_tokens,
                        calibration_profile,model_revision=None,testing=False):
    from .rotations import _names
    torch=_torch();names=_names(layer_names);ids=[int(name.split('.')[2]) for name in names]
    profile=_validated_calibration_profile(calibration_profile,model_fingerprint)
    if profile.get('max_tokens')!=max_tokens:
        raise ValueError('rotation calibration profile token budget differs')
    if len(rank_payloads)!=tp_size or {p['rank'] for p in rank_payloads}!=set(range(tp_size)):
        raise ValueError("complete unique TP rank payloads required")
    if global_q_heads%tp_size or not (global_kv_heads%tp_size==0 or tp_size%global_kv_heads==0):
        raise ValueError("unsupported global/TP head geometry")
    centered=profile.get('rotation_variant')=='perhead_center'
    if centered and (tp_size>global_kv_heads or global_kv_heads%tp_size):
        raise ValueError('perhead_center requires disjoint TP KV-head shards; replicated KV is unsupported')
    replication=max(1,tp_size//global_kv_heads);local_hk=max(1,global_kv_heads//tp_size)
    sums=torch.zeros((2,len(ids),head_dim,head_dim),dtype=torch.float64,device='cpu');counts=set()
    if centered:
        head_cov=torch.empty((2,len(ids),global_kv_heads,head_dim,head_dim),dtype=torch.float64,device='cpu')
        means=torch.empty((len(ids),global_kv_heads,head_dim),dtype=torch.float64,device='cpu')
    for rank in sorted(rank_payloads,key=lambda x:x['rank']):
        for key,want in [('tp_size',tp_size),('global_q_heads',global_q_heads),
                         ('global_kv_heads',global_kv_heads),('prompt_sha256',prompt_sha256),
                         ('model_fingerprint',model_fingerprint)]:
            if rank.get(key)!=want:raise ValueError('rank calibration identity differs: '+key)
        if set(rank['layers'])!=set(ids):raise ValueError('rank FULL layer coverage mismatch')
        for index,layer in enumerate(ids):
            data=rank['layers'][layer];counts.add(data['count'])
            if (data['head_dim'],data['local_kv_heads'],data['local_q_heads'])!=(head_dim,local_hk,global_q_heads//tp_size):
                raise ValueError('rank/layer head geometry differs')
            if centered:
                start=rank['rank']*local_hk
                q_cov,v_cov=local_head_covariances(data,testing=testing)
                head_cov[0,index,start:start+local_hk]=q_cov
                head_cov[1,index,start:start+local_hk]=v_cov
                means[index,start:start+local_hk]=data['k'].to(torch.float64).mean(0)
            else:
                sums[:,index].add_(local_covariance_sums(data,testing=testing))
    if len(counts)!=1 or not 0<next(iter(counts))<=max_tokens:raise ValueError('inconsistent selected prompt token count')
    tokens=next(iter(counts));cov=head_cov if centered else sums/(global_kv_heads*replication)
    cov=(cov+cov.transpose(-1,-2))/2
    if centered:
        mean_eigenvalue=cov[0].diagonal(dim1=-2,dim2=-1).sum(-1)/head_dim
        if not bool(torch.isfinite(mean_eigenvalue).all()) or bool((mean_eigenvalue<=0).any()):
            raise ValueError('per-head query covariance must have a positive finite trace')
        # Official center uses unit_mean_eig(M_q), but leaves S_v unscaled.
        cov[0]=cov[0]/mean_eigenvalue[...,None,None]
    flat=cov.reshape(-1,head_dim,head_dim)
    values,vectors=torch.linalg.eigh(flat)
    if not bool(torch.isfinite(values).all()&torch.isfinite(vectors).all()):raise ValueError('nonfinite FP64 eigensolver result')
    h=torch.ones((1,1),dtype=torch.float64,device='cpu')
    while h.shape[0]<head_dim:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)/math.sqrt(2)
    bits=head_dim.bit_length()-1;br=torch.tensor([int(f'{i:0{bits}b}'[::-1],2) for i in range(head_dim)],device='cpu')
    matrices=[]
    for u,e in zip(vectors,values):
        order=torch.argsort(e,descending=True);perm=torch.empty(head_dim,dtype=torch.long,device='cpu');perm[br]=order
        matrices.append(u@h@torch.eye(head_dim,dtype=torch.float64,device='cpu')[:,perm])
    axes=(2,len(ids),global_kv_heads) if centered else (2,len(ids))
    matrices=torch.stack(matrices).float().reshape(*axes,head_dim,head_dim).contiguous()
    values=values.reshape(*axes,head_dim);generation=uuid.uuid4().hex
    meta={'generation_id':generation,'model_path':str(model_path),'model_revision':model_revision,
        'model_fingerprint':model_fingerprint,'prompt_sha256':prompt_sha256,'tokens':tokens,
        'max_tokens':max_tokens,'global_q_heads':global_q_heads,'global_kv_heads':global_kv_heads,
        'tp_size':tp_size,'post_rope':True,'accumulator_dtype':'float64','created_at_unix':time.time(),
        'statistics_device':'cpu_offline','model_forward_device':'npu' if not testing else 'cpu_test',
        'official_commit':OFFICIAL_COMMIT,'test_only':testing,'calibration_profile':profile}
    def checkpoint(index,objective):
        entries={layer:{'layer_id':layer,'rotation':matrices[index,i],
                        'eigenvalues':values[index,i].float().contiguous()} for i,layer in enumerate(ids)}
        if centered and index==0:
            for i,layer in enumerate(ids):entries[layer]['k_mean']=means[i].float().contiguous()
        return {'format_version':2 if centered else 1,'objective':objective,
            'source_grouping':'layer_kv_head' if centered else 'layer','calibration':dict(meta),
            **({'global_kv_heads':global_kv_heads} if centered else {}),'layers':entries}
    return checkpoint(0,'qqt_r_h_pbr'),checkpoint(1,'sst_r_h_pbr')


def checkpoint_pair_artifact(k_state,v_state,*,layer_names,head_dim,model_fingerprint,
                             calibration_profile,testing=False,validate_math=True):
    from .rotations import _base_artifact,_names,ORTHOGONALITY_ATOL
    torch=_torch();names=_names(layer_names);ids={int(n.split('.')[2]) for n in names}
    profile=_validated_calibration_profile(calibration_profile,model_fingerprint)
    centered=profile.get('rotation_variant')=='perhead_center'
    global_heads=k_state.get('global_kv_heads') if centered else None
    if centered and (type(global_heads) is not int or global_heads<=0 or v_state.get('global_kv_heads')!=global_heads):
        raise ValueError('per-head pair requires matching positive global_kv_heads')
    matrix_shape=(global_heads,head_dim,head_dim) if centered else (head_dim,head_dim)
    states=[]
    for state,objective in ((k_state,'qqt_r_h_pbr'),(v_state,'sst_r_h_pbr')):
        if (state.get('format_version')!=(2 if centered else 1) or
                state.get('source_grouping')!=('layer_kv_head' if centered else 'layer') or state.get('objective')!=objective):
            raise ValueError('official calibrated pair schema/objective differs')
        layers=state.get('layers',{});keys=[int(i) for i in layers]
        if len(set(keys))!=len(keys) or set(keys)!=ids:raise ValueError('official FULL layer coverage differs')
        normalized={int(i):entry for i,entry in layers.items()};meta=state.get('calibration',{})
        if meta and (meta.get('accumulator_dtype')!='float64' or meta.get('post_rope') is not True):
            raise ValueError('official calibration requires FP64 post-RoPE statistics')
        if meta.get('test_only') and not testing:raise ValueError('CPU test calibration cannot enter production')
        if meta.get('model_fingerprint')!=model_fingerprint:
            raise ValueError('rotation model fingerprint missing or mismatched; regenerate for the selected model')
        if meta.get('calibration_profile')!=profile:
            raise ValueError('rotation calibration execution/profile missing or mismatched; regenerate for the selected configuration')
        if centered and meta.get('global_kv_heads')!=global_heads:
            raise ValueError('per-head metadata and matrix head counts differ')
        for layer,entry in normalized.items():
            matrix=entry.get('rotation')
            if entry.get('layer_id')!=layer or not torch.is_tensor(matrix) or matrix.dtype!=torch.float32 or matrix.shape!=matrix_shape:
                raise ValueError('official layer rotation geometry/dtype differs')
            if matrix.device.type!='cpu':raise ValueError('official pair must contain CPU constants')
            if validate_math and not bool(torch.isfinite(matrix).all()):raise ValueError('official pair must contain finite CPU constants')
            if validate_math and float((matrix.transpose(-1,-2)@matrix-torch.eye(head_dim,device='cpu')).abs().amax())>ORTHOGONALITY_ATOL:
                raise ValueError('official rotation is not orthogonal')
            if centered and any(key in entry for key in ('q_rotation','o_rotation')):
                raise ValueError('perhead_center does not accept non-orthogonal companion transforms')
            if centered and objective=='qqt_r_h_pbr':
                mean=entry.get('k_mean')
                if (not torch.is_tensor(mean) or mean.dtype!=torch.float32 or
                        mean.device.type!='cpu' or mean.shape!=(global_heads,head_dim)):
                    raise ValueError('per-head K mean geometry/dtype differs')
                if validate_math and not bool(torch.isfinite(mean).all()):
                    raise ValueError('per-head K mean must be finite')
        states.append(normalized)
    km,vm=k_state.get('calibration',{}),v_state.get('calibration',{})
    for field in ('generation_id','prompt_sha256','tokens','max_tokens','global_q_heads','global_kv_heads','tp_size','model_path','model_revision','accumulator_dtype','post_rope','test_only','calibration_profile'):
        if km.get(field)!=vm.get(field):raise ValueError('K/V calibration generation mismatch: '+field)
    result=_base_artifact(names,head_dim,model_fingerprint,OFFICIAL_COMMIT,'qqt_sst_r_h_pbr',testing)
    result.update(calibrated=True,accumulation_dtype='float64',calibration=dict(km),
        calibration_profile=profile,official_pair_objectives=[k_state['objective'],v_state['objective']])
    if centered:
        result.update(format_version=2,source_grouping='layer_kv_head',
                      global_kv_heads=global_heads,rotation_variant='perhead_center')
    for name in names:
        layer=int(name.split('.')[2]);result['layers'][name]={'layer_id':layer,'Rk':states[0][layer]['rotation'].contiguous(),'Rv':states[1][layer]['rotation'].contiguous()}
        if centered:result['layers'][name]['k_mean']=states[0][layer]['k_mean'].contiguous()
    return result


def _atomic_json(path,value):
    fd,temp=tempfile.mkstemp(prefix='.'+path.name,suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def publish_checkpoint_pair(k_state,v_state,directory,*,artifact,artifact_name='calibrated.pt'):
    torch=_torch();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    pending=directory/'.oscar.pending.json';complete=directory/'.oscar.complete.json'
    if Path(artifact_name).name!=artifact_name:raise ValueError('artifact must be in pair directory')
    with (directory/'.oscar.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);temporary=[]
        try:
            generation=k_state.get('calibration',{}).get('generation_id') or uuid.uuid4().hex
            records={K_FILENAME:k_state,V_FILENAME:v_state,artifact_name:artifact};written={}
            for name,state in records.items():
                fd,temp=tempfile.mkstemp(prefix='.'+name,suffix='.tmp',dir=directory);os.close(fd);temporary.append(temp)
                torch.save(state,temp)
                with open(temp,'rb') as stream:os.fsync(stream.fileno())
                written[name]={'path':temp,'sha256':hashlib.sha256(Path(temp).read_bytes()).hexdigest()}
            metadata={'generation_id':generation,'model_fingerprint':artifact['model_fingerprint'],
                'files':{name:row['sha256'] for name,row in written.items()}}
            _atomic_json(pending,metadata)
            for name,row in written.items():os.replace(row['path'],directory/name);temporary.remove(row['path'])
            _atomic_json(complete,metadata);pending.unlink()
        finally:
            for path in temporary:
                if os.path.exists(path):os.unlink(path)
            fcntl.flock(lock,fcntl.LOCK_UN)
    return directory/artifact_name


def verified_pair_files(directory,*,artifact_name='calibrated.pt'):
    directory=Path(directory)
    with (directory/'.oscar.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        if (directory/'.oscar.pending.json').exists():raise ValueError('incomplete rotation pair publication')
        manifest=json.loads((directory/'.oscar.complete.json').read_text())
        if set(manifest['files'])!={K_FILENAME,V_FILENAME,artifact_name}:raise ValueError('rotation pair manifest file set differs')
        for name,want in manifest['files'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=want:raise ValueError('rotation pair file hash mismatch')
        # Caller reads returned snapshots, not paths whose generation can race.
        torch=_torch()
        states=tuple(torch.load(directory/name,map_location='cpu',weights_only=True) for name in (K_FILENAME,V_FILENAME,artifact_name))
        for state in states[:2]:
            generation=state.get('calibration',{}).get('generation_id')
            if generation is not None and generation!=manifest['generation_id']:
                raise ValueError('rotation pair manifest generation differs')
        if states[2].get('model_fingerprint')!=manifest['model_fingerprint']:
            raise ValueError('rotation pair manifest model identity differs')
        return states
