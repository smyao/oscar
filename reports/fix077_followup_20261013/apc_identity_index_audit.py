from pathlib import Path
import sys, random, json
from types import SimpleNamespace
sys.path[:0] = [str(Path.cwd()), str(Path.cwd()/'tests')]
import pytest
from test_runtime_native import native_specs, native_groups
from test_prefix_caching import _coordinator_with_hashing
M=pytest.MonkeyPatch()
groups=native_groups.__wrapped__(native_specs.__wrapped__(M), M, SimpleNamespace(param=True))
coordinator=_coordinator_with_hashing(groups,M)
full=coordinator.single_type_managers[0]
B=full.block_size; S=64; R=256; ring=259
snap={}; reads=writes=reused=0; page_owners={}; seen=set(); examples=[]
rng=random.Random(13024)
def alloc(req,end):
    global reused
    for manager in coordinator.single_type_managers:
        manager.new_step_starts(); manager.allocate_new_blocks(req,end+3,end)
    for blk in full.req_to_blocks[req]:
        if blk.block_id in page_owners and page_owners[blk.block_id]!=req:
            reused+=1; page_owners[blk.block_id]=req
        elif blk.block_id not in page_owners: page_owners[blk.block_id]=req

def page(req,p): return full.req_to_blocks[req][p//B].block_id

def produce(arms):
    global writes
    slots=[]
    for req,begin,count in arms:
        slots.extend((req,p,page(req,p)*B+p%B) for p in range(begin,begin+count))
    for i,(req,p,slot) in enumerate(slots):
        pg,ip=divmod(slot,B)
        keep=p>=S
        if keep and i+ring<len(slots):
            later=slots[i+ring][2]
            if later//B==pg and later==slot+ring: keep=False
        if p<S: snap[pg,p]=(ip,req,p); writes+=1
        if keep: snap[pg,S+ip%ring]=(ip,req,p); writes+=1

def check(req,context,origin=None):
    global reads
    tail=min(context,max(S,context+1-R))
    for p in list(range(min(S,context)))+list(range(tail,context)):
        pg=page(req,p); ip=p%B; row=p if p<S else S+ip%ring
        expected_owner=(origin[0] if origin and p<origin[1] else req)
        got=snap.get((pg,row)); reads+=1
        if got!=(ip,expected_owner,p):
            raise AssertionError(dict(req=req,context=context,position=p,page=pg,row=row,expected=(ip,expected_owner,p),got=got))

def free(req):
    for manager in coordinator.single_type_managers: manager.free(req)

# Force reuse through repeated long cold requests, two concurrent flattened intervals.
for cycle in range(24):
    names=[f'cold{cycle}-0',f'cold{cycle}-1']; contexts=[0,0]
    targets=[rng.choice([20000,23000,27000,30000]),rng.choice([20000,23000,27000,30000])]
    while contexts != targets:
        arms=[]
        for j,n in enumerate(names):
            amount=min(rng.choice([128,768,1536,2048,2304]),targets[j]-contexts[j])
            if amount:
                alloc(n,contexts[j]+amount); check(n,contexts[j]); arms.append((n,contexts[j],amount))
                contexts[j]+=amount
        produce(arms)
    for turn in range(40):
        for n,ctx in zip(names,contexts): alloc(n,ctx+7); check(n,ctx)
        produce([(n,c,4) for n,c in zip(names,contexts)])
        # Worst-order target q4 writer already complete before first MTP full-q4 read.
        for n,c in zip(names,contexts): check(n,c)
        accepted=[rng.randrange(4),rng.randrange(4)]
        for draft in (1,2):
            arms=[]
            for j,n in enumerate(names):
                ctx=contexts[j]+accepted[j]+draft
                check(n,ctx); arms.append((n,ctx,1))
            produce(arms)
        contexts=[c+a+1 for c,a in zip(contexts,accepted)]
    for n in names: free(n)

# Exact native hash/refcount prefix hit, immutable shared full pages + private suffix.
seed='seed'; total=4*B
hashes=[i.to_bytes(4,'big') for i in range(total//coordinator.hash_block_size)]
req=SimpleNamespace(request_id=seed,block_hashes=hashes,num_prompt_tokens=total)
for c in range(768,total+1,768):
    alloc(seed,c); check(seed,c-768); produce([(seed,c-768,768)])
    for manager in coordinator.single_type_managers: manager.cache_blocks(req,c)
free(seed)
computed,hit=coordinator.find_longest_cache_hit(hashes,total)
assert hit==3*B
for fork in ('left','right'):
    for manager,blocks in zip(coordinator.single_type_managers,computed):
        manager.allocate_new_computed_blocks(fork,blocks,hit,0)
    alloc(fork,hit+3000)
    check(fork,hit,(seed,hit))
shared_before={k:v for k,v in snap.items() if k[0] in {b.block_id for b in computed[0]}}
produce([('left',hit,3000),('right',hit,3000)])
for fork in ('left','right'): check(fork,hit+3000,(seed,hit))
shared_after={k:v for k,v in snap.items() if k[0] in {b.block_id for b in computed[0]}}
assert shared_before==shared_after
print(json.dumps({'status':'passed_identity_index_model','native_physical_block':B,'sink':S,'recent':R,'ring':ring,'cold_pairs':24,'q4_mtp_cycles':960,'exact_rows_checked':reads,'writer_rows':writes,'physical_id_reassignments':reused,'native_prefix_hit':hit,'shared_snapshot_immutable':True,'scope':'native allocator plus Python translation of production writer/read index; no AscendC execution or NPU acceptance'}))
M.undo()
