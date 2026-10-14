# Archive #157-160: source counterexample only; no target attribution or frozen-kernel changes.
# This NumPy address model runs unmodified extracted function bodies, not Triton interpreter/NPU.
import ast
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

SRC = Path('/private/tmp/oscar-aisbench-stable-20261013/references/vllm-ascend/vllm_ascend/ops/triton/reject_sample.py')

@dataclass
class Ptr:
    name: str
    data: object
    offset: object = 0
    def __add__(self, offset):
        return Ptr(self.name, self.data, np.asarray(self.offset) + offset)
    def __sub__(self, offset):
        return self + (-offset)

class TL:
    constexpr = int
    def __init__(self, poison):
        self.poison = poison
        self.pid = 0
        self.cu_reads = 0
        self.events = []
    def program_id(self, axis):
        return self.pid
    arange = staticmethod(np.arange)
    where = staticmethod(np.where)
    range = staticmethod(range)
    def load(self, ptr, mask=True, other=None):
        offsets = np.asarray(ptr.offset)
        masks = np.broadcast_to(mask, offsets.shape)
        result = np.empty(offsets.shape, dtype=ptr.data.dtype)
        if ptr.name == 'cu':
            self.cu_reads += 1
        for idx in np.ndindex(offsets.shape):
            offset = int(offsets[idx])
            if masks[idx]:
                if offset < 0 or offset >= ptr.data.size:
                    self.events.append({'kind':'read_oob', 'pid':self.pid, 'tensor':ptr.name, 'offset':offset})
                    result[idx] = 0
                else:
                    result[idx] = ptr.data[offset]
            elif other is not None:
                result[idx] = other
            else:
                # One legal choice of unspecified masked values: start=0, end=3.
                value = 3 if self.poison and ptr.name == 'cu' and self.cu_reads == 2 else 0
                result[idx] = value
                self.events.append({'kind':'masked_unspecified', 'pid':self.pid, 'tensor':ptr.name, 'offset':offset, 'chosen':value})
        return result
    def store(self, ptr, value, mask=True):
        offsets = np.asarray(ptr.offset)
        masks = np.broadcast_to(mask, offsets.shape)
        values = np.broadcast_to(value, offsets.shape)
        for idx in np.ndindex(offsets.shape):
            if not masks[idx]:
                continue
            offset = int(offsets[idx])
            if offset < 0 or offset >= ptr.data.size:
                self.events.append({'kind':'write_oob', 'pid':self.pid, 'tensor':ptr.name, 'offset':offset, 'value':int(values[idx])})
            else:
                ptr.data[offset] = values[idx]
                self.events.append({'kind':'write', 'pid':self.pid, 'tensor':ptr.name, 'offset':offset, 'value':int(values[idx])})

names = {'bonus_renew', 'rejection_greedy_sample_triton', 'rejection_random_sample_kernel'}
tree = ast.parse(SRC.read_text())
funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
for node in funcs:
    node.decorator_list = []
module = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))


def model(drafts, greedy, vectorcores, poison, random_after):
    batch = len(drafts)
    width = max(drafts) + 1
    tl = TL(poison)
    namespace = {'tl':tl, 'get_element':lambda x, pos:x[pos]}
    exec(compile(module, str(SRC), 'exec'), namespace)
    grid = batch if batch <= vectorcores else vectorcores
    block = 1 if batch <= vectorcores else 1 << (((batch + grid - 1)//grid)-1).bit_length()
    cu = Ptr('cu', np.cumsum(drafts))
    draft = Ptr('draft', np.arange(sum(drafts)) + 10)
    target = Ptr('target', draft.data.copy())
    bonus = Ptr('bonus', np.arange(batch) + 20)
    output = Ptr('output', np.full(batch*width, -1, dtype=np.int32))
    greedy_ptr = None if greedy is None else Ptr('greedy', np.asarray(greedy))
    for pid in range(grid):
        tl.pid, tl.cu_reads = pid, 0
        namespace['rejection_greedy_sample_triton'](output,cu,draft,target,bonus,greedy_ptr,batch,width-1,block)
    if random_after:
        probs = Ptr('probs', np.zeros(sum(drafts)*64))
        recovered = Ptr('recovered', np.full(sum(drafts), 30))
        uniform = Ptr('uniform', np.full(sum(drafts), .5))
        for pid in range(grid):
            tl.pid, tl.cu_reads = pid, 0
            namespace['rejection_random_sample_kernel'](output,cu,draft,None,probs,None,bonus,recovered,uniform,greedy_ptr,width-1,64,64,batch,None,True,True,False,False,block)
    rows = output.data.reshape(batch,width)
    counts = ((rows != -1) & (rows < 64)).sum(axis=1)
    qsl = np.r_[0,np.cumsum(np.asarray(drafts)+1)]
    rejected = np.where(np.asarray(drafts)>0,np.asarray(drafts)+1-counts,0)
    anchors = qsl[1:]-1-rejected
    return {'drafts':drafts, 'greedy':greedy, 'vectorcores_model':vectorcores, 'grid':grid, 'block':block, 'poison':poison, 'rows':rows.tolist(), 'valid_counts':counts.tolist(), 'rejected_counts':rejected.tolist(), 'anchors':anchors.tolist(), 'qsl':qsl.tolist(), 'events':tl.events}

mixed_control = model([3,1],[1,0],32,False,True)
mixed_poison = model([3,1],[1,0],32,True,True)
assert mixed_control['valid_counts'] == [4,1]
assert mixed_poison['valid_counts'] == [4,3]
assert mixed_poison['rejected_counts'][1] == -1
assert mixed_poison['anchors'][1] == mixed_poison['qsl'][2]
padded = model([3]*9,None,8,True,False)
assert any(event['kind'] == 'write_oob' for event in padded['events'])
print(json.dumps({'evidence':'CPU NumPy source address model; unspecified choices injected; not actual Triton execution','mixed_control':mixed_control,'mixed_poison':mixed_poison,'B9_V8_padded':padded}, indent=2))
