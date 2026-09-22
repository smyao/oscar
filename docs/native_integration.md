# Native integration and physical cache contract

This document describes implemented host-side geometry, import hooks, spec
construction, and metadata seams. **It does not establish a functioning NPU
attention runtime.** The plugin rejects enabled dense FULL routing until a
concrete runtime provider passes its readiness check. Compilation, kernel
accuracy, cache lifecycle, graph capture, graph replay, and performance are
separate pending acceptance states.

Archive references: #17–20 (byte layout), #27 (incomplete service), #28/#77/#78
(imports), #31–33 (VllmConfig fields), #34/#36 (metadata capacity), #37–49
(transaction identity), #51/#52 (rank progress), #74–76 (native source baseline),
#84 (parent thread pools), #86 (calibration layer coverage), #94–97/#117/#122
(deployment and custom OPP environment). The archive was used for failure
evidence; no failed implementation repository was accessed.

## Verified reference seams

The checked local trees are vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
and vllm-ascend `19e436985102f4ed3aad36c137a6481653688a6c`; both were clean when
inspected. Paths below are relative to this project's `references/` directory.

| Native mechanism | Evidence | External action and invariant |
| --- | --- | --- |
| General plugin loaded in worker before worker class resolution | `vllm/vllm/v1/worker/worker_base.py:245` | Entry point `oscar_ascend.plugin:register`; stdlib imports only |
| Native platform backend selection | `vllm-ascend/vllm_ascend/platform.py:815` | Wrap classmethod; dense FULL resolves to external backend only after readiness |
| Custom cache-spec registration | `vllm-ascend/vllm_ascend/platform.py:1292` | Call native registration, then register `OscarFullAttentionSpec` |
| FULL and GDN spec generation | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4907` | Provider transforms FULL specs; original GDN objects remain intact |
| Shared raw allocation | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4080` | Provider must allocate each native pool once and never a second full BF16 history |
| Shared raw views | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4351` | Provider delegates GDN views to native logic and builds byte packed views for FULL |
| GDN SoA conv/SSM | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4696` | FULL may use only the SSM interval of its allocated physical page ID |
| Native page padding and base block | `vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py:94` | Single-K-page alignment can reserve `P=C+2*M`; preserve this full allocation budget |
| Virtual page conversion | `vllm-ascend/vllm_ascend/worker/block_table.py:52` | Retain native block tables and convert virtual IDs in kernel addressing |
| Group LCM and prefix hash GCD | `vllm/vllm/v1/core/kv_cache_utils.py:593` | Choose FULL block as multiple of both native Mamba block and kernel block |
| Layer grouping | `vllm/vllm/v1/core/kv_cache_utils.py:1158` | 17 FULL plus 48 GDN yields one 17-layer FULL group and three 16-layer GDN groups |
| Allocation by shared pools | `vllm/vllm/v1/core/kv_cache_utils.py:1302` | Count 17 physical tensors, not 65 independent layer tensors |
| Device batch metadata | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:3034` | Reuse device query starts, exact seq lengths, slots and block tables |
| Native metadata CPU readback | `vllm-ascend/vllm_ascend/attention/attention_v1.py:298` | Custom builder does not call native builder or deprecated CPU properties |
| Padded slots and block rows | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:3093` | Negative slot IDs never write; real capacities include native padding |
| Native FULL graphs retained | `vllm-ascend/vllm_ascend/platform.py:630` | Preserve requested FULL_DECODE_ONLY; older seam-map eager conclusion is stale |
| Draft eager flag scope | `vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:215` | Draft eager does not disable target graphs |
| GDN backend and acceptance | `vllm-ascend/vllm_ascend/ops/gdn.py:63`, `:196`, `:354` | Do not wrap GDN attention/state/acceptance kernels |
| Custom OPP environment | `vllm-ascend/vllm_ascend/utils.py:320` | Entries are vendor directories, not OPP roots; preserve native OPP |

## Packed byte ABI

One independent quantizer per `(token, local_kv_head)`. The PR's storage order
is K region followed by V region; each region contains packed codes, fp16 scale,
and fp16 zero. Codes store four 2-bit elements per byte, least significant bits
first. Metadata byte order is little endian.

For arbitrary K/V dimensions:

```text
K_codes = ceil(Dk/4)
V_codes = ceil(Dv/4)
slot = K_codes + 4 + V_codes + 4
token_bytes = local_kv_heads * slot
```

For `Dk=Dv=256`, offsets are K codes 0, K scale 64, K zero 66, V codes 68,
V scale 132, V zero 134, and total 136 bytes. No unexplained 160-byte row is
allocated. `SlotLayout` exposes these values; this geometry does not implement
or validate numerical quantization.

## GDN isolation and capacity derivation

Let `nb` be native physical block count, `C` native conv bytes per block, `M`
native SSM bytes per block, `P` native `MambaSpec.page_size_bytes`, `Bm` native
Mamba block tokens, and `K=128` kernel block tokens. Read `C/M` from the actual
Mamba spec shapes and dtypes, and P from its padded page budget. P can exceed
C+M; do not infer SSM dtype from P or assume historical model geometry.

```text
A = lcm(Bm, K)
B = floor(M / (token_bytes * A)) * A
raw shared tensor bytes = nb * P, where P >= C + M
native unused tail bytes = nb * (P - C - M)
FULL physical block b base = nb*C + b*M
token/head address = base + t*token_bytes + h*slot
```

Native `patch_mamba_config.py:94–119` first computes a FULL block so that
**one K page** equals M; K/V then consume 2*M, and the native padded budget
becomes P=C+2*M. Therefore P=801792 is compatible with BF16 SSM M=393216 and
C=15360. It does not prove a FP32 SSM M=786432. With one local KV head of D=256,
K bytes/token=512, the native block is 768. For this example, OSCAR B=2304;
its payload uses only its own actual SSM interval, while the 393216 bytes of
native padding per page remain charged. This is a conservative isolated layout,
not a claim that all native padding has been reclaimed.

Zero capacity fails explicitly. The new scheduler LCM is `lcm(B,Bm)=B`, since
`B` is a multiple of Bm. With prefix caching enabled and native align mode,
`patch_mamba_config.py:143–146` sets Bm to the native FULL block. In other modes
Bm can equal max_model_len and fail the capacity check; those modes still need
a separately proved design, not a silent change to GDN block semantics.

Native GDN page `g` owns `[g*C,(g+1)*C)` and
`[nb*C+g*M,nb*C+(g+1)*M)`. A FULL page `b` owns a subset of its SSM interval.
Therefore FULL cannot overlap any conv interval, and FULL `b` cannot overlap
GDN SSM `g` when `b != g`. This proof relies on native allocator ownership:
simultaneously live FULL and GDN groups must not receive the same physical page
ID. Such allocator/device ownership still needs a real integration probe.

Using an AoS formula `b*(C+M)` is incorrect. With nb=19, C=15360, M=786432,
the supposed FULL page-1 start is 801792, which lies in GDN page-0 SSM
`[291840,1078272)`. The regression test includes this concrete counterexample.

Native virtual page IDs are `v=b*(B/K)+i`. Thus:

```text
b = v // (B/K)
t = (v % (B/K))*K + kernel_token_offset
slot = v*K + kernel_token_offset
# equivalently: b = slot // B; t = slot % B
```

Store kernels may consume native int32 slots; signed negative slots are padding
and have no storage address. Attention needs both B and K when interpreting the
native virtual block table.

`packed_view()` produces `[nb,B,H,slot]` with byte strides
`[M,H*slot,slot,1]` and storage offset `raw.storage_offset()+nb*C`. It does not
allocate a second history buffer. Native raw int8 storage can be reinterpreted
as uint8 without copying when the operator schema requires uint8.

The physical pool budget is `nb * shared_tensors * P`, counted once. Add
the actual bounded window arena, rotations, metadata, graph and kernel
workspaces. `MemoryBudget` requires these quantities explicitly and labels
`nb*B` as a FULL capacity **upper bound**. Usable request capacity must subtract
the null block, native GDN reservations, per-request rounding, and all extras.

## Runtime handoff interface

`integration/runtime_api.py` owns process-local state; `VllmConfig` receives no
undeclared runtime field. Root runtime code installs either a concrete provider
or a deferred factory:

```python
install_runtime(provider)
# or install_runtime_factory(factory)
```

The provider supplies these methods:

```text
assert_ready()
get_impl_cls() -> real AttentionImpl class
transform_kv_cache_specs(runner, native_specs) -> specs
allocate_kv_cache_tensors(runner, config, native_allocate) -> raw tensors
reshape_kv_cache_tensors(runner, config, raw, native_reshape) -> views
```

`assert_ready()` must validate real operator capability and initialization. It
must not infer readiness from import success or a reference implementation.
Missing provider/methods and provider failures propagate as explicit errors.
`transform_native_specs()` and `packed_view()` are ready for provider reuse;
they do not implement the allocator or complete lifecycle themselves.

Hooks preserve `__wrapped__`, function signatures and original descriptors.
Unregistration restores original classmethod/staticmethod/function objects and
refuses to overwrite another integration's later changes. The meta-path finder
delegates to the existing host loader and patches only after module execution.
Existing fully initialized modules are handled immediately. Registration while
a required module is halfway through import fails explicitly.

## Metadata and graph boundary

`from_common()` reuses tensor objects without slicing, values inspection,
host readback, Python per-request loops, or modifications to native buffers.
Native exact device `seq_lens` controls async MTP visibility; optimistic CPU
mirrors must not decide accepted lengths. The frozen metadata carries actual
request/token counts, padded tensors and maximum dimensions.

`MetadataCapacity.from_native_buffers()` records actual rows, offset count,
token slots and block-table columns. It deliberately does not derive columns
from `ceil(max_model_len/128)`; archive #36 had 2052 native columns versus a
wrong 2048 plugin capacity.

The builder advertises the intended `UNIFORM_BATCH` graph capability and records
capture origin. This is an interface declaration, not a passed graph test.
The runtime must still preallocate workspaces and establish safe dummy capture
state. A Python capture flag must **not** permanently suppress captured writes,
because replay needs to update the real cache. Capture return and real-request
replay require separate target-NPU evidence.

## Verification and remaining work

Executed locally using stdlib `unittest`: layout/metadata tests and lightweight
plugin tests. They cover PR byte offsets, generalized head dimensions, native
LCM geometry, explicit AoS corruption, SoA disjointness, virtual mapping,
memory accounting, metadata readback traps, historical capacity failure,
fresh-interpreter import differences, future import hooks, readiness rejection,
binding preservation and restoration. These tests do not import torch/NPU or
claim native worker execution.

Still pending: import against installed native dependencies, concrete runtime
provider, real raw allocation/GDN view delegation, bounded window ownership and
generation protocol, prefix sharing/reuse, MTP rollback, real AscendC calls,
all-rank worker probes, graph capture/replay and paired accuracy/performance.
There is no native FULL fallback when OSCAR is enabled.
