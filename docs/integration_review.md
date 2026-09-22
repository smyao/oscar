# Independent integration review

Reviewed the newly written deployment, build, rotation preparation, primitive
probe, readiness reporting, and native hook code. Only this workspace's
authorized reference trees and archive logs were consulted. No failed OSCAR
implementation was read. Archive references: #27/#28/#31–36/#74–78/#84/#94/#95/
#117/#122; native source evidence is catalogued in `native_integration.md`.

## Findings fixed during this review

| Finding | Consequence | Fix and evidence |
| --- | --- | --- |
| A nonempty `VLLM_PLUGINS` list appended OSCAR but could omit Ascend | Native platform could remain unavailable despite an installed plugin | Root changed `target_env()` to retain existing entries and require both `ascend` and `oscar_ascend`; contract test passes |
| Device/config validation happened before initial status creation | First configuration failure had no `status.json`, repeating archive #94/#95 | Root creates status first and writes failed config status plus `config.log`; tested by a real subprocess |
| Terminal unimplemented full-service gate raised while status remained running | User could misread a failed deployment as still active | Root records `failed_phase=full-service-probe`; terminal-state test passes |
| Native integrity comparison ran only after successful readiness | Any earlier phase failure skipped the post-run mutation check | Root moved comparison to `finally`; injected real process exit 7 plus integrity exit 3 retains 7 and records both |
| CMake searched Torch/pybind11 without querying the selected Python environment | Installed pip packages could still fail clean configure | CMake now queries `torch.utils.cmake_prefix_path` and `pybind11.get_cmake_dir`; source review only, target configure not executed |
| Build signature omitted package versions and CANN version contents | An in-place toolchain upgrade could reuse stale device objects | Root added package versions and CANN version-file hashes; source review only |
| Existing selector LRU cache survived external hook installation/restoration | Cached native FULL could bypass OSCAR, or cached OSCAR survive unregister | Plugin now clears the already-loaded `_cached_get_attn_backend` cache without importing native modules; test passes |
| A future module import could fail halfway through applying descriptors | Partial hooks could survive a failed import attempt | Callback installation now restores only its own partial changes before rethrowing; test passes |
| Initial layout assumed native `P=C+M` and rejected page padding | The target's legal BF16 SSM layout could fail spec conversion | Traced `patch_mamba_config.py:94–119`, added explicit native padded P, retained true C/M for SoA addresses and charged tail padding; host regression passes |

No tests were weakened to accept these failures. The command preservation test
parses the user's Appendix A with `shlex` and compares every option/value to
the generated command, normalizing only spelling of underscores versus
hyphens and JSON formatting. It confirms TP4, MTP3/draft eager, async scheduling,
FULL_DECODE_ONLY/capture sizes, W8A8, BF16 Mamba dtypes, model/limits, host/port,
media options and rope overrides remain equivalent.

## Native spec and descriptor review

`OscarFullAttentionSpec` inherits the actual native `FullAttentionSpec` and is
registered with `FullAttentionManager` under its own uniform base. This prevents
raw BF16 and packed OSCAR specs from being treated as one uniform type.
`merge()` retains the complete geometry and explicitly rejects differences;
calling the native superclass merge would reconstruct the subclass without its
required conv/SSM fields and is deliberately avoided. Native grouping dispatches
`layer_specs[0].merge(layer_specs)` in `kv_cache_utils.py:867`, so this override
is the relevant seam. The full native dependency stack has not been imported
or executed on target in this review.

For each transformed FULL spec, block size is a multiple of both the original
Mamba block size and kernel block 128. Native GDN spec instances, shapes, dtypes,
and block identities remain untouched by the conversion helper. Prefix hash
and scheduler LCM still follow the native `resolve_kv_cache_block_sizes()`
implementation. The native patch aligns one K page to SSM, then reserves K/V,
so padded P is commonly C+2*M. The target BF16 example C=15360/M=393216 can
have P=801792/nativeB=768/OSCAR B=2304, not the earlier inferred FP32 SSM.
Modes using Mamba block=max_model_len exceed this layout capacity and remain
an explicit unsupported implementation gap. Numerical cache behavior requires
the missing runtime layer.

Classmethod/staticmethod/function descriptors retain their original binding and
introspection signatures. Unregister restores exact original objects; inherited
attributes are removed from the child class rather than shadowing the parent.
Late registration supports fully initialized native modules, and refuses
in-flight native imports. It does not replace already-constructed model layer
instances: registration must precede model construction.

## Current implementation and target boundary

The concrete RuntimeProvider, AttentionImpl, allocator/GDN delegation, native page snapshots, MTP slot-derived positions, graph workspace, fused CV/rotation/store, status guard, service probes and cleanup are implemented. Official CANN CPU-debug and cross-compilation evidence is in reports/vm; additional real native method execution tests cover grouping, manager/hash/LCM, shape derivation and rejected-token draft updates.

Remaining acceptance: target NPU execution, actual capture/replay, model quality, memory benefit and paired performance. Direct launch is the chosen native-supported independent-library interface; no custom OPP vendor package is produced. H18 literal sublinear reads remains an explicit conflict.

## Local verification

Executed:

```text
python3 -m unittest discover -s tests -p 'test_layout.py' -v
python3 -m unittest discover -s tests -p 'test_plugin.py' -v
python3 -m unittest discover -s tests -p 'test_integration_contracts.py' -v
```

The reviewed suite comprises 15 layout/metadata tests, 11 plugin tests, and
10 integration contract tests in the initial review. The final suite is recorded
in reports/pytest.log. CANN VM cross-compilation and official CPU debugging have
since passed. Target device completion, actual worker binding, graph capture,
graph replay, model accuracy and performance remain not run.
