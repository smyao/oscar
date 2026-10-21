# Calibration and rotation audit, 2026-10-21

Archive references: G19–G24 / #23–25 require one aligned sample and finite covariance; #86 separates target FULL layers from MTP; #97 prohibits treating a missing calibrated artifact as a working calibration. This is a source/CPU algebra audit only. No model forward, target connection, NPU validation, performance experiment, or LongBench acceptance was performed.

Official source: `FutureMLS-Lab/OSCAR`, `sglang-rebase` resolved by this audit to `38680852e52c10bd6b9e9a64ed9bc815e742859b`; individual URLs and hashes are in `upstream_manifest.json`. The local production fitter still identifies the earlier `339368e` V1 recipe. The current official calibration, startup prompt driver, offline rotation fitter, and quantized-prefill helper have byte-identical SHA256 to the saved `339368e` source hashes. There is no evidence that these capabilities were introduced only after the local October 18 audit.

## What actually matches the official V1 startup fitter

- The default input is the same GPQA-Diamond CSV, same five fields, same seed-0 option shuffle, and identical question template. All 198 rendered message records compare equal against functions extracted from the current official source. Both call `apply_chat_template(tokenize=True, add_generation_prompt=True, return_dict=False)` and stop before the first whole prompt would exceed the 30,000-token budget. The real model tokenizer was not executed in this host audit.
- The official startup path is also prefill-only. Its request has `max_new_tokens=0`; local vLLM uses `max_tokens=1`, sampling the prefill logits without a subsequent decode forward. The official path excludes prefix sharing using per-prompt `extra_key`; local calibration disables APC. Both require each prompt to fit one prefill chunk. Therefore, short GPQA prompts and absent decode samples are **not a local-versus-official implementation discrepancy**. The official README itself warns that short-prompt calibration can degrade at unseen long-context positions; that is a shared quality limitation, not proof of the user's five-point cause.
- Both observe post-RoPE Q/K/V before their attention calculation. The official pool starts with identity rotations. Although it writes an INT2 cache, fresh unchunked calibration prompts have no cached prefix and the prefill attention concatenates the unquantized current K/V. The INT2 writes therefore do not imply that lower-layer cache quantization corrupts the higher-layer calibration sample.
- Both accumulate each KV head's grouped-query Gram in FP64, divide by token count times GQA ratio, form V covariance weighted by `k^T M_q k`, normalize each head by its energy sum clamped at `1e-12`, sum over KV heads/TP, and average into **one rotation per layer**. Both symmetrize, run `eigh`, and form `U @ H @ Pbr`, with `perm[bit_reverse] = argsort(eigenvalues, descending=True)`.
- Direct CPU execution of the current official covariance/composition methods against local code gave identical sufficient statistics and identical FP32 K/V matrices in two cases: D8/Q6/KV1/TP4/2111 tokens and D64/Q12/KV2/TP2/615 tokens. This is implementation-level equation evidence, not cross-device or model acceptance. See `calibration_equation_comparison.json`.

Source locations: official `entrypoints/oscar_startup_calibration.py:34–51,90–119,150–185,188–239`; `environ.py:957–960`; `layers/radix_attention.py:179–195`; `layers/attention/triton_backend.py:1448–1512`; `mem_cache/oscar_calibration.py:58–95,318–372,417–458,646–691`. Local `tools/calibrate_model_rotations.py:23–86,159–184`; `oscar_ascend/rotation_calibration.py:85–175`; `oscar_ascend/integration/rotation_capture.py:108–134`.

## Actual execution differences and their limits

Official startup batches up to 32 prompts into a page-rounded prefill chunk; local calibration runs one prompt at a time, eager, in an isolated native Ascend engine. Official fitting uses device FP64 (CUDA on the referenced route); local fitting uses CPU FP64 after collecting the NPU sample. Official Qwen3.8 defaults to TP2; this project uses TP4. Given identical Q/K/V samples, the grouping/reduction equations agree, but these engine/backend choices do not guarantee identical samples or eigenvectors.

Eigenvector signs are not unique. Two valid eigenbases of the same covariance can give different INT2 reconstruction after `U @ H @ Pbr`; a D8 host example has zero eigen residual for both bases and reconstruction L2 errors 5.3072 versus 3.9791. This establishes a cross-backend reproducibility limitation. It does **not** identify a bug in either eigensolver or justify inventing a sign-normalization rule absent from the official recipe.

No W8A8-specific compensation, alternate covariance, eigenvalue order, rotation direction, or head grouping exists in the local calibration/rotation code. W8A8 changes the actual native model execution via its `quantization` parameter, and therefore the real Q/K/V samples. There is no evidence from this host audit that the observed original-versus-W8A8 quality difference has been causally assigned.

## V1 startup fitting is not the complete current published recipe

The linked `rotation/run/qwen3.8-27b.sh` calls `_common.sh::need_rot`. By default `STARTUP_CALIBRATE=0`: it loads already supplied per-model RotationZoo files, and does not necessarily run V1 startup fitting. It also does not establish which matrix schema or calibration corpus the supplied PT contains. `eval_oscar_gpqa.sh` is a GPQA driver by default, not evidence of lossless LongBench performance.

The current official README distinguishes the latest published results: most dense/hybrid rows use **per-(layer, KV head) rotations, key centering, and per-head K clipping**. The local default remains V1 shared-layer qqt/sst, no key mean, and global K/V clipping. This is a concrete missing-capability difference, but does not establish the exact Qwen3.8 result artifact. The public RotationZoo snapshot `35c854c4cb6deffa62b93c1939053ae9343a7df6` (last modified 2026-08-11) lists no Qwen3.8 directory/files, and the Qwen3.8 directory request returned 404 in the parent audit. An image may contain an unpublicized artifact; it was not inspected. Therefore the Qwen3.8 published PT must not be asserted to be V2 merely from the README wording.

Official V2 support:

- `memory_pool.py:282–408` preserves a leading KV-head matrix axis; `411–464` loads optional `q_rotation`, `o_rotation`, and `k_mean` fields.
- `unified_kv_pool.py:544–630` loads these fields and slices global KV heads by TP rank. Missing companion matrices alias the orthogonal R matrices; missing mean disables centering.
- `quantized_kv_prefill.py:126–140` applies per-head matrices via `einsum` and repeats them across GQA query groups; `176–201` uses the query companion, subtracts the per-head mean from K, and rotates K/V.
- `unified_kv_pool.py:145–199,474–508` obtains per-layer/per-head K clipping from the `SGLANG_OSCAR_K_CLIP_PER_LAYER` / `_PER_HEAD` JSON paths, then shards the table and converts ratios to `floor(ratio * D)` indices. Clip tables are not automatically inferred from the matrix PT. The Qwen3.8 script does not set these two environment variables; the generic launch pins scalar K=0.96 / V=0.92 and otherwise inherits external per-head settings.
- The local artifact loader requires format 1, grouping `layer`, and `[D,D]` matrices. It does not implement V2 per-head matrices, means, or distinct query/output companion transforms.

README evidence: lines 19,28,88 identify the current recipe; lines 231–245 describe V1 startup and its long-context limitation; lines 251–274 distinguish `perhead`, `center`, and output-aware variants. The Qwen3.8 published table reports GPQA 88.4 → 86.9 and KL@128K 0.029; those are not LongBench scores or a no-loss guarantee.

## Safe mathematical implementation requirements if V2 is adopted

Per-head rotations must preserve global KV-head identity through TP slicing and GQA broadcasting. Centering must apply consistently to historical INT2, HP windows, and current keys. The official implementation does that for all sources (`unified_kv_pool.py:1374–1383`). This project keeps exact sources in their original space, so simply subtracting a mean only in the INT2 writer would change relative attention weights. It must either center every source or restore `q dot mean * attention_scale` to centered-history logits/LSE before merging with original-space exact sources.

The D64 host counterexample in `calibration_center_counterexample.json` has equal original history/raw logits and expected output 0.5. Centering history alone changes the output to 0.0000453979; centering both sources or applying the required history correction restores 0.5. Likewise, non-orthogonal transforms require separate `Q=Rk^{-T}` and `O=Rv^{-T}`; reusing the current orthogonal transpose rule would be mathematically wrong.

No production correction is proposed solely from the short-prompt or eigenvector-sign observations. The concrete alignment work is to identify the desired official PT/variant and implement its complete per-head/centering/clip contract, while separately resolving any verified attention source-window discrepancies reported by the attention audit.

## Confirmed and not verified

Confirmed: V1 equations and all GPQA message text match; official startup is short-prompt prefill-only; official default script loads supplied PTs; current code supports V2/centering/head clipping while the local default does not; no local W8A8-specific calibration branch was found.

Not verified: the actual Qwen3.8 published PT/schema/calibration data/head clip table, actual model Q/K/V equality across Ascend/vLLM and CUDA/SGLang, target eigensolver sign differences, or the cause/share of the user's five-point LongBench loss. No claimed recovery or quality improvement follows from these host comparisons.
