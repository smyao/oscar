# Exact-current numerical audit

This is a CPU algebra counterexample using the actual recipe3 fitter on
512 disclosed synthetic observations per TP rank. It does not use user
benchmark questions, real model activations, an arbitrary fabricated rotation,
or a clip search. The four-rank geometry is Q24/KV4/D256; inputs and all seeds
are recorded by the script.

Run from the repository root:

```bash
.venv/bin/python -m csrc.experiments.native_exact_current_20261028.counterexample --output reports/precision_stability_20261028/native_exact_current/counterexample.json
.venv/bin/python -m pytest -q csrc/experiments/native_exact_current_20261028/test_counterexample.py
```

The dense reference computes attention independently in original-space FP64
with BF16 interfaces. For one token the answer must equal V exactly. The
existing cold path's BF16 rotation and inverse can change that answer even
without INT2. Multi-token comparisons isolate the added finite-precision
coordinate transforms; they are not NPU FIA execution.

The cache comparison calls the production `rotate_inputs` and `store_rotated`
producers and records their four writer input tensors. Given identical
incoming K/V, old precomputed rotations and later store-only rotations agree
bitwise. This is not a claim that downstream layer activations remain equal
after improving an earlier attention output.

Any implementation must preserve raw-Q validation, native FIA parameters,
K centering, K/V rotations, clipping, writer/snapshot ownership and output
validation. It removes two GEMMs and preserves the existing copy count: the
native FIA result still enters contiguous preallocated output storage because
the native wrapper does not guarantee output contiguity. It adds no buffer or
new synchronization. This is a new exact-current precision policy, not a
claim that the explicitly selected v4 cold-rotation policy was a lifecycle
bug. Hardware performance and LongBench acceptance remain separate.
