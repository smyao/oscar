`retention_results.json` is the bounded host comparison. Reproduce with
`.venv/bin/python tests/test_prefix_checkpoint_retention.py`. It executes the
native coordinator aligned publication, managers/pool, actual plugin lease hook,
and request 21 native `get_computed_blocks`. Source hashes are included.

`validation.json` and `validation.log` bind the final source and 233 related host
checks. `user_server.log` is the supplied field log, distinct from synthetic data.

The initial manager-only exploration bypassed native coordinator publication;
its counts are not used as production evidence. The final comparison executes
the inherited native method, including its alignment and EAGLE behavior.

All comparison identities and decode schedules are synthetic. No NPU, GDN tensor
contents, model quality or performance were executed or established.
