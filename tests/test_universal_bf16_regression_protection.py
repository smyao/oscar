"""Archive #126/#145/#148/#172/#174: source and route protection.

The fixed hashes were read from git show 232b563b:<path> and 34bc2b75:<path>.
They protect the complete decoder and unchanged current-only/C4/routing path,
while retaining the independently changed C16 from 34bc2b75. No context-length
parameter or new threshold is permitted in those pinned dispatch/runtime files.
This does not assert that every 32K stage is original: warm C16 still differs
from 232b563b, and no source/host check proves device or end-to-end speed.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import pytest

from oscar_ascend.integration.current_only_plan import plan_current_suffix
from oscar_ascend.ops.cv_dispatch import OFFICIAL_CV_MAP, select_cv_op
from tools import generate_official_attention as generated
from tools import install_serve, observe_serve
from tools.serving_variants import variant_config

ROOT = Path(__file__).resolve().parents[1]
BASE_232B_SHA256 = {
    "csrc/kernels/attention_cv_official_decode.cpp": "817a5cbb7738e9ad572e39796838cd915a902c98301c452e21caefe6ceaebc22",
    "csrc/kernels/attention_cv_official_cluster4.cpp": "5069fa48b89a72d78b269b42c50be833007dd24e3bd05eceda0a62ceec722858",
    "oscar_ascend/integration/current_only_dispatch.py": "14b588f4a86c21660b023f3469cc17ecae84d592698b983cabf1faa7a7b08646",
    "oscar_ascend/integration/current_only_plan.py": "7f5fa1d9ab5a1f48c396eb9978f16b8fe8257a9115df5a4989b45825c4f2d366",
    "oscar_ascend/integration/current_attention.py": "79da9df9abecd6f02f386afac83b9c21e4d347fe02fcfe8376465f54192a823c",
    "oscar_ascend/ops/cv_dispatch.py": "40a27b068cf051a8dee562340c3ea9c2133311de48c422fde9243267c7cbf5b6",
    "oscar_ascend/integration/impl.py": "069ce6aa7038f1b9378efaf42eaaf4b9b8a400257b3304dd537442c7d49ebe3a",
    "oscar_ascend/runtime.py": "4660b4a6c60384528a1c238215618ecb12d38926c0ae80a38da0c42b71accd9f",
    "tools/target_cli.py": "ca282f3b52f866159f2f8ad7ab9d3f5b616cdd0b707dfd43f979b6caff6e07b5",
    "configs/target.json": "855b35cc37935ef52e61308a742c758b7ae7081b27d826ade33e6a89e00fc416",
}
RETAINED_34BC_SHA256 = {
    "csrc/kernels/attention_cv_official_cluster16.cpp": "1bcf9ccfe0181f3103b255f67589845a9d83b4e232baa9aa11561b61326d8264",
    "csrc/kernels/attention_official_finite.h": "776e507213aba7e7089dcc5c4706944859f4916929e9ad1688a22b90d1abd695",
}


def digest(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


@pytest.mark.parametrize("path,expected", sorted(BASE_232B_SHA256.items()))
def test_restored_and_unchanged_production_bytes_match_232b(path, expected):
    source = (ROOT / path).read_bytes()
    if path == 'configs/target.json':
        # User subsequently authorized only the default 8989 -> 7878 port.
        # Retain the original full-file hash after undoing that exact edit.
        assert source.count(b'"port": 7878') == 1
        source = source.replace(b'"port": 7878', b'"port": 8989')
    assert digest(source) == expected


@pytest.mark.parametrize("path,expected", sorted(RETAINED_34BC_SHA256.items()))
def test_complete_c16_and_certificate_remain_34bc(path, expected):
    assert digest((ROOT / path).read_bytes()) == expected


def test_default_and_packaged_generation_cannot_reenable_decode_cast_experiment():
    path = ROOT / "csrc/kernels/attention_cv_official_decode.cpp"
    expected = BASE_232B_SHA256[str(path.relative_to(ROOT))]
    assert inspect.signature(generated.decode).parameters["optimize"].default is False
    assert digest(generated.decode()) == expected
    assert digest(generated.decode(optimize=False)) == expected
    assert digest(generated.outputs()[path]) == expected
    assert digest(generated.cluster(16)) == RETAINED_34BC_SHA256[
        "csrc/kernels/attention_cv_official_cluster16.cpp"]
    assert digest(generated.cluster(4)) == BASE_232B_SHA256[
        "csrc/kernels/attention_cv_official_cluster4.cpp"]


def test_variant_source_only_adds_pinned_optr_and_retired_policy_cleanup():
    text = (ROOT / 'tools/serving_variants.py').read_text()
    cleanup = """    # User withdrew the unsuccessful exact-current delivery. A reused
    # generated config must not carry its retired policy into any variant.
    selected.pop('oscar_current_only_policy', None)
"""
    base = subprocess.check_output(['git', 'show',
        'ca46f57e0267f15aadba08d6913cef81b46e21b1:tools/serving_variants.py'],
        cwd=ROOT, text=True)
    assert base.count(cleanup) == 1
    assert base.index(cleanup) == base.index('    selected = dict(config)\n') + len(
        '    selected = dict(config)\n')
    # This hash is the accepted 614c source, not a hash of the withdrawal.
    assert digest(base.replace(cleanup, '')) == '354061eb242f8730ef0326674240b339f4f6539ad563c0924c8c0ceb3e7a5c4f'
    # The user's requested source commit is an independent, immutable oracle
    # for the reviewed OptR-only variant delta. Preserve new-6's retirement
    # immediately after copying the input; do not repin a hash of our edits.
    source = subprocess.check_output(['git', 'show',
        'e7dc8a92000e0d9b49800eadda6d1d16f978436f:tools/serving_variants.py'],
        cwd=ROOT, text=True)
    anchor = '    selected = dict(config)\n'
    assert source.count(anchor) == 1
    expected = source.replace(anchor, anchor + "    selected.pop('oscar_current_only_policy', None)\n")
    assert ast.dump(ast.parse(text)) == ast.dump(ast.parse(expected))
    # test_optr_migration_scope additionally executes all four variants,
    # both APC modes and legacy recipe/budget choices against the base.


@pytest.mark.parametrize('prefix', [False, True])
def test_candidate_configuration_matches_accepted_performance_base(prefix):
    expected = json.loads((ROOT / 'reports/precision_stability_20261028/accepted_candidate_config.json').read_text())
    expected['enable_prefix_caching'] = prefix
    expected.update(port=7878, rotation_refinement='optr')
    assert candidate_config(prefix) == expected


@pytest.mark.parametrize('variant', [None, 'candidate', 'native', 'baseline'])
@pytest.mark.parametrize('retired', ['native_exact', 'rotated_v4', 'stale_unknown_value'])
def test_reused_generated_config_cannot_reenable_retired_current_policy(variant, retired):
    clean = candidate_config()
    stale = dict(clean, oscar_current_only_policy=retired)
    assert variant_config(stale, variant) == variant_config(clean, variant)
    assert stale['oscar_current_only_policy'] == retired  # Do not mutate caller input.


def candidate_config(prefix=True):
    config = json.loads((ROOT / "configs/target.json").read_text())
    config["enable_prefix_caching"] = prefix
    return variant_config(config, "candidate")


def selected_operator(tokens, longest, *, later_draft=False):
    config = candidate_config()
    names = ("fast_unpack", "mixed_cv", "striped_cache", "decode_bundle",
             "c4_member_pipeline", "mdl_decode", "c16_member_pipeline",
             "hybrid_l1_decode", "c16_l1_reuse", "bf16_attention")
    flags = {name: config.get("experimental_" + name, False) for name in names}
    selected = select_cv_op(16, 6, 1, tokens, longest,
                            q1_draft=later_draft, **flags)
    assert config["experimental_official_math"] is True
    return OFFICIAL_CV_MAP[selected]


@pytest.mark.parametrize("tokens,longest,later_draft,operator", [
    (128, 4, False, "attention_cv_official_decode_out"),
    (512, 4, False, "attention_cv_official_decode_out"),
    (128, 1, True, "attention_cv_official_decode_q1_out"),
    (16384, 1, True, "attention_cv_official_decode_q1_out"),
    (1536, 1536, False, "attention_cv_official_cluster4_out"),
    (8191, 8191, False, "attention_cv_official_cluster4_out"),
    (8192, 8192, False, "attention_cv_official_cluster16_out"),
    (13824, 13824, False, "attention_cv_official_cluster16_out"),
    (16384, 16384, False, "attention_cv_official_cluster16_out"),
])
def test_actual_candidate_keeps_existing_shape_routes(tokens, longest, later_draft, operator):
    # These are pre-existing query-shape routes. Whole-file pins above reject
    # new context parameters or thresholds without keyword-based false alarms.
    assert selected_operator(tokens, longest, later_draft=later_draft) == operator


@pytest.mark.parametrize("fresh,tails", [
    (3072, (1024,)),
    (4608, (2304, 1280)),
    (13824, (1536, 1024)),
])
def test_short_prefill_current_only_and_c4_paths_are_fully_preserved(fresh, tails):
    # Isolated APC cold-request 4K/8K/16K production-policy replay shapes;
    # these are route fixtures, not a reconstruction of the user's traffic.
    plan = plan_current_suffix([0, fresh], [fresh], actual_tokens=fresh,
                               padded_tokens=fresh)
    assert plan.prefix_requests == 0 and plan.fresh_tokens == fresh
    context = fresh
    for count in tails:
        assert plan_current_suffix([0, count], [context + count], actual_tokens=count,
                                   padded_tokens=count) is None
        assert selected_operator(count, count) == "attention_cv_official_cluster4_out"
        context += count
    # 32K's warm13824 chunk is intentionally NOT described as restored.
    assert selected_operator(13824, 13824) == "attention_cv_official_cluster16_out"


@pytest.mark.parametrize("entry", [install_serve.main, observe_serve.main])
@pytest.mark.parametrize("prefix", [False, True])
def test_candidate_entrypoints_preserve_16k_budget_and_explicit_apc(entry, prefix, capsys):
    option = "--enable-prefix-caching" if prefix else "--no-enable-prefix-caching"
    assert entry(["--variant", "candidate", "--plan", option]) == 0
    report = json.loads(capsys.readouterr().out)
    argv = report["vllm_argv"]
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "16384"
    assert option in argv
    assert ("--no-enable-prefix-caching" if prefix else "--enable-prefix-caching") not in argv
    assert report["runtime_selection"]["prefix_caching"] is prefix
    assert report["runtime_selection"]["aligned_c16_operator"] == "attention_cv_official_cluster16_out"
    assert "--scheduler-cls" not in argv
    assert candidate_config(prefix)["max_num_batched_tokens"] == 16384
