"""Archive #129/#148/#157: scoped source transform and native-task proof identity."""
import json
from pathlib import Path
import pytest
from csrc.experiments.c16_anchor_skip_20261015 import generate as gen
from csrc.experiments.c16_anchor_skip_20261015.host_replay import PREPARE_SOURCE_SHA,sha


def test_transform_preserves_first_pass_and_every_other_method():
    text=(gen.ROOT/gen.SOURCE).read_text();new,proof=gen.apply_skip(text)
    assert proof['first_pass_byte_identical'] and proof['outside_Process_byte_identical']
    assert new.replace(gen.AFTER,gen.BEFORE)==text
    assert new.count('token += Min64(bucketWidth,end-token)-1;')==1


def test_changed_process_rejected_instead_of_extending_proof_silently():
    text=(gen.ROOT/gen.SOURCE).read_text().replace('const int64_t scheduleTile=4;', 'const int64_t scheduleTile=8;')
    with pytest.raises(ValueError,match='re-prove native reachability'):gen.apply_skip(text)


def test_composition_accepts_changes_outside_process_without_changing_them():
    text=(gen.ROOT/gen.SOURCE).read_text();tag='// Independent finite transform owns only Softmax.\n'
    new,_=gen.apply_skip(tag+text)
    assert new.startswith(tag)
    assert new.replace(gen.AFTER,gen.BEFORE)==tag+text


def test_proof_uses_pinned_producer_and_actual_cpu_prepare_witnesses():
    assert sha((gen.ROOT/'csrc/kernels/attention_tasks.cpp').read_bytes())==PREPARE_SOURCE_SHA
    witnesses=json.loads((gen.HERE/'fixtures/provenance.json').read_text())
    assert {w['task_rows'] for w in witnesses.values()}=={1008,2016}
    assert all(w['attention_numerics_claimed'] is False for w in witnesses.values())
    assert 'arbitrary_overlapping_table_not_covered' in (gen.HERE/'host_replay.cpp').read_text()
