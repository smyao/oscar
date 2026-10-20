"""Archive #70-73/#145/#164: derive only the supplied C16 diagnostic scope.

No model, SDK or NPU is imported. Printed call milliseconds are rounded;
actor milliseconds are recomputed from recorded ticks/50MHz. No cross-actor
time is added, and diagnostic shares are never applied to the old 1199ms.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'user_summary.txt'
    text = source.read_text()
    calls, actors, consistency, ranks = [], {}, {}, []
    validation = None
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith('C16_VALIDATION '):
            validation = {'line': number, 'text': line}
        if line.startswith('C16_CONSISTENCY '):
            rank = int(re.search(r'rank=(\d+)', line)[1])
            winners = json.loads(re.search(r'winners=(\{.*?\})', line)[1])
            consistency[rank] = {'line': number, 'winners': winners}
        if line.startswith('C16_CALL '):
            match = re.search(r'r(\d+)/c(\d+) (target|draft)\[\d+\].*baseline_ms=([\d.]+) diag_ms=([\d.]+).*valid=(\w+) timebase=(\w+)', line)
            assert match is not None
            rank, call, role, base, diag, valid, clock = match.groups()
            assert valid == clock == 'True'
            assert 'N/S=16384/1 req=1 context/query=32768/16384' in line
            calls.append({'rank': int(rank), 'call': int(call), 'role': role,
                          'baseline_ms': float(base), 'diagnostic_ms': float(diag), 'line': number})
        if line.startswith('C16_ACTOR '):
            tag = line.split()[1]
            cycles = int(re.search(r'total_cycles=(\d+)', line)[1])
            assert 'closed=True' in line
            actors[tag] = {'line': number, 'total_cycles': cycles, 'total_ms': cycles / 50000,
                           'other_cycles': int(re.search(r'other=(\d+)', line)[1]),
                           'phases': {}, 'categories': {}}
        if line.startswith('C16_CATEGORIES '):
            actor = actors[line.split()[1]]
            actor['category_line'] = number
            for name, cycles in re.findall(r'(compute|data_movement|wait|mixed)=(\d+)cy', line):
                cycles = int(cycles)
                actor['categories'][name] = {'cycles': cycles, 'ms': cycles / 50000,
                                            'percent': cycles * 100 / actor['total_cycles']}
        if line.startswith('C16_PHASES '):
            actor = actors[line.split()[1]]
            phases = re.findall(r'(\w+)=[\d.]+ms/(\d+)cy/(\d+)x\([\d.]+%,(\w+)\)', line)
            assert phases
            for name, cycles, count, category in phases:
                cycles = int(cycles)
                assert name not in actor['phases']
                actor['phases'][name] = {'cycles': cycles, 'ms': cycles / 50000, 'count': int(count),
                                         'percent': cycles * 100 / actor['total_cycles'],
                                         'category': category, 'line': number}
    assert validation and 'issues=0 calls=68' in validation['text']
    assert len(calls) == 68 and len(actors) == 8 and len(consistency) == 4
    assert 'status=complete source_status=completed' in text
    for rank in range(4):
        rows = [call for call in calls if call['rank'] == rank]
        assert len(rows) == 17 and sum(call['role'] == 'target' for call in rows) == 16
        assert len({call['call'] for call in rows}) == 17
        assert consistency[rank]['winners'] == {'softmax_math': 16}
        baseline = sum(call['baseline_ms'] for call in rows)
        diagnostic = sum(call['diagnostic_ms'] for call in rows)
        ranks.append({'rank': rank, 'baseline_sum_from_rounded_calls_ms': baseline,
                      'diagnostic_sum_from_rounded_calls_ms': diagnostic,
                      'diagnostic_overhead_percent': (diagnostic / baseline - 1) * 100,
                      'versus_previous_c16_percent': (baseline / 1189.457 - 1) * 100})
    for actor in actors.values():
        assert sum(p['cycles'] for p in actor['phases'].values()) + actor['other_cycles'] == actor['total_cycles']
        for category, value in actor['categories'].items():
            assert sum(p['cycles'] for p in actor['phases'].values() if p['category'] == category) == value['cycles']
    aiv = {name: actor for name, actor in actors.items() if '/AIV' in name}
    for actor in aiv.values():
        for label, names in {
            'unpack_kv': ['unpack_k', 'unpack_v'],
            'softmax_plus_all_casts': ['softmax_math', 'bf16_cast'],
            'state_roundtrip': ['state_restore', 'state_spill'],
            'score_and_p_transfers': ['softmax_score_load', 'softmax_publish_p'],
            'kv_read_unpack_publish_excluding_cast': ['history_load', 'unpack_k', 'unpack_v', 'publish_kv'],
        }.items():
            cycles = sum(actor['phases'][name]['cycles'] for name in names)
            actor[label] = {'cycles': cycles, 'ms': cycles / 50000,
                            'percent': cycles * 100 / actor['total_cycles']}
        actor['compute_over_explicit_movement'] = actor['categories']['compute']['cycles'] / actor['categories']['data_movement']['cycles']
    result = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'lines': len(text.splitlines()), 'validation': validation,
              'calls': calls, 'ranks': ranks, 'consistency': consistency, 'representative_actors': actors,
              'scope': 'four representative calls, actor-local exclusive diagnostic intervals; shares are not production-wall or PMU saturation',
              'rounding': 'call-ms sums use printed rounded inputs; actor intervals use supplied raw cycles'}
    (ROOT / 'derived_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'ranks': ranks, 'representative_aiv': {
        name: {'softmax_ms': a['phases']['softmax_math']['ms'], 'cast_ms': a['phases']['bf16_cast']['ms'],
               'unpack_ms': a['unpack_kv']['ms'], 'history_ms': a['phases']['history_load']['ms'],
               'compute_over_movement': a['compute_over_explicit_movement']}
        for name, a in aiv.items()}}, indent=2))


if __name__ == '__main__':
    main()
