"""Bounded diagnostic summary; never changes probe results or acceptance."""
import argparse
import json
from pathlib import Path


def summarize(report):
    keys = ('status', 'first_error', 'device_completion', 'precision', 'error_domain',
            'graph_capture', 'graph_replay', 'cv_performance', 'full_attention_performance')
    result = {k: report.get(k) for k in keys}
    result['progress'] = report.get('progress')
    result['cpu_screen'] = (report.get('cpu_screen') or {}).get('status')
    errors = (report.get('performance_gate') or {}).get('blocking_errors') or []
    result['blocking_error_count'] = len(errors)
    result['first_blocking_errors'] = errors[:12]
    result['cases'] = []
    def failures(node, path, found):
        if len(found) >= 4:
            return
        if isinstance(node, dict):
            if node.get('passed') is False and 'failed_elements' in node:
                found.append({'path': path, **{k: node.get(k) for k in
                    ('failed_elements', 'nonfinite_mismatches', 'max_abs')}})
            for k, v in node.items():
                failures(v, path + '.' + k, found)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                failures(v, path + '.' + str(i), found)
    cases = report.get('cases') or []
    result['completed_case_count'] = len(cases)
    failed_cases=[case for case in cases if case.get('status')!='passed']
    result['failed_case_count']=len(failed_cases)
    result['omitted_failed_cases']=max(0,len(failed_cases)-8)
    for case in failed_cases[:8]:
        row = {k: case.get(k) for k in ('case', 'status')}
        row['precision'] = (case.get('precision') or {}).get('passed')
        graph = case.get('graph') or {}
        row['graph_replay'] = graph.get('replay')
        row['graph_precision'] = graph.get('precision')
        if row['status'] != 'passed':
            row['first_failures'] = []
            failures(case.get('precision'), 'precision', row['first_failures'])
            failures(graph, 'graph', row['first_failures'])
        result['cases'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.report.read_text())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
