"""Archive #148/#155/#170-172: parse supplied observations, never invent per-case evidence.

Read-only attachment analysis. Raw precision metrics are parsed once per
evaluation, not counted again from the duplicated rounded summary tables.
Top-three selection is stated explicitly; tied cutoff alternatives are retained.
No model, NPU, benchmark rerun, production edit, or performance inference occurs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

METRICS = ("accuracy", "accuracy_easy", "accuracy_hard", "accuracy_short")
DEFAULTS = {
    "candidate": str(Path(__file__).with_name("candidate_client.txt")),
    "baseline": str(Path(__file__).with_name("baseline_client.txt")),
}


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values), "min": min(values),
            "max": max(values), "range": max(values)-min(values),
            "sample_stddev": statistics.stdev(values) if len(values)>1 else None}


def parse(path, role):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if "Current exp folder:" in line]
    rounds = []
    for number, begin in enumerate(starts, 1):
        end = starts[number] if number < len(starts) else len(lines)
        block = lines[begin:end]

        def one(pattern):
            found = [(begin+i+1, re.search(pattern, line)) for i,line in enumerate(block)]
            found = [(line,m) for line,m in found if m]
            if len(found) != 1:
                raise ValueError(f"{role} round{number}: expected exactly one {pattern}, got {len(found)}")
            return found[0]

        exp_line, exp = one(r"Current exp folder: (.+)$")
        metric_line, metric = one(r"Task vllm-api-stream-chat/LongBenchv2: (\{.+\})$")
        values = ast.literal_eval(metric[1])
        if set(values) != set(METRICS) or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in values.values()):
            raise ValueError("unexpected raw metric schema")
        progress_line, progress = one(r"^Progress:.*?\| (\d+)/(\d+) \[(\d+):(\d+)<")
        finish_line, finish = one(r"POST=(\d+).*RECV=(\d+).*FAIL=(\d+).*FINISH=(\d+)")
        elapsed_line, elapsed = one(r"Api infer task time elapsed: ([\d.]+)s")
        rate_line, rate = one(r"Traffic request rate: (\d+(?:\.\d+)?) RPS with burstiness (\d+(?:\.\d+)?)")
        concurrency_line, concurrency = one(r"Debug mode, run with concurrency: (\d+)")
        warmup_line, warmup = one(r"Warmup finished Total Count: (\d+) Success Count: (\d+) Failed Count: (\d+)")
        eval_done_line, _ = one(r"Evaluation tasks completed\.")
        versions = set(re.findall(r"LongBenchv2\s+([0-9]+)\s+accuracy\s+gen", "\n".join(block)))
        if len(versions)!=1:raise ValueError("missing/ambiguous dataset version")
        # Independently validate the duplicated, rounded printed metric tables.
        for name in METRICS:
            printed = re.findall(rf"LongBenchv2\s+\d+\s+{name}\s+gen\s+([\d.]+)", "\n".join(block))
            markdown = re.findall(rf"\| LongBenchv2 \| \d+ \| {name} \| gen \| ([\d.]+) \|", "\n".join(block))
            if len(printed)!=1 or len(markdown)!=1 or any(float(v)!=round(values[name],2) for v in printed+markdown):
                raise ValueError(f"raw/rounded summary mismatch: {role}/{number}/{name}")
        rounds.append({
            "round":number,"experiment":exp[1],"date":lines[begin][1:11],
            "dataset_version":next(iter(versions)),"metrics":values,
            "raw_metrics_literal":metric[1],
            "progress":{"finished":int(progress[1]),"total":int(progress[2]),
                        "duration_seconds":int(progress[3])*60+int(progress[4])},
            "transport":dict(zip(("post","recv","fail","finish"),map(int,finish.groups()))),
            "api_infer_task_seconds":float(elapsed[1]),"request_rate_rps":float(rate[1]),
            "burstiness":float(rate[2]),"concurrency":int(concurrency[1]),
            "warmup":dict(zip(("total","success","failed"),map(int,warmup.groups()))),
            "source_lines":{"experiment":exp_line,"metrics":metric_line,"progress":progress_line,
                            "finish":finish_line,"api_elapsed":elapsed_line,"rps":rate_line,
                            "concurrency":concurrency_line,"warmup":warmup_line,"evaluation_done":eval_done_line},
        })
    if not rounds:raise ValueError("no complete rounds")
    suspicious = [(i+1,line) for i,line in enumerate(lines) if re.search(r"Traceback|\[ERROR\]|parse error|failed to|timeout|truncat",line,re.I)]
    optional = {key:[i+1 for i,line in enumerate(lines) if re.search(pattern,line,re.I)] for key,pattern in {
        "temperature":"temperature", "top_p":r"top_p", "top_k":r"top_k", "seed":r"\bseed\b",
        "input_output_tokens":r"prompt_tokens|completion_tokens|total_tokens|input tokens|output tokens",
        "max_tokens":r"max_tokens|max_new_tokens", "finish_reason":"finish_reason",
        "prefix_cache_hit_rate":r"prefix cache hit|prefix_cache_hit", "loaded_commit":r"614c739|git commit|commit hash",
    }.items()}
    return {"role":role,"source":{"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),
            "line_count":len(lines),"contains_baseline_log_header":"cat baseline.log" in text},
            "rounds":rounds,"parse_warnings":suspicious,"optional_field_source_lines":optional}


def top3(rounds, metric):
    selected = sorted(rounds,key=lambda r:(-r["metrics"][metric],r["round"]))[:3]
    cutoff=selected[-1]["metrics"][metric]
    higher=[r for r in rounds if r["metrics"][metric]>cutoff]
    tied=[r for r in rounds if r["metrics"][metric]==cutoff]
    options=[higher+list(chosen) for chosen in itertools.combinations(tied,3-len(higher))]
    means={name:[statistics.mean(r["metrics"][name] for r in option) for option in options] for name in METRICS}
    return {"selection_metric":metric,"tie_break_for_display":"earliest chronological round",
            "selected_rounds":[r['round'] for r in selected],
            "cutoff_value":cutoff,"tied_at_cutoff_rounds":[r['round'] for r in tied],
            "possible_selected_round_sets":[sorted(r['round'] for r in option) for option in options],
            "means":{name:statistics.mean(r['metrics'][name] for r in selected) for name in METRICS},
            "means_over_cutoff_tie_choices":{name:{"min":min(v),"max":max(v)} for name,v in means.items()}}


def denominators(rounds, metric, upper):
    return [n for n in range(1,upper+1) if all(abs(r['metrics'][metric]*n/100-round(r['metrics'][metric]*n/100))<1e-9 for r in rounds)]


def make_report(data):
    all_rounds=[r for group in data.values() for r in group['rounds']]
    size=127
    candidates={m:denominators(all_rounds,m,size) for m in METRICS}
    pairs=[(e,h) for e in candidates['accuracy_easy'] for h in candidates['accuracy_hard'] if e+h==size]
    if pairs!=[(43,84)] or candidates['accuracy']!=[127] or candidates['accuracy_short']!=[127]:
        raise ValueError(f"unexpected inferred denominator possibilities {candidates}/{pairs}")
    denom={'accuracy':127,'accuracy_easy':43,'accuracy_hard':84,'accuracy_short':127}
    for group in data.values():
        rounds=group['rounds']
        for r in rounds:
            r['inferred_correct_counts']={m:round(r['metrics'][m]*denom[m]/100) for m in METRICS}
            if r['inferred_correct_counts']['accuracy_easy']+r['inferred_correct_counts']['accuracy_hard']!=r['inferred_correct_counts']['accuracy']:
                raise ValueError('subgroup counts do not add to overall')
        group['all_rounds_stats']={m:stats([r['metrics'][m] for r in rounds]) for m in METRICS}
        group['timing_stats']={key:stats([r[key] if key=='api_infer_task_seconds' else r['progress']['duration_seconds'] for r in rounds]) for key in ('api_infer_task_seconds','progress_duration_seconds')}
        group['top3_per_metric']={m:top3(rounds,m) for m in METRICS}
        group['top3_by_overall_accuracy']=top3(rounds,'accuracy')
    comparisons={}
    for m in METRICS:
        comparisons[m]={
            'all_rounds_baseline_minus_candidate_pp':data['baseline']['all_rounds_stats'][m]['mean']-data['candidate']['all_rounds_stats'][m]['mean'],
            'independent_top3_baseline_minus_candidate_pp':data['baseline']['top3_per_metric'][m]['means'][m]-data['candidate']['top3_per_metric'][m]['means'][m],
            'overall_top3_baseline_minus_candidate_pp':data['baseline']['top3_by_overall_accuracy']['means'][m]-data['candidate']['top3_by_overall_accuracy']['means'][m],
        }
    return {'schema':1,'date':'2026-10-28','sources':data,'comparisons':comparisons,
            'denominator_inference':{'counts':denom,'method':'Unique integer denominators <=127 jointly fitting all20 raw percentage vectors, with easy+hard=127; assumes ordinary correct/count accuracy and an easy/hard partition. Subgroup labels/counts were not explicitly printed.','alternatives_before_partition':candidates},
            'limits':['Only10 aggregate evaluations per source; no127 per-item answers, predictions, IDs or per-item scores are present.',
                      'No input/output token counts, finish_reason, empty-output count, truncation count or answer-parser error count are printed. FAIL=0 is request completion, not proof all answers parsed or were untruncated.',
                      'No temperature/top_p/top_k/seed/max_tokens values are printed; cannot establish identical sampling.',
                      'RPS differs(10 vs0), dates differ(10/22 vs10/28). Shell labels node205/node93 identify client terminals, not necessarily serving hosts. Full engine/model launch configurations are absent. Do not assign accuracy differences to RPS or infer different NPU machines.',
                      'Native/candidate roles follow user identification plus date/experiment/header evidence; these client logs do not contain loaded OSCAR/commit/PT identities.',
                      'Averages of per-metric independent top3 can combine different rounds. Unified overall top3 retains cutoff ties, with chronological tie break only for display.',
                      'Top3 summaries are selected best runs and do not replace the full10-run variability summary.'],
            'NPU':'not_run','performance_tests':'not_run','production_modified':False}


def markdown(report):
    lines=['# LongBenchv2 附件定量核对（2026-10-28）','',
           '仅解析用户提供的客户端日志；未重跑模型、未连接真机、未改生产。每组10个完整轮次；每轮127/127、FAIL=0。原始指标仅从evaluation字典抽取，重复展示的两位小数表不重复计数。','',
           '## 完整十轮','', '| 组 | 轮 | overall | easy | hard | short | 推断正确数 overall/easy/hard | 进度条时间 | Api infer task(s) |', '|---|---:|---:|---:|---:|---:|---|---|---:|']
    for role in ('baseline','candidate'):
        g=report['sources'][role]
        for r in g['rounds']:
            v=r['metrics'];c=r['inferred_correct_counts'];s=r['progress']['duration_seconds']
            lines.append(f"| {role} | {r['round']} | "+' | '.join(f'{v[m]:.8f}' for m in METRICS)+f" | {c['accuracy']}/{c['accuracy_easy']}/{c['accuracy_hard']} | {s//60:02d}:{s%60:02d} | {r['api_infer_task_seconds']:.2f} |")
    lines += ['','## 两种top3口径','', '| 指标 | baseline独立top3 | candidate独立top3 | 差(pp) | baseline按overall选轮 | candidate按overall选轮 | 差(pp) |','|---|---:|---:|---:|---:|---:|---:|']
    for m in METRICS:
        b=report['sources']['baseline'];c=report['sources']['candidate'];d=report['comparisons'][m]
        vals=(b['top3_per_metric'][m]['means'][m],c['top3_per_metric'][m]['means'][m],d['independent_top3_baseline_minus_candidate_pp'],b['top3_by_overall_accuracy']['means'][m],c['top3_by_overall_accuracy']['means'][m],d['overall_top3_baseline_minus_candidate_pp'])
        lines.append('| '+m+' | '+' | '.join(f'{v:.8f}' for v in vals)+' |')
    lines += ['', '按overall统一选轮：baseline=[1,5,8]；candidate显示=[2,8,3]，其中第三名3/5/9轮并列；选择3或9时同表，选5时easy/hard均值分别48.83720930/39.68253968。overall与short均值不变。不能将任一tie-break当成用户已指定。', '', '## 全部轮次统计','', '| 指标 | baseline均值 | candidate均值 | 差(pp) | baseline范围 | candidate范围 | baseline样本标准差 | candidate样本标准差 |','|---|---:|---:|---:|---|---|---:|---:|']
    for m in METRICS:
        b=report['sources']['baseline']['all_rounds_stats'][m];c=report['sources']['candidate']['all_rounds_stats'][m]
        lines.append(f"| {m} | {b['mean']:.8f} | {c['mean']:.8f} | {b['mean']-c['mean']:.8f} | {b['min']:.8f}–{b['max']:.8f} | {c['min']:.8f}–{c['max']:.8f} | {b['sample_stddev']:.8f} | {c['sample_stddev']:.8f} |")
    lines += ['', '这两组附件中，candidate总体均值和最高轮次低于baseline；baseline十轮overall范围与标准差反而更大。因此这些日志不支持“原生更稳定”的全十轮统计结论，不能只挑最高三轮来描述稳定性。', '', '## 时间统计','', '| 口径 | baseline十轮均值(s) | candidate十轮均值(s) | candidate/base |', '|---|---:|---:|---:|']
    for key in ('progress_duration_seconds','api_infer_task_seconds'):
        b=report['sources']['baseline']['timing_stats'][key]['mean'];c=report['sources']['candidate']['timing_stats'][key]['mean']
        lines.append(f'| {key} | {b:.3f} | {c:.3f} | {c/b:.8f} |')
    lines += ['', '首轮进度条是14:02和18:42；十轮进度条均值为12:27.3和18:26.2，不能把首轮时间当成全部轮次均值。Api infer task为日志另一独立口径，不和进度条时间混算。', '', '## 证据边界','',
              '两组均为dataset version324293、并发32、burstiness1.0、127题。baseline为10/22、10 RPS，文本首行cat baseline.log；candidate为10/28、0 RPS并明确同时提交。终端标签分别node205/node93，仅标识客户端终端，不能据此推断推理服务器或NPU机器不同。附件没有完整服务配置；不能据RPS差异归咎精度。', '',
              '在普通correct/count计分且easy/hard互斥覆盖全集假设下，全部20个原始百分比唯一对应overall127、easy43、hard84、short127；各轮easy正确数+hard正确数=overall正确数。它是分母推断，不是附件直接打印的分组count。', '']
    for limit in report['limits']:lines.append('- '+limit)
    lines += ['', '## 来源','']
    for role,g in report['sources'].items():
        src=g['source'];lines.append(f"- {role}: `{src['path']}`；{src['bytes']} bytes；SHA256 `{src['sha256']}`。各轮具体原文行号在log_analysis.json中。")
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for role,path in DEFAULTS.items():parser.add_argument('--'+role,type=Path,default=Path(path))
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    report=make_report({role:parse(getattr(args,role),role) for role in DEFAULTS})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'log_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (args.output_dir/'log_analysis.md').write_text(markdown(report))
    print(json.dumps({'rounds':{r:len(g['rounds']) for r,g in report['sources'].items()},'comparisons':report['comparisons']},indent=2))


if __name__=='__main__':main()
