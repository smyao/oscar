from tools.summarize_bf16 import summarize


def test_finite_accuracy_failure_is_visible_without_traceback():
    metric={'passed':False,'failed_elements':1536,'max_abs':3.6953125,'nonfinite_mismatches':0}
    report={'status':'failed','device_completion':'passed','precision':'failed',
            'performance_gate':{'blocking_errors':['accuracy']*100},
            'cases':[{'case':'cancel','status':'failed','precision':{'passed':False,'comparisons':{'candidate':{'output':metric}}}}]}
    s=summarize(report)
    assert s['blocking_error_count']==100 and len(s['first_blocking_errors'])==12
    assert s['cases'][0]['first_failures'][0]['max_abs']==3.6953125
    assert s['device_completion']=='passed' and s['status']=='failed'


def test_exception_partial_report_and_graph_failures_remain_distinct():
    assert summarize({'status':'failed','first_error':'compile'})['first_error']=='compile'
    r={'cases':[{'case':'q1','status':'failed','precision':{'passed':True},'graph':{'replay':'passed','precision':'failed'}}]}
    row=summarize(r)['cases'][0]
    assert row['precision'] is True and row['graph_precision']=='failed'
