"""Archive #94/#129/#145/#154: official zero-dump mask, no FD mutation.

Logging-only adapter for fixed historical replayers. Remove their entire
readlink/dup2 daemon before execution. All kernels, inputs, numeric gates and
task/AIC/AIV profile gates remain byte-for-byte in the original runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(data):return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def transform(source):
    text=source;edits=[]
    def edit(old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError(('logging-only anchor',old[:60],text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count))
    if '    def quiet_dumps():' in text:
        begin=text.index('    def quiet_dumps():');end=text.index('    library=',begin)
    elif '    def quiet():' in text:
        begin=text.index('    def quiet():');end=text.index('    acl=C.CDLL',begin)
    else:raise ValueError('known historical polling daemon required')
    old=text[begin:end]
    if 'os.dup2' not in old or 'thread.start()' not in old:raise ValueError('daemon shape changed')
    edit(old,'    # Official LOG.core_enable_mask=0 suppresses dumps; no polling thread exists.\n')
    if 'stop=threading.Event();' in text:
        edit('stop=threading.Event();','safe_logging_no_event=None;',text.count('stop=threading.Event();'))
    # The small C16 runner places the event at a line end instead.
    if ';stop=threading.Event()' in text:edit(';stop=threading.Event()',';safe_logging_no_event=None')
    edit('stop.set();thread.join(timeout=2)','safe_logging_no_daemon_to_join=True')
    if 'os.dup2' in text or '/proc/self/fd' in text:raise ValueError('residual descriptor mutation')
    restored=text
    for old,new,count in reversed(edits):
        if restored.count(new)!=count:raise ValueError('logging inverse identity changed')
        restored=restored.replace(new,old)
    if restored!=source:raise ValueError('original runner not restored')
    return text


def main():
    p=argparse.ArgumentParser();p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();source=a.runner.read_text()
    if sha(source)!=a.expected_sha256:raise ValueError('fixed original runner changed')
    config=Path(os.environ['CAMODEL_CONFIG_PATH'])
    detail=json.loads((config/'config.json').read_text());stars=json.loads((config/'config_stars.json').read_text())
    if detail['LOG']['core_enable_mask']!=['0x0']:raise ValueError('official zero-dump logging required')
    arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    if '--output' not in arguments:raise ValueError('owned output required')
    output=Path(arguments[arguments.index('--output')+1]).resolve()
    text=transform(source)
    record={'status':'logging_only_adapter','runner':str(a.runner.resolve()),'original_sha256':sha(source),
        'adapted_sha256':sha(text),'inverse_runner_bytes':True,'polling_thread_created':False,
        'descriptor_mutation':False,'LOG':detail['LOG'],'L2CACHE':detail['L2CACHE'],'config_stars':stars,
        'numeric_and_profile_gates':'original runner unchanged','performance_claim':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_name(output.name+'-safe-logging.json').write_text(json.dumps(record,indent=2)+'\n')
    sys.argv=[str(a.runner),*arguments]
    exec(compile(text,str(a.runner),'exec'),{'__file__':str(a.runner.resolve()),'__name__':'__main__'})


if __name__=='__main__':main()
