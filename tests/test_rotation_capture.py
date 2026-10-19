"""Native API/lifetime host doubles only, not NPU or model-quality evidence."""
import csv,json,sys,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
from oscar_ascend.integration.rotation_capture import RotationCaptureWorker,_capture_forward
from tools import calibrate_model_rotations as cli


class Tape:
    def __init__(self,**geometry):self.geometry=geometry;self.rows=[];self.count=0
    def observe(self,q,k,v):self.rows.append((q.clone(),k.clone(),v.clone()));self.count+=len(q)
    def export_payload(self):
        if self.count!=self.geometry['token_budget']:raise ValueError('incomplete samples')
        return {'count':self.count}


class Native:
    __module__='vllm_ascend.attention.attention_v1'
    head_size=8;num_heads=4;num_kv_heads=2
    def forward(self,layer,q,k,v,cache,meta,*args,**kwargs):return kwargs.get('output','native-result')


def metadata(n=3,*,padding=False):
    return NS(num_actual_tokens=n,num_prefills=2 if padding else 1,num_decodes=0,num_decode_tokens=0,
        attn_state=NS(name='PrefillNoCache'),actual_seq_lengths_q=[n,n+2] if padding else [n],
        seq_lens_list=[n,1] if padding else [n])


def tensors(n=5):return torch.arange(n*32.).reshape(n,32),torch.zeros(n,16),torch.ones(n,16)


@pytest.fixture
def worker(monkeypatch,tmp_path):
    from oscar_ascend import rotation_calibration
    monkeypatch.setenv('OSCAR_ENABLED','0')
    distributed=types.ModuleType('vllm.distributed');distributed.get_tensor_model_parallel_rank=lambda:0
    monkeypatch.setitem(sys.modules,'vllm.distributed',distributed)
    monkeypatch.setattr(rotation_calibration,'OfflineLayerAccumulator',Tape)
    impl=Native();layer=NS(layer_name='language_model.model.layers.3.self_attn.attn',impl=impl)
    model=NS(named_modules=lambda:[('layer',layer)])
    instance=RotationCaptureWorker();instance.get_model=lambda:model
    spec={'layer_ids':[3],'head_dim':8,'tp_size':1,'global_q_heads':4,'global_kv_heads':2,
        'token_count':3,'prompt_lengths':[3],'prompt_sha256':'prompt','model_fingerprint':'model',
        'output_dir':str(tmp_path)}
    return instance,impl,spec


def test_afterwarmup_capture_excludes_padding_and_restores_native_method(worker):
    instance,impl,spec=worker;q,k,v=tensors()
    assert impl.forward(None,q,k,v,None,None)=='native-result'
    result=instance.oscar_rotation_capture_start(spec)
    assert result['state']=='armed_after_warmup'
    collector=instance._oscar_rotation_capture[2][3]
    q[3:]=float('nan')
    assert impl.forward(None,q,k,v,None,metadata(padding=True),output='real-native')=='real-native'
    assert collector.count==3 and torch.isfinite(collector.rows[0][0]).all()
    assert collector.rows[0][0].shape==(3,4,8)
    finished=instance.oscar_rotation_capture_finish()
    assert finished['status']=='complete' and Path(finished['path']).is_file()
    assert 'forward' not in impl.__dict__ and instance._oscar_rotation_capture is None
    assert instance.oscar_rotation_capture_abort()['status']=='not_armed'


@pytest.mark.parametrize('case',['warmup','decode','history','chunked','duplicate','key-short'])
def test_nonprefill_and_repeat_frames_never_enter_samples(worker,case):
    instance,impl,spec=worker;instance.oscar_rotation_capture_start(spec);q,k,v=tensors();meta=metadata()
    if case=='warmup':meta=None
    elif case=='decode':meta.attn_state=NS(name='DecodeOnly')
    elif case=='history':meta.seq_lens_list=[8]
    elif case=='chunked':meta.num_actual_tokens=2;meta.actual_seq_lengths_q=[2];meta.seq_lens_list=[3]
    elif case=='key-short':k=k[:2]
    else:impl.forward(None,q,k,v,None,meta)
    with pytest.raises(RuntimeError):impl.forward(None,q,k,v,None,meta)
    collector=instance._oscar_rotation_capture[2][3]
    assert collector.count==(3 if case=='duplicate' else 0)
    instance.oscar_rotation_capture_abort();assert 'forward' not in impl.__dict__


def test_partial_start_and_incomplete_finish_restore_owned_hooks(worker):
    instance,impl,spec=worker
    with pytest.raises(RuntimeError,match='coverage'):
        instance.oscar_rotation_capture_start({**spec,'layer_ids':[3,7]})
    assert 'forward' not in impl.__dict__
    instance.oscar_rotation_capture_start(spec)
    with pytest.raises(ValueError,match='incomplete'):instance.oscar_rotation_capture_finish()
    assert 'forward' not in impl.__dict__ and instance._oscar_rotation_capture is None


def test_sampler_rejects_production_and_reentrant_arm(worker,monkeypatch):
    instance,_,spec=worker;monkeypatch.setenv('OSCAR_ENABLED','1')
    with pytest.raises(RuntimeError,match='isolated native'):instance.oscar_rotation_capture_start(spec)
    monkeypatch.setenv('OSCAR_ENABLED','0');instance.oscar_rotation_capture_start(spec)
    with pytest.raises(RuntimeError,match='already armed'):instance.oscar_rotation_capture_start(spec)
    instance.oscar_rotation_capture_abort()


def test_gpqa_seed0_and_canonical_prompt_id_hash(tmp_path):
    path=tmp_path/'gpqa.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cli.GPQA_FIELDS);writer.writeheader()
        writer.writerow(dict(zip(cli.GPQA_FIELDS,['question','correct','wrong1','wrong2','wrong3'])))
    text=cli.load_messages(path)[0][0]['content']
    assert text.endswith('question\n\nA) wrong3\nB) wrong1\nC) correct\nD) wrong2')
    import hashlib
    assert cli.prompt_ids_sha256([[1,2],[3]])==hashlib.sha256(b'[[1,2],[3]]').hexdigest()


def test_whole_prompt_budget_stops_without_truncating_or_skipping():
    class Tokenizer:
        def apply_chat_template(self,ids,**kwargs):
            assert kwargs=={'tokenize':True,'add_generation_prompt':True,'return_dict':False}
            return {'input_ids':[ids]}
    prompts=[list(range(n)) for n in (10000,12000,9000,1)]
    selected=cli.render_prompt_ids(Tokenizer(),prompts,30000,16384)
    assert selected==prompts[:2]
    with pytest.raises(ValueError,match='one prefill chunk'):
        cli.render_prompt_ids(Tokenizer(),[list(range(17000))],30000,16384)


@pytest.mark.parametrize('failure',[None,'generate','start'])
@pytest.mark.parametrize('bundled', [False, True])
def test_public_llm_rpc_sampling_and_shutdown_are_isolated(monkeypatch,tmp_path,failure,bundled):
    from tools import model_config
    monkeypatch.setattr(model_config,'apply_model_options',lambda value,**_:value)
    monkeypatch.setenv('OMP_NUM_THREADS','1')
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps({'num_hidden_layers':4,'full_attention_interval':4,
        'head_dim':8,'num_attention_heads':4,'num_key_value_heads':2}))
    source=tmp_path/'prompts.jsonl';source.write_text('"promptA"\n"promptB"\n')
    config={'model':str(model),'devices':[0,1,2,3],'tensor_parallel_size':1,'max_model_len':262144,
        'max_num_batched_tokens':16384,'gpu_memory_utilization':.7,'rotation_calibration_prompts':str(source)}
    if bundled:
        import urllib.request
        monkeypatch.setattr(urllib.request, 'urlopen', lambda *_a, **_k: pytest.fail('startup must be offline'))
        del config['rotation_calibration_prompts']
        # Two complete three-token prompts; remaining bundled records exceed
        # the budget. The model below remains an explicit host double.
        config['rotation_calibration_tokens'] = 6
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config));before=config_path.read_bytes()
    transformer=types.ModuleType('transformers')
    transformer.AutoTokenizer=NS(from_pretrained=lambda *a,**k:NS(apply_chat_template=lambda *_a,**_k:[7,8,9]))
    monkeypatch.setitem(sys.modules,'transformers',transformer)
    calls=[];options={};run_spec={}
    class LLM:
        def __init__(self,**kwargs):
            options.update(kwargs);calls.append('warmup')
            self.llm_engine=NS(engine_core=NS(shutdown=lambda **_:calls.append('shutdown')))
        def collective_rpc(self,name,**kwargs):
            calls.append(name)
            if name.endswith('_start'):
                run_spec.update(kwargs['args'][0])
                if failure=='start':raise RuntimeError('native start failed')
                return [{'rank':0,'layers':[3],'state':'armed_after_warmup'}]
            if name.endswith('_finish'):
                path=Path(run_spec['output_dir'])/'rank0.pt';path.write_bytes(b'host-double-not-a-model-sample')
                return [{'rank':0,'layers':[3],'tokens':6,'status':'complete','path':str(path)}]
            return [{'status':'not_armed'}]
        def generate(self,prompts,sampling,**kwargs):
            assert prompts==[{'prompt_token_ids':[7,8,9]}] and sampling=={'temperature':0,'max_tokens':1}
            calls.append('prefill')
            if failure=='generate':raise RuntimeError('native forward failed')
            return [NS(finished=True)]
    vllm=types.ModuleType('vllm');vllm.LLM=LLM;vllm.SamplingParams=lambda **kwargs:kwargs
    monkeypatch.setitem(sys.modules,'vllm',vllm)
    out=tmp_path/'capture'
    if failure:
        with pytest.raises(RuntimeError,match='native .* failed'):cli.collect(config_path,out)
        assert not (out/'capture-manifest.json').exists()
    else:
        cli.collect(config_path,out)
        report=json.loads((out/'capture-manifest.json').read_text())
        assert report['token_count']==6 and report['prompt_lengths']==[3,3] and report['rank_sha256']
        if bundled:
            from tools.calibration_data import GPQA_SHA256
            assert report['prompt_source_sha256'] == GPQA_SHA256
            assert Path(report['prompt_source']) == out/'gpqa_diamond.csv'
        assert calls.count('prefill')==2
    assert calls[-2:]==['oscar_rotation_capture_abort','shutdown']
    assert calls[0]=='warmup' and config_path.read_bytes()==before
    assert options['worker_extension_cls'].endswith('.RotationCaptureWorker')
    assert options['enforce_eager'] and not options['enable_prefix_caching'] and not options['enable_chunked_prefill']
    assert options['max_model_len']==4 and options['max_num_batched_tokens']==16384
    assert 'speculative_config' not in options and options['dtype']=='bfloat16'
