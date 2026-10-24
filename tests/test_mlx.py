"""Optional native MLX tests. Run in an MLX environment; no Torch dependency."""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
import mlx.nn as nn
from mlx_lm.models.llama import Model, ModelArgs

import sr_association_lab as lab


def tiny_model(seed=713,quantized=False):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    mx.set_default_device(mx.cpu);mx.random.seed(seed)
    symbols=['<pad>','<bos>','<eos>','<unk>']+sorted(pre_tokenizers.ByteLevel.alphabet())
    backend=Tokenizer(models.BPE({t:i for i,t in enumerate(symbols)},merges=[],unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);backend.decoder=decoders.ByteLevel()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<bos>',eos_token='<eos>',unk_token='<unk>',pad_token='<pad>',model_max_length=2048)
    args=ModelArgs(model_type='llama',hidden_size=64,num_hidden_layers=2,intermediate_size=128,num_attention_heads=4,num_key_value_heads=2,rms_norm_eps=1e-5,vocab_size=len(tok),max_position_embeddings=2048,tie_word_embeddings=False)
    m=Model(args);m.eval()
    if quantized: nn.quantize(m,group_size=32,bits=4)
    mx.eval(m.parameters());mx.synchronize()
    config=dataclasses.asdict(args);config.update(bos_token_id=1,eos_token_id=2,pad_token_id=0)
    if quantized: config['quantization']={'group_size':32,'bits':4}
    return lab.MLXModelHandle(m,config,eos_ids=[2]),tok


def test_native_cache_and_hooks_match_full_prefix():
    handle,tok=tiny_model()
    hooks=lab.MLXHookRecorder(handle,layers='all',patterns=['model.layers.0.self_attn','model.layers.0.self_attn.q_proj'])
    original_paths=list(hooks.sites)
    cache=None
    prefix=[]
    for ids in ([4,5,6],[7],[8,9],[10]):
        prefix.extend(ids)
        raw,states,dtypes,raw_dtype,fms,cms,cache=lab.mlx_record_forward(handle,ids,len(prefix),cache,True,hooks)
        assert fms>0 and cms>=0
        assert all(c.offset==len(prefix) for c in cache)
        # Independent full-prefix execution; no previous cache.
        hooks.clear();out=handle.model(mx.array([prefix]),cache=None)
        mx.eval(out,list(hooks.snapshots.values()));expected_states,_=hooks.take()
        np.testing.assert_allclose(raw,np.array(out[0,-1,:]),rtol=2e-5,atol=2e-5)
        for name in states: np.testing.assert_allclose(states[name],expected_states[name],rtol=2e-5,atol=2e-5)
        assert list(states)==[x['name'] for x in original_paths]
    hooks.close()
    assert handle._active_recorder is None
    assert all(type(x).__name__!='OutputObserver' for _,x in handle.model.named_modules())


def test_nested_wrappers_leave_output_and_objects_unchanged():
    handle,tok=tiny_model();native=handle.model
    old={n:m for n,m in native.named_modules()};inputs=mx.array([[4,5,6]])
    before=native(inputs);mx.eval(before)
    h=lab.MLXHookRecorder(handle,layers='all',patterns=['model.layers.0.self_attn','model.layers.0.self_attn.q_proj'])
    out=native(inputs);mx.eval(out,list(h.snapshots.values()));h.take();h.close();h.close()
    assert dict(native.named_modules())==old
    np.testing.assert_array_equal(np.array(before),np.array(out))


def test_bad_hooks_restoration_and_owner_check():
    model,tok=tiny_model()
    with pytest.raises(ValueError,match='matched no modules'):lab.MLXHookRecorder(model,patterns=['does.not.exist'])
    assert model._active_recorder is None
    h=lab.MLXHookRecorder(model,layers='all')
    with pytest.raises(ValueError,match='current MLX hook'):lab.MLXHookRecorder(model,layers='all')
    h.close()
    with pytest.raises(ValueError,match='Layer index'):lab.MLXHookRecorder(model,layers='99')
    assert model._active_recorder is None


def test_isolated_encoding_matches_native_hidden_oracle():
    handle,tok=tiny_model();texts=['cat','a window']
    # Make an unrelated nonempty native cache first; encoder must not see it.
    h=lab.MLXHookRecorder(handle,enabled=False)
    *_,cache=lab.mlx_record_forward(handle,[4,6,8],3,None,True,h);h.close()
    actual,ids,meta=lab.mlx_encode_isolated(handle,tok,texts,max_context=100)
    expected=[]
    for seq in ids:
        hidden=handle.model.model(mx.array([seq]),cache=None)
        mx.eval(hidden);expected.append(np.array(hidden[0]).mean(axis=0))
    np.testing.assert_allclose(actual,np.stack(expected),rtol=2e-6,atol=2e-6)
    assert meta['hidden_site']=='model.norm' and not meta['native_cache_used']
    assert all(c.offset==3 for c in cache)
    assert handle._active_recorder is None
    with pytest.raises(ValueError,match='empty'):lab.mlx_encode_isolated(handle,tok,[''])
    with pytest.raises(ValueError,match='exceeds context'):lab.mlx_encode_isolated(handle,tok,['window'],max_context=2)
    assert handle._active_recorder is None


@pytest.mark.parametrize('quantized',[False,True])
def test_local_loader_quantization_and_tokenizer(tmp_path,quantized):
    original,tok=tiny_model(quantized=quantized)
    original.model.save_weights(str(tmp_path/'model.safetensors'))
    (tmp_path/'config.json').write_text(json.dumps(original.config_dict))
    tok.save_pretrained(tmp_path)
    a=argparse.Namespace(model=str(tmp_path),tokenizer=None,revision=None,device='cpu',dtype='auto',device_map=None,load_in_4bit=False,attn_implementation='eager',adapter=None,local_files_only=True,trust_remote_code=False,seed=31)
    loaded,tokenizer=lab.load_mlx_model(a)
    ids=tokenizer('cat',add_special_tokens=False)['input_ids']
    assert ids==tok('cat',add_special_tokens=False)['input_ids']
    assert 2 in loaded.generation_config.eos_token_id
    x=mx.array([ids]);expected=original.model(x);actual=loaded.model(x);mx.eval(expected,actual)
    np.testing.assert_array_equal(np.array(expected),np.array(actual))
    h=lab.MLXHookRecorder(loaded,layers='all');r=lab.mlx_record_forward(loaded,ids,len(ids),None,True,h);h.close()
    np.testing.assert_allclose(r[0],np.array(actual[0,-1,:]),atol=2e-5,rtol=2e-5)


def test_loader_rejects_backend_mismatches_before_loading():
    for kwargs,match in [({'device_map':'auto'},'device-map'),({'load_in_4bit':True},'bitsandbytes'),({'device':'cuda'},'MLX --device'),({'attn_implementation':'sdpa'},'Transformers')]:
        a=argparse.Namespace(model='/unused',device_map=None,load_in_4bit=False,attn_implementation='eager',device='cpu')
        for k,v in kwargs.items():setattr(a,k,v)
        with pytest.raises(ValueError,match=match):lab.load_mlx_model(a)


def test_shared_recorder_live_n_plus_one_rows_match_oracle(tmp_path):
    model,tok=tiny_model();session=lab.LiveSession()
    hooks=lab.make_hooks(model,layers='all');rng=np.random.default_rng(43)
    try:
        for i,ids in enumerate(([4,5,6],[7,8])):
            before=list(session.token_ids)
            d=tmp_path/f'event{i}'
            result=lab.record_generation(model,tok,ids,d,options=lab.GenerationOptions(max_new_tokens=3,temperature=.8,top_p=.9,max_context=2048,save_policy_logits=True),hooks=hooks,eos_ids=[],session=session,rng=rng)
            assert result['valid_rows']==result['n_tokens']+1==4
            assert result['appended_prompt_ids']==list(ids)
            assert result['prompt_ids']==before+list(ids)
            assert result['cache_before_tokens']==len(before)
            assert session.token_ids==result['prompt_ids']+result['generated_ids']
            assert all(c.offset==len(session.token_ids) for c in session.past_key_values)
            raw=np.load(d/result['arrays']['raw_logits'])
            states={name:np.load(d/info['file']) for name,info in result['hidden'].items()}
            for row in range(result['valid_rows']):
                prefix=result['prompt_ids']+result['generated_ids'][:row]
                hooks.clear();oracle=model.model(mx.array([prefix]),cache=None)
                mx.eval(oracle,list(hooks.snapshots.values()));oracle_states,_=hooks.take()
                np.testing.assert_allclose(raw[row],np.array(oracle[0,-1,:]),rtol=2e-5,atol=2e-5)
                for name in states:np.testing.assert_allclose(states[name][row],oracle_states[name],rtol=2e-5,atol=2e-5)
        assert result['cache_reused']
    finally:hooks.close()
    assert model._active_recorder is None


def test_mutated_mlx_cache_error_invalidates_session(tmp_path,monkeypatch):
    model,tok=tiny_model();session=lab.LiveSession();hooks=lab.make_hooks(model,layers='all')
    options=lab.GenerationOptions(max_new_tokens=2,max_context=2048)
    lab.record_generation(model,tok,[4,5],tmp_path/'ok',options=options,hooks=hooks,eos_ids=[],session=session)
    before=list(session.token_ids);old_cache=session.past_key_values
    original=lab.mlx_record_forward
    def fail_after_mutation(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('injected after actual native cache update')
    monkeypatch.setattr(lab,'mlx_record_forward',fail_after_mutation)
    with pytest.raises(RuntimeError,match='actual native cache'):
        lab.record_generation(model,tok,[7,8],tmp_path/'bad',options=options,hooks=hooks,eos_ids=[],session=session)
    assert all(c.offset==len(before)+2 for c in old_cache)
    assert not session.valid and session.past_key_values is None and session.token_ids==before
    assert json.loads((tmp_path/'bad/meta.json').read_text())['session_invalidated']
    session.reset();assert session.valid and not session.token_ids
    hooks.close()


@pytest.mark.parametrize('command',['experiment','imagine'])
def test_native_production_trials_and_posthoc_similarity(tmp_path,command):
    model,tok=tiny_model();checkpoint=tmp_path/'model';checkpoint.mkdir()
    model.model.save_weights(str(checkpoint/'model.safetensors'))
    (checkpoint/'config.json').write_text(json.dumps(model.config_dict));tok.save_pretrained(checkpoint)
    out=tmp_path/'output'
    common=[command,'--backend','mlx','--model',str(checkpoint),'--device','cpu','--local-files-only',
            '--seeds','43,44','--condition','both','--max-new-tokens','3','--max-context','2048',
            '--layers','all','--no-plot','--out',str(out)]
    if command=='experiment':
        bank=tmp_path/'bank.json';bank.write_text(json.dumps({'words':['moon','door'],'language':'en'}))
        args=lab.make_parser().parse_args(common+['--bank',str(bank)])
        lab.run_experiment(args)
    else:
        args=lab.make_parser().parse_args(common+['--scene','A door opens.','--language','en','--steps','2'])
        lab.run_imagination(args)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['backend']=='mlx'
    rows=lab.read_jsonl(out/'responses.jsonl');assert len(rows)==8
    for trial_id in {r['trial_id'] for r in rows}:
        group=[r for r in rows if r['trial_id']==trial_id]
        first=json.loads((out/group[0]['event_directory']/'meta.json').read_text())
        last=json.loads((out/group[1]['event_directory']/'meta.json').read_text())
        assert not first['cache_reused']
        assert last['cache_reused']==(group[1]['condition']=='live')
        assert last['valid_rows']==last['n_tokens']+1
    assert len(lab.read_jsonl(out/'analysis/responses.jsonl'))==8
    metadata=json.loads((out/'analysis/metadata.json').read_text())
    assert metadata['performed_after_generation']


def test_unrecognized_internal_norm_is_not_treated_as_final_hidden():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__();self.norm=nn.RMSNorm(8)
        def __call__(self,x):return self.norm(x)
    class Unrecognized(nn.Module):
        def __init__(self):
            super().__init__();self.emb=nn.Embedding(260,8);self.internal=Inner();self.head=nn.Linear(8,260)
        def __call__(self,ids,cache=None):return self.head(self.internal(self.emb(ids)))
    _,tok=tiny_model();handle=lab.MLXModelHandle(Unrecognized(),{'vocab_size':260,'max_position_embeddings':2048})
    with pytest.raises(ValueError,match='No recognized final normalization'):
        lab.mlx_encode_isolated(handle,tok,['word'])
    assert handle._active_recorder is None


def test_no_cache_generation_matches_native_cached_rows(tmp_path):
    model,tok=tiny_model();hooks=lab.make_hooks(model,layers='all')
    results=[]
    try:
        for caching in (True,False):
            d=tmp_path/str(caching)
            meta=lab.record_generation(model,tok,[4,5,6],d,options=lab.GenerationOptions(max_new_tokens=4,temperature=0,max_context=2048,use_cache=caching),hooks=hooks,eos_ids=[])
            results.append((meta,np.load(d/'raw_logits.npy')))
        assert results[0][0]['generated_ids']==results[1][0]['generated_ids']
        np.testing.assert_allclose(results[0][1],results[1][1],rtol=2e-5,atol=2e-5)
        assert results[1][0]['cache_after_tokens']==0
    finally:hooks.close()
