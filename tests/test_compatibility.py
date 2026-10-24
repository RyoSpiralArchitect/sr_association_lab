"""Backend compatibility regressions, independent of downloaded model weights."""
import argparse
import dataclasses
import json

import numpy as np
import pytest

import sr_association_lab as lab


@pytest.mark.parametrize("temperature", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_temperature_is_rejected_before_sampling(temperature):
    with pytest.raises(ValueError, match="[Ff]inite"):
        lab.sampling_distribution(np.array([0.0, 1.0]), temperature, 0, 1.0)


def mlx_fixture():
    pytest.importorskip("mlx.core")
    from test_mlx import tiny_model
    return tiny_model()


def loader_args(path, **overrides):
    values = dict(model=str(path), tokenizer=None, revision=None, device="cpu", dtype="auto",
                  device_map=None, load_in_4bit=False, attn_implementation="eager",
                  adapter=None, local_files_only=True, trust_remote_code=False, seed=43)
    return argparse.Namespace(**(values | overrides))


@pytest.mark.parametrize("modern", [False, True])
def test_loader_supports_both_signatures_and_offline_snapshot_lookup(tmp_path, monkeypatch, modern):
    handle, tokenizer = mlx_fixture()
    import mlx_lm
    import mlx_lm.utils
    import huggingface_hub

    (tmp_path / "config.json").write_text(json.dumps(handle.config_dict))
    calls, downloads = [], []

    def legacy_load(path, *, return_config, tokenizer_config, adapter_path):
        calls.append((path, return_config, tokenizer_config, adapter_path))
        return handle.model, tokenizer, handle.config_dict

    def modern_load(path, *, return_config, tokenizer_config, adapter_path, trust_remote_code):
        assert trust_remote_code is False
        return legacy_load(path, return_config=return_config,
                           tokenizer_config=tokenizer_config, adapter_path=adapter_path)

    def download(model, **kwargs):
        downloads.append((model, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(mlx_lm, "load", modern_load if modern else legacy_load)
    monkeypatch.delattr(mlx_lm.utils, "DEFAULT_ALLOW_PATTERNS", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    loaded, adapted = lab.load_mlx_model(loader_args("offline-test/nonexistent", revision="pinned-revision"))
    assert loaded.model is handle.model and len(calls) == 1
    assert calls[0][2] == {"trust_remote_code": False, "local_files_only": True}
    assert downloads[0][1]["revision"] == "pinned-revision"
    assert downloads[0][1]["local_files_only"] is True
    assert "model*.safetensors" in downloads[0][1]["allow_patterns"]
    assert adapted("lamp")["input_ids"] == tokenizer("lamp")["input_ids"]


def test_legacy_custom_code_requires_explicit_trust_before_loader(tmp_path, monkeypatch):
    mlx_fixture()
    import mlx_lm
    (tmp_path / "config.json").write_text(json.dumps({"model_file": "custom.py"}))
    called = []

    def loader(path, **kwargs):
        called.append(path)
        raise AssertionError("Custom code loader must not run without explicit trust")

    monkeypatch.setattr(mlx_lm, "load", loader)
    with pytest.raises(ValueError, match="trust-remote-code"):
        lab.load_mlx_model(loader_args(tmp_path))
    assert called == []


def test_explicit_fast_tokenizer_stays_callable(tmp_path):
    handle, tokenizer = mlx_fixture()
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    handle.model.save_weights(str(checkpoint / "model.safetensors"))
    (checkpoint / "config.json").write_text(json.dumps(handle.config_dict))
    tokenizer.save_pretrained(checkpoint)
    override = tmp_path / "tokenizer"
    tokenizer.save_pretrained(override)
    _, adapted = lab.load_mlx_model(loader_args(checkpoint, tokenizer=str(override)))
    assert adapted.name_or_path == str(override)
    assert adapted("a lamp")["input_ids"] == tokenizer("a lamp")["input_ids"]
    assert len(adapted) == len(tokenizer)


def test_native_gpt2_hooks_cache_and_isolated_encoding_match_full_prefix(tmp_path):
    _, tokenizer = mlx_fixture()
    import mlx.core as mx
    from mlx_lm.models.gpt2 import Model, ModelArgs

    args = ModelArgs(model_type="gpt2", n_ctx=256, n_positions=256, n_embd=48,
                     n_head=4, n_layer=3, layer_norm_epsilon=1e-5, vocab_size=len(tokenizer))
    native = Model(args)
    native.eval()
    mx.eval(native.parameters())
    handle = lab.MLXModelHandle(native, dataclasses.asdict(args), eos_ids=[2])
    originals = dict(native.named_modules())
    hooks = lab.make_hooks(handle, layers="all")
    session = lab.LiveSession()
    try:
        assert [site["name"] for site in hooks.sites] == [
            "model.h.0", "model.h.1", "model.h.2", "model.ln_f"]
        for event_id, ids in enumerate(([4, 5, 6], [7, 8])):
            directory = tmp_path / str(event_id)
            meta = lab.record_generation(handle, tokenizer, ids, directory,
                options=lab.GenerationOptions(max_new_tokens=3, max_context=256, temperature=0),
                hooks=hooks, eos_ids=[], session=session)
            raw = np.load(directory / "raw_logits.npy")
            hidden = {name: np.load(directory / entry["file"])
                      for name, entry in meta["hidden"].items()}
            assert meta["valid_rows"] == meta["n_tokens"] + 1 == 4
            assert meta["cache_reused"] == bool(event_id)
            for row in range(meta["valid_rows"]):
                prefix = meta["prompt_ids"] + meta["generated_ids"][:row]
                hooks.clear()
                oracle = native(mx.array([prefix]), cache=None)
                mx.eval(oracle, list(hooks.snapshots.values()))
                states, _ = hooks.take()
                np.testing.assert_allclose(raw[row], np.array(oracle[0, -1]), rtol=2e-5, atol=2e-5)
                for name, state in states.items():
                    np.testing.assert_allclose(hidden[name][row], state, rtol=2e-5, atol=2e-5)
            assert all(c.offset == len(session.token_ids) for c in session.past_key_values)
    finally:
        hooks.close()
    assert dict(native.named_modules()) == originals
    means, ids, metadata = lab.mlx_encode_isolated(handle, tokenizer, ["a lamp", "a station"], 256)
    for actual, sequence in zip(means, ids):
        expected = native.model(mx.array([sequence]), cache=None)[0].mean(axis=0)
        mx.eval(expected)
        np.testing.assert_allclose(actual, np.array(expected), rtol=2e-5, atol=2e-5)
    assert metadata["hidden_site"] == "model.ln_f"
    assert handle._active_recorder is None
