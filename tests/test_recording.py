"""Measurement-contract tests; all model weights/tokenizers are created offline.

Run with: python -m pytest -q tests/test_recording.py
These test the recorder, not linguistic or psychological claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")

import sr_association_lab as lab


def _prompt(tokenizer):
    return tokenizer.encode("mirror water", add_special_tokens=True)


def _options(**changes):
    values = dict(
        max_new_tokens=4,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        seed=43,
        save_logits=True,
        save_policy_logits=False,
        hidden_dtype="float32",
        stop_on_newline=False,
        max_context=512,
    )
    values.update(changes)
    return lab.GenerationOptions(**values)


def _record(model, tokenizer, path, *, enabled=True, options=None, eos_ids=None):
    hooks = lab.HookRecorder(model, layers="all", enabled=enabled)
    try:
        return lab.record_generation(
            model,
            tokenizer,
            _prompt(tokenizer),
            path,
            options=options or _options(),
            hooks=hooks,
            eos_ids=[] if eos_ids is None else eos_ids,
        )
    finally:
        hooks.close()


def _first_sequence_tensor(output):
    if isinstance(output, torch.Tensor):
        return output if output.ndim == 3 else None
    if isinstance(output, (tuple, list)):
        for value in output:
            found = _first_sequence_tensor(value)
            if found is not None:
                return found
    if isinstance(output, dict):
        for value in output.values():
            found = _first_sequence_tensor(value)
            if found is not None:
                return found
    return None


def _read_tokens(path):
    return [json.loads(line) for line in (path / "tokens.jsonl").read_text().splitlines() if line]


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_cached_logits_and_every_hook_row_match_full_prefix_including_final_consume(tmp_path, architecture):
    """Row t contains prefix+generated[:t], never the not-yet-consumed next token."""
    model, tokenizer = lab.make_tiny_model(architecture=architecture, seed=17)
    model.eval()
    path = tmp_path / architecture
    meta = _record(model, tokenizer, path)
    generated = meta["generated_ids"]
    assert meta["status"] == "complete"
    assert meta["stop_reason"] == "length"
    assert meta["n_tokens"] == len(generated) == 4
    assert meta["valid_rows"] == len(generated) + 1
    assert meta["hidden"], "Requested decoder hooks must produce saved states."

    saved_logits = np.load(path / "raw_logits.npy", mmap_mode="r")
    saved_hidden = {
        name: np.load(path / details["file"], mmap_mode="r")
        for name, details in meta["hidden"].items()
    }
    modules = dict(model.named_modules())
    captured = {}
    handles = []
    for name in saved_hidden:
        def capture(_module, _inputs, output, name=name):
            tensor = _first_sequence_tensor(output)
            assert tensor is not None, f"No [batch, sequence, width] tensor at {name}"
            captured[name] = tensor[0, -1, :].detach().float().cpu().numpy().copy()
        handles.append(modules[name].register_forward_hook(capture))
    prompt = _prompt(tokenizer)
    try:
        with torch.no_grad():
            for row in range(meta["valid_rows"]):
                captured.clear()
                input_ids = torch.tensor([prompt + generated[:row]], dtype=torch.long)
                output = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    return_dict=True,
                )
                expected_logits = output.logits[0, -1, :].detach().float().cpu().numpy()
                np.testing.assert_allclose(saved_logits[row], expected_logits, rtol=2e-4, atol=2e-5)
                for name, array in saved_hidden.items():
                    np.testing.assert_allclose(array[row], captured[name], rtol=2e-4, atol=2e-5)
    finally:
        for handle in handles:
            handle.remove()

    tokens = _read_tokens(path)
    assert len(tokens) == len(generated)
    for t, token in enumerate(tokens):
        assert token["t"] == t
        assert token["token_id"] == generated[t]
        assert token["prefix_len"] == len(prompt) + t
        assert token["readout_position"] == len(prompt) + t - 1
        assert token["predictor_row"] == t
        assert token["consumed_state_row"] == t + 1


def test_read_only_hooks_do_not_change_greedy_generation(tmp_path):
    model, tokenizer = lab.make_tiny_model(seed=29)
    with_hooks = _record(model, tokenizer, tmp_path / "hooks")
    without_hooks = _record(model, tokenizer, tmp_path / "no_hooks", enabled=False)
    assert with_hooks["generated_ids"] == without_hooks["generated_ids"]
    assert without_hooks["hidden"] == {}
    rows = with_hooks["valid_rows"]
    np.testing.assert_allclose(
        np.load(tmp_path / "hooks" / "raw_logits.npy")[:rows],
        np.load(tmp_path / "no_hooks" / "raw_logits.npy")[:rows],
        rtol=0,
        atol=0,
    )


def test_eos_is_recorded_and_consumed_once_without_an_extra_sample(tmp_path):
    model, tokenizer = lab.make_tiny_model(seed=31)
    model.eval()
    ids = torch.tensor([_prompt(tokenizer)], dtype=torch.long)
    with torch.no_grad():
        eos = int(model(input_ids=ids).logits[0, -1].argmax())
    path = tmp_path / "eos"
    meta = _record(model, tokenizer, path, eos_ids=[eos])
    assert meta["stop_reason"] == "eos"
    assert meta["generated_ids"] == [eos]
    assert meta["n_tokens"] == 1
    assert meta["valid_rows"] == 2
    assert len(_read_tokens(path)) == 1
    with torch.no_grad():
        full = model(input_ids=torch.tensor([_prompt(tokenizer) + [eos]], dtype=torch.long))
    np.testing.assert_allclose(
        np.load(path / "raw_logits.npy")[1],
        full.logits[0, -1].float().numpy(),
        rtol=2e-4,
        atol=2e-5,
    )


def test_temperature_and_top_k_preserve_raw_logits():
    raw = np.array([2.0, 1.0, 0.0, -2.0], dtype=np.float32)
    before = raw.copy()
    policy, probabilities = lab.sampling_distribution(raw, 2.0, 2, 1.0)
    np.testing.assert_array_equal(raw, before)
    expected = np.exp(np.array([1.0, 0.5]))
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities[:2], expected, rtol=1e-6)
    np.testing.assert_array_equal(probabilities[2:], 0)
    assert np.isneginf(policy[2:]).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_nucleus_keeps_the_token_crossing_the_probability_threshold():
    raw = np.log(np.array([0.60, 0.25, 0.15], dtype=np.float64))
    policy, probabilities = lab.sampling_distribution(raw, 1.0, 0, 0.7)
    np.testing.assert_allclose(probabilities, np.array([0.6 / 0.85, 0.25 / 0.85, 0]), atol=1e-7)
    assert np.isfinite(policy[:2]).all()
    assert np.isneginf(policy[2])


def test_greedy_policy_is_a_point_mass_without_zero_temperature_division():
    raw = np.array([0.0, 5.0, 1.0], dtype=np.float32)
    policy, probabilities = lab.sampling_distribution(raw, 0.0, 0, 1.0)
    np.testing.assert_array_equal(probabilities, [0.0, 1.0, 0.0])
    assert policy[1] == 0
    assert np.isneginf(policy[[0, 2]]).all()


def test_saved_raw_and_policy_logits_are_distinct_and_reconstructible(tmp_path):
    model, tokenizer = lab.make_tiny_model(seed=37)
    path = tmp_path / "policy"
    options = _options(temperature=0.6, top_k=3, top_p=0.8, save_policy_logits=True)
    meta = _record(model, tokenizer, path, options=options)
    raw = np.load(path / "raw_logits.npy")
    policy = np.load(path / "policy_logits.npy")
    with torch.no_grad():
        expected = model(input_ids=torch.tensor([_prompt(tokenizer)], dtype=torch.long)).logits[0, -1]
    np.testing.assert_allclose(raw[0], expected.float().numpy(), rtol=2e-4, atol=2e-5)
    for t, token in enumerate(_read_tokens(path)):
        expected_policy, p = lab.sampling_distribution(raw[t], 0.6, 3, 0.8)
        np.testing.assert_allclose(policy[t], expected_policy, rtol=1e-6, atol=1e-6)
        assert p[token["token_id"]] > 0
        assert token["policy_surprisal_nats"] == pytest.approx(-np.log(p[token["token_id"]]), abs=1e-6)
    assert meta["valid_rows"] == meta["n_tokens"] + 1


def test_context_overflow_is_explicit_and_not_silently_truncated(tmp_path):
    model, tokenizer = lab.make_tiny_model(seed=41)
    options = _options(max_context=len(_prompt(tokenizer)) - 1)
    with pytest.raises(ValueError, match="(?i)context|budget|length"):
        _record(model, tokenizer, tmp_path / "overflow", options=options)


def test_forward_failure_leaves_truthful_readable_partial_metadata(tmp_path, monkeypatch):
    model, tokenizer = lab.make_tiny_model(seed=43)
    original = model.forward
    calls = 0

    def fail_second_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("intentional test fault after first sampled token")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", fail_second_forward)
    path = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="intentional test fault"):
        _record(model, tokenizer, path)
    meta = json.loads((path / "meta.json").read_text())
    assert meta["status"] in {"error", "failed", "interrupted"}
    assert 1 <= meta["valid_rows"] <= meta["n_tokens"] + 1
    tokens = _read_tokens(path)
    assert len(tokens) == meta["n_tokens"] == len(meta["generated_ids"])
    assert meta["n_tokens"] == 1
    saved = np.load(path / "raw_logits.npy", mmap_mode="r")
    assert np.isfinite(saved[:meta["valid_rows"]]).all()


def test_history_is_explicit_and_prompt_builder_does_not_mutate_it():
    history = [{"role": "user", "content": "old cue"}, {"role": "assistant", "content": "old response"}]
    before = json.loads(json.dumps(history))
    carried = lab.build_messages("association", "new cue", history)
    independent = lab.build_messages("association", "new cue", [])
    assert history == before
    assert any(message["content"] == "old response" for message in carried)
    assert not any(message["content"] == "old response" for message in independent)
    assert carried[-1]["role"] == independent[-1]["role"] == "user"
    assert "new cue" in carried[-1]["content"]
