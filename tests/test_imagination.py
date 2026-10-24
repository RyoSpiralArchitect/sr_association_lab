"""Sentence and origin/cache contracts, with real offline model forwards.

Forced logits are used only to reach specific textual boundary cases. The
underlying transformer and its KV cache still execute normally; these fixtures
assert recorder behavior, not learned linguistic ability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")

import sr_association_lab as lab


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sentence_options(**changes):
    values = dict(max_new_tokens=16, temperature=0.0, top_k=0, top_p=1.0,
                  seed=43, max_context=2048, save_logits=True,
                  hidden_dtype="float32", stop_on_newline=False,
                  stop_on_sentence=True)
    values.update(changes)
    return lab.GenerationOptions(**values)


def add_atomic_text(model, tokenizer, text):
    """One real vocabulary token can cross a surface sentence boundary."""
    from tokenizers import AddedToken
    tokenizer.add_tokens([AddedToken(text, normalized=False, special=False)])
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == 1 and tokenizer.decode(ids, skip_special_tokens=True) == text
    return ids[0]


@pytest.mark.parametrize("text", [
    "灯りが揺れる。", "窓が開く！", "水音がする？", "A lamp flickers.",
    "At 3.14 a lamp flickers.",
])
def test_sentence_surface_accepts_one_complete_sentence(text):
    boundary = lab.sentence_boundary(text, final=True)
    assert boundary is not None
    assert boundary["end"] == len(text)
    assert not boundary["overshoot"].strip()
    checked = lab.check_one_sentence(text, "sentence")
    assert checked["surface_ok"] and checked["eligible"]
    assert checked["text"] == text


@pytest.mark.parametrize("text,stop", [
    ("", "eos"),
    ("A lamp flickers", "eos"),
    ("A lamp flickers. Another window opens.", "sentence"),
    ("灯りが揺れる。別の窓が開く。", "sentence"),
    ("A lamp flickers.", "length"),
])
def test_invalid_or_censored_sentence_preserves_the_full_surface(text, stop):
    checked = lab.check_one_sentence(text, stop)
    assert not checked["eligible"]
    assert checked["reasons"]
    assert checked["text"] == text.strip()
    if stop == "length":
        assert "length_censored" in checked["reasons"]


@pytest.mark.parametrize("scene,eligible", [
    ("灯りが揺れる。", True),
    ("Rain. Wind", False),
])
def test_sentence_stop_consumes_complete_sampled_token_even_when_it_contains_overshoot(tmp_path, monkeypatch, scene, eligible):
    model, tokenizer = lab.make_tiny_model(seed=151)
    forced_id = add_atomic_text(model, tokenizer, scene)
    original_forward = model.forward
    forward_ids = []

    def scripted_forward(*positional, **kwargs):
        forward_ids.append(kwargs["input_ids"][0].tolist())
        result = original_forward(*positional, **kwargs)
        result.logits.fill_(-100.0)
        result.logits[..., forced_id] = 100.0
        return result

    monkeypatch.setattr(model, "forward", scripted_forward)
    prompt = tokenizer.encode("A room.", add_special_tokens=True)
    session = lab.LiveSession()
    hooks = lab.HookRecorder(model, layers="0")
    path = tmp_path / "sentence"
    try:
        result = lab.record_generation(model, tokenizer, prompt, path,
            options=sentence_options(), hooks=hooks, session=session, eos_ids=[])
    finally:
        hooks.close()
    assert result["response"] == scene  # No first-sentence slicing/repair.
    assert result["stop_reason"] == "sentence"
    assert result["generated_ids"] == [forced_id]
    assert result["n_tokens"] == 1 and result["valid_rows"] == 2
    assert result["terminal"]["consumed_token_id"] == forced_id
    assert session.token_ids == prompt + [forced_id]
    assert forward_ids == [prompt, [forced_id]]
    assert lab.check_one_sentence(result["response"], result["stop_reason"])["eligible"] is eligible
    rows = lab.read_jsonl(path / "tokens.jsonl")
    assert len(rows) == 1 and rows[0]["text_so_far"] == scene
    assert np.load(path / "raw_logits.npy").shape[0] >= result["valid_rows"]


def test_loading_origin_verifies_hash_and_preserves_provided_scene(tmp_path):
    scene = "A blue door stands at the end of a narrow hallway."
    path = tmp_path / "origin.json"
    lab.atomic_json(path, {"scene": scene, "sha256": lab.scene_hash(scene),
                           "source": "provided", "language": "en"})
    loaded = lab.load_origin_scene(path)
    assert loaded["scene"] == scene and loaded["sha256"] == lab.scene_hash(scene)
    saved = read_json(path)
    saved["scene"] = "A red door stands at the end of a narrow hallway."
    lab.atomic_json(path, saved)
    with pytest.raises(ValueError, match="(?i)hash|checksum|changed"):
        lab.load_origin_scene(path)


def imagine_args(out, *extra):
    return lab.make_parser().parse_args([
        "imagine", "--model", "offline-test-fixture", "--out", str(out),
        "--steps", "3", "--max-new-tokens", "2", "--max-context", "2048",
        "--layers", "0", "--no-logits", "--no-plot", "--similarity", "self",
        "--template", "plain", "--cpu-threads", "1", *extra,
    ])


def test_imagination_uses_frozen_origin_continuous_state_and_repeated_seeds(tmp_path, monkeypatch):
    """Exercise production orchestration, indices, exact ledgers and self encoding."""
    model, tokenizer = lab.make_tiny_model(seed=157)
    monkeypatch.setattr(lab, "load_model", lambda args: (model, tokenizer))
    scene = "A blue door stands at the end of a narrow hallway."
    out = tmp_path / "imagine"
    args = imagine_args(out, "--scene", scene, "--language", "en",
                        "--seeds", "43,44", "--repeats", "2", "--condition", "both")
    expected_trials = 2 * 2 * 2
    expected_responses = expected_trials * 3
    scoring_inputs = []
    original_forward = model.forward

    def observe_forward(*positional, **kwargs):
        if kwargs.get("output_hidden_states"):
            committed = lab.read_jsonl(out / "responses.jsonl")
            assert len(committed) == expected_responses
            assert kwargs["use_cache"] is False and kwargs.get("past_key_values") is None
            assert not any(
                getattr(hook, "__qualname__", "").startswith("HookRecorder.")
                for module in model.modules() for hook in module._forward_hooks.values()
            )
            scoring_inputs.append(kwargs["input_ids"][0].tolist())
        return original_forward(*positional, **kwargs)

    monkeypatch.setattr(model, "forward", observe_forward)
    assert lab.run_imagination(args) == out
    manifest = read_json(out / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["completed_trials"] == expected_trials
    assert manifest["completed_responses"] == expected_responses
    origin = read_json(out / "origin.json")
    assert origin["scene"] == scene and origin["sha256"] == lab.scene_hash(scene)
    schedule = read_json(out / "schedule.json")
    assert len(schedule) == expected_trials
    rows = lab.read_jsonl(out / "responses.jsonl")
    assert len(rows) == expected_responses
    grouped = {}
    for row in rows:
        grouped.setdefault(row["trial_id"], []).append(row)
    repeated = {}
    for trial_id, trial_rows in grouped.items():
        previous = None
        for position, row in enumerate(trial_rows):
            meta = read_json(out / row["event_directory"] / "meta.json")
            assert meta["generation"]["stop_on_sentence"] is True
            if row["condition"] == "reset" or position == 0:
                assert meta["past_tokens"] == 0 and meta["cache_reused"] is False
                assert scene in meta["prompt_text"]
            else:
                assert previous is not None
                assert meta["cache_reused"] is True
                assert meta["past_tokens"] == previous["cache_after_tokens"]
                assert meta["prompt_ids"] == (previous["prompt_ids"] + previous["generated_ids"]
                                             + meta["appended_prompt_ids"])
                assert scene not in meta["prompt_text"]  # No re-presented origin/history.
                assert meta["rng_state_before"] == previous["rng_state_after"]
            key = (row["sample_seed"], row["condition"], position)
            repeated.setdefault(key, []).append(meta["generated_ids"])
            previous = meta
        saved_session = read_json(out / "trials" / trial_id / "session.json")
        assert saved_session["token_ids"] == previous["prompt_ids"] + previous["generated_ids"]
    assert all(len(versions) == 2 and versions[0] == versions[1] for versions in repeated.values())
    texts = read_json(out / "analysis" / "embedding_texts.json")
    assert scene in texts
    assert len(texts) == len(set(texts))
    assert scoring_inputs == [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    scored = lab.read_jsonl(out / "analysis" / "responses.jsonl")
    assert len(scored) == expected_responses


def test_generated_origin_cache_and_private_prompt_do_not_leak_into_trials(tmp_path, monkeypatch):
    model, tokenizer = lab.make_tiny_model(seed=163)
    scene = "A lamp waits by the window."
    origin_token = add_atomic_text(model, tokenizer, scene)
    monkeypatch.setattr(lab, "load_model", lambda args: (model, tokenizer))
    marker = "ORIGIN_ONLY_MARKER_7901"
    prompt_file = tmp_path / "origin_prompt.txt"
    prompt_file.write_text(marker + ": Write one initial scene sentence.", encoding="utf-8")
    out = tmp_path / "generated"
    args = imagine_args(out, "--origin-prompt", str(prompt_file), "--origin-attempts", "1",
                        "--origin-max-new-tokens", "16", "--language", "en",
                        "--seeds", "43,44", "--condition", "live", "--similarity", "none")
    original_forward = model.forward
    original_record = lab.record_generation
    state = {"origin_active": False, "origin_session": None, "origin_calls": 0, "trial_calls": 0}

    def scripted_forward(*positional, **kwargs):
        result = original_forward(*positional, **kwargs)
        if state["origin_active"]:
            result.logits.fill_(-100.0)
            result.logits[..., origin_token] = 100.0
        return result

    def observe_record(*positional, **kwargs):
        if kwargs.get("metadata", {}).get("role") == "origin_generation":
            state["origin_calls"] += 1
            state["origin_session"] = kwargs["session"]
            state["origin_active"] = True
            try:
                result = original_record(*positional, **kwargs)
                assert kwargs["session"].past_key_values is not None
                assert kwargs["session"].token_ids
                return result
            finally:
                state["origin_active"] = False
        origin_session = state["origin_session"]
        assert origin_session is not None
        assert origin_session.past_key_values is None and origin_session.token_ids == []
        assert origin_session.reset_count == 1
        assert kwargs["session"] is not origin_session
        assert marker not in kwargs["metadata"]["prompt_text"]
        if kwargs["metadata"].get("step_index", kwargs["metadata"].get("position", kwargs["metadata"].get("step"))) == 0:
            assert kwargs["session"].past_key_values is None and kwargs["session"].token_ids == []
        state["trial_calls"] += 1
        return original_record(*positional, **kwargs)

    monkeypatch.setattr(model, "forward", scripted_forward)
    monkeypatch.setattr(lab, "record_generation", observe_record)
    lab.run_imagination(args)
    assert state["origin_calls"] == 1 and state["trial_calls"] == 6
    origin = read_json(out / "origin.json")
    assert origin["scene"] == scene and origin["source"] == "model-generated"
    assert origin["provenance"]["cache_discarded_before_trials"] is True
    for row in lab.read_jsonl(out / "responses.jsonl"):
        meta = read_json(out / row["event_directory"] / "meta.json")
        assert marker not in tokenizer.decode(meta["prompt_ids"])


def test_run_imagination_defaults_to_live_one_sentence_steps(tmp_path, monkeypatch):
    model, tokenizer = lab.make_tiny_model(seed=167)
    sentence = "灯りが揺れる。"
    forced_id = add_atomic_text(model, tokenizer, sentence)
    monkeypatch.setattr(lab, "load_model", lambda args: (model, tokenizer))
    original_forward = model.forward

    def scripted_forward(*positional, **kwargs):
        result = original_forward(*positional, **kwargs)
        result.logits.fill_(-100.0)
        result.logits[..., forced_id] = 100.0
        return result

    monkeypatch.setattr(model, "forward", scripted_forward)
    out = tmp_path / "run"
    args = lab.make_parser().parse_args([
        "run", "--model", "offline-test-fixture", "--protocol", "imagination",
        "--cue", "無人の駅に灯りがある。", "--steps", "2", "--out", str(out),
        "--max-new-tokens", "16", "--max-context", "2048", "--no-hooks",
        "--no-logits", "--no-plot", "--template", "plain", "--cpu-threads", "1",
    ])
    lab.run_session(args)
    events = lab.read_jsonl(out / "events.jsonl")
    assert len(events) == 2
    assert all(e["response"] == sentence and e["stop_reason"] == "sentence" for e in events)
    first = read_json(out / events[0]["directory"] / "meta.json")
    second = read_json(out / events[1]["directory"] / "meta.json")
    assert first["generation"]["stop_on_sentence"] is True
    assert first["past_tokens"] == 0
    assert second["cache_reused"] is True
    assert second["prompt_ids"] == first["prompt_ids"] + first["generated_ids"] + second["appended_prompt_ids"]


def test_scene_similarity_links_exact_predecessors_within_each_trial_and_keeps_missing_values():
    # Deliberately interleaved and unsorted, with a missing step and an invalid
    # earlier response. Referents are trial/step identities, never file adjacency.
    rows = [
        {"trial_id": "b", "step_index": 1, "condition": "reset", "eligible": True},
        {"trial_id": "a", "step_index": 0, "condition": "live", "eligible": True},
        {"trial_id": "b", "step_index": 0, "condition": "reset", "eligible": False},
        {"trial_id": "a", "step_index": 1, "condition": "live", "eligible": False},
        {"trial_id": "c", "step_index": 2, "condition": "live", "eligible": True},
        {"trial_id": "a", "step_index": 2, "condition": "live", "eligible": False},
        {"trial_id": "d", "step_index": 0, "condition": "live", "eligible": False},
    ]
    vectors = np.array([[2, 0], [3, 0], [0, 4], [-5, 0], [7, 0], [0, 0], [np.nan, 1]], dtype=np.float32)
    scored, matrix = lab.scene_similarity_rows(rows, np.array([10, 0]), vectors)
    expected = np.array([
        [1, 1, 0, -1, 1, np.nan, np.nan],
        [1, 1, 0, -1, 1, np.nan, np.nan],
        [0, 0, 1, 0, 0, np.nan, np.nan],
        [-1, -1, 0, 1, -1, np.nan, np.nan],
        [1, 1, 0, -1, 1, np.nan, np.nan],
        [np.nan] * 7, [np.nan] * 7,
    ], dtype=np.float32)
    np.testing.assert_allclose(matrix, expected, atol=1e-7, equal_nan=True)
    assert scored[0]["previous_response_index"] == 2
    assert scored[0]["previous_scene_cosine"] == pytest.approx(0)
    assert scored[0]["previous_scene_used_as_context"] is False
    assert scored[0]["previous_response_eligible"] is False
    assert scored[1]["previous_scene_kind"] == "origin"
    assert scored[1]["previous_scene_cosine"] == pytest.approx(1)
    assert scored[1]["previous_scene_used_as_context"] is True
    assert scored[1]["previous_response_index"] is None
    assert scored[2]["origin_cosine"] == pytest.approx(0)
    assert scored[3]["previous_response_index"] == 1
    assert scored[3]["previous_scene_cosine"] == pytest.approx(-1)
    assert scored[3]["previous_scene_used_as_context"] is True
    assert scored[3]["nearest_prior_scene_index"] == -1  # Stable origin-first tie.
    assert scored[4]["previous_scene_kind"] == "missing"
    assert scored[4]["previous_scene_cosine"] is None
    assert scored[4]["previous_scene_used_as_context"] is None
    assert scored[4]["nearest_prior_scene_index"] == -1
    assert scored[5]["previous_response_index"] == 3
    assert scored[5]["previous_response_eligible"] is False
    for index in (5, 6):
        assert scored[index]["origin_cosine"] is None
        assert scored[index]["previous_scene_cosine"] is None
        assert scored[index]["nearest_prior_scene_index"] is None
        assert scored[index]["nearest_prior_scene_cosine"] is None


@pytest.mark.parametrize("rows", [
    [{"trial_id": "a", "step_index": 0}, {"trial_id": "a", "step_index": 0}],
    [{"trial_id": "a", "step_index": 0.5}],
    [{"trial_id": "a", "step_index": True}],
    [{"trial_id": "a", "step_index": -1}],
])
def test_scene_similarity_rejects_ambiguous_step_identities(rows):
    with pytest.raises(ValueError, match="(?i)duplicate|nonnegative integer"):
        lab.scene_similarity_rows(rows, np.array([1, 0]), np.ones((len(rows), 2)))
