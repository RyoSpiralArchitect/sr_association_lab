"""Experiment invariants, using only offline random model fixtures.

The important oracle is an independent, uncached full-prefix forward over the
actual token ledger. These tests do not assert linguistic or semantic findings.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
torch = pytest.importorskip("torch")

import sr_association_lab as lab


def options(**changes):
    values = dict(max_new_tokens=3, temperature=0.0, top_k=0, top_p=1.0,
                  seed=43, max_context=1024, save_logits=True,
                  hidden_dtype="float32", stop_on_newline=False)
    values.update(changes)
    return lab.GenerationOptions(**values)


def record(model, tokenizer, ids, path, session, *, generation=None, rng=None):
    hooks = lab.HookRecorder(model, layers="all")
    try:
        return lab.record_generation(model, tokenizer, ids, path,
                                     options=generation or options(), hooks=hooks,
                                     eos_ids=[], session=session, rng=rng)
    finally:
        hooks.close()


def tokens(path):
    return [json.loads(line) for line in (path / "tokens.jsonl").read_text().splitlines()]


def first_sequence_tensor(value):
    if isinstance(value, torch.Tensor) and value.ndim == 3:
        return value
    if isinstance(value, (tuple, list)):
        return next((t for x in value if (t := first_sequence_tensor(x)) is not None), None)
    if isinstance(value, dict):
        return next((t for x in value.values() if (t := first_sequence_tensor(x)) is not None), None)
    return None


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_live_turns_match_uncached_exact_ledger_and_do_not_replay_history(tmp_path, monkeypatch, architecture):
    model, tokenizer = lab.make_tiny_model(architecture, seed=71)
    session = lab.LiveSession()
    initial = tokenizer.encode("mirror", add_special_tokens=True)
    first = record(model, tokenizer, initial, tmp_path / "first", session)
    old_ledger = initial + first["generated_ids"]
    assert session.token_ids == old_ledger
    assert session.valid and session.past_key_values is not None

    suffix = tokenizer.encode("\nwater", add_special_tokens=False)
    seen_input_lengths = []
    original_forward = model.forward

    def spy_forward(*args, **kwargs):
        seen_input_lengths.append(int(kwargs["input_ids"].shape[-1]))
        return original_forward(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(model, "forward", spy_forward)
        second = record(model, tokenizer, suffix, tmp_path / "second", session)

    # One suffix-only forward, then one token per decode and final consumption.
    assert seen_input_lengths == [len(suffix)] + [1] * second["n_tokens"]
    actual_prompt = old_ledger + suffix
    assert second["prompt_ids"] == actual_prompt
    assert second["appended_prompt_ids"] == suffix
    assert second["past_tokens"] == second["cache_before_tokens"] == len(old_ledger)
    assert second["new_prompt_tokens"] == len(suffix)
    assert second["cache_reused"] is True
    assert session.token_ids == actual_prompt + second["generated_ids"]
    assert second["cache_after_tokens"] == len(session.token_ids)
    assert second["valid_rows"] == second["n_tokens"] + 1

    raw = np.load(tmp_path / "second" / "raw_logits.npy", mmap_mode="r")
    hidden = {name: np.load(tmp_path / "second" / info["file"], mmap_mode="r")
              for name, info in second["hidden"].items()}
    captured = {}
    handles = []
    for name, module in model.named_modules():
        if name in hidden:
            def capture(_module, _args, output, name=name):
                tensor = first_sequence_tensor(output)
                assert tensor is not None
                captured[name] = tensor[0, -1].detach().float().cpu().numpy().copy()
            handles.append(module.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            for row in range(second["valid_rows"]):
                captured.clear()
                prefix = actual_prompt + second["generated_ids"][:row]
                full = torch.tensor([prefix], dtype=torch.long)
                result = model(input_ids=full, attention_mask=torch.ones_like(full),
                               use_cache=False, return_dict=True)
                np.testing.assert_allclose(raw[row], result.logits[0, -1].float().numpy(),
                                           rtol=2e-4, atol=2e-5)
                for name, saved in hidden.items():
                    np.testing.assert_allclose(saved[row], captured[name], rtol=2e-4, atol=2e-5)
    finally:
        for handle in handles:
            handle.remove()


def test_reset_releases_cache_and_next_event_matches_a_fresh_session(tmp_path):
    model, tokenizer = lab.make_tiny_model(seed=73)
    session = lab.LiveSession()
    record(model, tokenizer, tokenizer.encode("old history"), tmp_path / "old", session)
    assert session.token_ids and session.past_key_values is not None
    session.reset()
    assert session.token_ids == [] and session.past_key_values is None
    assert session.valid and session.reset_count == 1
    ids = tokenizer.encode("new cue")
    reset = record(model, tokenizer, ids, tmp_path / "reset", session)
    fresh = record(model, tokenizer, ids, tmp_path / "fresh", lab.LiveSession())
    assert reset["past_tokens"] == 0
    assert reset["cache_reused"] is False
    assert reset["generated_ids"] == fresh["generated_ids"]
    np.testing.assert_allclose(np.load(tmp_path / "reset" / "raw_logits.npy"),
                               np.load(tmp_path / "fresh" / "raw_logits.npy"), rtol=0, atol=0)


def test_live_context_overflow_fails_before_any_forward_or_session_mutation(tmp_path, monkeypatch):
    model, tokenizer = lab.make_tiny_model(seed=79)
    session = lab.LiveSession()
    record(model, tokenizer, tokenizer.encode("old"), tmp_path / "old", session)
    before_ids = session.token_ids.copy()
    before_cache = session.past_key_values
    suffix = tokenizer.encode("new cue")

    def forbidden_forward(*args, **kwargs):
        pytest.fail("The context budget must be checked before mutating a live cache.")

    monkeypatch.setattr(model, "forward", forbidden_forward)
    limited = options(max_context=len(before_ids) + len(suffix) + 2, max_new_tokens=3)
    with pytest.raises(ValueError, match="(?i)context|budget|length"):
        record(model, tokenizer, suffix, tmp_path / "overflow", session, generation=limited)
    assert session.token_ids == before_ids
    assert session.past_key_values is before_cache
    assert session.valid


def test_mutated_cache_failure_invalidates_session_until_reset(tmp_path, monkeypatch):
    model, tokenizer = lab.make_tiny_model(seed=83)
    session = lab.LiveSession()
    record(model, tokenizer, tokenizer.encode("old"), tmp_path / "old", session)
    old_ids = session.token_ids.copy()
    original_forward = model.forward

    def fail_after_model_mutates_cache(*args, **kwargs):
        original_forward(*args, **kwargs)
        raise RuntimeError("intentional fault after cache mutation")

    with monkeypatch.context() as patch:
        patch.setattr(model, "forward", fail_after_model_mutates_cache)
        with pytest.raises(RuntimeError, match="intentional fault"):
            record(model, tokenizer, tokenizer.encode("next"), tmp_path / "failed", session)
    assert not session.valid
    # No success ledger may be published for the failed turn.
    assert session.token_ids == old_ids
    with pytest.raises((ValueError, RuntimeError), match="(?i)invalid|reset"):
        record(model, tokenizer, tokenizer.encode("next"), tmp_path / "unsafe_reuse", session)
    session.reset()
    recovered = record(model, tokenizer, tokenizer.encode("safe"), tmp_path / "recovered", session)
    assert recovered["status"] == "complete" and session.valid
    assert recovered["past_tokens"] == 0


def test_live_session_rejects_disabled_cache(tmp_path):
    model, tokenizer = lab.make_tiny_model(seed=89)
    with pytest.raises(ValueError, match="(?i)cache|live"):
        record(model, tokenizer, tokenizer.encode("cue"), tmp_path / "bad",
               lab.LiveSession(), generation=options(use_cache=False))


def test_response_rng_continues_across_turns_and_repeats_reproduce_exact_ids(tmp_path):
    model, tokenizer = lab.make_tiny_model(seed=97)
    generation = options(temperature=0.9, top_k=31, top_p=0.94, max_new_tokens=4)

    def trial(name):
        session = lab.LiveSession()
        rng = np.random.default_rng(1234)
        first = record(model, tokenizer, tokenizer.encode("river"), tmp_path / name / "first",
                       session, generation=generation, rng=rng)
        # Creating an order plan must not consume this response RNG.
        lab.build_trial_plan(["river", "glass", "stone"], [1, 2, 3], order="shuffle",
                             order_seeds=(101, 102))
        second = record(model, tokenizer, tokenizer.encode("\nglass"), tmp_path / name / "second",
                        session, generation=generation, rng=rng)
        assert first["rng_state_after"] == second["rng_state_before"]
        assert first["rng_state_before"] != first["rng_state_after"]
        return first, second, session.token_ids.copy()

    first_run = trial("a")
    repeated = trial("b")
    assert first_run[2] == repeated[2]
    for a, b in zip(first_run[:2], repeated[:2]):
        assert a["generated_ids"] == b["generated_ids"]
        assert a["rng_state_before"] == b["rng_state_before"]
        assert a["rng_state_after"] == b["rng_state_after"]


def test_model_time_excludes_final_consume_and_latency_precedes_disk_write_completion(tmp_path):
    model, tokenizer = lab.make_tiny_model(seed=101)
    path = tmp_path / "timing"
    meta = record(model, tokenizer, tokenizer.encode("cue"), path, lab.LiveSession())
    rows = tokens(path)
    assert meta["first_token_model_ms"] == pytest.approx(rows[0]["instrumented_forward_ms"])
    assert meta["response_model_ms"] == pytest.approx(sum(r["instrumented_forward_ms"] for r in rows))
    assert meta["terminal_consume_ms"] == pytest.approx(meta["terminal"]["instrumented_forward_ms"])
    assert meta["first_token_latency_ms"] == pytest.approx(rows[0]["sample_ready_ms"])
    assert meta["response_wall_ms"] == pytest.approx(rows[-1]["sample_ready_ms"])
    assert 0 <= meta["first_token_latency_ms"] <= meta["response_wall_ms"] <= meta["event_wall_ms"]
    assert all(0 <= r["sample_ready_ms"] <= r["token_ready_ms"] for r in rows)


def test_chat_suffix_preserves_exact_prior_sampled_ids(tmp_path, monkeypatch):
    model, tokenizer = lab.make_tiny_model(seed=103)
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}"
        "{{ '<' + message['role'] + '>' + message['content'] + eos_token }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    )
    session = lab.LiveSession()
    first_prompt = lab.build_live_prompt(tokenizer, "association", "river", session, template="chat")
    first = record(model, tokenizer, first_prompt["ids"], tmp_path / "first", session,
                   generation=options(max_context=2048))
    ledger = session.token_ids.copy()
    original_decode = tokenizer.decode

    def never_roundtrip_history(ids, *args, **kwargs):
        if isinstance(ids, (list, tuple)) and list(ids) == ledger:
            pytest.fail("The complete sampled ledger must not be decoded and rebuilt.")
        return original_decode(ids, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tokenizer, "decode", never_roundtrip_history)
        suffix = lab.build_live_prompt(tokenizer, "association", "stone", session, template="chat")
    assert session.token_ids == ledger
    assert "stone" in suffix["text"] and "river" not in suffix["text"]
    second = record(model, tokenizer, suffix["ids"], tmp_path / "second", session,
                    generation=options(max_context=2048))
    assert second["prompt_ids"] == ledger + suffix["ids"]
    assert second["prompt_ids"][:len(ledger)] == first["prompt_ids"] + first["generated_ids"]


def test_turn_boundary_deduplicates_consumed_special_eos_but_never_ordinary_text():
    _, tokenizer = lab.make_tiny_model(seed=107)
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}"
        "{{ '<' + message['role'] + '>' + message['content'] + eos_token }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    )
    session = lab.LiveSession(token_ids=tokenizer.encode("water") + [tokenizer.eos_token_id])
    suffix = lab.build_live_prompt(tokenizer, "association", "stone", session, template="chat")
    complete_suffix_ids = tokenizer.encode(suffix["text"], add_special_tokens=False)
    assert complete_suffix_ids[0] == tokenizer.eos_token_id
    assert suffix["boundary_dedup_ids"] == [tokenizer.eos_token_id]
    assert suffix["ids"] == complete_suffix_ids[1:]
    assert session.token_ids[-1] == tokenizer.eos_token_id

    # Newline stopping must not justify stripping lexical/whitespace overlap.
    session = lab.LiveSession(token_ids=tokenizer.encode("water\n"))
    plain = lab.build_live_prompt(tokenizer, "association", "stone", session, template="plain")
    assert plain["boundary_dedup_ids"] == []
    assert plain["ids"] == tokenizer.encode(plain["text"], add_special_tokens=False)


def test_bank_parser_reports_rejections_duplicates_and_extras_without_filler():
    raw = '["river", "river", "ice cream", "", "stone", "glass", "leaf"]'
    parsed = lab.parse_word_bank(raw, 3)
    assert parsed["accepted"] is True
    assert parsed["words"] == ["river", "stone", "glass"]
    assert "river" in parsed["duplicates"]
    assert parsed["rejected"]
    assert "leaf" in parsed["extras"]
    short = lab.parse_word_bank('["river", "river"]', 3)
    assert short["accepted"] is False and short["words"] == ["river"]
    assert len(short["words"]) < 3


def test_bank_line_fallback_and_hash_replay_are_order_sensitive():
    parsed = lab.parse_word_bank("1. river\n2. glass\n- stone\n", 3)
    assert parsed["accepted"] is True
    assert parsed["words"] == ["river", "glass", "stone"]
    words = lab.normalize_word_bank(parsed["words"])
    reloaded = json.loads(json.dumps(words, ensure_ascii=False))
    assert lab.word_bank_hash(words) == lab.word_bank_hash(reloaded)
    assert lab.word_bank_hash(words) != lab.word_bank_hash(list(reversed(words)))
    with pytest.raises(ValueError):
        lab.normalize_word_bank(["river", "ice cream"])
    with pytest.raises(ValueError):
        lab.normalize_word_bank(["river", "river"])


def test_loading_frozen_bank_checks_the_saved_content_hash(tmp_path):
    words = ["river", "glass", "stone"]
    path = tmp_path / "bank.json"
    saved = {"words": words, "sha256": lab.word_bank_hash(words), "source": "test_fixture"}
    path.write_text(json.dumps(saved), encoding="utf-8")
    loaded = lab.load_word_bank(path)
    assert loaded["words"] == words and loaded["sha256"] == saved["sha256"]
    saved["words"] = ["river", "leaf", "stone"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)checksum|hash|changed"):
        lab.load_word_bank(path)


def test_order_randomness_is_independent_from_sample_seed_repeat_and_condition():
    words = ["river", "glass", "stone", "leaf", "bell", "rain", "bird", "road"]
    before = words.copy()
    plan = lab.build_trial_plan(words, seeds=[43, 44, 45], repeats=2,
                                order="shuffle", order_seeds=(101, 102),
                                conditions=("live", "reset"))
    assert words == before
    assert len(plan) == 3 * 2 * 2 * 2
    assert len({p["trial_id"] for p in plan}) == len(plan)
    orders = {}
    for row in plan:
        assert sorted(row["bank_order"]) == list(range(len(words)))
        orders.setdefault(row["order_seed"], set()).add(tuple(row["bank_order"]))
    assert set(orders) == {101, 102}
    assert all(len(value) == 1 for value in orders.values())
    assert orders[101] != orders[102]
    changed_seeds = lab.build_trial_plan(words, seeds=[900], repeats=1,
                                         order="shuffle", order_seeds=(101, 102))
    for row in changed_seeds:
        assert tuple(row["bank_order"]) in orders[row["order_seed"]]
    fixed = lab.build_trial_plan(words, seeds=[43, 44], repeats=2,
                                 order="fixed", order_seeds=(101, 102),
                                 conditions=("live", "reset"))
    assert len(fixed) == 2 * 2 * 2
    assert all(p["bank_order"] == list(range(len(words))) and p["order_seed"] is None for p in fixed)


@pytest.mark.parametrize("text,stop,surface,eligible", [
    ("river", "eos", True, True),
    (" 水 \n", "newline", True, True),
    ("red fox", "eos", False, False),
    ("water\nstone", "eos", False, False),
    ("", "eos", False, False),
    ("forest", "length", True, False),
])
def test_one_word_check_preserves_entire_response_and_marks_censoring(text, stop, surface, eligible):
    result = lab.check_one_word(text, stop)
    assert result["surface_ok"] is surface
    assert result["eligible"] is eligible
    assert result["text"] == text.strip()
    if not eligible:
        assert result["reasons"]


def test_experiment_reuses_frozen_bank_and_repeats_exact_ids_then_encodes_isolated_texts(tmp_path, monkeypatch):
    """Exercise the real runner, recorders, on-disk indices and self encoder."""
    model, tokenizer = lab.make_tiny_model(seed=109)
    monkeypatch.setattr(lab, "load_model", lambda args: (model, tokenizer))
    words = ["moss", "quartz"]
    bank_path = tmp_path / "fixed_bank.json"
    bank_path.write_text(json.dumps({"words": words, "sha256": lab.word_bank_hash(words), "language": "en"}))
    out = tmp_path / "experiment"
    args = lab.make_parser().parse_args([
        "experiment", "--model", "offline-test-fixture", "--out", str(out),
        "--bank", str(bank_path), "--seeds", "43,44",
        "--repeats", "2", "--condition", "both", "--max-new-tokens", "2",
        "--max-context", "1024", "--layers", "0", "--no-logits", "--no-plot",
        "--similarity", "self", "--template", "plain", "--cpu-threads", "1",
    ])
    expected_rows = 2 * 2 * 2 * 2
    scoring_inputs = []
    generation_with_hooks = []
    original_forward = model.forward

    def observe_forward(*positional, **kwargs):
        # Transformers may install its own persistent hooks when hidden states
        # are first requested. Only the harness's observation hooks must close.
        active_hooks = sum(
            getattr(hook, "__qualname__", "").startswith("HookRecorder.")
            for module in model.modules() for hook in module._forward_hooks.values()
        )
        if kwargs.get("output_hidden_states"):
            # Analysis starts after all committed responses, outside the trial loop.
            committed = lab.read_jsonl(out / "responses.jsonl")
            assert len(committed) == expected_rows
            assert active_hooks == 0
            assert kwargs["use_cache"] is False and kwargs.get("past_key_values") is None
            scoring_inputs.append(kwargs["input_ids"][0].tolist())
        else:
            generation_with_hooks.append(active_hooks > 0)
        return original_forward(*positional, **kwargs)

    monkeypatch.setattr(model, "forward", observe_forward)
    assert lab.run_experiment(args) == out
    assert generation_with_hooks and all(generation_with_hooks)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["requested_bank_language"] is None
    assert manifest["effective_association_language"] == "en"
    assert manifest["completed_trials"] == 8 and manifest["completed_responses"] == expected_rows
    assert json.loads((out / "bank.json").read_text())["sha256"] == lab.word_bank_hash(words)
    rows = lab.read_jsonl(out / "responses.jsonl")
    generated_by_case = {}
    for row in rows:
        meta = json.loads((out / row["event_directory"] / "meta.json").read_text())
        assert row["cue"] in meta["prompt_text"]
        assert words[1 - row["bank_index"]] not in meta["prompt_text"]
        if row["condition"] == "reset" or row["position"] == 0:
            assert meta["past_tokens"] == 0 and not meta["cache_reused"]
        else:
            assert meta["past_tokens"] > 0 and meta["cache_reused"]
        key = (row["sample_seed"], row["condition"], row["position"])
        generated_by_case.setdefault(key, []).append(meta["generated_ids"])
        assert not (out / row["event_directory"] / "raw_logits.npy").exists()
    assert all(len(versions) == 2 and versions[0] == versions[1]
               for versions in generated_by_case.values())

    analysis = out / "analysis"
    texts = json.loads((analysis / "embedding_texts.json").read_text())
    assert scoring_inputs == [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    assert len(texts) == len(set(texts))
    matrix = np.load(analysis / "similarity.npy")
    assert matrix.shape == (expected_rows, len(words)) and matrix.dtype == np.float32
    bank_vectors = np.load(analysis / "bank_vectors.npy")
    response_vectors = np.load(analysis / "response_vectors.npy")
    np.testing.assert_allclose(matrix, np.clip(response_vectors @ bank_vectors.T, -1, 1),
                               rtol=2e-6, atol=2e-6, equal_nan=True)
    summaries = json.loads((analysis / "cue_summary.json").read_text())
    assert len(summaries) == 4
    assert all(summary["independent_seed_count"] == 2 and summary["n"] == 4 for summary in summaries)


def test_model_generated_bank_cache_is_released_before_any_association_trial(tmp_path, monkeypatch):
    """Script bank logits while retaining real model forward/KV and production orchestration."""
    model, tokenizer = lab.make_tiny_model(seed=113)
    monkeypatch.setattr(lab, "load_model", lambda args: (model, tokenizer))
    words = ["moss", "quartz"]
    scripted_bank = tokenizer.encode(json.dumps(words), add_special_tokens=False) + [tokenizer.eos_token_id]
    marker = "BANK_ONLY_MARKER_9821"
    prompt_file = tmp_path / "bank_prompt.txt"
    prompt_file.write_text(marker + ": return two words as a JSON array.")
    out = tmp_path / "generated_experiment"
    args = lab.make_parser().parse_args([
        "experiment", "--model", "offline-test-fixture", "--out", str(out),
        "--bank-size", "2", "--bank-language", "en", "--bank-attempts", "1",
        "--bank-max-new-tokens", "64", "--bank-prompt", str(prompt_file),
        "--n-seeds", "1", "--condition", "live", "--max-new-tokens", "2",
        "--max-context", "1024", "--no-hooks", "--no-logits", "--no-plot",
        "--similarity", "none", "--template", "plain", "--cpu-threads", "1",
    ])
    original_forward = model.forward
    original_record = lab.record_generation
    state = {"bank_active": False, "bank_step": 0, "bank_session": None, "trial_calls": 0}

    def scripted_forward(*positional, **kwargs):
        result = original_forward(*positional, **kwargs)
        if state["bank_active"]:
            token_id = scripted_bank[min(state["bank_step"], len(scripted_bank) - 1)]
            state["bank_step"] += 1
            result.logits.fill_(-100.0)
            result.logits[..., token_id] = 100.0
        return result

    def observe_record(*positional, **kwargs):
        bank_call = kwargs.get("metadata", {}).get("role") == "bank_generation"
        if bank_call:
            state["bank_session"] = kwargs["session"]
            state["bank_active"] = True
            try:
                result = original_record(*positional, **kwargs)
                assert kwargs["session"].past_key_values is not None
                assert kwargs["session"].token_ids
                return result
            finally:
                state["bank_active"] = False
        bank_session = state["bank_session"]
        assert bank_session is not None
        assert bank_session.past_key_values is None and bank_session.token_ids == []
        assert bank_session.reset_count == 1
        assert kwargs["session"] is not bank_session
        assert marker not in kwargs["metadata"]["prompt_text"]
        if state["trial_calls"] == 0:
            assert kwargs["session"].past_key_values is None and kwargs["session"].token_ids == []
        state["trial_calls"] += 1
        return original_record(*positional, **kwargs)

    monkeypatch.setattr(model, "forward", scripted_forward)
    monkeypatch.setattr(lab, "record_generation", observe_record)
    lab.run_experiment(args)
    assert state["trial_calls"] == 2
    bank = json.loads((out / "bank.json").read_text())
    assert bank["words"] == words and bank["source"] == "model-generated"
    assert bank["provenance"]["cache_discarded_before_trials"] is True
    bank_trace = json.loads((out / "bank_generation" / "attempt_00" / "meta.json").read_text())
    assert bank_trace["generated_ids"] == scripted_bank
    assert marker in tokenizer.decode(bank_trace["prompt_ids"])
    for row in lab.read_jsonl(out / "responses.jsonl"):
        meta = json.loads((out / row["event_directory"] / "meta.json").read_text())
        assert marker not in tokenizer.decode(meta["prompt_ids"])
        assert words[1 - row["bank_index"]] not in meta["prompt_text"]


def test_all_bank_similarity_ranking_and_seen_future_partitions_use_trial_order():
    bank_vectors = np.eye(3, dtype=np.float32)
    # Interleave another trial to ensure previous-response links cannot leak.
    rows = [
        {"trial_id": "a", "position": 0, "bank_index": 2},
        {"trial_id": "a", "position": 1, "bank_index": 0},
        {"trial_id": "b", "position": 0, "bank_index": 1},
        {"trial_id": "a", "position": 2, "bank_index": 1},
    ]
    response_vectors = np.array([[0, 0, 2], [3, 4, 0], [0, 0, 0], [-2, 0, 0]], dtype=np.float32)
    schedule = [{"trial_id": "a", "bank_order": [2, 0, 1]},
                {"trial_id": "b", "bank_order": [1, 2, 0]}]
    scored, matrix = lab.similarity_rows(rows, bank_vectors, response_vectors, schedule)
    expected = np.array([[0, 0, 1], [.6, .8, 0], [np.nan] * 3, [-1, 0, 0]], dtype=np.float32)
    np.testing.assert_allclose(matrix, expected, atol=1e-6, equal_nan=True)
    assert scored[0]["nearest_bank_index"] == 2 and scored[0]["cue_rank"] == 1
    assert scored[0]["previously_seen_max_cosine"] is None
    assert scored[0]["future_bank_max_cosine"] == pytest.approx(0)
    assert scored[1]["cue_cosine"] == pytest.approx(.6)
    assert scored[1]["nearest_bank_index"] == scored[1]["nearest_other_bank_index"] == 1
    assert scored[1]["cue_rank"] == 2
    assert scored[1]["previously_seen_max_cosine"] == pytest.approx(0)
    assert scored[1]["future_bank_max_cosine"] == pytest.approx(.8)
    assert scored[1]["previous_response_cosine"] == pytest.approx(0)
    assert scored[2]["cue_cosine"] is None and scored[2]["nearest_bank_index"] is None
    assert scored[2]["previous_response_cosine"] is None
    assert scored[3]["cue_rank"] == 1  # Competition rank: tied maxima share rank one.
    assert scored[3]["nearest_other_bank_index"] == 2
    assert scored[3]["previously_seen_max_cosine"] == pytest.approx(0)
    assert scored[3]["future_bank_max_cosine"] is None
    assert scored[3]["previous_response_cosine"] == pytest.approx(-.6)


def test_optional_sentence_transformer_path_uses_its_saved_modules_and_prompt_config(tmp_path):
    """Build a real local encoder; no network or pretrained language claim."""
    st = pytest.importorskip("sentence_transformers")
    modules = pytest.importorskip("sentence_transformers.sentence_transformer.modules")
    model, tokenizer = lab.make_tiny_model(seed=127)
    transformer_path = tmp_path / "tiny_transformer"
    model.save_pretrained(transformer_path)
    tokenizer.save_pretrained(transformer_path)
    transformer = modules.Transformer(
        str(transformer_path), model_kwargs={"local_files_only": True}, max_seq_length=128,
    )
    encoder = st.SentenceTransformer(
        modules=[transformer, modules.Pooling(embedding_dimension=48, pooling_mode="mean"), modules.Normalize()],
        device="cpu", prompts={"study": "Probe: "}, default_prompt_name="study",
    )
    encoder_path = tmp_path / "sentence_encoder"
    encoder.save_pretrained(str(encoder_path), create_model_card=False)
    out = tmp_path / "observations"
    out.mkdir()
    words = ["river", "window"]
    lab.atomic_json(out / "bank.json", {"words": words, "sha256": lab.word_bank_hash(words)})
    schedule = [{"trial_id": "a", "bank_order": [0, 1]}, {"trial_id": "b", "bank_order": [0, 1]}]
    lab.atomic_json(out / "schedule.json", schedule)
    for trial_id, event_id, cue_index, response in [("a", 0, 0, "water"), ("a", 1, 1, "light"), ("b", 0, 0, "")]:
        lab.append_jsonl(out / "responses.jsonl", {
            "trial_id": trial_id, "event_id": event_id, "position": event_id,
            "bank_index": cue_index, "cue": words[cue_index], "response": response,
            "eligible": bool(response), "condition": "live", "sample_seed": 43,
            "repeat_index": 0 if trial_id == "a" else 1, "order_seed": None,
        })
    args = lab.make_parser().parse_args([
        "experiment", "--model", "unused-generative-model", "--out", str(out),
        "--similarity", "sentence-transformer", "--embedding-model", str(encoder_path),
        "--embedding-device", "cpu", "--local-files-only", "--no-plot",
    ])
    destination = lab.analyze_experiment(out, args)
    texts = json.loads((destination / "embedding_texts.json").read_text())
    assert texts == ["river", "window", "water", "light"]
    expected = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    saved_bank = np.load(destination / "bank_vectors.npy")
    saved_responses = np.load(destination / "response_vectors.npy")
    np.testing.assert_allclose(saved_bank, expected[:2], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(saved_responses[:2], expected[2:], rtol=2e-5, atol=2e-6)
    assert np.isnan(saved_responses[2]).all()
    similarity = np.load(destination / "similarity.npy")
    np.testing.assert_allclose(similarity[:2], expected[2:] @ expected[:2].T, rtol=2e-5, atol=2e-6)
    assert np.isnan(similarity[2]).all()
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["method"] == "sentence-transformer"
    # Save the effective dictionary, including package-provided defaults.
    assert metadata["prompts"] == encoder.prompts
    assert metadata["prompts"]["study"] == "Probe: "
    assert metadata["default_prompt_name"] == "study"
    assert metadata["max_seq_length"] == 128
    assert metadata["saved_pipeline_verified"] is True
    assert set(metadata["modules"].values()) == {"Transformer", "Pooling", "Normalize"}
    assert metadata["row_keys"] == [["a", 0], ["a", 1], ["b", 0]]


def test_sentence_transformer_mode_rejects_plain_checkpoint_without_saved_pipeline(tmp_path):
    pytest.importorskip("sentence_transformers")
    model, tokenizer = lab.make_tiny_model(seed=131)
    checkpoint = tmp_path / "plain_causal_lm"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    assert not (checkpoint / "modules.json").exists()
    out = tmp_path / "observations"
    out.mkdir()
    lab.atomic_json(out / "bank.json", {"words": ["river"], "sha256": lab.word_bank_hash(["river"])})
    lab.atomic_json(out / "schedule.json", [])
    (out / "responses.jsonl").touch()
    args = lab.make_parser().parse_args([
        "experiment", "--model", str(checkpoint), "--out", str(out),
        "--similarity", "sentence-transformer", "--embedding-model", str(checkpoint),
        "--embedding-device", "cpu", "--local-files-only", "--no-plot",
    ])
    with pytest.raises(ValueError, match=r"modules\.json.*--similarity self"):
        lab.analyze_experiment(out, args)
    assert not (out / "analysis" / "similarity.npy").exists()


def test_final_index_snapshots_restore_committed_rows_after_a_missing_last_append(tmp_path, monkeypatch):
    """A missing append must not silently drop a completed event from analysis."""
    model, tokenizer = lab.make_tiny_model(seed=137)
    monkeypatch.setattr(lab, "load_model", lambda args: (model, tokenizer))
    bank_path = tmp_path / "bank.json"
    lab.atomic_json(bank_path, {"words": ["moss", "quartz"], "language": "en"})
    out = tmp_path / "experiment"
    args = lab.make_parser().parse_args([
        "experiment", "--model", "offline-test-fixture", "--out", str(out),
        "--bank", str(bank_path), "--n-seeds", "1", "--max-new-tokens", "1",
        "--max-context", "1024", "--no-hooks", "--no-logits", "--no-plot",
        "--similarity", "none", "--template", "plain", "--cpu-threads", "1",
    ])
    original_append = lab.append_jsonl
    original_analyze = lab.analyze_experiment
    dropped = []
    analyzed = []

    def lose_last_append(path, row):
        if row.get("event_id") == 1:
            dropped.append(str(path))
            return
        return original_append(path, row)

    def require_complete_indices_before_analysis(path, args, **kwargs):
        root_rows = lab.read_jsonl(path / "responses.jsonl")
        trial_rows = lab.read_jsonl(path / "trials" / "t0000" / "events.jsonl")
        assert [r["event_id"] for r in root_rows] == [0, 1]
        assert [r["event_id"] for r in trial_rows] == [0, 1]
        for row in root_rows:
            meta = json.loads((path / row["event_directory"] / "meta.json").read_text())
            assert meta["status"] == "complete" and meta["response"] == row["response"]
        assert "## R1" in (path / "transcript.md").read_text()
        assert "## Event 1" in (path / "trials" / "t0000" / "transcript.md").read_text()
        analyzed.append(True)
        return original_analyze(path, args, **kwargs)

    monkeypatch.setattr(lab, "append_jsonl", lose_last_append)
    monkeypatch.setattr(lab, "analyze_experiment", require_complete_indices_before_analysis)
    lab.run_experiment(args)
    assert len(dropped) == 2 and analyzed == [True]
    assert len(lab.read_jsonl(out / "analysis" / "responses.jsonl")) == 2
    assert json.loads((out / "manifest.json").read_text())["completed_responses"] == 2
