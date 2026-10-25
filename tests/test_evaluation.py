"""Saved-run evaluation contracts; tiny classifiers test plumbing, not semantics."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sr_association_lab as lab


def fixture_run(path, *, repeats=1):
    path.mkdir()
    origin = {"scene": "A lamp shines at a station.", "language": "en"}
    origin["sha256"] = lab.scene_hash(origin["scene"])
    schedule, rows = [], []
    for seed in (43, 44):
        for repeat in range(repeats):
            for condition in ("live", "reset"):
                trial = dict(trial_id=f"t{len(schedule):04d}", sample_seed=seed,
                             repeat_index=repeat, condition=condition, steps=3)
                schedule.append(trial)
                texts = (["A woman carries a red suitcase.", "She carries the red suitcase.", "She carries the red suitcase."]
                         if condition == "live" else
                         ["A woman carries a red suitcase.", "A lamp shines above the platform.", "A suitcase rests by the door."])
                for step, text in enumerate(texts):
                    rows.append({**trial, "step_index": step, "event_id": step, "response": text,
                                 "origin_sha256": origin["sha256"], "eligible": True})
    lab.atomic_json(path / "origin.json", origin)
    lab.atomic_json(path / "schedule.json", schedule)
    lab.atomic_jsonl(path / "responses.jsonl", rows)
    lab.atomic_json(path / "manifest.json", {"kind": "imagination_experiment", "status": "complete",
        "origin_sha256": origin["sha256"], "completed_responses": len(rows), "completed_trials": len(schedule)})
    return origin, rows


def args_for(root, *flags):
    return lab.make_parser().parse_args(["evaluate", str(root), *map(str, flags)])


def test_evaluate_saved_run_without_models_preserves_sources_and_escapes_report(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _, rows = fixture_run(root)
    rows[0]["response"] = "<script>alert('not executable')</script> A red suitcase."
    lab.atomic_jsonl(root / "responses.jsonl", rows)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    monkeypatch.setattr(lab, "load_model", lambda *a, **k: pytest.fail("Generation model must not load"))
    out = lab.run_scene_evaluation(args_for(root))
    assert all((root / name).read_bytes() == data for name, data in before.items())
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["nli"] is None
    assert metadata["source_sha256"] == {name: hashlib.sha256(data).hexdigest() for name, data in before.items()}
    report = (out / "report.html").read_text()
    assert "<script>alert" not in report and "&lt;script&gt;alert" in report
    assert "NLI was not run" in report
    scored = lab.read_jsonl(out / "responses.jsonl")
    assert [r["response"] for r in scored] == [r["response"] for r in rows]
    assert all(r["evaluation"]["previous_nli_contradiction"] is None for r in scored)
    with pytest.raises(ValueError, match="already exists"):
        lab.run_scene_evaluation(args_for(root))


@pytest.mark.parametrize("damage", ["duplicate", "missing", "origin", "condition", "seed", "incomplete"])
def test_evaluation_rejects_ambiguous_or_incomplete_input(tmp_path, damage):
    root = tmp_path / "run"
    _, rows = fixture_run(root)
    if damage == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif damage == "missing":
        rows[1]["step_index"] = rows[1]["event_id"] = 7
    elif damage == "origin":
        rows[1]["origin_sha256"] = "wrong"
    elif damage == "condition":
        rows[1]["condition"] = "reset"
    elif damage == "seed":
        rows[1]["sample_seed"] = 99
    else:
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["status"] = "running"
        lab.atomic_json(root / "manifest.json", manifest)
    lab.atomic_jsonl(root / "responses.jsonl", rows)
    with pytest.raises(ValueError):
        lab.run_scene_evaluation(args_for(root))
    assert not (root / "evaluation").exists()


def test_pairs_keep_reset_predecessors_out_of_context_and_bound_history(tmp_path):
    origin, rows = fixture_run(tmp_path / "run")
    scored, pairs = lab.build_scene_evaluation(origin, list(reversed(rows)), history_window=1)
    for pair in pairs:
        assert pair["used_as_context"] == (pair["premise_step"] == -1 or pair["condition"] == "live")
        if pair["premise_step"] >= 0:
            assert pair["premise_step"] == pair["step_index"] - 1
    later = next(r for r in scored if r["condition"] == "live" and r["step_index"] == 2)
    assert later["evaluation"]["history_responses_omitted"] == 1
    assert later["evaluation"]["exact_repeat_step"] == 1
    reset_pair = next(p for p in pairs if p["condition"] == "reset" and p["is_previous"] and p["premise_step"] >= 0)
    reset_pair.update(status="scored", probabilities={"entailment": .005, "neutral": .005, "contradiction": .99})
    lab.attach_scene_nli(scored, pairs)
    row = next(r for r in scored if (r["trial_id"], r["step_index"]) == (reset_pair["trial_id"], reset_pair["step_index"]))
    assert row["evaluation"]["previous_nli_contradiction"] == .99
    assert row["evaluation"]["max_context_contradiction"] is None
    with pytest.raises(ValueError, match="budget"):
        lab.build_scene_evaluation(origin, rows, max_pairs=1)


def test_lexical_evidence_is_exact_spans_not_paraphrase_or_semantic_score():
    left, right = "The ＲＥＤ suitcase stood by a lamp.", "A red suitcase moved."
    result = lab.scene_lexical_pair(left, right)
    assert {r["unit"] for r in result["shared_units"]} == {"red", "suitcase"}
    for record in result["shared_units"]:
        for field, source in (("premise_span", left), ("hypothesis_span", right)):
            span = record[field]
            assert source[span["start"]:span["end"]] == span["text"]
    assert lab.scene_lexical_pair("", "A lamp.")["jaccard"] is None
    assert [u["unit"] for u in lab.scene_units("灯り。扉")]==["灯", "り", "扉"]
    assert lab.scene_lexical_pair("A suitcase.", "Some luggage.")["jaccard"] == 0


def test_paired_summary_uses_seed_clusters_excludes_initial_and_reports_coverage(tmp_path):
    origin, rows = fixture_run(tmp_path / "run", repeats=2)
    scored, _ = lab.build_scene_evaluation(origin, rows)
    # Four repeats are still only two distinct seeds. First turns are not effects.
    for r in scored:
        r["evaluation"]["previous_nli_entailment"] = (100 if r["step_index"] == 0 else
            (0.8 if r["condition"] == "live" else 0.2))
    scored[1]["eligible"] = False
    summary = lab.summarize_scene_evaluation(scored)
    effect = summary["paired"]["all_responses"]["previous_nli_entailment"]
    assert effect["distinct_seed_count"] == 2 and effect["matched_response_pairs"] == 8
    assert effect["mean_seed_difference"] == pytest.approx(.6)
    assert all(r["repeat_count"] == 2 for r in effect["by_seed"])
    assert summary["paired"]["both_surface_eligible"]["previous_nli_entailment"]["matched_response_pairs"] == 7
    assert len(summary["initial_pairs"]) == 4 and all(r["text_matches"] for r in summary["initial_pairs"])
    incomplete = [r for r in scored if not (r["sample_seed"] == 44 and r["condition"] == "reset")]
    assert lab.summarize_scene_evaluation(incomplete)["unmatched_post_initial_keys"] == 4


def test_nli_labels_must_be_explicit_and_order_is_not_guessed():
    config = SimpleNamespace(num_labels=3, id2label={0: "contradiction", 1: "ENTAILMENT", 2: "neutral"})
    assert lab.nli_label_indices(config) == {"contradiction": 0, "entailment": 1, "neutral": 2}
    with pytest.raises(ValueError, match="not guessed"):
        lab.nli_label_indices(SimpleNamespace(num_labels=3, id2label={i: f"LABEL_{i}" for i in range(3)}))


def make_classifier(path):
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer_impl = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "A": 2, "woman": 3, "lamp": 4, ".": 5}, unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl, unk_token="[UNK]", pad_token="[PAD]", model_max_length=32)
    config = BertConfig(vocab_size=6, hidden_size=12, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=32, num_labels=3,
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
        label2id={"contradiction": 0, "entailment": 1, "neutral": 2})
    model = BertForSequenceClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([-2., 3., 0.]))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return model, tokenizer


def test_nli_real_classifier_alignment_bounds_and_provenance(tmp_path):
    model, tokenizer = make_classifier(tmp_path / "model")
    pairs = [{"pair_id": "short", "premise": "A woman.", "hypothesis": "A lamp."},
             {"pair_id": "long", "premise": "woman " * 40, "hypothesis": "A lamp."},
             {"pair_id": "empty", "premise": "", "hypothesis": "A lamp."}]
    progress = []
    results, controls, metadata = lab.score_scene_nli(pairs, str(tmp_path / "model"), max_length=32,
        batch_size=2, progress_callback=lambda items: progress.append(copy.deepcopy(items)))
    import torch
    with torch.inference_mode():
        direct = model.eval()(**tokenizer("A woman.", "A lamp.", return_tensors="pt")).logits[0]
    expected = torch.softmax(direct, dim=-1).tolist()
    assert results[0]["probabilities"] == pytest.approx(dict(contradiction=expected[0], entailment=expected[1], neutral=expected[2]))
    assert results[1]["status"] == "input_too_long" and results[1]["probabilities"] is None
    assert results[1]["token_count"] > 32
    assert len(results[0]["input_ids"]) == results[0]["token_count"]
    assert results[2]["status"] == "empty_text"
    assert progress[0][0]["status"] == "pending" and progress[-1][0]["status"] == "scored"
    assert metadata["truncation"] is False and metadata["effective_max_length"] == 32
    assert metadata["local_file_sha256"]["model.safetensors"]
    assert len(controls) == 3 and sum(c["passed"] for c in controls) == 1  # Constant classifier is not a validated judge.
    assert metadata["control_status"] == "failed"


def test_source_mutation_during_scoring_fails_without_complete_report(tmp_path, monkeypatch):
    root = tmp_path / "run"
    fixture_run(root)
    def replace_source(pairs, *args, **kwargs):
        with (root / "responses.jsonl").open("a") as stream:
            stream.write("\n")
        return pairs, [], {"control_count": 0, "control_pass_count": 0}
    monkeypatch.setattr(lab, "score_scene_nli", replace_source)
    with pytest.raises(ValueError, match="changed during"):
        lab.run_scene_evaluation(args_for(root, "--nli-model", "unused"))
    assert json.loads((root / "evaluation/metadata.json").read_text())["status"] == "error"
    assert not (root / "evaluation/report.html").exists()


def test_interrupt_has_explicit_receipt_and_budget_precedes_model_load(tmp_path, monkeypatch):
    root = tmp_path / "run"
    fixture_run(root)
    calls = []
    def interrupt(*args, **kwargs):
        calls.append(True)
        raise KeyboardInterrupt
    monkeypatch.setattr(lab, "score_scene_nli", interrupt)
    with pytest.raises(ValueError, match="budget"):
        lab.run_scene_evaluation(args_for(root, "--nli-model", "unused", "--max-pairs", 1))
    assert calls == [] and not (root / "evaluation").exists()
    with pytest.raises(KeyboardInterrupt):
        lab.run_scene_evaluation(args_for(root, "--nli-model", "unused"))
    assert json.loads((root / "evaluation/metadata.json").read_text())["status"] == "interrupted"
