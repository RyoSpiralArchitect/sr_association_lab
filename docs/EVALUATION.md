# Evaluate saved scene experiments

`evaluate RUN [--out NEW_DIRECTORY]` reads a finished `imagine` recording.
It never loads the generation model, changes sampled text or eligibility,
reuses trial KV/RNG, or requires the older cosine analysis. Existing reports
are not overwritten. The default destination is `RUN/evaluation`.

`report.html` works offline, with a seed/repeat selector, side-by-side live/reset
text, measurements and exact lexical spans. JSON files retain all observations.

## Separate measurements

| Measurement | Observation | Limit |
| --- | --- | --- |
| Exact prior-output repetition | Same trial, NFKC/casefold and collapsed whitespace; punctuation retained | Not paraphrase, or proof a reset output saw its predecessor |
| Repeated trigram fraction | `1 - unique / total` three-unit sequences within one response | Not narrative quality |
| Previous lexical Jaccard | Set overlap after fixed English stop-word removal, with source offsets | Not entity identity, coreference or synonyms |
| Directed NLI | Joint premise → hypothesis classifier: entailment/neutral/contradiction | Not calibrated truth probability or narrative consistency |

Lexical units are coarse Unicode alphanumeric spans and individual CJK characters,
not linguistic segmentation. Original character offsets/text and the stop-word
list are retained. Empty content sets and fewer than three units yield null for
the corresponding metric. No weighted quality aggregate or automatic verdict is
produced. Failed surface responses remain visible; cosine remains a separate proxy.

## Optional local NLI

Install PyTorch and the shared requirements. A compatible classifier must have
safe tensor weights and explicit semantic `id2label` for all three labels.
Generic `LABEL_0` mappings and binary classifiers are rejected, not guessed.

One English example is
[MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli):

```bash
python sr_association_lab.py evaluate runs/night-station \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
  --nli-revision 6f5cf0a2b59cabb106aca4c287eed12e357e90eb \
  --allow-download --nli-device cpu --nli-batch-size 4 \
  --out runs/night-station/evaluation-nli
```

This example is not a Japanese evaluator; the chosen model's language/domain
limits apply. Acquisition is explicit and respects environment network settings.
Scene texts are not sent to remote inference services. Remote model code is disabled.

The input limit is the minimum of `--nli-max-length` (512 by default) and declared
model/tokenizer limits. Full pairs, including special tokens, are tokenized first.
Oversized pairs get `input_too_long` and null scores; there is no truncation or
replacement text. Empty pairs are unavailable too. Input IDs, counts, raw logits,
label mapping, package versions, device/dtypes and model identity are saved.
Local model files are hashed; Hub models record resolved commits when available.

Three fixed controls check entailment, explicit negation and an unrelated hypothesis.
Inputs, expected labels, scores and failures are saved. These are smoke checks,
not narrative calibration or validation. Softmax scores remain uncalibrated.
Novel events may be neutral; temporal changes and uncertain referents may produce
false contradiction candidates. No threshold automatically declares an error.

Four additional narrative probes expose sensitivity to temporal framing and
different referents. The unmarked standing → sitting pair has no gold NLI label;
compare its scores with an explicit earlier → later version. Separate cases
use another traveler and a subsequent new event. The three explicit cases are
expected neutral, but passing them still does not validate judgments of real
stories. Full probe inputs/scores appear in `probes.json` and the HTML report.

## Scope, bounds and aggregation

Each response is paired with the frozen origin. Live adds up to the last
`--history-window` (default 8) prior responses from the same trial; omitted counts
are explicit. Reset adds its adjacent output as a diagnostic with
`used_as_context=false`, excluded from the maximum-context contradiction field.
No other trial supplies history. Each full recorded response is one unit,
including malformed/multi-sentence output; no hidden splitting or repair occurs.

`--max-pairs` defaults to 4096 experiment pairs, plus three controls and four
probes. Exceeding it fails before model loading. Each NLI batch commits a progress snapshot;
interrupted/error metadata never claims completion. Source hashes are checked
again before writing a completed report to reject inputs changed during scoring.

Differences match `(sample_seed, repeat_index, step_index)`. Step 0 is a text-match
control, excluded from effects. Average steps within repeat, repeats within seed,
then seeds equally. Report per-seed differences, matched counts, missing pairs,
and both all-output and both-surface-eligible populations. Repeats do not add
seeds. These are descriptive summaries without p-values or response-independence
claims. Matching seeds does not align later draws after different output lengths.

Maximum-context contradiction only selects review candidates. It is excluded from
paired metrics: longer histories offer more chances for a high score. Adjacent
outputs also have different conditioning semantics in live and reset.

## Methodological basis

[Cross-Encoder documentation](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)
describes joint sentence-pair classification, distinct from independently encoded
cosine. [SummaC](https://aclanthology.org/2022.tacl-1.10/) motivates attention to
NLI input granularity. This implementation is not SummaC and does not inherit its
benchmark results; creative continuation also differs from faithful summarization.

For stronger claims, annotate entity identity, time, state changes and
contradictions with evidence, then validate against held-out human judgments.
Keep those judgments separate from model suggestions and surface eligibility.
