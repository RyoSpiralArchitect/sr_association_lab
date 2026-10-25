# Trace schema

The runtime remains one file: `sr_association_lab.py`. Generation requires a new
output directory. `plot` reads saved records without loading model weights and
may replace diagnostic PNGs. This schema describes fixed-bank association, fixed-origin active imagination,
and their per-trial traces. PyTorch and MLX share the event format. Standalone
`run` uses the same trace recorder. Only `sr_association_lab.py` is required as
application source; development fragments are not runtime modules.

## Units and identity

- **Bank:** one ordered, immutable list of stimulus words.
- **Origin:** one immutable scene text from which imagination trials start.
- **Trial:** one traversal of the bank, or one sequence of scene outputs, under
  a sampling seed, repeat index, and `live` or `reset` condition. Bank trials
  additionally specify presentation order.
- **Event:** one cue and its entire response, potentially several subword tokens.
- **Row:** one model forward readout at the last consumed position. A completed
  event with N sampled tokens has N+1 valid rows, including final consume.

`schema_version` is `association-lab/0.3`. Older bank recordings remain readable by the offline tools.
This identifier describes the saved format, not an application release. The
manifest identifies the harness as `sr_association_lab` and records its exact
source SHA-256; application release numbers are not maintained. IDs and array positions are zero-based;
`cue_rank` is one-based, with tied values sharing the same rank.
Trial IDs are strings such as `t0000`; event directories are zero-padded integers
such as `events/000000`. `bank_index` always refers to the original bank order,
while `position` refers to presentation order within a bank trial. Imagination
uses `step_index`; step 0 is the first generated sentence after the fixed
origin, not the origin itself.

## Fixed bank

`bank.json` contains:

- `words`: the accepted ordered list of strings.
- `sha256`: SHA-256 of the canonical UTF-8 JSON string array, with
  `ensure_ascii=False` and separators `(',', ':')`, after validation,
  NFC normalization, and outer-whitespace removal.
- `source` and generation/import provenance.

Bank items must be nonempty single-span strings passing the surface validator.
Duplicates are detected by casefolded text. Loading a bank with an existing
checksum verifies it; an inconsistent checksum is an error. JSON arrays,
objects with a `words` array, and UTF-8 one-word-per-line files can be imported.

Generated banks prefer a JSON array. The parser also accepts numbered/bulleted
lines and records `format`, `requested_size`, `valid_count`, `rejected`,
`duplicates`, and `extras`. `accepted` requires at least the requested number of
valid unique words; `words` takes the first requested number. The harness never
invents missing entries. Exhaustion of attempts is an explicit failure.
Standalone `bank --out DIR` saves `DIR/bank.json`, `DIR/manifest.json`,
`DIR/bank_generation/attempts.json`, and per-attempt `meta.json` and
`tokens.jsonl` under `DIR/bank_generation/attempt_NN/`.
Each attempt starts a fresh session and uses `bank_seed + attempt_index`.
`--bank-language` selects both generation and association-instruction language.
For `experiment`, the flag defaults to unset and resolves by precedence:
explicit flag, saved bank `language`, then `ja`. The manifest records
`requested_bank_language` (null when omitted) and
`effective_association_language`. Standalone `bank` defaults to `ja`.
`--bank-prompt` takes a UTF-8 file path.

The bank-generation session/cache is discarded before response trials. No
response trial receives the complete bank in its prompt.

## Fixed-bank experiment files

| Path | Contents |
| --- | --- |
| `manifest.json` | `kind='bank_experiment'`, schema/source identity, configuration, model/runtime information, explicit synthetic/untrained indicators, bank checksum, timing/randomness/similarity definitions, status and completed trial count |
| `bank.json` | Frozen bank and provenance |
| `bank_generation/` | Bank-generation attempt records, when this experiment generates its bank |
| `schedule.json` | Ordered list of trial-plan records |
| `responses.jsonl` | One completed cue-response summary per row, in recording order |
| `vocab.json` | Shared vocabulary indexed by token ID |
| `transcript.md` | All responses, with numeric bank/response labels |
| `trials/TRIAL_ID/` | Per-trial run traces, described below |
| `analysis/` | Post-generation vector/scalar analysis, when enabled |
| `plots/` | Experiment-level diagnostic PNGs and `index.json` mapping numeric labels to records |

Each schedule record contains `trial_id`, `sample_seed`, `repeat_index`,
`order_seed` (null for fixed order), `condition`, and `bank_order` (a permutation
of original bank indices).

The response RNG is initialized once per trial and continues across cues.
Bank generation, presentation shuffling, and analysis do not consume it.
Shuffled permutations depend only on `order_seed`. Repeating the same sampling
seed is a repeat measurement, not another independent stochastic seed.
Different response lengths may consume different numbers of random draws, so
matching trial seeds does not imply draw-by-draw alignment at every later cue.

## Fixed-origin active imagination

`imagine` creates `kind='imagination_experiment'`. It either accepts `--scene`,
imports `--scene-file`, or generates an origin when neither is provided.
`origin.json` records the scene, checksum, language, source and provenance.
Replaying a saved origin verifies its checksum; a mismatch fails explicitly.
The complete saved text is the common starting stimulus for every trial.
`sha256` is SHA-256 over `NFC(scene).strip().encode("utf-8")`, without JSON
quoting. A supplied origin may contain multiple sentences; the one-sentence
rule applies to generated trial responses and to model-generated origins.
Non-JSON UTF-8 files are accepted as origin text. Language precedence is an
explicit `--language`, then saved-origin `language`, then `ja`; the manifest
records requested and effective values separately.

Origin generation uses its own seed and fresh session. After an origin is
accepted, its generation session/cache is discarded before the scene trials.
The origin-generation instruction, rejected attempts and their KV states are
not part of a response trial. Clearing this cache does not unload weights or
claim to reset device-level warm-up effects. Each attempted origin generation
is recorded under `origin_generation/attempt_NN/` together with its token
trace; `origin_generation/attempts.json` records `attempt`, actual `seed`,
directory, stop/censor status, `cache_discarded=true` and `sentence_check`.
Attempt i uses `origin_seed + i`. Acceptance requires
`check_one_sentence(...)["eligible"]`; exhausting the bound is an explicit
error. Generated-origin provenance records the model, backend, revision and
resolved commit where available, actual rendered request, template,
`origin_seed`, `accepted_attempt`, attempt records and
`cache_discarded_before_trials=true`.

A live trial starts from an empty model-native cache, receiving the scene and
one-sentence rule. Subsequent turns append a fixed continuation cue and turn
boundaries to the exact consumed token ledger. Generated token IDs survive
unchanged; the full history is not decoded and tokenized again. All completed
outputs, including sentence-rule failures, remain in the live trajectory.
There is no retry or rejection-sampling of a trial response.

The `reset` imagination condition starts **every step from the same fixed
origin**, without any earlier response in the input. It does not pass the
previous scene as a new standalone cue. The response RNG still advances
through the trial rather than being reset at each step. `both` schedules live
and reset with the same origin, sampling seeds and repeat indices.

| Path | Imagination contents |
| --- | --- |
| `manifest.json` | `kind='imagination_experiment'`, origin identity, model/backend/runtime, config, state/sampling/sentence/timing definitions and status |
| `origin.json` | Frozen scene, hash and provenance |
| `origin_generation/` | Generation attempts and acceptance records, when the origin was generated |
| `schedule.json` | Trial IDs, sample seeds, repeat indices and live/reset conditions |
| `responses.jsonl` | Completed scene outputs in recording order |
| `trials/TRIAL_ID/` | Shared event/array/token trace format; trial kind identifies imagination |
| `analysis/origin_vectors.npy` | `[1, D]` normalized origin vector |
| `analysis/response_vectors.npy` | `[R, D]` normalized response vectors in analysis-record order |
| `analysis/similarity.npy` | `[R, R]` response-to-response cosine, with both axes in the same analysis-record order |
| `analysis/responses.jsonl` and `.csv` | Responses extended with origin and preceding-scene cosine and their identities |
| `analysis/step_summary.json` | Descriptive eligible-only summaries grouped by step index and condition |
| `plots/` | Scene trajectory, timing, context lengths, global and within-trial cosine matrices |

Imagination response rows identify `trial_id`, `event_id`, `step_index`,
`sample_seed`, `repeat_index`, `condition`, `cue`, `origin_sha256` and
`event_directory`. The rows retain
`response`, `unit_text`, `stop_reason`, `censored`, `sentence_check`, `eligible`,
reasons, token counts, cache-reuse information and the shared timing metrics.
`event_id` equals `step_index` within a trial. An origin is not a generated
response row and has no response-time observation in these trials.

### One-sentence stopping and eligibility

`GenerationOptions.stop_on_sentence` activates the punctuation surface policy.
Each sampled token is decoded into the complete generated prefix. The first
candidate ending uses `.。!?！？`, groups adjacent terminal punctuation and
waits for trailing closing delimiters where necessary. Common abbreviations,
initials and numeric dots are handled conservatively. This deterministic
heuristic does not claim grammatical sentence segmentation.

`sentence_check` contains:

- `text`: the decoded response stripped only at the edges for checking.
- `surface_ok`, `eligible`, `reasons` and the policy label.
- `boundary`: null if no ending was found; otherwise `start`, `end`,
  `terminal`, `complete`, `pending_closers`, `overshoot` and `policy`.

Boundary positions are Unicode string offsets in the checked text, not token
indices or byte positions. A streaming dot or unfinished delimiter can require
additional tokens before the boundary is decidable. No sampled token or
suffix is removed to repair an overshoot. The full response and exact IDs stay
in the event; `text_after_first_sentence` marks non-whitespace text after the
first ending. Other reasons include `empty_or_no_content`, `multiple_lines`,
`missing_terminal`, `unclosed_delimiter` and `length_censored`.

`surface_ok` excludes the length-cap condition itself. `eligible` requires
`surface_ok` and a stop reason other than `length`. Thus a complete recording
can still be ineligible or censored. Punctuation failure at EOS is recorded as
a failure, without inventing a closing period. One-sentence stopping is also
the default for standalone `run --protocol imagination|imagination_en`;
`--no-stop-sentence` disables that stopping rule for `run`.

### Scene similarity fields

All vectors are computed after all trials, using isolated texts and the shared
encoding methods below. Nonempty ineligible responses are also encoded; their
status remains visible. JSON uses null for unavailable comparisons.

| Field | Definition |
| --- | --- |
| `origin_cosine` | Response against the fixed origin |
| `previous_scene_cosine` | Step 0 against the origin; later steps against exactly step−1 in the same trial |
| `previous_scene_kind` | `origin`, `response`, or `missing` |
| `previous_response_index` | Global response-record index of the predecessor, or null |
| `previous_response_eligible` | Predecessor eligibility, or null for the origin/missing |
| `previous_scene_used_as_context` | True for the origin or an observed live predecessor; false for a reset predecessor; null if missing |
| `nearest_prior_scene_index` | Most similar previously recorded scene in the same trial, with `-1` for the origin; null if none is available |
| `nearest_prior_scene_cosine` | Cosine at that index |

The helper resolves predecessor identity by `(trial_id, step_index)`, not file
adjacency. Interleaved trial records are allowed; duplicate trial/step keys
are rejected. Missing step−1 yields a missing predecessor rather than bridging
the gap. Output row order and both matrix axes preserve the input record order.
Nearest-prior ties select the origin first, then the earliest observed prior
step. These comparisons retain predecessor ineligibility; a failed sentence is
not silently skipped. For reset trials, adjacent-output similarity does not
assert that the earlier output conditioned the later one.

`analysis/metadata.json` records `origin_sha256`, `row_keys` as trial/event
pairs and encoding method. `embedding_texts.json` records the distinct input
strings; self encoding also records input token IDs in metadata. Offline plotting checks row identity,
response text, origin hash and matrix dimensions before combining analysis
with authoritative root measurements. `step_summary.json` groups by
`(step_index, condition)` and summarizes eligible available values only.
`n`, `eligible_n`, distinct-seed counts, seeds and repeat indices remain
explicit. Means and population standard deviations are descriptive; repeats
of one seed do not increase the number of distinct sampling seeds.

## Association response summary rows

`responses.jsonl` contains:

- Identity: `trial_id`, `event_id`, `sample_seed`, `repeat_index`, `order_seed`,
  `condition`, `position`, `bank_index`, `cue`, and `event_directory`.
- Output: `response`, `stop_reason`, `censored`, `surface_ok`, `eligible`,
  and `reasons`.
- Lengths: `past_tokens`, `new_prompt_tokens`, `generated_tokens`.
- Times in milliseconds: `first_token_model_ms`, `first_token_latency_ms`,
  `response_model_ms`, `response_wall_ms`, `terminal_consume_ms`, `event_wall_ms`.
- Distribution summaries: `raw_entropy_mean`, `raw_surprisal_mean`.

`response` retains the full decoded output. No first-word extraction or lexical
repair occurs. `surface_ok` is a surface heuristic for an unquoted, single-span
word-like response; it is not Japanese morphological segmentation. A response
ending at the generation cap is `censored=true`, with `length_censored` in
`reasons`, and is not `eligible`. Other failures can make a response ineligible
without censoring it. Trailing whitespace is ignored by the surface check but
preserved in the raw response.

Only completed recordings enter the complete-event and response indices.
An error or interrupt may leave a partial event directory with truthful status
and committed `valid_rows`; allocated tails are not observations.

## Per-trial / standalone run files

- `manifest.json`: schema identifier and source hash; model/tokenizer
  configuration, identifiers and resolved revision when available; runtime,
  devices, dtype, hook sites, sampling/timing/alignment/context definitions;
  status and count of completed events. Trial manifests also identify the
  bank, trial, sampling seed, presentation order, repeat, and state condition.
- `vocab.json`: list of tokenizer pieces indexed by vocabulary token ID for
  standalone runs. Bank and imagination experiments share the experiment-root vocabulary.
- `model_config.json`: Backend model configuration for standalone runs.
  Experiment/trial manifests retain the model configuration.
- `events.jsonl`: completed-event index, including cue, response, context,
  parent link, token count, stop reason, and relative event path.
- `transcript.md`: event-number-to-text mapping.
- `session.json`: final exact consumed token ledger and logical cache length
  for a bank or imagination trial. Standalone live runs use
  `sessions/GROUP_ID.json` instead; standalone imagination also saves each
  supplied starting scene under `origins/GROUP_ID.json`.
  KV tensors are not serialized; these are not process-resume checkpoints.
- `notes.jsonl`: optional human observations made using `/note`.
- `plots/`: per-run PNGs; selected-event plots use `plots/event_NNNNNN/`.

## Event files: `events/NNNNNN/`

### `meta.json`

Includes exact complete `prompt_ids`, appended IDs, generated IDs, readable
prompt information, messages when available, response, cue/protocol,
context/parent, stop reason, array names/dtypes/widths, `n_tokens`, `valid_rows`,
status, timings, and final consume-only metadata.

The shared live-trace fields include:

- `appended_prompt_ids`: only the newly introduced cue/turn-boundary IDs.
- `past_tokens`: committed consumed IDs before this event.
- `new_prompt_tokens`: number of appended IDs.
- `prompt_tokens`: complete prefix length; equals past plus new prompt tokens.
- `cache_before_tokens`, `cache_after_tokens`: logical consumed-prefix lengths,
  not a dump or measurement of physically resident KV tensors.
- `cache_reused`, `live_session`: whether a previous session/cache was used and
  whether this event belongs to a live session.
- `rng_state_before`, `rng_state_after`: response generator state for the event.
- `template`: the selected wire-format label; `boundary_dedup_ids`: exact
  special-token prefix removed to avoid duplicate closing tokens.
- `session_reset_count`: explicit session reset counter where provided.
- Trial identity and bank position or scene step for experiment events.
- Backend identity and backend-native cache policy where available.
- `sentence_check` when sentence stopping is enabled.
- `session_invalidated` on relevant failures.

`prompt_text` contains the initial rendered prompt or the later appended
suffix, not necessarily the entire accumulated conversation.
`prompt_ids` is the authoritative input ledger. A readable string cannot prove
that sampled IDs survived a decode/encode round trip. `unit_text` may be stripped
for subsequent chain cues; `response` retains the complete decoded response.
`response_with_special_tokens` exposes decoded special tokens as an additional
inspection aid. A complete event can still have `stop_reason='length'` and
`censored=true`.

### `tokens.jsonl`

Includes `t`, `token_id`, `token_piece`, `decoded_piece`, `is_special`,
`text_so_far`, `phase` (`prefill`, `append_prefill`, or `decode`), `prefix_len`,
`readout_position`, `consumed_token_id`, `predictor_row`, `consumed_state_row`,
raw/policy entropy and selected-token surprisal, raw rank/top-1 probability/
top-two logit gap/EOS mass, previous-distribution JS divergence,
`top_raw` candidates, and timing components. Entropies, surprisal, and
divergences are in nats. Prefix/readout positions refer to the complete ledger,
including previously cached turns.

### Arrays

`raw_logits.npy` stores unmodified full-vocabulary model logits as float32.
`--no-logits` omits the full arrays while retaining scalar token statistics.

Optional `policy_logits.npy` stores float32 scores after the harness's CPU
float64 temperature/top-k/top-p policy. Greedy policy has score 0 at argmax and
-inf elsewhere. The model's computation dtype is recorded separately from
the storage dtype.

`hook_NNN.npy` has shape `[capacity, width]`, stored as float32 or requested
float16. `meta.hidden` maps exact module paths to files, widths, source dtypes,
and stored dtypes. Hooks take the first output tensor `[1, sequence, width]`,
last position only, detached and cloned without modifying the output. They do
not serialize the entire KV cache or attention matrix. Q/K/V projection hooks
represent the actual module output, which may precede head reshaping/RoPE.

## Row alignment and final consume

All arrays allocate `max_new_tokens+1` rows. Only `[:meta.valid_rows]` is data.

| Row | Prefix consumed | Distribution predicts |
| --- | --- | --- |
| `0` | Previous exact ledger + appended cue/boundaries | `generated[0]` |
| `r` | Complete prompt + `generated[:r]` | `generated[r]` when `r < N` |
| `N` | Complete prompt + all generated IDs | Unsampled next token |

Token `t` is sampled from logits row `t`. Its consumed-token state is hook row
`t+1`. Completed events have `valid_rows=N+1`, including EOS if sampled. Final
consume does not sample another token; it places the last generated token in
the live cache exactly once before the next cue. `content_token_indices`
excludes special IDs and individually decoded whitespace-only pieces; it is
not word segmentation.

## State and chat boundaries

`live` maintains exact consumed IDs and their live KV cache in one process.
Later event calls supply only `appended_prompt_ids`. The first forward consumes
that suffix against the existing cache; later forwards consume one sampled
token each. Historical generated IDs are never decoded and re-encoded to
construct the live prefix.

The initial chat turn includes the protocol rule and current cue. Later
suffixes use the model's chat template to derive the assistant closing,
new user cue, and assistant opening. They are tokenized independently, without
adding special tokens again. Only an exact already-consumed leading run of
special closing IDs is deduplicated. Boundary IDs and the selected wire format
are recorded. This defined suffix-tokenization protocol does not claim equality
with tokenizing a fully rendered conversation as one string. In `plain` mode,
fixed textual User/Assistant boundaries are appended instead.

`reset` starts a new cache for each event and presents only its rule/current cue.
Standalone `run` also retains the legacy `carry` condition: readable messages
are carried forward, re-rendered, and prefetched into a fresh cache each event.
`live` is the default for imagination; `/reset` releases the session and
history. Trials begin from empty response sessions, and state is never shared
between trials or with bank generation.

Context overflow is checked before mutating a session. No silent truncation,
cache rebuilding, or historical retokenization is allowed. Failed forwards can
mutate model cache objects, so a session is invalidated on such failures and
cannot be reused until reset. `--no-cache` is incompatible with `live`.

## Timing

`instrumented_forward_ms` synchronizes discovered CUDA/MPS devices on PyTorch
and includes hook cloning. On MLX it explicitly evaluates model logits,
observer snapshots and cache state and synchronizes before stopping the timer;
constructing the lazy computation graph alone is not treated as execution. `capture_to_cpu_ms`, `sampling_stats_ms`, and `array_write_ms`
are separate. `sample_ready_ms` is cumulative event latency after sampling and
sampling statistics, before writing that row's arrays. `token_ready_ms` is
measured later, after array writing and token-record preparation.

| Event metric | Definition |
| --- | --- |
| `first_token_model_ms` | Predictor forward time for row 0 |
| `first_token_latency_ms` | `sample_ready_ms` for the first sample |
| `response_model_ms` | Sum of predictor forward times, rows 0 through N-1 |
| `response_wall_ms` | `sample_ready_ms` for the last sample |
| `terminal_consume_ms` | Final consume-only forward time, row N |
| `event_wall_ms` | Generation through final consume-row array writing; excludes outer event-index I/O |

These values do not form an additive decomposition of `event_wall_ms`:
capture, control flow, serialization, prior-row I/O, and terminal work occur
at different points. Model load, bank/origin generation, and post-trial similarity computation are
outside response timing. Context growth, appended cue length,
and generated token count remain available alongside latency; first token
readiness need not equal readiness of a complete word or visible character.
There is no dedicated warm-up: a newly generated bank or origin precedes
response trials, while imported-bank/origin runs may retain cold-start effects
in their first response. The experiment manifest records this condition.
Backend, device, dtype, quantization, selected observation sites and model
architecture affect these instrumented times; a common schema is not a claim
of identical timing conditions across hardware or runtimes.

## Post-generation analysis

Analysis starts after all response generation, with hooks closed and response
caches discarded. It cannot consume the response RNG or alter its live state.

`analysis/metadata.json` identifies the method (`self`, `sentence-transformer`,
or `none`), display label, model ID/revision, effective pooling/
configuration, bank/origin checksum, normalization, row count, and row correspondence
with `responses.jsonl`.

| Array | Shape and meaning |
| --- | --- |
| `bank_vectors.npy` | `[B, D]` normalized float32 vectors in original bank order |
| `response_vectors.npy` | `[R, D]` normalized float32 vectors in response-record order |
| `similarity.npy` (association) | `[R, B]` response-to-every-bank-word cosine matrix |
| `origin_vectors.npy` (imagination) | `[1, D]` normalized origin vector |
| `similarity.npy` (imagination) | `[R, R]` symmetric response-to-response cosine matrix |

Empty decoded responses and zero/nonfinite vectors use NaN rather than invented
vectors or zero similarity. A nonempty self-encoder input that tokenizes to no
tokens or exceeds its context budget raises an analysis error; it is not
silently truncated. Unavailable JSON scalar values are null. The cosine matrix
contains cosine values, not separately normalized rows or probabilities.

`self` uses the same LM in isolated per-text forwards: no historical prompt or
KV cache, raw text tokens, mean of final hidden states, then normalization.
PyTorch reads the final returned hidden states. MLX observes the recognized
final-normalization module over all input positions. An architecture without
a recognized final norm cannot silently substitute another layer; select
`sentence-transformer` or `none` for that model. Its
label is **isolated-LM representation cosine**. This is distinct from comparing
the contextual generation-time hook states. `sentence-transformer` requires
an explicit encoder model with a saved `modules.json`; plain LM checkpoints
are rejected instead of automatically receiving a pooling layer. It uses the
verified model's `encode(normalize_embeddings=True)` pipeline and records
`saved_pipeline_verified=true`, effective module configuration/prompts, and
`resolved_commits` when discoverable. `embedding_texts.json` lists deduplicated
input strings; response strings are stripped only at the edges. Self metadata
also records the exact encoder input IDs. For Sentence Transformers,
`encoder_config.txt` records the module configuration, and metadata records
the encoder's maximum sequence length and prompt settings. `truncation_policy`
explicitly records that Sentence Transformers uses its native `max_seq_length`
and may truncate inputs beyond that limit.
`--similarity none` omits vector arrays but still writes response tables,
metadata, and cue summaries without cosine fields.

For association, `analysis/responses.jsonl` extends the response summaries with:

- `cue_cosine`.
- `nearest_bank_index`, `nearest_bank_cosine`.
- `nearest_other_bank_index`, `nearest_other_cosine` (current cue excluded).
- `cue_rank`, `mean_bank_cosine`.
- `previous_response_cosine` within the same trial.
- `previously_seen_max_cosine`, `future_bank_max_cosine`, based on that trial's
  presentation order rather than original bank order.

No previous/future/other item means the corresponding field is null. The raw
nonempty responses remain available for inspection even when ineligible.
Primary comparison plots/statistics use eligible rows and expose rejected or
censored counts instead of silently repairing outputs.
`cue_rank` equals one plus the number of strictly larger cosine values;
nearest-item ties choose the lowest original bank index.

`analysis/responses.csv` provides the corresponding flat table.
`analysis/cue_summary.json` groups by cue/condition. Its `n` counts all recorded
responses, `eligible_n` counts eligible responses, and `unique_responses` is a
sorted list of edge-stripped eligible responses. `distinct_seed_count` counts
distinct recorded `sample_seed` values; `eligible_seed_count` counts distinct
seeds represented among eligible responses. `independent_seed_count` remains
as a compatibility alias of `distinct_seed_count`, without asserting
statistical independence. `sample_seeds`, `repeat_indices`, and `order_seeds`
provide the underlying identities; repeats do not add seeds.
For each timing/cue-cosine field, `_mean`, `_std`, and `_n` summarize eligible,
available values. Standard deviation uses the population denominator
(`numpy.std`, ddof=0). These are descriptive summaries, including when multiple
orders or repeats share the same sampling seed.

## Backend-specific execution

`--backend torch` selects PyTorch/Transformers; `--backend mlx` selects MLX-LM.
Both use the same explicit token-generation loop, NumPy sampling policy,
response RNG state, event writer, raw/policy statistics, token ledger and
N+1-row alignment. Inference imports are lazy: saved-file analysis does not
require either generation framework, and MLX generation does not import torch.
SentenceTransformer is a separate optional encoder dependency.

MLX creates its prompt cache using `make_prompt_cache(native_model)`. The
harness does not add a rotating cache or silently trim context. A model may
itself use sliding-window or recurrent cache semantics; those native policies
remain in effect. `cache_*_tokens`, `past_tokens` and the session ledger report
logical consumed-prefix lengths, not physical KV tensor residency. No KV
serialization or process-restart continuation is provided.

MLX observers are local reversible module wrappers installed through
`update_modules`; the original output is returned unchanged. The wrapper
captures the last position of the first rank-3 batch-one output. Selected
sites must execute once per forward. Nested selections and restored original
modules preserve the model's normal callable interface; unsupported output
shapes fail explicitly. Observed tensors are evaluated within the measured
forward and converted to NumPy in the separate capture phase.

MLX runtime metadata records native model configuration, backend/package
versions, device, resolved model path/revision where available, dtypes,
quantization, adapter and hook locations. `--dtype auto` preserves checkpoint
precision. Explicit dtype casting affects floating parameters; packed integer
quantization weights remain packed. `--adapter` is an MLX-LM native adapter on
MLX and a PEFT adapter on PyTorch. `--load-in-4bit`, `--device-map` and
non-default PyTorch attention selections are rejected on MLX; quantization is
loaded from the chosen MLX checkpoint. No backend fallback occurs.

## Plots

Experiment plots are `latency_similarity.png`, `bank_affinity.png`,
`cue_variation.png`, and `trial_progress.png`. They expose the cosine method,
eligibility, and synthetic/untrained status when applicable. Bank-affinity
columns use original bank indices and include the whole bank. `plots/index.json`
maps numeric labels back to bank words and response records.

Imagination adds `scene_path.png`, `scene_latency.png`, `scene_progress.png`,
`scene_similarity_matrix.png` and `scene_recurrence_NN.png`. The latter shows
origin-plus-scene matrices separately for every trial, up to 12 per figure.
All decoded responses remain visible, with sentence failures and length caps
marked. Global matrix axes follow the root response index after identity-based
alignment. Per-trial matrix O is the fixed origin and S0, S1, ... are generated
step indices. Color scales use a labelled observed range; trajectory cosine
axes are autoscaled. Reset lines are labelled origin restarts, and adjacency
comparisons do not imply conditioning in that condition. No fit, significance
test or dimensionality reduction is applied to these scene cosine matrices.

Per-trial state drift compares consecutive consumed-token states within one
event/site. Event PCA fits one shared within-run basis to mean content-token
states and projects endpoints into it, preferring final norm. Full-vocabulary
heatmaps include every ID and explicitly bin when needed. Finite values and
-inf masks are displayed separately. Source arrays retain all original values
at the stored dtype.

## Independent scene evaluation

`evaluate RUN [--out NEW_DIRECTORY]` writes additive format
`association-lab/scene-evaluation/1`; the generation format is unchanged.
Input is the finished experiment's manifest, origin, schedule and authoritative
root responses. Hashes, counts and complete, unambiguous identities are checked.
Derived cosine files are not required.

| File | Contents |
| --- | --- |
| `metadata.json` | Input/source hashes, status, configuration, lexical policy, optional NLI identity |
| `responses.jsonl` | Original fields plus `evaluation`; unchanged text and eligibility |
| `pairs.jsonl` | Premise/hypothesis text, identity, predecessor/context flags, optional logits/scores/input IDs |
| `summary.json` | Counts, initial text matches, per-seed differences, population/coverage definitions |
| `controls.json` | NLI smoke controls, or empty when disabled |
| `probes.json` | Temporal-framing and referent probes, or empty when disabled; no gold label for the ambiguous unmarked transition |
| `nli_progress.jsonl` | Latest atomic batch snapshot including controls and probes; pending rows are not scores |
| `report.html` | Standalone escaped report with seed/repeat selection |

Pair status is `not_scored`, `pending`, `scored`, `empty_text` or `input_too_long`.
Unavailable scores are null. Premise step `-1` is the origin; other indices refer
only to the same trial. Reset adjacency has `used_as_context=false`. Full NLI
input IDs include pair special tokens; no truncation is applied. Named softmax
scores are uncalibrated. See [evaluation methods](EVALUATION.md) for definitions,
budgets, aggregation, provenance and interpretation limits.

## Primary implementation references

- [Transformers: cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation)
- [Transformers: chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers: generation outputs](https://huggingface.co/docs/transformers/main/en/internal/generation_utils)
- [Sentence Transformers: semantic textual similarity](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
- [Sentence Transformers: encode API](https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode)
- [PyTorch: forward hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)

- [MLX-LM: model loading and native cache](https://github.com/ml-explore/mlx-lm)
- [MLX: lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [MLX: synchronize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html)
- [MLX: Module.update_modules](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update_modules.html)
