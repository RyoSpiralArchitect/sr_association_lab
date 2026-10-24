# Validation

The tests exercise real small Transformer forwards and caches with random,
locally initialized weights. They check recorder behavior, not learned language
ability. No pretrained model download is required for the test suite.

## Measurement contracts

| Suite | Coverage |
| --- | --- |
| `tests/test_recording.py` | Raw/policy logits, sampling, EOS, N+1 alignment, hooks, partial writes |
| `tests/test_sessions.py` | Exact token ledgers, continuing/reset KV, RNG, banks, ordering, post-generation similarity |
| `tests/test_imagination.py` | Frozen origins, origin-cache disposal, sentence stopping, scene identity and similarity |
| `tests/test_mlx.py` | Native MLX caches, observer restoration, quantized loading, end-to-end trials and encoding |
| `tests/test_compatibility.py` | Loader API compatibility, tokenizer overrides, finite sampling parameters, native GPT-2 oracles |

Cached logits and selected hidden states are compared with independent
full-prefix forwards. A final consume-only forward must provide the N+1th row.
Failure tests verify that potentially mutated caches cannot be reused and that
committed trace data remain readable.

## Local validation

The standalone repository's renamed suites passed **79 tests, with no failures
or skips**, on 2026-10-24. Runtime: Python 3.12.6, PyTorch 2.12.1,
Transformers 4.57.6, MLX 0.31.2, MLX-LM 0.31.3, NumPy 2.2.3, and pytest 9.0.2.
The suite completed in 8.08 seconds. A `TRANSFORMERS_CACHE` deprecation warning
did not affect the result. User-site startup patches were disabled for this
test run; the separately recorded Metal checks used the normal Python startup.

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg \
  python -m pytest -q
```

An additional bounded run used a local GPT-2 checkpoint on Apple Silicon Metal:
the English scene command in the README completed four seeds and eight steps
per seed, followed by self encoding and five figures. It produced 32 responses,
396 generated tokens, and 428 valid recorder rows. Thirty responses passed the
surface-only sentence check; one reached the per-response length cap.

The same parameters with Japanese instructions stopped at the model's native
context limit: 954 prompt tokens plus a reserved 96 new tokens exceeded 1024.
Thirteen completed responses and their token ledgers were retained. This run
did not complete the four trials or post-generation analysis.

All 45 completed responses were checked for token-ledger continuity, N+1 row
counts, saved logit ranks, array shapes, and finite values in committed rows.
Six saved rows, including a 906-token prefix, were compared with fresh
cache-free MLX forwards. All logits and selected hidden states met
`rtol=1e-4, atol=1e-4`; bitwise equality was not required.

These trained-model runs establish a bounded execution check. They do not
establish scene quality, Japanese fluency, general capability, or performance
comparability between backends. Private machine-local traces are not bundled
with the source repository.

## Continuous integration

[GitHub Actions](https://github.com/RyoSpiralArchitect/sr_association_lab/actions/workflows/tests.yml)
runs PyTorch tests on Linux with Python 3.11 and 3.12, and native MLX tests on an
Apple Silicon runner. MLX fixtures select the CPU device; the separate Metal
check above is not implied by the CI label. PyTorch CI also exercises the public
demo, plotting, and saved-trace inspection commands.

Optional-backend tests skip if that backend is absent. Each CI job imports its
target backend explicitly before running tests. The PyTorch jobs include the
optional Sentence Transformers integration tests.
