# Contributing

Issues and pull requests are welcome. Describe the observable problem, the model
architecture and backend involved, and a small reproduction where possible.

The runtime lives in `sr_association_lab.py`. Keep it usable as one Python file;
tests and documentation live separately. Use descriptive test names rather than
release-numbered copies of the implementation or test suites.

## Run the relevant tests

```bash
python -m pip install pytest
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Install the backend you are changing. Native MLX tests can also run on their own:

```bash
python -m pytest -q tests/test_mlx.py tests/test_compatibility.py
```

The tests use locally constructed random models and tokenizers. For a new
architecture, compare cached forwards against independent full-prefix forwards,
including the terminal consume row and selected hidden states.

## Preserve the measurement contracts

- Keep complete sampled text and token IDs, including failed surface checks.
- Preserve exact token histories in live sessions. Do not silently re-tokenize,
  truncate, retry, or repair a generated response.
- Keep raw logits separate from sampling scores and record only committed rows.
- Close local observers and invalidate a live session after a failed forward
  that may have mutated its cache.
- Run isolated similarity analysis after generation, without trial KV or RNG.
- Describe what a test establishes. Recorder correctness alone does not
  establish linguistic quality or a psychological interpretation.

Update the English README and the linked Japanese README when public behavior
changes. Update `docs/SCHEMA.md` for saved-format changes, preserving readability
of earlier recordings where feasible. Application changes are tracked through
Git commits and source hashes; there is no release-number update step.

Use small synthetic fixtures in contributions. Generated runs, local model
weights, credentials, and machine-specific logs are excluded by `.gitignore`.
Contributions are made under the project's [MIT License](LICENSE).
