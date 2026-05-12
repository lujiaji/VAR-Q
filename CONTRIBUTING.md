# Contributing

Thanks for improving VAR-Q. Keep the repository lightweight and reproducible.

## Repository Hygiene

- Do not commit model checkpoints, generated images/videos, benchmark outputs, or local experiment queues.
- Keep third-party model repositories under `third_party/<repo>` and out of git.
- Do not vendor or patch third-party backend source code. New backend support should prefer runtime hooks or thin launchers.
- Public JSON configs should not contain local checkpoint paths, dataset paths, or machine-specific paths.

## Code Changes

- Main VAR-Q logic belongs in `VAR_Q/`.
- KIVI, FLexGen, KVQuant, and other comparison baselines belong in `ablation/`.
- `VAR_Q` and `ablation` should not depend on each other at runtime except through the hook routing layer.
- If you modify quantization, packing, config loading, or hook behavior, add or update a lightweight smoke test under `tests/`.

## Testing

Run the public checks before opening a pull request:

```bash
python -m py_compile VAR_Q/*.py VAR_Q/hooks/*.py ablation/*.py scripts/*.py
pytest tests/
```

Do not add CI jobs that download model weights or run large benchmark suites.

## Backend Integrations

When adding a backend:

- Follow the backend's official installation instructions.
- Add the upstream repository to `third_party/README.md`.
- Add a lightweight launcher or adapter that resolves paths relative to the VAR-Q repo.
- Prefer `install_varq_hooks(...)` after model construction instead of editing backend source files.
- Add a mock or no-checkpoint smoke test when possible.
