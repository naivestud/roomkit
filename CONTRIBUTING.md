# Contributing to roomkit

Thanks for considering a contribution! roomkit is a small, focused toolkit
and the bar is "readable, tested, documented".

## Getting started

```bash
git clone https://github.com/naivestud/roomkit
cd roomkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"          # add [neural] too if you touch roomkit.neural
```

Run the checks before pushing:

```bash
pytest -q
ruff check .
```

## Layout

- `src/roomkit/` — the package, one subpackage per capability
  (`signal`, `denoise`, `dereverb`, `acoustics`, `vad`, `codec`,
  `diarization`, `metrics`, `neural`).
- `tests/` — mirrors the package; every module has a test file.
- `docs/` — long-form documentation; `README.md` stays a quick tour.
- `examples/` — self-contained scripts that generate their own audio.

## Conventions

- NumPy-style docstrings with Args/Returns sections on public functions.
- Type hints on public signatures; keep the numpy-only core free of torch
  imports (torch belongs in `roomkit.neural` only).
- Deterministic tests: seed every RNG (`np.random.default_rng(seed)`), and
  never download data or model weights in the test suite.
- One concern per commit; conventional-commit-style messages preferred.

## Reporting bugs

Open an issue with the bug-report template: a minimal reproduction and the
output of `python -c "import roomkit; print(roomkit.__version__)"` goes a
long way.
