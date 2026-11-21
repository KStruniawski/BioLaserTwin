# Contributing to BioLaserTwin

Contributions, bug reports, and feature requests are welcome via GitHub Issues and Pull Requests.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks Before Submitting a Pull Request

1. Ensure linting passes with zero warnings:
   ```bash
   ruff check biolaser_twin tests examples
   ```
2. Run the fast verification tests and examples:
   ```bash
   pytest tests/test_verification.py
   python examples/illustrative_spectra96.py
   python examples/forward_to_inverse.py
   ```
3. Keep `biolaser_twin.simulator` isolated from `biolaser_twin.kinetics_nominal` and `biolaser_twin.solvers`.
