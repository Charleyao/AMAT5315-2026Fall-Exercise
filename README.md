# AMAT5315-2026Fall-Exercise

Weekly exercises for AMAT5315 (Fall 2026). Each week is a mostly self-contained
course exercise containing its own SPEC, implementation, and pytest tests,
developed using TDD/BDD.

## Repository layout

```
weekN/
  SPEC.md            # Specification/requirements for that week's task
  <module>.py        # Implementation
  test_<module>.py   # Pytest tests
```

## Current exercises

- **week1** — Monte Carlo estimate of π (`estimate_pi`)

## Requirements

- Python 3.x (developed on Python 3.14)
- pytest

## Install pytest

```sh
python -m pip install pytest
```

## Run the tests

From a week folder:

```sh
cd week1
python -m pytest -q
```

Or from the repository root:

```sh
python -m pytest week1 -q
```

Drop `-q` to see more detailed output.
