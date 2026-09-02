# AGENTS.md

This repository holds yaoyiyi's weekly exercises for AMAT5315, organized in
week1/, week2/, and so on. Each week is a mostly self-contained course exercise
containing its own SPEC, implementation, and tests for that week's TDD/BDD
task.

## About the owner

- Name: yaoyiyi
- Role: student in AMAT5315
- Background: advanced materials / physics-related research; experienced with
  MATLAB and COMSOL; basic Python/programming experience.
- Currently learning: software engineering practices such as TDD/BDD, Git,
  testing, and agent-assisted coding.

## Repository layout

```
weekN/
  SPEC.md            # Specification/requirements for that week's task
  <module>.py        # Implementation
  test_<module>.py   # Pytest tests
```

Each week is treated as its own self-contained exercise. The estimate_pi Monte
Carlo example in week1/ is an early exercise; future weeks may contain
different, unrelated problems rather than continuously extending estimate_pi.

## Workflow (TDD/BDD)

1. Read (or write) the SPEC/requirements first.
2. Write or update tests before the implementation when the exercise requires
   TDD.
3. Confirm the new test fails for the expected reason (red).
4. Implement the minimum code needed to make the test pass (green).
5. Refactor only after tests pass.
6. Run pytest to verify:

   ```sh
   cd weekN
   python -m pytest -q
   ```

7. Keep random behavior reproducible with an explicit seed when appropriate.

## Code style and preferences

- Keep code simple, readable, and beginner-friendly.
- Use clear variable/function names and short comments where useful.
- Explain non-obvious implementation decisions.
- Use Python and pytest unless the exercise specifies otherwise.
- Match the exact function signatures, defaults, and return types in the SPEC.
- Do not over-engineer; prioritize satisfying the SPEC and tests clearly and
  correctly.

## Scope and boundaries

- Keep each week's files inside its own week folder unless the course
  instructions say otherwise.
- Follow the naming and structure already present in the repository rather than
  inventing a new convention.
- Do not make unnecessary changes outside the current exercise.
- Avoid modifying previous completed weeks unless explicitly asked.
- Do not delete or substantially restructure existing coursework without asking
  first.

## Guardrails

- Never commit API keys, tokens, passwords, or other private information.
- Avoid committing generated files, large artifacts, virtual environments,
  caches, or unrelated files (e.g. __pycache__/, *.pyc, .pytest_cache/,
  codex-*.tar.gz).
- Do not modify a SPEC or test just to make a failing test pass; fix the
  implementation instead.

## Communication

- Code, files, and commit messages: English.
- Explanations to the owner may be in Chinese if helpful.
- When possible, state the exact command to run to test the code.

Memory probe: W1-MEMORY-5315
