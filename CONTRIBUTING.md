# Contributing to Play_with_RAG_system

Thank you for contributing! This repository is a hands-on RAG engineering tutorial built as a series of standalone Python versions (V01–V10).

## How to contribute
1. Read `ROADMAP.md` and `AGENTS.md` first — they define scope, rules, and the order of work.
2. Pick one milestone at a time. The roadmap is the single source of truth for what to build next.
3. Create a branch from `develop`, implement the milestone, and open a pull request.
4. Every version must satisfy the shared definition of done in `ROADMAP.md` (tests, docs, CLI contract, independence check).
5. Never invent benchmark results, test outcomes, or completed releases. Report only what was actually run, with hardware and model details.

## Coding standards
- No RAG orchestration frameworks (LangChain, LlamaIndex, Haystack, …) in core code.
- Python standard library for early versions; add dependencies only when justified.
- Type hints, small functions, explicit error handling, deterministic tests without Ollama.
- Each version is independently installable; never import runtime code from another version directory.

## Before merging
- Tests and quality checks pass (or failures are explicitly explained).
- README and diagrams for the version are updated.
- `ROADMAP.md` status is updated and the release checklist is satisfied.
- No secrets, virtual environments, model weights, generated indexes, or personal data are committed.

## License
By contributing, you agree your contributions are licensed under the repository's LICENSE.