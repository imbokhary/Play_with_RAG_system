# Release checklist

A milestone is complete only when every item below is satisfied and the ROADMAP.md status is updated.

## Implementation
- [ ] Working implementation with standalone dependencies and configuration.
- [ ] CLI/JSON contract implemented (`docs/cli-contract.md`).
- [ ] No cross-version runtime imports; no runtime reads of `../shared-resources/`.
- [ ] No RAG orchestration framework in core code.

## Tests
- [ ] Unit tests for deterministic logic (no Ollama required).
- [ ] Ollama-dependent tests marked separately.
- [ ] Tests and quality checks pass (ruff, pytest); failures explicitly explained.
- [ ] Fresh installation follows the documented workflow.

## Docs
- [ ] Version README follows the template (`docs/version-template.md`).
- [ ] Mermaid diagram of the implemented pipeline added.
- [ ] Troubleshooting and limitations documented.
- [ ] CHANGELOG.md updated (semver) for the version.
- [ ] ROADMAP.md status updated (`Planned` → `Implemented`).

## Release
- [ ] Tag created: `vXX-<name>/1.0.0`.
- [ ] Release notes: what the reader will learn, new features, install/run instructions, tests, benchmark results, known limitations, README and diagram links.
- [ ] No secrets, virtual environments, model weights, generated indexes, or personal data committed.
- [ ] Benchmark results reported honestly with hardware and model details (never invented).