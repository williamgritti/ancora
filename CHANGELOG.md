# Changelog

## 0.1.1

### Changed
- Distribution name is `ancora-rag`; the import name stays `ancora`. The PyPI
  name `ancora` is held by an empty placeholder (0.0.0.dev0, no summary, no
  author, one release), so it was unavailable.

### Added
- `[project.urls]` so the PyPI page links back to the repository, the issue
  tracker and the changelog. Without it, anyone arriving via `pip install`
  had no route to the source.

## 0.1.0

First release.

### Added
- `Ancora` engine: retrieve → generate → verify → decide.
- `GroundingChecker` with IDF-weighted claim coverage and a hard numeric guard.
- `Policy` for the answer / escalate / refuse decision.
- BM25 retrieval with an optional dense-embedding blend.
- Sentence-aware chunking that preserves character offsets for citations.
- Locale-aware number canonicalisation (pt-BR, it-IT, en-US).
- Light multilingual stemmer for Portuguese, Spanish, Italian and English.
- Provider adapters for OpenAI and Anthropic, plus `ScriptedProvider` and
  `EchoProvider` for deterministic testing.
- `ancora ask` and `ancora audit` command line entry points.
- JSON audit trail on every answer.

### Fixed during development
- Numeric guard checked the union of all retrieved chunks, so an unrelated
  document could license a figure. Scoped to lexically related chunks.
- A claim's own numbers counted toward lexical coverage, letting an invented
  figure select the chunk that would then license it. Numeric tokens are now
  excluded from coverage.
- No stemming, so verbatim source sentences could score 0.0 on retrieval and
  be wrongly escalated.
