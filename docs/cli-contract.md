# CLI / JSON contract

Versioned contract that every RAG version implements from the version where the feature exists. Schema version: `1`.

## Supported commands

```bash
# Ingest documents (V01 onward; incremental from V03)
python rag.py ingest --docs data/documents --json

# Full RAG query (retrieve + generate)
python rag.py query "What is RAG?" --top-k 5 --json

# Retrieval only (no LLM call; used by the eval runner for retrieval metrics)
python rag.py query "What is RAG?" --top-k 5 --retrieve-only --json
```

## Output schema for `query --json`

```json
{
  "schema_version": "1",
  "question": "What is RAG?",
  "answer": "string, or null when --retrieve-only",
  "abstained": false,
  "chunks": [
    {
      "chunk_id": "sha1-or-stable-id",
      "source": "data/documents/intro.md",
      "score": 0.83,
      "rank": 1,
      "text": "chunk text"
    }
  ],
  "timings_ms": { "retrieval": 12.4, "generation": 1840.0, "total": 1855.1 },
  "models": { "chat": "gemma3:4b", "embedding": "embeddinggemma" },
  "config": { "chunk_size": 800, "chunk_overlap": 100, "top_k": 5, "retrieval": "dense" }
}
```

## Rules
- Always write **only** JSON to stdout when `--json` is set; logs go to stderr.
- Exit code `0` on success, non-zero on failure, with an error JSON: `{"error": {"code": "...", "message": "..."}}`.
- `chunk_id` values must be stable across runs for the same document content so metrics can match them against the dataset.
- Document this contract in `docs/cli-contract.md` during M0 and treat it as versioned (`schema_version`).