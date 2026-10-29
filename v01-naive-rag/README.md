# V01 — Naive RAG from scratch

The smallest complete Retrieval-Augmented Generation (RAG) pipeline, built with **no RAG framework** — just Python standard library and Ollama.

## 1. What you will learn

- The complete RAG pipeline end to end: load → chunk → embed → store → retrieve → generate.
- How to call Ollama (`/api/embed`, `/api/chat`) with plain `urllib`.
- How cosine similarity and top-k retrieval work, implemented in pure Python.
- How to build a context-grounded prompt and return source-attributed answers.
- How to implement the shared CLI/JSON contract (`docs/cli-contract.md`).
- How to test deterministic logic (similarity, chunking, ranking) without Ollama.

**Prerequisites:** Python 3.10+, Ollama running locally. No prior RAG knowledge assumed.

## 2. Why this version exists

RAG is often taught through frameworks (LangChain, LlamaIndex, …) that hide the pipeline. This version shows what is actually happening under the hood: one file, no vector database, no orchestration library. Later versions (V02–V10) add chunking reliability, persistence, evaluation, hybrid search, and architecture — but they all build on this same naive baseline.

## 3. Architecture diagram

```mermaid
flowchart TD
    A[Local documents (.txt/.md)] --> B[Fixed-size chunking]
    B --> C[Ollama /api/embed]
    C --> D[Store chunks + vectors (index.json)]
    E[User question] --> F[Ollama /api/embed]
    F --> G[Cosine similarity top-k]
    D --> G
    G --> H[Top-k context]
    H --> I[Build grounded prompt]
    I --> J[Ollama /api/chat]
    J --> K[Answer with sources]
```

## 4. Conceptual explanation

### Embeddings
An embedding model turns a piece of text into a vector (a list of numbers). Similar texts produce nearby vectors. V01 uses Ollama `/api/embed` with the `embeddinggemma` model.

### Cosine similarity
Cosine similarity measures the angle between two vectors: 1 = identical direction, 0 = orthogonal, -1 = opposite. It is implemented in pure Python (`cosine_similarity` in `rag.py`). For normalized vectors it equals the dot product.

### Brute-force top-k retrieval
The question is embedded, compared against every stored vector, and the top-k most similar chunks are returned. Exact and simple; cost grows linearly with corpus size (V03 covers approximate indexes such as HNSW).

### Stable chunk ids
Each chunk id is a SHA-1 of `source|position|text`, so re-ingesting the same document produces identical ids. This lets evaluation match chunks to questions without relying on positional chunk ids.

### Prompt construction
Retrieved chunks are assembled into a prompt that instructs the model to answer using only the supplied context, to say so when insufficient, and to treat context as untrusted reference data (reduces prompt-injection risk).

## 5. Project structure

```
v01-naive-rag/
├── rag.py                # Single-file pipeline (ingest + query)
├── config.json           # Models, endpoints, chunking, top-k
├── ruff.toml             # Lint configuration
├── README.md             # This tutorial chapter
├── index.json            # Generated at runtime (ingestion index; not committed)
├── data/
│   └── documents/        # Sample documents (.txt/.md)
│       ├── intro-to-rag.md
│       ├── chunking.md
│       └── retrieval-basics.md
├── tests/
│   └── test_rag.py       # Unit tests (no Ollama required)
└── .github/
    └── workflows/
        └── v01-ci.yml    # CI: lint + unit tests
```

`index.json` is generated when you run `ingest` and is gitignored.

## 6. Installation and setup

1. **Python 3.10+** and **Ollama** running (`http://localhost:11434`).
2. Pull the models:
   ```bash
   ollama pull embeddinggemma
   ollama pull gemma3:4b
   ```
3. Verify Ollama:
   ```bash
   curl http://localhost:11434/api/tags
   ```
4. (Optional) For linting, install ruff:
   ```bash
   pip install ruff
   ```

Models, endpoints, chunk size/overlap, and top-k are configurable in `config.json` — no code changes required.

## 7. Run the application

From the `v01-naive-rag/` folder:

```bash
# 1. Ingest the sample documents (creates index.json)
python rag.py ingest --docs data/documents --json

# 2. Ask a question (retrieve + generate)
python rag.py query "What is RAG?" --top-k 5

# 3. Show detailed source scores
python rag.py query "What is RAG?" --top-k 5 --verbose

# 4. Machine-readable JSON output (full metadata)
python rag.py query "What is RAG?" --top-k 5 --json

# 5. Retrieval only (no LLM call; useful for retrieval metrics)
python rag.py query "What is RAG?" --top-k 5 --retrieve-only --json
```

Example human-readable output (default):
```
RAG stands for Retrieval-Augmented Generation. It is a technique that gives a
language model access to external information at answer time.

Sources:
  [1] data/documents/intro-to-rag.md
  [2] data/documents/intro-to-rag.md
  [3] data/documents/chunking.md
```

With `--verbose`:
```
RAG stands for Retrieval-Augmented Generation. It is a technique that gives a
language model access to external information at answer time.

Sources:
  [1] data/documents/intro-to-rag.md (score=0.73)
  [2] data/documents/intro-to-rag.md (score=0.67)
  [3] data/documents/chunking.md (score=0.46)
```

### CLI contract (`--json`)
When `--json` is set, **only JSON** is written to stdout (logs go to stderr), matching `docs/cli-contract.md` (schema_version `1`). The JSON includes full metadata (chunk ids, scores, texts, timings, models, config) for programmatic use:
```json
{
  "schema_version": "1",
  "question": "What is RAG?",
  "answer": "...",
  "abstained": false,
  "chunks": [
    {"chunk_id": "...", "source": "...", "score": 0.91, "rank": 1, "text": "..."}
  ],
  "timings_ms": {"retrieval": 12.4, "generation": 1840.0, "total": 1855.1},
  "models": {"chat": "qwen2.5-coder:7b", "embedding": "nomic-embed-text:latest"},
  "config": {"chunk_size": 800, "chunk_overlap": 100, "top_k": 5, "retrieval": "dense"}
}
```

## 8. Code walkthrough

Execution order for `query`:
1. `load_config` — reads `config.json` with defaults.
2. `load_index` — loads `index.json` (chunks + vectors).
3. `retrieve` — embeds the question via Ollama `/api/embed`, then `retrieve_by_vector` ranks chunks by cosine similarity and returns top-k.
4. `build_prompt` — assembles `SYSTEM_PROMPT` + context + question.
5. `generate_answer` — calls Ollama `/api/chat` and returns the answer.

Execution order for `ingest`:
1. `load_documents` — reads `.txt`/`.md` from `docs_dir`.
2. `chunk_text` — fixed-size windows with overlap.
3. `stable_chunk_id` — deterministic SHA-1 id per chunk.
4. `build_index` — embeds all chunks via Ollama `/api/embed`.
5. `save_index` — writes `index.json`.

## 9. Experiments

Try these and record the effect (model: `embeddinggemma`/`gemma3:4b`, Ollama version, corpus, hardware):

| Variable | Values to try | What to observe |
|---|---|---|
| `--top-k` | 1, 3, 5, 10 | Answer quality vs. prompt length vs. generation latency |
| `chunk_size` | 400, 800, 1200 | Retrieval precision vs. context per chunk |
| `chunk_overlap` | 0, 100, 200 | Boundary context loss vs. duplicate storage |

Record the exact commands, model names/tags, and timings from `timings_ms`. Never fabricate results.

## 10. Testing and quality

Unit tests run without Ollama (deterministic logic only):

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v   # Windows
python -m unittest discover -s tests -v                      # Linux/macOS
```

Coverage: cosine similarity (hand-computed examples), fixed-size chunking + overlap, stable chunk ids, prompt construction, retrieval ranking with hand-computed vectors.

Lint:
```bash
.venv\Scripts\ruff check v01-naive-rag   # Windows
ruff check v01-naive-rag                 # Linux/macOS
```

Ollama-dependent end-to-end checks are run manually against a local Ollama instance and are not part of the unit suite.

## 11. Security and performance

- Retrieved context is treated as untrusted reference data, not instructions.
- Never log full document contents or secrets.
- Brute-force retrieval is O(n) in corpus size — fine for the sample corpus, not for large collections (see V03/V05).
- All model calls are local via Ollama; no documents leave your machine.

## 12. What changed and why

This is the first version; there is no previous version. V02 adds PDF ingestion, cleaning, configurable chunking, and stable metadata. V03 adds persistent indexes and incremental re-ingestion.

## 13. References and further reading

- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md (`/api/embed`, `/api/chat`)
- "Retrieval-Augmented Generation for Large Language Models" (Lewis et al., 2020)
- Shared sample documents: `../shared-resources/sample-documents/` (MIT license)
- CLI contract: `../../docs/cli-contract.md`
- Roadmap and conventions: `../../ROADMAP.md`, `../../AGENTS.md`