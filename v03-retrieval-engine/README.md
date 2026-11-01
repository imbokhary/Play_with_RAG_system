# V02 — Document ingestion and chunking

Make document processing reliable and explainable. V02 builds on V01's naive
RAG pipeline with a production-grade ingestion layer: PDF support, text
cleaning, multiple chunking strategies, stable chunk IDs, duplicate detection,
and an ingestion report.

## 1. What you will learn

- How to load `.txt`, `.md`, and `.pdf` documents (PDF via `pypdf`).
- How to clean raw text (whitespace normalization, encoding repair).
- Three chunking strategies: fixed-size, token-aware, and Markdown-heading-aware.
- How stable chunk IDs enable duplicate detection across re-ingestions.
- How an ingestion report surfaces per-file counts and errors.
- How malformed files (empty files, corrupt PDFs, bad encoding) are handled.
- How to seed an evaluation dataset for retrieval metrics.

## 2. Why this version exists

V01 proved the end-to-end RAG pipeline works, but its ingestion was minimal:
fixed-size chunking only, no PDF support, no cleaning, no duplicate detection,
and no ingestion report. V02 makes document processing reliable and explainable
before V03 adds persistence and incremental re-ingestion.

## 3. Architecture diagram

```mermaid
flowchart TD
    A[Documents (.txt/.md/.pdf)] --> B[Load + detect format]
    B --> C[Clean text]
    C --> D[Chunk (fixed/token/markdown)]
    D --> E[Stable chunk ID (SHA-1)]
    E --> F[Duplicate detection]
    F --> G[Embed via Ollama /api/embed]
    G --> H[Store chunks + vectors (index.json)]
    H --> I[Ingestion report]
    E2[User question] --> F2[Embed question]
    F2 --> G2[Cosine similarity top-k]
    H --> G2
    G2 --> H2[Top-k context]
    H2 --> I2[Ollama /api/chat]
    I2 --> J[Answer with sources]
```

## 4. Conceptual explanation

### Document loading
V02 loads `.txt`, `.md`, and `.pdf` files. PDFs are parsed with `pypdf`, which
extracts text page by page and labels each page. Empty files and corrupt PDFs
are reported as ingestion errors rather than crashing the pipeline.

### Text cleaning
Raw text can contain null bytes, control characters, and irregular whitespace.
`clean_text()` normalizes whitespace, strips control characters, and collapses
excessive newlines so chunks are predictable.

### Chunking strategies
- **fixed**: Character-based windows with overlap (same as V01).
- **token**: Approximate token count (4 chars/token rule of thumb), breaking
  at sentence boundaries.
- **markdown**: Splits on `#`/`##` headings, keeping each heading with its
  content; falls back to fixed-size for oversized sections.

### Stable chunk IDs
Each chunk ID is `SHA-1(source | position | text)`, so re-ingesting the same
document produces identical IDs. This enables duplicate detection: chunks already
in the index are skipped, preventing duplicate storage.

### Ingestion report
After ingest, the pipeline reports: documents processed, new chunks created,
duplicates skipped, total chunks, and per-file details (format, chunk count,
duplicate count, character count). Errors are listed separately.

### Query (unchanged from V01)
Embed the question via Ollama `/api/embed`, rank chunks by cosine similarity,
return top-k, then generate an answer via Ollama `/api/chat` with source
citations.

## 5. Project structure

```
v02-ingestion-and-chunking/
├── rag.py                # Single-file pipeline (ingest + query)
├── config.json           # Models, endpoints, chunking, top-k
├── ruff.toml             # Lint configuration
├── requirements.txt      # pypdf (PDF support)
├── README.md             # This tutorial chapter
├── index.json            # Generated at runtime (not committed)
├── data/
│   └── documents/        # Sample documents (.txt/.md/.pdf)
│       ├── intro-to-rag.md
│       ├── chunking.md
│       ├── retrieval-basics.md
│       ├── embedding-models.txt
│       └── embedding-models.pdf
├── tests/
│   ├── test_ingestion.py # Unit tests (no Ollama required)
│   └── fixtures/
│       └── eval-dataset.json  # 24-question seed evaluation dataset
└── .github/
    └── workflows/
        └── v02-ci.yml    # CI: lint + unit tests
```

`index.json` is generated when you run `ingest` and is gitignored.

## 6. Installation and setup

1. **Python 3.10+** and **Ollama** running (`http://localhost:11434`).
2. Pull the models:
   ```bash
   ollama pull nomic-embed-text:latest
   ollama pull qwen2.5-coder:7b
   ```
3. Verify Ollama:
   ```bash
   curl http://localhost:11434/api/tags
   ```
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt   # pypdf for PDF support
   pip install ruff                  # for linting
   ```

Models, endpoints, chunk size/overlap, chunk strategy, and top-k are configurable
in `config.json`.

## 7. Run the application

From the `v02-ingestion-and-chunking/` folder:

```bash
# 1. Ingest documents (creates index.json)
python rag.py ingest --docs data/documents --json

# 2. Re-ingest the same documents (duplicates skipped)
python rag.py ingest --docs data/documents --json

# 3. Ask a question (retrieve + generate)
python rag.py query "What is RAG?" --top-k 5

# 4. Show detailed source scores
python rag.py query "What is RAG?" --top-k 5 --verbose

# 5. Machine-readable JSON output
python rag.py query "What is RAG?" --top-k 5 --json

# 6. Retrieval only (no LLM call)
python rag.py query "What is RAG?" --top-k 5 --retrieve-only --json
```

Example human-readable output:
```
RAG stands for Retrieval-Augmented Generation. It is a technique that gives a
language model access to external information at answer time.

Sources:
  [1] data/documents/intro-to-rag.md
  [2] data/documents/retrieval-basics.md
```

Example `--json` ingest output (includes ingestion report):
```json
{
  "schema_version": "1",
  "report": {
    "documents_processed": 5,
    "new_chunks": 12,
    "duplicates_skipped": 0,
    "total_chunks": 12,
    "per_file": {
      "data/documents/intro-to-rag.md": {
        "format": "md",
        "chunks": 3,
        "duplicates": 0,
        "chars": 1200
      }
    }
  },
  "load_errors": [],
  "index_path": "index.json",
  "embedding_model": "nomic-embed-text:latest",
  "embedding_dimension": 768
}
```

## 8. Code walkthrough

Execution order for `ingest`:
1. `load_documents` — reads `.txt`/`.md`/`.pdf` from `docs_dir`, returns docs + errors.
2. `clean_text` — normalizes whitespace, strips control characters.
3. `chunk_text` — splits into chunks using the configured strategy.
4. `stable_chunk_id` — assigns deterministic SHA-1 IDs.
5. Duplicate detection — skips chunks whose ID already exists in the index.
6. `embed_texts` — embeds all new chunks via Ollama `/api/embed` (one batch).
7. `save_index` — writes `index.json`.

Execution order for `query`:
1. `load_index` — loads `index.json`.
2. `retrieve` — embeds the question, ranks chunks by cosine similarity, returns top-k.
3. `build_prompt` — assembles `SYSTEM_PROMPT` + context + question.
4. `generate_answer` — calls Ollama `/api/chat` and returns the answer.

## 9. Experiments

Try these and record the effect (model: `nomic-embed-text`/`qwen2.5-coder`, Ollama version, corpus, hardware):

| Variable | Values to try | What to observe |
|---|---|---|
| `chunk_strategy` | fixed, token, markdown | Retrieval hit rate vs. chunk quality |
| `chunk_size` | 400, 800, 1200 | Precision vs. context per chunk |
| `chunk_overlap` | 0, 100, 200 | Boundary context loss vs. duplicate storage |
| `--top-k` | 1, 3, 5, 10 | Answer quality vs. prompt length vs. latency |

Run `ingest` then `query --retrieve-only --json` for each variant and measure
retrieval hit rate against `tests/fixtures/eval-dataset.json`. Never fabricate
results.

## 10. Testing and quality

Unit tests run without Ollama (deterministic logic only):

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v   # Windows
python -m unittest discover -s tests -v                      # Linux/macOS
```

Coverage: text cleaning, fixed/token/markdown chunking, stable chunk ids,
approximate token count, chunk dispatcher, document loading (txt/md/pdf,
empty files, bad encoding, corrupt PDFs), duplicate detection, stable
chunk id hashing.

Lint:
```bash
.venv\Scripts\ruff check v02-ingestion-and-chunking   # Windows
ruff check v02-ingestion-and-chunking                  # Linux/macOS
```

Ollama-dependent end-to-end checks are run manually against a local Ollama
instance and are not part of the unit suite.

## 11. Security and performance

- Retrieved context is treated as untrusted reference data, not instructions.
- Never log full document contents or secrets.
- Brute-force retrieval is O(n) in corpus size — fine for the sample corpus.
- PDF parsing via `pypdf` can be slow for large documents.
- All model calls are local via Ollama; no documents leave your machine.

## 12. What changed and why

V01 → V02: Added PDF ingestion (`pypdf`), text cleaning, three chunking
strategies (fixed/token/markdown), stable chunk IDs with duplicate detection,
ingestion reports, malformed-file handling, and a seeded evaluation dataset.

V02 → V03 (planned): Persistent index with model/dimension validation,
incremental ingestion (skip unchanged files), explicit rebuild, NumPy
optimization, and optional ANN index demo.

## 13. References and further reading

- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- pypdf: https://pypdf.readthedocs.io/
- "Retrieval-Augmented Generation for Large Language Models" (Lewis et al., 2020)
- CLI contract: `../../docs/cli-contract.md`
- Roadmap and conventions: `../../ROADMAP.md`, `../../AGENTS.md`