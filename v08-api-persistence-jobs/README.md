"""V08 - API, persistence and jobs.

New in V08 (vs V07):
    - FastAPI service with REST endpoints for ingest and query
    - SQLite persistence with Alembic migrations
    - Streaming ingestion jobs with status tracking
    - Background job queue using SQLite

Design goals (per ROADMAP.md V08):
    - No RAG orchestration framework.
    - FastAPI for HTTP API, SQLite for persistence.
    - Educational code, type-annotated, modular.
    - Implements docs/cli-contract.md (schema_version 1).

Run:
    cd v08-api-persistence-jobs
    pip install -r requirements.txt
    # Create SQLite database:
    python -m rag_app init-db
    # Start the API server:
    uvicorn rag_app.main:app --host 0.0.0.0 --port 8000
    # Or run via CLI:
    python -m rag_app ingest --docs data/documents
    python -m rag_app query "What is RAG?"
    python -m rag_app eval --index ./index.json --dataset ./tests/fixtures/eval-dataset.json

CLI commands:
    python -m rag_app ingest --docs data/documents --json
    python -m rag_app query "What is RAG?" --top-k 5 --json
    python -m rag_app ingest --jobs  # background ingestion job
    python -m rag_app query "What is RAG?" --stream  # streaming query

References:
- FastAPI docs: https://fastapi.tiangolo.com/
- SQLAlchemy docs: https://docs.sqlalchemy.org/
- Alembic docs: https://alembic.readthedocs.io/
"""