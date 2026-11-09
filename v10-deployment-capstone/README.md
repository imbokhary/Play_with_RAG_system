"""V10 - Deployment and capstone.

Docker, CI/CD, load tests, final integrated application.

New in V10 (vs V09):
    - Dockerfile for containerized deployment
    - CI/CD pipeline configuration (GitHub Actions)
    - Load testing with lightweight stress testing
    - Final integrated application with all features combined
    - Release checklist and version tagging

Design goals (per ROADMAP.md V10):
    - No RAG orchestration framework.
    - Docker for containerized deployment.
    - CI/CD pipeline for automated testing and releases.
    - Load testing for performance verification.
    - Final integrated application combining all previous versions' features.
    - Release checklist and version tagging.

Run:
    cd v10-deployment-capstone
    pip install -r requirements.txt
    # Build Docker image:
    docker build -t rag-system:latest .
    # Run Docker container:
    docker run -p 8000:8000 rag-system:latest
    # Or run via CLI:
    python -m rag_app ingest --docs data/documents
    python -m rag_app query "What is RAG?" --top-k 5
    python -m rag_app eval --index ./index.json --dataset ./tests/fixtures/eval-dataset.json
    # Run load test:
    python -m rag_app load-test --target-url http://localhost:8000/query --concurrent-users 10 --requests 100

CI/CD:
    The project uses GitHub Actions for continuous integration.
    On push to develop:
    - Run unit tests (no Ollama required)
    - Run lint checks with ruff
    - Build Docker image
    - Tag image with version
    On tag creation (v01, v02, ...):
    - Build and push Docker image
    - Create release notes
    - Update CHANGELOG

Release checklist:
    - [ ] Unit tests pass (no Ollama required)
    - [ ] Ruff lint passes
    - [ ] Docker image builds successfully
    - [ ] Docker image passes basic health check
    - [ ] CHANGELOG entry added
    - [ ] Version tag created
    - [ ] Release notes drafted

## Release History

| Version | Release Date | Features |
|---------|-------------|----------|
| V01 | Initial release | Naive RAG pipeline |
| V02 | + PDF ingestion, cleaning, chunking strategies |
| V03 | + Persistent index, incremental ingestion |
| V04 | + Evaluation harness, metrics |
| V05 | + Hybrid search (BM25 + dense + RRF) |
| V06 | + Query rewriting, reranking, parent-child, abstention |
| V07 | + Clean architecture, type hints, modular package |
| V08 | + FastAPI service, SQLite persistence, jobs |
| V10 | + Docker, CI/CD, load tests, final integrated application |
"""