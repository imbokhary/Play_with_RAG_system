# V10 - Deployment and capstone
# Docker, CI/CD, load tests, final integrated application

from rag_app.rag import (
    load_config,
    clean_text,
    chunk_text,
    stable_chunk_id,
    approx_token_count,
    normalize_vector,
    cosine_similarity,
    dot_product,
)

from rag_app.index import (
    load_index,
    save_index,
    validate_index,
    build_index,
    retrieve,
)

from rag_app.eval_harness import (
    run_eval,
    compute_metrics_for_question,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    abstention_rate,
)

from rag_app.hybrid import (
    hybrid_search,
    bm25_retrieve,
    dense_retrieve,
    rrf_fuse,
)

from rag_app.jobs import (
    submit_job,
    get_job_status,
    list_jobs,
    ingest_job,
    query_job,
)

__all__ = [
    "load_config",
    "clean_text",
    "chunk_text",
    "stable_chunk_id",
    "approx_token_count",
    "normalize_vector",
    "cosine_similarity",
    "dot_product",
    "load_index",
    "save_index",
    "validate_index",
    "build_index",
    "retrieve",
    "run_eval",
    "compute_metrics_for_question",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "abstention_rate",
    "hybrid_search",
    "bm25_retrieve",
    "dense_retrieve",
    "rrf_fuse",
    "submit_job",
    "get_job_status",
    "list_jobs",
    "ingest_job",
    "query_job",
]