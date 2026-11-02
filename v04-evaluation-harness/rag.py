from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCHEMA_VERSION = "1"

DEFAULT_CONFIG = {
    "ollama_base_url": "http://localhost:11434",
    "chat_model": "qwen2.5-coder:7b",
    "embedding_model": "nomic-embed-text:latest",
    "chunk_size": 800,
    "chunk_overlap": 100,
    "chunk_strategy": "fixed",
    "top_k": 5,
    "score_threshold": 0.0,
    "batch_size": 32,
    "normalize": True,
    "docs_dir": "data/documents",
    "index_path": "index.json",
}

HTTP_TIMEOUT = 120
# ---------------------------------------------------------------------------
# Metric functions (pure-Python, unit-testable without Ollama)
# ---------------------------------------------------------------------------


def relevance_judgment(q: dict, ranked_chunks: list[dict], top_k: int) -> dict:
    """Determine which retrieved chunks are relevant for this question.

    A chunk is relevant if its source file appears in q['relevant_sources'].
    """
    relevant_sources = set(q.get("relevant_sources", []))
    _ = q.get("relevant_passages", [])  # stored for future extensibility
    relevant_chunk_indices = []
    for i, chunk in enumerate(ranked_chunks[:top_k]):
        if chunk.get("source") in relevant_sources:
            relevant_chunk_indices.append(i)
    return {
        "relevant_at": relevant_chunk_indices,
        "is_answerable": q.get("answerable", False),
        "type": q.get("type", "factual"),
    }


def precision_at_k(relevant: list[int], k: int) -> float:
    """Precision at k: fraction of the top-k results that are relevant."""
    if not relevant or k == 0:
        return 0.0
    relevant_top_k = [pos for pos in relevant if pos < k]
    return len(relevant_top_k) / k


def recall_at_k(relevant: list[int], k: int, total_relevant: int) -> float:
    """Recall at k: fraction of all relevant items found in top-k."""
    if not relevant or k == 0 or total_relevant == 0:
        return 0.0
    relevant_top_k = [pos for pos in relevant if pos < k]
    return len(relevant_top_k) / total_relevant


def mean_reciprocal_rank(relevant: list[int], k: int) -> float:
    """MRR: reciprocal of the rank of the first relevant item, capped at k."""
    if not relevant:
        return 0.0
    first_relevant = min(pos for pos in relevant if pos < k)
    # first_relevant is zero-based; rank is 1-based
    rank = first_relevant + 1
    if rank > k:
        return 0.0
    return 1.0 / rank


def abstention_rate(relevant: list[int], k: int, is_answerable: bool) -> float:
    """Fraction of cases where no relevant chunk was found in top-k."""
    if not is_answerable:
        return 1.0 if not relevant else 0.0
    return 1.0 if not relevant else 0.0


def compute_metrics_for_question(
    question: dict,
    get_ranked_chunks_func,
    top_k: int = 5,
) -> dict:
    """Compute all metrics for a single eval question."""
    question_text = question["question"]
    ranked = get_ranked_chunks_func(question_text, top_k=top_k)
    rel = relevance_judgment(question, ranked, top_k)
    p = precision_at_k(rel["relevant_at"], top_k)
    total_rel = len(rel["relevant_at"])
    r = recall_at_k(rel["relevant_at"], top_k, total_rel)
    mrr = mean_reciprocal_rank(rel["relevant_at"], top_k)
    abstain = abstention_rate(rel["relevant_at"], top_k, rel["is_answerable"])
    return {
        "question_id": question["id"],
        "question": question_text,
        "type": question["type"],
        "answerable": rel["is_answerable"],
        "relevant_chunk_positions": rel["relevant_at"],
        "precision_at_k": round(p, 6),
        "recall_at_k": round(r, 6),
        "mrr": round(mrr, 6),
        "abstained": abstain,
        "difficulty": question.get("difficulty", "easy"),
    }
# ---------------------------------------------------------------------------
# Index loading helpers (mirror V03's load_index / retrieve patterns)
# ---------------------------------------------------------------------------


def load_index(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_dataset(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"eval dataset not found: {path}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Benchmark report generation
# ---------------------------------------------------------------------------

def run_eval(
    index_path: str,
    dataset_path: str,
    top_k: int,
    config_path: str | None = None,
) -> dict:
    """Run the evaluation harness and return a benchmark report."""
    index = load_index(index_path)
    if index is None:
        raise RuntimeError(f"index not found: {index_path}")

    dataset = _load_dataset(dataset_path)

    results = []
    for q in dataset["questions"]:
        result = compute_metrics_for_question(q, lambda qq, tk: [], top_k=top_k)
        result["stub"] = True
        results.append(result)

    n = len(results)
    avg_precision = round(sum(r["precision_at_k"] for r in results) / n, 6) if n else 0.0
    avg_recall = round(sum(r["recall_at_k"] for r in results) / n, 6) if n else 0.0
    avg_mrr = round(sum(r["mrr"] for r in results) / n, 6) if n else 0.0
    abstain_count = sum(1 for r in results if r["abstained"])
    abstain_rate_val = round(abstain_count / n, 6) if n else 0.0

    report = {
        "dataset_version": dataset.get("dataset_version", "1.0.0"),
        "corpus_version": dataset.get("corpus_version", "1.0.0"),
        "top_k": top_k,
        "total_questions": n,
        "average_precision_at_k": avg_precision,
        "average_recall_at_k": avg_recall,
        "average_mrr": avg_mrr,
        "abstention_rate": abstain_rate_val,
        "per_question": results,
    }
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V04 RAG evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run evaluation on an index + dataset")
    p_eval.add_argument("--index", required=True, help="Path to index.json")
    p_eval.add_argument("--dataset", required=True, help="Path to eval-dataset.json")
    p_eval.add_argument("--top-k", type=int, default=5, help="Top-k retrieval")
    p_eval.add_argument("--json", action="store_true", help="Output JSON report")

    p_compare = sub.add_parser("compare", help="Compare multiple indexes against the same dataset")
    p_compare.add_argument("--indexes", nargs="+", required=True, help="Path(s) to index.json files")
    p_compare.add_argument("--dataset", required=True, help="Path to eval-dataset.json")
    p_compare.add_argument("--top-k", type=int, default=5, help="Top-k retrieval")

    args = parser.parse_args(argv)

    if args.command == "eval":
        report = run_eval(args.index, args.dataset, args.top_k)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            n = report["total_questions"]
            print(f"Benchmark report (top-k={report['top_k']}, {report['total_questions']} questions):")
            print(f"  Avg Precision@k: {report['average_precision_at_k']}")
            print(f"  Avg Recall@k:      {report['average_recall_at_k']}")
            print(f"  Avg MRR:           {report['average_mrr']}")
            print(f"  Abstention rate:   {report['abstention_rate']}")
            print(f"  Dataset: {report['dataset_version']} / {report['corpus_version']}")
            for r in report["per_question"][:3]:
                print(f"  - {r['question_id']}: P@{report['top_k']}={r['precision_at_k']}, R@{report['top_k']}={r['recall_at_k']}, MRR={r['mrr']}, abstained={r['abstained']}")
        return 0

    if args.command == "compare":
        reports = []
        for idx_path in args.indexes:
            r = run_eval(idx_path, args.dataset, args.top_k)
            r["index_path"] = idx_path
            reports.append(r)
        if args.json:
            print(json.dumps(reports, indent=2))
        else:
            for r in reports:
                print(f'Index: {r["index_path"]}')
                print(f"  Avg P@{r['top_k']}: {r['average_precision_at_k']}")
                print(f"  Avg R@{r['top_k']}: {r['average_recall_at_k']}")
                print(f"  Avg MRR: {r['average_mrr']}")
                print(f"  Abstain: {r['abstention_rate']}")
                print()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
def embed_texts(texts: list[str], config: dict) -> list[list[float]]:
    """Embed a batch of texts via Ollama /api/embed.

    In the harness we return dummy vectors so the code is runnable without
    Ollema; real use replaces this with the real Ollama call.
    """
    if not texts:
        return []
    # Dummy vectors of the expected dimension (768 for nomic-embed-text)
    dim = 768
    return [[float(i % dim) / dim for i in range(dim)] for _ in texts]


def embed_texts_batched(texts: list[str], config: dict) -> list[list[float]]:
    """Embed texts in batches of config['batch_size']."""
    batch_size = max(1, int(config.get("batch_size", 32)))
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(embed_texts(texts[i:i + batch_size], config))
    return out


if __name__ == "__main__":
    sys.exit(main())