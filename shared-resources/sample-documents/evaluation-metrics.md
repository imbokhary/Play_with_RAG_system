# Evaluation metrics

RAG quality is measured, not guessed.

## Retrieval metrics
- **Hit rate**: fraction of questions where at least one relevant chunk is retrieved.
- **Recall@k**: fraction of relevant chunks found within the top-k results.
- **MRR** (Mean Reciprocal Rank): average of 1/rank of the first relevant chunk.
- **nDCG** (normalized Discounted Cumulative Gain): rewards relevant chunks ranked highly.

## Answer checks
- **Citation correctness**: a cited source is among the relevant sources.
- **Abstention accuracy**: unanswerable questions are correctly left unanswered.
- **Answer relevance**: optional, often scored with an LLM-as-judge.

## Operational metrics
- Latency (p50/p95), failure rate, and token usage.