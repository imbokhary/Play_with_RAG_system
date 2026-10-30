# Reranking

Retrieval returns a candidate set; reranking re-orders it before the model reads it.

## Why rerank
Retrievers optimize for recall (bring back many possibly relevant chunks). A reranker optimizes for precision (put the truly relevant chunks first), often using a cross-encoder that scores question and chunk together.

## Cost
Reranking adds latency and compute. It is worth enabling only when evaluation shows it improves the target metric enough to justify the cost.