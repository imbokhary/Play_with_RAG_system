# Hybrid search

Dense retrieval finds semantically similar chunks; keyword search finds exact terms. Hybrid search combines both.

## BM25
BM25 is a term-frequency based keyword ranking. It rewards documents containing rare query terms and normalizes by document length.

## Reciprocal Rank Fusion (RRF)
RRF merges rankings from keyword and dense search without needing comparable scores. For each chunk, its fused score is the sum of 1 / (k + rank) over all rankings.

## When each wins
- Keyword search wins for exact terms, IDs, and codes.
- Dense search wins for paraphrased questions.
- Hybrid search wins overall by covering both cases.