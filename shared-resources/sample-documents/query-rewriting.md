# Query rewriting

Users ask vague questions; retrievers need specific ones.

## Multi-query retrieval
Rewrite the question into several paraphrases, retrieve for each, and merge results. This catches relevant chunks that use different wording than the question.

## Trade-off
More queries mean more embedding and retrieval work, so latency rises. Enable only when evaluation shows a clear recall gain.