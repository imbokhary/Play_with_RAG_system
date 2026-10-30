# Abstention

Not every question can be answered from the corpus.

## When to abstain
If the top retrieved chunks have low similarity to the question, or no chunk supports an answer, the system should say "I don't know" instead of hallucinating.

## Measurement
Evaluation includes unanswerable questions and measures abstention accuracy: how often the system correctly declines to answer.