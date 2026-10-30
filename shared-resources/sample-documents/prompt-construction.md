# Prompt construction

The retrieved chunks are combined into a prompt that tells the language model what to do.

## Grounding
The prompt contains the retrieved context and instructs the model to answer using only that context. If the context is insufficient, the model should say so rather than hallucinate.

## Context as untrusted data
Retrieved text is treated as untrusted reference data, not as instructions. This reduces the risk of prompt injection: a malicious document cannot easily tell the model to ignore its system prompt.

## Citation
The prompt asks the model to name the sources it used, so answers are attributable and verifiable.