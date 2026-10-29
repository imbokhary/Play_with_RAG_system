# Introduction to RAG

RAG stands for Retrieval-Augmented Generation. It is a technique that gives a language model access to external information at answer time.

## Why RAG matters
A language model knows only what it learned during training. Its knowledge has a cutoff date, it can hallucinate facts, and it cannot see your private documents. RAG solves this by retrieving relevant documents from a corpus and feeding them to the model as context.

## The RAG pipeline
1. Load and chunk documents.
2. Create embeddings for each chunk.
3. Store chunks and vectors in an index.
4. Embed the user question.
5. Search the index for the most similar chunks.
6. Compose a prompt with the retrieved context and the question.
7. Ask the language model to answer, citing sources.

RAG is a pipeline, not a framework. You can build it with a few hundred lines of Python and standard library tools plus a local model runtime such as Ollama.