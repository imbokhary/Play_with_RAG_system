# Choosing an embedding model

The embedding model determines what "similar" means for your RAG system.

## Dimensions
Embedding models produce vectors of a fixed dimension (for example, 384, 768, or 1024 numbers). Chunks and questions must be embedded with the same model, because vectors from different models are not comparable.

## Reindexing
If you change the embedding model, every stored vector was computed with the old model and must be recomputed. A correct RAG system stores the embedding model name and dimension as index metadata and refuses to query an index built with a different model or dimension.

## Local-first
This tutorial uses Ollama as the primary embedding runtime so embeddings stay local and free. Remote OpenAI-compatible embedding endpoints are optional and introduced through provider interfaces in later versions.