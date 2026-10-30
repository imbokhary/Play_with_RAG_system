# Vector storage

V01 stores chunks and vectors in memory using JSON. Later versions add persistent local indexes.

## What is stored
For each chunk: id, source file, text, embedding vector, and metadata (model name, dimension, chunking config).

## Persistence
The index survives restart by writing to disk. Re-ingesting unchanged files creates no duplicates because each chunk has a stable content-based id.

## Model binding
The index records the embedding model name and dimension. Queries are rejected if they use a different model or dimension than the index was built with.