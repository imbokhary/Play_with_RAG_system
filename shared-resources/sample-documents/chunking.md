# Chunking and overlap

Documents are usually too long to fit into a single model prompt, so they are split into smaller pieces called chunks.

## Fixed-size chunking
The simplest method cuts text into fixed-size windows (for example, 800 characters) with a fixed step. Every window becomes one chunk with its own embedding.

## Chunk overlap
Chunk overlap repeats the end of one chunk at the start of the next. If a sentence falls across a chunk boundary, the overlapping region keeps the context together so the retriever can still find it.

## Trade-offs
- Small chunks: precise matches, but each chunk may lack context.
- Large chunks: more context per chunk, but lower precision and more tokens sent to the model.
- Overlap: reduces boundary loss at the cost of storing more, near-duplicate chunks.

## Metadata
Each chunk stores its source file, position, and a stable id derived from its content so the same document produces the same chunks on re-ingestion.