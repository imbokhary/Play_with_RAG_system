# Retrieval basics: embeddings and similarity

Retrieval turns text into numbers so a question can be matched against stored documents.

## Embeddings
An embedding model converts a piece of text into a vector (a list of numbers). Similar texts produce nearby vectors. The distance or angle between vectors measures semantic similarity.

## Cosine similarity
Cosine similarity measures the angle between two vectors. A value of 1 means identical direction, 0 means orthogonal, and -1 means opposite. For normalized vectors, cosine similarity equals the dot product.

## Brute-force search
Brute-force search embeds the question, then compares it against every stored vector and returns the top-k most similar. It is exact and simple, but its cost grows linearly with corpus size. For larger corpora, approximate nearest neighbor indexes such as HNSW are used instead.