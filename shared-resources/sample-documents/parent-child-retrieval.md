# Parent-child retrieval

Small chunks embed and retrieve well, but large passages answer better.

## Idea
Index small child chunks for retrieval. When a child is relevant, return its parent document (or a larger surrounding chunk) as context for the model. This combines precise lookup with rich context.

## Complexity
It adds mapping between children and parents and a second lookup step, so it is introduced only after the baseline retrieval is measured.