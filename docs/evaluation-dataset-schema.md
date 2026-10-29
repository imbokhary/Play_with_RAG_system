# Evaluation dataset schema

Created in M0 (schema + a tiny example), seeded with 20–30 questions in V02, and grown in V04.

## Schema

```json
{
  "dataset_version": "1.0.0",
  "corpus_version": "1.0.0",
  "questions": [
    {
      "id": "q001",
      "question": "What does chunk overlap do?",
      "type": "factual",
      "answerable": true,
      "relevant_sources": ["chunking.md"],
      "relevant_passages": ["Overlap repeats the end of one chunk at the start of the next."],
      "reference_answer": "Overlap preserves context across chunk boundaries.",
      "difficulty": "easy"
    },
    {
      "id": "q020",
      "question": "What is the capital of Mars?",
      "type": "unanswerable",
      "answerable": false,
      "relevant_sources": [],
      "relevant_passages": [],
      "reference_answer": null,
      "difficulty": "easy"
    }
  ]
}
```

## Field reference
| Field | Type | Description |
|---|---|---|
| `dataset_version` | string | Semver of this dataset file. |
| `corpus_version` | string | Semver of the source corpus it was built against. |
| `questions[].id` | string | Stable question id. |
| `questions[].question` | string | The user question. |
| `questions[].type` | string | `factual`, `multi-chunk`, `keyword-heavy`, `paraphrased`, or `unanswerable`. |
| `questions[].answerable` | boolean | Whether the corpus contains enough evidence to answer. |
| `questions[].relevant_sources` | string[] | Source file names that contain the answer (matched by file + passage text, not chunk id). |
| `questions[].relevant_passages` | string[] | Passage text that supports the answer. |
| `questions[].reference_answer` | string|null | Gold answer when answerable, else `null`. |
| `questions[].difficulty` | string | `easy`, `medium`, or `hard`. |

## Dataset requirements
- Use a corpus with a clear license (documented in `shared-resources/sample-documents/LICENSE-NOTES.md`). 30–50 short documents are enough.
- Include question types: factual, multi-chunk (needs two passages), keyword-heavy (exact terms, IDs), paraphrased, and unanswerable (to test abstention).
- Match relevance by source file and passage text, not by chunk ID, so the dataset survives changes to chunking.
- Version the dataset. Results are only comparable on the same dataset and corpus version.

## Example file
See `shared-resources/datasets/example-dataset.json` for a tiny M0 example.