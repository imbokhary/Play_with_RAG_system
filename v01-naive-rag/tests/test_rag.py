"""Unit tests for V01 naive RAG.

These tests exercise deterministic logic WITHOUT a running Ollama instance:
cosine similarity, chunking, stable chunk ids, prompt construction, and
retrieval ranking with hand-computed vectors.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the package root importable so `import rag` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


class TestCosineSimilarity(unittest.TestCase):
    """Cosine similarity against hand-computed examples."""

    def test_identical_vectors(self):
        self.assertAlmostEqual(rag.cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(rag.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite(self):
        self.assertAlmostEqual(rag.cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_known_example(self):
        # cosine([3, 4], [6, 8]) = (18 + 32) / (5 * 10) = 1.0
        self.assertAlmostEqual(rag.cosine_similarity([3.0, 4.0], [6.0, 8.0]), 1.0)

    def test_zero_vector(self):
        self.assertEqual(rag.cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_mismatched_dimensions_raise(self):
        with self.assertRaises(ValueError):
            rag.cosine_similarity([1.0, 0.0], [1.0])


class TestChunking(unittest.TestCase):
    """Fixed-size chunking with overlap."""

    def test_fixed_size(self):
        text = "a" * 1000
        chunks = rag.chunk_text(text, 300, 50)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len(chunks[0]), 300)
        self.assertEqual(len(chunks[-1]), 250)

    def test_overlap(self):
        text = "abcdefghij"  # length 10
        chunks = rag.chunk_text(text, 4, 2)
        self.assertEqual(chunks, ["abcd", "cdef", "efgh", "ghij"])

    def test_invalid_chunk_size_raise(self):
        with self.assertRaises(ValueError):
            rag.chunk_text("text", 0, 0)

    def test_invalid_overlap_raise(self):
        with self.assertRaises(ValueError):
            rag.chunk_text("text", 4, 4)


class TestStableChunkId(unittest.TestCase):
    """Chunk ids must be stable across runs for the same content."""

    def test_deterministic(self):
        id1 = rag.stable_chunk_id("src.md", 0, "hello world")
        id2 = rag.stable_chunk_id("src.md", 0, "hello world")
        self.assertEqual(id1, id2)

    def test_different_position_differs(self):
        self.assertNotEqual(
            rag.stable_chunk_id("src.md", 0, "text"),
            rag.stable_chunk_id("src.md", 1, "text"),
        )

    def test_different_content_differs(self):
        self.assertNotEqual(
            rag.stable_chunk_id("src.md", 0, "text a"),
            rag.stable_chunk_id("src.md", 0, "text b"),
        )


class TestPromptConstruction(unittest.TestCase):
    """Context-grounded prompt construction."""

    def test_prompt_contains_context_and_question(self):
        chunks = [
            {
                "rank": 1,
                "source": "doc.md",
                "text": "RAG retrieves external information.",
            }
        ]
        prompt = rag.build_prompt("What is RAG?", chunks)
        self.assertIn("RAG retrieves external information.", prompt)
        self.assertIn("What is RAG?", prompt)
        self.assertIn("untrusted reference data", prompt)

    def test_prompt_uses_all_chunks(self):
        chunks = [
            {"rank": 1, "source": "a.md", "text": "first chunk"},
            {"rank": 2, "source": "b.md", "text": "second chunk"},
        ]
        prompt = rag.build_prompt("Q?", chunks)
        self.assertIn("first chunk", prompt)
        self.assertIn("second chunk", prompt)


class TestRetrieval(unittest.TestCase):
    """Retrieval ranking with hand-computed vectors (no Ollama needed)."""

    def _index(self):
        return {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "source": "a.md",
                    "text": "alpha",
                    "vector": [1.0, 0.0],
                },
                {
                    "chunk_id": "c2",
                    "source": "b.md",
                    "text": "beta",
                    "vector": [0.0, 1.0],
                },
                {
                    "chunk_id": "c3",
                    "source": "c.md",
                    "text": "gamma",
                    "vector": [1.0, 0.0],
                },
            ]
        }

    def test_top_k_and_ordering(self):
        results = rag.retrieve_by_vector([1.0, 0.0], self._index(), {"top_k": 2})
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[1]["rank"], 2)
        # identical vector scores highest
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_orthogonal_ranks_last(self):
        results = rag.retrieve_by_vector([1.0, 0.0], self._index(), {"top_k": 3})
        self.assertEqual(results[-1]["chunk_id"], "c2")
        self.assertAlmostEqual(results[-1]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()