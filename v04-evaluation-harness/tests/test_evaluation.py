"""Unit tests for V04 evaluation harness (deterministic, no Ollama)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


class TestRelevanceJudgment(unittest.TestCase):
    def test_relevant_source_matched(self):
        q = {"relevant_sources": ["chunking.md"], "relevant_passages": []}
        chunks = [{"source": "chunking.md", "text": "some text"}]
        rel = rag.relevance_judgment(q, chunks, top_k=5)
        self.assertIn(0, rel["relevant_at"])

    def test_irrelevant_source_not_matched(self):
        q = {"relevant_sources": ["other.md"], "relevant_passages": []}
        chunks = [{"source": "chunking.md", "text": "some text"}]
        rel = rag.relevance_judgment(q, chunks, top_k=5)
        self.assertEqual(rel["relevant_at"], [])

    def test_answerable_flag(self):
        q = {"relevant_sources": [], "relevant_passages": [], "answerable": True}
        chunks = [{"source": "chunking.md", "text": "some text"}]
        rel = rag.relevance_judgment(q, chunks, top_k=5)
        self.assertTrue(rel["is_answerable"])

    def test_unanswerable_flag(self):
        q = {"relevant_sources": [], "relevant_passages": [], "answerable": False}
        chunks = [{"source": "chunking.md", "text": "some text"}]
        rel = rag.relevance_judgment(q, chunks, top_k=5)
        self.assertFalse(rel["is_answerable"])


class TestPrecisionRecallMRR(unittest.TestCase):
    def test_precision_at_k(self):
        self.assertEqual(rag.precision_at_k([0, 2, 4], k=5), 3 / 5)
        self.assertEqual(rag.precision_at_k([], k=3), 0.0)
        self.assertEqual(rag.precision_at_k([1, 2], k=1), 0.0)

    def test_recall_at_k(self):
        self.assertEqual(rag.recall_at_k([0, 2], k=5, total_relevant=5), 2 / 5)
        self.assertEqual(rag.recall_at_k([], k=3, total_relevant=3), 0.0)

    def test_mean_reciprocal_rank(self):
        self.assertEqual(rag.mean_reciprocal_rank([0, 3, 5], k=5), 1.0 / 1)  # first at pos 0 -> rank 1
        self.assertEqual(rag.mean_reciprocal_rank([2, 4], k=5), 1.0 / 3)  # first at pos 2 -> rank 3
        self.assertEqual(rag.mean_reciprocal_rank([], k=5), 0.0)

    def test_abstention_rate(self):
        self.assertEqual(rag.abstention_rate([], k=5, is_answerable=True), 1.0)
        self.assertEqual(rag.abstention_rate([1], k=5, is_answerable=True), 0.0)
        self.assertEqual(rag.abstention_rate([], k=5, is_answerable=False), 1.0)
        self.assertEqual(rag.abstention_rate([1], k=5, is_answerable=False), 0.0)


class TestComputeMetricsForQuestion(unittest.TestCase):
    def setUp(self):
        self.config = {
            "chunk_strategy": "fixed",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "embedding_model": "test-model",
            "normalize": True,
            "top_k": 5,
            "score_threshold": 0.0,
            "batch_size": 32,
        }
        self._orig = rag.embed_texts
        rag.embed_texts = lambda texts, cfg: [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def tearDown(self):
        rag.embed_texts = self._orig

    def test_basic(self):
        q = {
            "id": "q001",
            "question": "What is RAG?",
            "type": "factual",
            "answerable": True,
            "relevant_sources": ["intro-to-rag.md"],
            "relevant_passages": ["RAG retrieves relevant external information."],
            "reference_answer": "RAG stands for Retrieval-Augmented Generation.",
            "difficulty": "easy",
        }
        # When no chunks are actually relevant, should still return a valid result
        result = rag.compute_metrics_for_question(q, lambda qq, top_k: [], top_k=5)
        self.assertIn("question_id", result)
        self.assertIn("precision_at_k", result)
        self.assertIn("recall_at_k", result)
        self.assertIn("mrr", result)
        self.assertIn("abstained", result)


class TestLoadDataset(unittest.TestCase):
    def test_load_example_dataset(self):
        path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "eval-dataset.json"
        with open(path, "r", encoding="utf-8") as fh:
            dataset = json.load(fh)
        self.assertIn("dataset_version", dataset)
        self.assertIn("corpus_version", dataset)
        self.assertIn("questions", dataset)
        self.assertGreaterEqual(len(dataset["questions"]), 10)


if __name__ == "__main__":
    unittest.main()