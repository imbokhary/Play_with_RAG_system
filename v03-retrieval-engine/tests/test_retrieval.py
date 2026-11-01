"""Unit tests for V03 retrieval engine (deterministic, no Ollama)."""

from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


class TestCleanText(unittest.TestCase):
    def test_normal_whitespace(self):
        self.assertEqual(rag.clean_text("hello   world"), "hello world")

    def test_control_chars_removed(self):
        self.assertEqual(rag.clean_text("hello\x00world"), "hello world")

    def test_stripped(self):
        self.assertEqual(rag.clean_text("  hello  "), "hello")


class TestStableChunkId(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(
            rag.stable_chunk_id("src.md", 0, "hello world"),
            rag.stable_chunk_id("src.md", 0, "hello world"),
        )

    def test_matches_sha1(self):
        src, pos, txt = "x.md", 3, "some text"
        expected = hashlib.sha1(f"{src}|{pos}|{txt}".encode()).hexdigest()
        self.assertEqual(rag.stable_chunk_id(src, pos, txt), expected)


class TestChunkingFixed(unittest.TestCase):
    def test_fixed_size(self):
        text = "a" * 1000
        chunks = rag.chunk_text_fixed(text, 300, 50)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len(chunks[0]), 300)

    def test_overlap(self):
        self.assertEqual(rag.chunk_text_fixed("abcdefghij", 4, 2), ["abcd", "cdef", "efgh", "ghij"])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            rag.chunk_text_fixed("text", 0, 0)
        with self.assertRaises(ValueError):
            rag.chunk_text_fixed("text", 4, 4)


class TestChunkingTokenAware(unittest.TestCase):
    def test_basic(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = rag.chunk_text_token_aware(text, 20, 5)
        self.assertGreater(len(chunks), 0)


class TestChunkingMarkdown(unittest.TestCase):
    def test_heading_split(self):
        text = "# H1\n\nContent one.\n\n# H2\n\nContent two."
        chunks = rag.chunk_text_markdown(text, 100, 10)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("# H1"))


class TestChunkTextDispatcher(unittest.TestCase):
    def test_fixed_strategy(self):
        self.assertEqual(
            rag.chunk_text("abcdefghij", 4, 2, "fixed"),
            ["abcd", "cdef", "efgh", "ghij"],
        )

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            rag.chunk_text("text", 4, 2, "bogus")


class TestApproxTokenCount(unittest.TestCase):
    def test_positive(self):
        self.assertGreaterEqual(rag.approx_token_count("hello"), 1)

class TestVectorMath(unittest.TestCase):
    def test_normalize_unit_length(self):
        v = rag.normalize_vector([3.0, 4.0])
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=9)

    def test_normalize_zero_unchanged(self):
        self.assertEqual(rag.normalize_vector([0.0, 0.0]), [0.0, 0.0])

    def test_dot_product(self):
        self.assertAlmostEqual(rag.dot_product([1.0, 2.0], [3.0, 4.0]), 11.0)

    def test_cosine_identical(self):
        self.assertAlmostEqual(rag.cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0, places=9)

    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(rag.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=9)

    def test_normalized_dot_equals_cosine(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        cos = rag.cosine_similarity(a, b)
        dot = rag.dot_product(rag.normalize_vector(a), rag.normalize_vector(b))
        self.assertAlmostEqual(cos, dot, places=9)

    def test_numpy_helpers_optional(self):
        if rag.HAS_NUMPY:
            a = [1.0, 2.0, 3.0]
            b = [4.0, 5.0, 6.0]
            self.assertAlmostEqual(
                rag.cosine_similarity_numpy(a, b),
                rag.cosine_similarity(a, b),
                places=9,
            )
        else:
            with self.assertRaises(RuntimeError):
                rag.cosine_similarity_numpy([1.0], [1.0])


class TestLoadDocuments(unittest.TestCase):
    def test_loads_txt_and_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("hello txt", encoding="utf-8")
            (Path(tmp) / "b.md").write_text("# hi", encoding="utf-8")
            docs, errors = rag.load_documents(tmp)
            self.assertEqual(errors, [])
            self.assertEqual(len(docs), 2)

    def test_empty_file_reported_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "empty.txt").write_text("", encoding="utf-8")
            docs, errors = rag.load_documents(tmp)
            self.assertEqual(docs, [])
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["error"], "empty file")

    def test_bad_encoding_falls_back_to_latin1(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "latin.txt").write_bytes(b"caf\xe9")
            docs, errors = rag.load_documents(tmp)
            self.assertEqual(errors, [])
            self.assertEqual(len(docs), 1)

    def test_corrupt_pdf_reported_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bad.pdf").write_bytes(b"this is not a pdf")
            docs, errors = rag.load_documents(tmp)
            self.assertEqual(docs, [])
            self.assertEqual(len(errors), 1)

    def test_missing_dir_returns_empty(self):
        docs, errors = rag.load_documents("/definitely/not/a/dir")
        self.assertEqual(docs, [])
        self.assertEqual(errors, [])


class TestDuplicateDetection(unittest.TestCase):
    def setUp(self):
        self.config = {
            "chunk_strategy": "fixed",
            "chunk_size": 20,
            "chunk_overlap": 2,
            "embedding_model": "test-model",
            "normalize": True,
            "batch_size": 8,
        }
        self._orig = rag.embed_texts
        rag.embed_texts = lambda texts, config: [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    def tearDown(self):
        rag.embed_texts = self._orig

    def test_first_ingest_creates_chunks(self):
        docs = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        index, report = rag.build_index(docs, self.config, None)
        self.assertGreater(report["new_chunks"], 0)
        self.assertEqual(report["duplicates_skipped"], 0)
        self.assertEqual(report["files_skipped"], 0)

    def test_reingest_skips_unchanged(self):
        docs = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        first_index, first_report = rag.build_index(docs, self.config, None)
        second_index, second_report = rag.build_index(docs, self.config, first_index)
        self.assertEqual(second_report["new_chunks"], 0)
        self.assertEqual(second_report["files_skipped"], 1)
        self.assertEqual(second_report["total_chunks"], first_report["total_chunks"])

    def test_rebuild_reprocesses(self):
        docs = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        first_index, _ = rag.build_index(docs, self.config, None)
        second_index, second_report = rag.build_index(docs, self.config, first_index, rebuild=True)
        self.assertGreater(second_report["new_chunks"], 0)
        self.assertEqual(second_report["files_skipped"], 0)

    def test_changed_content_reprocesses(self):
        docs1 = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        idx1, _ = rag.build_index(docs1, self.config, None)
        docs2 = [{"source": "a.txt", "text": "different content here " * 5, "format": "txt"}]
        idx2, report2 = rag.build_index(docs2, self.config, idx1)
        self.assertGreater(report2["new_chunks"], 0)
        self.assertEqual(report2["files_skipped"], 0)

    def test_index_metadata_stored(self):
        docs = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        index, _ = rag.build_index(docs, self.config, None)
        self.assertEqual(index["embedding_model"], "test-model")
        self.assertGreater(index["embedding_dimension"], 0)
        self.assertIn("a.txt", index["files"])
        self.assertIn("content_hash", index["files"]["a.txt"])

    def test_config_change_reprocesses(self):
        docs = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        idx, _ = rag.build_index(docs, self.config, None)
        cfg2 = dict(self.config, chunk_size=10)
        idx2, report2 = rag.build_index(docs, cfg2, idx)
        self.assertEqual(report2["files_skipped"], 0)


class TestValidateIndex(unittest.TestCase):
    def test_missing_index_raises(self):
        with self.assertRaises(RuntimeError):
            rag.validate_index({"embedding_model": "m"}, None)

    def test_model_mismatch_raises(self):
        index = {
            "embedding_model": "other",
            "embedding_dimension": 4,
            "chunks": [{"vector": [0.0] * 4}],
        }
        with self.assertRaises(RuntimeError):
            rag.validate_index({"embedding_model": "m"}, index)

    def test_dimension_mismatch_raises(self):
        index = {
            "embedding_model": "m",
            "embedding_dimension": 4,
            "chunks": [{"vector": [0.0] * 3}],
        }
        with self.assertRaises(RuntimeError):
            rag.validate_index({"embedding_model": "m"}, index)

    def test_valid_passes(self):
        index = {
            "embedding_model": "m",
            "embedding_dimension": 4,
            "chunks": [{"vector": [0.0] * 4}],
        }
        rag.validate_index({"embedding_model": "m"}, index)


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.config = {
            "chunk_strategy": "fixed",
            "chunk_size": 20,
            "chunk_overlap": 2,
            "embedding_model": "test-model",
            "normalize": True,
            "top_k": 2,
            "score_threshold": 0.0,
            "batch_size": 8,
        }
        self._orig = rag.embed_texts
        rag.embed_texts = lambda texts, config: [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def tearDown(self):
        rag.embed_texts = self._orig

    def test_retrieve_returns_top_k(self):
        docs = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        index, _ = rag.build_index(docs, self.config, None)
        for c in index["chunks"]:
            c["vector"] = [1.0, 0.0, 0.0, 0.0]
        chunks, abstained = rag.retrieve("q", index, self.config)
        self.assertFalse(abstained)
        self.assertLessEqual(len(chunks), self.config["top_k"])
        self.assertEqual(chunks[0]["rank"], 1)

    def test_score_threshold_abstains(self):
        docs = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        index, _ = rag.build_index(docs, self.config, None)
        for c in index["chunks"]:
            c["vector"] = [0.0, 1.0, 0.0, 0.0]
        cfg = dict(self.config, score_threshold=0.5)
        chunks, abstained = rag.retrieve("q", index, cfg)
        self.assertTrue(abstained)
        self.assertEqual(chunks, [])

    def test_validate_on_retrieve(self):
        index = {"embedding_model": "other", "embedding_dimension": 4, "chunks": []}
        with self.assertRaises(RuntimeError):
            rag.retrieve("q", index, self.config)


class TestFileContentHash(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(rag.file_content_hash("abc"), rag.file_content_hash("abc"))

    def test_differs(self):
        self.assertNotEqual(rag.file_content_hash("abc"), rag.file_content_hash("xyz"))


if __name__ == "__main__":
    unittest.main()

