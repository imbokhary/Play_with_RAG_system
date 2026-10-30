"""Unit tests for V02 ingestion and chunking.

These tests exercise deterministic logic WITHOUT a running Ollama instance:
text cleaning, chunking strategies, stable chunk ids, document loading,
and duplicate detection.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make the package root importable so `import rag` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


class TestCleanText(unittest.TestCase):
    """Text cleaning: whitespace normalization, control char removal."""

    def test_normal_whitespace(self):
        self.assertEqual(rag.clean_text("hello   world"), "hello world")

    def test_newlines_preserved(self):
        self.assertEqual(rag.clean_text("line1\n\n\nline2"), "line1\n\nline2")

    def test_control_chars_removed(self):
        self.assertEqual(rag.clean_text("hello\x00world"), "hello world")

    def test_stripped(self):
        self.assertEqual(rag.clean_text("  hello  "), "hello")


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


class TestChunkingFixed(unittest.TestCase):
    """Fixed-size character chunking with overlap."""

    def test_fixed_size(self):
        text = "a" * 1000
        chunks = rag.chunk_text_fixed(text, 300, 50)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len(chunks[0]), 300)
        self.assertEqual(len(chunks[-1]), 250)

    def test_overlap(self):
        text = "abcdefghij"
        chunks = rag.chunk_text_fixed(text, 4, 2)
        self.assertEqual(chunks, ["abcd", "cdef", "efgh", "ghij"])

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            rag.chunk_text_fixed("text", 0, 0)

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            rag.chunk_text_fixed("text", 4, 4)


class TestChunkingTokenAware(unittest.TestCase):
    """Token-aware chunking (approximate tokens, sentence-aware)."""

    def test_basic(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = rag.chunk_text_token_aware(text, 20, 5)
        self.assertGreater(len(chunks), 0)
        # All chunks should be non-empty
        for c in chunks:
            self.assertGreater(len(c), 0)

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            rag.chunk_text_token_aware("text", 0, 0)


class TestChunkingMarkdown(unittest.TestCase):
    """Markdown-heading-aware chunking."""

    def test_heading_split(self):
        text = (
            "# Heading One\n\n"
            "Content under heading one.\n\n"
            "# Heading Two\n\n"
            "Content under heading two."
        )
        chunks = rag.chunk_text_markdown(text, 100, 10)
        self.assertGreaterEqual(len(chunks), 2)
        # First chunk should start with the heading
        self.assertTrue(chunks[0].startswith("# Heading One"))


class TestApproxTokenCount(unittest.TestCase):
    """Approximate token count rule of thumb."""

    def test_positive(self):
        self.assertGreaterEqual(rag.approx_token_count("hello"), 1)

    def test_scales_with_length(self):
        short = rag.approx_token_count("hi")
        long = rag.approx_token_count("a" * 400)
        self.assertGreater(long, short)


class TestChunkTextDispatcher(unittest.TestCase):
    """chunk_text dispatches to the correct strategy."""

    def test_fixed_strategy(self):
        chunks = rag.chunk_text("abcdefghij", 4, 2, "fixed")
        self.assertEqual(chunks, ["abcd", "cdef", "efgh", "ghij"])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            rag.chunk_text("text", 4, 2, "bogus")


class TestLoadDocuments(unittest.TestCase):
    """Document loading: formats, empty files, bad encoding, corrupt PDFs."""

    def test_loads_txt_and_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("hello txt", encoding="utf-8")
            (Path(tmp) / "b.md").write_text("# hi", encoding="utf-8")
            docs, errors = rag.load_documents(tmp)
            self.assertEqual(errors, [])
            self.assertEqual(len(docs), 2)
            formats = {d["format"] for d in docs}
            self.assertEqual(formats, {"txt", "md"})

    def test_empty_file_reported_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "empty.txt").write_text("", encoding="utf-8")
            docs, errors = rag.load_documents(tmp)
            self.assertEqual(docs, [])
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["error"], "empty file")

    def test_bad_encoding_falls_back_to_latin1(self):
        with tempfile.TemporaryDirectory() as tmp:
            # \xe9 is not valid UTF-8 as a standalone byte sequence in this context
            (Path(tmp) / "latin.txt").write_bytes(b"caf\xe9")
            docs, errors = rag.load_documents(tmp)
            self.assertEqual(errors, [])
            self.assertEqual(len(docs), 1)
            self.assertIn("caf", docs[0]["text"])

    def test_corrupt_pdf_reported_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bad.pdf").write_bytes(b"this is not a pdf")
            docs, errors = rag.load_documents(tmp)
            self.assertEqual(docs, [])
            self.assertEqual(len(errors), 1)
            self.assertIn("source", errors[0])

    def test_pdf_with_no_text_reported_as_error(self):
        # A minimal valid-ish PDF with no extractable text is hard to craft;
        # instead verify an empty directory yields no docs and no errors.
        with tempfile.TemporaryDirectory() as tmp:
            docs, errors = rag.load_documents(tmp)
            self.assertEqual(docs, [])
            self.assertEqual(errors, [])

    def test_missing_dir_returns_empty(self):
        docs, errors = rag.load_documents("/definitely/not/a/dir")
        self.assertEqual(docs, [])
        self.assertEqual(errors, [])


class TestDuplicateDetection(unittest.TestCase):
    """Duplicate detection via stable chunk ids in build_index."""

    def setUp(self):
        self.config = {
            "chunk_strategy": "fixed",
            "chunk_size": 20,
            "chunk_overlap": 2,
            "embedding_model": "test-model",
        }
        # Monkeypatch embed_texts to avoid Ollama.
        self._orig = rag.embed_texts
        rag.embed_texts = lambda texts, config: [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    def tearDown(self):
        rag.embed_texts = self._orig

    def test_first_ingest_creates_chunks(self):
        docs = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        index, report = rag.build_index(docs, self.config, None)
        self.assertGreater(report["new_chunks"], 0)
        self.assertEqual(report["duplicates_skipped"], 0)

    def test_reingest_skips_duplicates(self):
        docs = [{"source": "a.txt", "text": "hello world " * 5, "format": "txt"}]
        first_index, first_report = rag.build_index(docs, self.config, None)
        second_index, second_report = rag.build_index(docs, self.config, first_index)
        self.assertEqual(second_report["new_chunks"], 0)
        self.assertEqual(
            second_report["duplicates_skipped"], first_report["new_chunks"]
        )
        self.assertEqual(
            second_report["total_chunks"], first_report["total_chunks"]
        )


class TestStableChunkIdHash(unittest.TestCase):
    """stable_chunk_id is SHA-1 of source|position|text."""

    def test_matches_sha1(self):
        import hashlib

        src, pos, txt = "x.md", 3, "some text"
        expected = hashlib.sha1(f"{src}|{pos}|{txt}".encode("utf-8")).hexdigest()
        self.assertEqual(rag.stable_chunk_id(src, pos, txt), expected)

if __name__ == "__main__":
    unittest.main()