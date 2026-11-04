from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

SCHEMA_VERSION = "1"

DEFAULT_CONFIG = {
    "ollama_base_url": "http://localhost:11434",
    "chat_model": "qwen2.5-coder:7b",
    "embedding_model": "nomic-embed-text:latest",
    "chunk_size": 800,
    "chunk_overlap": 100,
    "chunk_strategy": "fixed",
    "top_k": 5,
    "score_threshold": 0.0,
}
# ---------------------------------------------------------------------------
# Text Processing
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a JSON file, falling back to defaults."""
    config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                user_cfg: Dict[str, Any] = json.load(fh)
            config.update(user_cfg)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            sys.stderr.write(f"Warning: Could not load config from {path}: {e}\n")
    return config


def clean_text(text: str) -> str:
    """Clean raw document text.

    - Replaces null bytes and control characters (except newline/tab).
    - Collapses runs of whitespace into a single space (preserves newlines).
    - Strips leading/trailing whitespace.
    """
    # Replace null bytes and other control chars except \n and \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Collapse runs of whitespace into a single space (preserves newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse triple+ newlines into double newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_token_count(text: str) -> int:
    """Approximate token count for English text (~4 chars/token)."""
    return max(1, int(len(text) / 4.0))


def chunk_text_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-size character chunking with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = end - overlap

    return chunks


def chunk_text_token_aware(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Token-aware chunking (approximate tokens, sentence-aware)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    char_chunk_size = int(chunk_size * 4.0)  # 4 chars per token
    char_overlap = int(overlap * 4.0)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current: str = ""

    for sentence in sentences:
        candidate: str = (current + " " + sentence).strip() if current else sentence
        if approx_token_count(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if approx_token_count(sentence) > chunk_size:
                # Sentence itself is too large; break it with fixed chunking
                chunks.extend(chunk_text_fixed(sentence, char_chunk_size, char_overlap))
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def chunk_text_markdown(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Markdown-heading-aware chunking.

    Splits on `#`/`##` headings, keeping each heading with its content.
    Large sections are further split with fixed-size chunking.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    import re as re_mod

    # Split on markdown headings (#, ##, ###, etc.)
    parts = re_mod.split(r"(?=^#{1,6}\s)", text, flags=re_mod.MULTILINE)
    sections: list[str] = [p.strip() for p in parts if p.strip()]

    chunks: list[str] = []
    for section in sections:
        if approx_token_count(section) <= chunk_size:
            chunks.append(section)
        else:
            # Large section: fall back to fixed chunking
            chunks.extend(chunk_text_fixed(section, chunk_size, overlap))

    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int, strategy: str = "fixed") -> list[str]:
    """Dispatch to the configured chunking strategy.

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap
        strategy: Chunking strategy (fixed | token | markdown)

    Returns:
        List of chunk texts
    """
    if strategy == "fixed":
        return chunk_text_fixed(text, chunk_size, overlap)
    elif strategy == "token":
        return chunk_text_token_aware(text, chunk_size, overlap)
    elif strategy == "markdown":
        return chunk_text_markdown(text, chunk_size, overlap)
    else:
        raise ValueError(f"unknown chunk strategy: {strategy}. Use 'fixed', 'token', or 'markdown'")