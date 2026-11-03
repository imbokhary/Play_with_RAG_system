from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # optional dependency
    np = None
    HAS_NUMPY = False

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
    "batch_size": 32,
    "k1": 1.5,   # BM25 parameter
    "b": 0.75,   # BM25 parameter
    "normalize": True,
    "docs_dir": "data/documents",
    "index_path": "index.json",
}

HTTP_TIMEOUT = 120