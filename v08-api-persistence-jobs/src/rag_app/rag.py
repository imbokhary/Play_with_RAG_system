from __future__ import annotations

import argparse
import json
import math
import re
import sys
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import fastapi
    import uvicorn
    HAS_FASTAPI = True
except ImportError:  # optional dependency
    fastapi = None
    uvicorn = None
    HAS_FASTAPI = False

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
    "k1": 1.5,
    "b": 0.75,
    "normalize": True,
    "max_query_rewrite_steps": 3,
    "diversity_lambda": 0.5,
    "batch_size": 32,
    "docs_dir": "data/documents",
    "index_path": "index.json",
    "api_host": "0.0.0.0",
    "api_port": 8000,
    "api_workers": 1,
}

HTTP_TIMEOUT = 120