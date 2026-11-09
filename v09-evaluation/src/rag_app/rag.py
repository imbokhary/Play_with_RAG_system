from __future__ import annotations

import argparse
import json
import math
import re
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