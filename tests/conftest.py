"""Pytest configuration: add src/ to sys.path and seed RNGs."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Force CPU for deterministic tests unless overridden.
os.environ.setdefault("LLM_DEVICE", "cpu")

import pytest  # noqa: E402

from llm.tensor import set_seed  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    set_seed(0)
