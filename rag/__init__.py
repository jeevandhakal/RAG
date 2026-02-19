"""Modular RAG package for document Q&A with retrieval-augmented generation.

This package provides:
- Document loading and chunking (documents.py)
- Chroma vector store (vectorstore.py)
- Basic and secure QA chains (qa.py, secure_qa.py)
- Guardrails, prompt injection defense, and evaluation
- End-to-end pipeline orchestration (pipeline.py)
"""

__version__ = "0.2.0"

from .config import Settings
from .guardrails import ErrorCode
from .pipeline import RagPipeline
from .services.secure_qa import SecureQueryResult

__all__ = [
    "__version__",
    "RagPipeline",
    "Settings",
    "ErrorCode",
    "SecureQueryResult",
]
