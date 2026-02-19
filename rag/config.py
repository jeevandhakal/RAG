"""Application configuration and settings."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings and runtime configuration.

    Paths, processing parameters, and API keys. API keys are loaded from
    environment variables (e.g., via .env file).
    """

    # Paths
    data_dir: str = "data"
    output_dir: str = "output"
    persist_directory: str = "chroma_db"

    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    k: int = 3

    # LLM & Embeddings
    model: str = "gemini-2.5-flash"
    temperature: float = 1.0

    # Guardrails & Security (Assignment 3)
    retrieval_threshold: float = 0.3
    llm_timeout_seconds: int = 30
    max_response_words: int = 500

    # API keys (loaded from environment)
    jina_api_key: Optional[str] = os.getenv("JINA_API_KEY")
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")

    def validate(self) -> list[str]:
        """Return a list of validation error messages (if any)."""
        errors: list[str] = []
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY not found in environment variables.")
        if not self.jina_api_key:
            errors.append("JINA_API_KEY not found in environment variables.")
        return errors
