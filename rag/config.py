import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


# Load environment variables from .env if present
load_dotenv()


@dataclass
class Settings:
    """Application settings and runtime configuration."""

    # Paths
    data_dir: str = "data"
    output_dir: str = "output"
    persist_directory: str = "chroma_db"

    # Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    k: int = 3

    # LLM & Embeddings
    model: str = "gemini-2.5-flash"
    temperature: float = 1.0

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
