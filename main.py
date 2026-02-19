"""RAG system entry point.

Run with:
    uv run python main.py              # Interactive mode
    uv run python main.py --batch      # Batch mode (Assignment 2)
    uv run python main.py --assignment3  # Secure RAG tests (Assignment 3)
"""

import logging
import sys

from rag.cli import main

# Configure logging: INFO for rag package, WARNING for third-party
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logging.getLogger("rag").setLevel(logging.INFO)

if __name__ == "__main__":
    main()
