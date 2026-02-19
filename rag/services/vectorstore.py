"""Chroma vector store service for document embeddings."""

import logging
import os
import shutil
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manage creation and loading of the Chroma vector store."""

    def __init__(
        self,
        embedding_function: Embeddings,
        persist_directory: str,
    ) -> None:
        """Initialize the service.

        Args:
            embedding_function: Embedding model for document vectors.
            persist_directory: Directory to persist the Chroma database.
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.vector_store: Optional[Chroma] = None

    def exists(self) -> bool:
        """Return True if the vector store directory exists."""
        return os.path.exists(self.persist_directory)

    def load(self) -> Chroma:
        """Load existing vector store from disk."""
        logger.info("Loading existing vector store...")
        self.vector_store = Chroma(
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        return self.vector_store

    def build(
        self,
        chunks: List[Document],
        force_recreate: bool = False,
    ) -> Chroma:
        """Build vector store from document chunks.

        Args:
            chunks: List of document chunks to embed and store.
            force_recreate: If True, remove existing store before building.
        """
        if force_recreate and self.exists():
            logger.info("Rebuilding vector store: removing existing index...")
            shutil.rmtree(self.persist_directory, ignore_errors=True)

        logger.info("Creating embeddings and vector store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        return self.vector_store
