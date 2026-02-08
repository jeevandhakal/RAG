import os
import shutil
from typing import List, Optional

from langchain_chroma import Chroma

class VectorStoreService:
    """Manage creation and loading of the Chroma vector store."""

    def __init__(self, embedding_function, persist_directory: str):
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.vector_store: Optional[Chroma] = None

    def exists(self) -> bool:
        return os.path.exists(self.persist_directory)

    def load(self) -> Chroma:
        print("Loading existing vector store...")
        self.vector_store = Chroma(
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        return self.vector_store

    def build(self, chunks: List, force_recreate: bool = False) -> Chroma:
        print("Creating embeddings and vector store...")

        if force_recreate and self.exists():
            print("Rebuilding vector store: removing existing index...")
            shutil.rmtree(self.persist_directory, ignore_errors=True)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        return self.vector_store
