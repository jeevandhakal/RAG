"""Document loading and splitting for the RAG pipeline."""

import glob
import logging
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_documents(data_dir: str) -> List[Document]:
    """Load all PDF documents from the given directory.

    Args:
        data_dir: Path to directory containing PDF files.

    Returns:
        List of LangChain Document objects (one per page).
    """
    documents: List[Document] = []
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return []

    logger.info("Loading %d documents...", len(pdf_files))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            logger.info("Loaded %s: %d pages", pdf_file, len(docs))
        except Exception as exc:
            logger.error("Error loading %s: %s", pdf_file, exc)

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into chunks for embedding.

    Args:
        documents: List of documents to split.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Split into %d chunks.", len(chunks))
    return chunks
