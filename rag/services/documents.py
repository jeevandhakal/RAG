import glob
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(data_dir: str) -> List:
    """Load all PDF documents from the given directory."""
    documents: List = []
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return []

    print(f"Loading {len(pdf_files)} documents...")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {pdf_file}: {len(docs)} pages")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    return documents


def split_documents(documents: List, chunk_size: int, chunk_overlap: int) -> List:
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks
