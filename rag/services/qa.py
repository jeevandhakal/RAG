"""Question-answering chain for basic RAG (no guardrails)."""

from typing import Any

from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 1.0,
) -> ChatGoogleGenerativeAI:
    """Initialize the Google Generative AI chat model.

    Args:
        model: Model name (e.g., gemini-2.5-flash).
        temperature: Sampling temperature (0-2).

    Returns:
        Configured ChatGoogleGenerativeAI instance.
    """
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def build_qa_chain(
    vector_store: Chroma,
    llm: Any,
    k: int = 3,
) -> RetrievalQA:
    """Create the RetrievalQA chain from a vector store and LLM.

    Args:
        vector_store: Chroma vector store for retrieval.
        llm: Language model for answer generation.
        k: Number of documents to retrieve.

    Returns:
        Configured RetrievalQA chain with source documents.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
