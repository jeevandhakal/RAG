from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


def get_llm(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    """Initialize the Google Generative AI chat model."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def build_qa_chain(vector_store, llm, k: int) -> RetrievalQA:
    """Create the RetrievalQA chain from a vector store and LLM."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
