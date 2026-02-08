from typing import List, Optional
import os

from langchain_community.embeddings import JinaEmbeddings

from .config import Settings
from .services.documents import load_documents, split_documents
from .services.vectorstore import VectorStoreService
from .services.qa import get_llm, build_qa_chain


class RagPipeline:
    """End-to-end orchestration for a RAG workflow."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_function: Optional[JinaEmbeddings] = None
        self.vector_service: Optional[VectorStoreService] = None
        self.qa_chain = None

    def _init_embeddings(self):
        if not self.settings.jina_api_key:
            raise ValueError("JINA_API_KEY not set. Please configure your environment.")
        self.embedding_function = JinaEmbeddings(
            jina_api_key=self.settings.jina_api_key,
            model_name="jina-embeddings-v2-base-en",
        )

    def setup(self, rebuild: bool = False) -> None:
        """Prepare vector store and QA chain."""
        # Initialize embeddings and vector store service
        self._init_embeddings()
        self.vector_service = VectorStoreService(
            embedding_function=self.embedding_function,
            persist_directory=self.settings.persist_directory,
        )

        if self.vector_service.exists() and not rebuild:
            vector_store = self.vector_service.load()
        else:
            documents = load_documents(self.settings.data_dir)
            if not documents:
                raise RuntimeError("No documents found to build the vector store.")
            chunks = split_documents(
                documents,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            vector_store = self.vector_service.build(chunks, force_recreate=rebuild)

        llm = get_llm(model=self.settings.model, temperature=self.settings.temperature)
        self.qa_chain = build_qa_chain(vector_store=vector_store, llm=llm, k=self.settings.k)

    def query(self, question: str) -> str:
        if not self.qa_chain:
            return "System not initialized."

        result = self.qa_chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result["source_documents"]

        formatted_output = f"Question: {question}\nAnswer: {answer}\n\nSources:"
        seen_sources = set()
        for doc in source_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            source_id = f"{source} (Page {page})"
            if source_id not in seen_sources:
                formatted_output += f"\n- {source_id}"
                formatted_output += f"\n  Snippet: \n{doc.page_content[:100]}..."
                seen_sources.add(source_id)

        return formatted_output

    def run_batch(self, queries: List[str], output_path: str) -> None:
        print(f"Running batch queries and saving to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for q in queries:
                response = self.query(q)
                f.write(response + "\n" + "=" * 50 + "\n")
                print(response)
                print("-" * 30)
