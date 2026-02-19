"""Evaluation metrics for RAG answer quality.

Provides faithfulness checking (LLM-based) and retrieval relevance scoring
for measuring answer quality and retrieval effectiveness.
"""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT: str = """You are evaluating whether an answer is supported by the provided context.

Context:
{context}

Answer:
{answer}

Is this answer fully supported by the context above? Answer with exactly one word: Yes or No."""


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma L2 distance to 0-1 similarity score.

    Chroma returns distance where lower = more similar.
    """
    return 1.0 / (1.0 + distance) if distance >= 0 else 1.0


def evaluate_faithfulness(
    answer: str,
    context: str,
    llm: BaseChatModel,
) -> Tuple[str, bool]:
    """Use LLM to check if answer is supported by the retrieved context.

    Args:
        answer: The generated answer to evaluate.
        context: The retrieved context used to generate the answer.
        llm: The language model for evaluation.

    Returns:
        Tuple of (score_string, is_faithful). score_string is "Yes", "No", or "N/A".
    """
    if not answer or not context:
        return "N/A", False

    try:
        prompt = FAITHFULNESS_PROMPT.format(
            context=context[:3000], answer=answer[:1000]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip().upper() if hasattr(response, "content") else ""
        is_faithful = "YES" in text[:10]
        return "Yes" if is_faithful else "No", is_faithful
    except Exception as exc:
        logger.warning("Faithfulness evaluation failed: %s", exc)
        return "N/A", False


def compute_retrieval_relevance(
    docs_with_scores: List[Tuple[Document, float]],
    similarity_threshold: float = 0.5,
) -> Tuple[int, Optional[float], bool]:
    """Compute retrieval relevance metrics from Chroma results.

    Chroma returns L2 distance (lower = more similar). Converts to 0-1 similarity.

    Args:
        docs_with_scores: List of (Document, distance) tuples from Chroma.
        similarity_threshold: Minimum similarity to consider retrieval successful.

    Returns:
        Tuple of (num_chunks, top_similarity_score, below_threshold).
    """
    if not docs_with_scores:
        return 0, None, True

    scores = [_distance_to_similarity(score) for _, score in docs_with_scores]
    top_score = max(scores) if scores else None
    below_threshold = top_score is not None and top_score < similarity_threshold

    return len(docs_with_scores), top_score, below_threshold
