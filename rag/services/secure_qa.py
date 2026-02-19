"""Secure RAG query with guardrails, prompt injection defense, and evaluation."""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..guardrails import (
    ErrorCode,
    apply_input_guardrails,
    check_response_length,
)
from ..prompt_defense import (
    JAILBREAK_REFUSAL,
    SYSTEM_PROMPT_HARDENED,
    sanitize_input,
    validate_output,
    wrap_context_for_llm,
)
from ..evaluation import (
    compute_retrieval_relevance,
    evaluate_faithfulness,
)

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SecureQueryResult:
    """Structured result for secure RAG queries (Assignment 3 output format)."""

    query: str
    guardrails_triggered: List[str] = field(default_factory=list)
    error_code: Optional[ErrorCode] = None
    retrieved_chunks: int = 0
    top_similarity_score: Optional[float] = None
    answer: str = ""
    faithfulness_score: str = "N/A"
    injection_blocked: bool = False


def query_secure(
    question: str,
    vector_store: "VectorStore",
    llm: BaseChatModel,
    *,
    k: int = 3,
    retrieval_threshold: float = 0.3,
    timeout_seconds: int = 30,
    max_response_words: int = 500,
    run_faithfulness: bool = True,
) -> SecureQueryResult:
    """Execute a secure RAG query with guardrails, prompt injection defense, and evaluation.

    Applies input guardrails (length, off-topic, PII), injection detection,
    retrieval with confidence threshold, LLM call with timeout, output validation,
    and optional faithfulness evaluation.

    Args:
        question: The user's question.
        vector_store: Chroma vector store for retrieval.
        llm: Chat model for answer generation.
        k: Number of chunks to retrieve.
        retrieval_threshold: Minimum similarity score to accept retrieval.
        timeout_seconds: LLM call timeout.
        max_response_words: Maximum response length in words.
        run_faithfulness: Whether to run faithfulness evaluation.

    Returns:
        SecureQueryResult with answer, guardrails triggered, and evaluation scores.
    """
    result = SecureQueryResult(query=question)

    # --- Input guardrails ---
    input_result = apply_input_guardrails(question)
    if not input_result.should_proceed:
        result.guardrails_triggered = input_result.triggered
        result.error_code = input_result.error_code
        result.answer = input_result.refusal_message or "Request blocked."
        return result

    if input_result.warning:
        result.guardrails_triggered.append("pii_detection")
        result.error_code = ErrorCode.PII_DETECTED
        result.answer = input_result.warning + " "

    query_to_use = input_result.sanitized_query or question

    # --- Prompt injection defense: input sanitization ---
    _, injection_blocked, injection_triggered = sanitize_input(query_to_use)
    if injection_blocked:
        result.guardrails_triggered.extend(injection_triggered)
        result.error_code = ErrorCode.POLICY_BLOCK
        result.answer = (result.answer or "") + JAILBREAK_REFUSAL
        result.injection_blocked = True
        return result

    # --- Retrieval with scores ---
    try:
        docs_with_scores = vector_store.similarity_search_with_score(query_to_use, k=k)
    except Exception as exc:
        logger.debug("Retrieval failed: %s", exc)
        result.error_code = ErrorCode.RETRIEVAL_EMPTY
        result.answer = "I don't have enough information to answer that."
        result.guardrails_triggered.append("retrieval_error")
        return result

    num_chunks, top_score, below_threshold = compute_retrieval_relevance(
        docs_with_scores, similarity_threshold=retrieval_threshold
    )
    result.retrieved_chunks = num_chunks
    result.top_similarity_score = round(top_score, 4) if top_score is not None else None

    # --- Output guardrail: refusal on low confidence ---
    if not docs_with_scores or below_threshold:
        result.guardrails_triggered.append("retrieval_empty_or_low_confidence")
        result.error_code = ErrorCode.RETRIEVAL_EMPTY
        result.answer = "I don't have enough information to answer that."
        return result

    # Build context with instruction-data separation
    context_parts = [doc.page_content for doc, _ in docs_with_scores]
    context_str = "\n\n---\n\n".join(context_parts)
    wrapped_context = wrap_context_for_llm(context_str)

    system_content = f"""{SYSTEM_PROMPT_HARDENED}

Use the following retrieved context to answer the question. If the context does not contain enough information, say "I don't have enough information to answer that."

{wrapped_context}"""

    # --- LLM call with timeout (cross-platform via ThreadPoolExecutor) ---
    def _invoke_llm() -> object:
        return llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=query_to_use),
            ]
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke_llm)
            response = future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        result.error_code = ErrorCode.LLM_TIMEOUT
        result.guardrails_triggered.append("llm_timeout")
        result.answer = "Request timed out. Please try again."
        return result
    except Exception as exc:
        logger.warning("LLM invocation failed: %s", exc)
        result.answer = f"An error occurred: {exc!s}"
        return result

    answer = response.content if hasattr(response, "content") else str(response)

    # --- Output guardrail: response length ---
    length_result = check_response_length(answer, max_words=max_response_words)
    if length_result.sanitized_query:
        answer = length_result.sanitized_query
        result.guardrails_triggered.append("response_length_limit")

    # --- Output validation (prompt injection defense) ---
    is_valid, _ = validate_output(answer, query_to_use)
    if not is_valid:
        result.guardrails_triggered.append("output_validation_failed")
        result.error_code = ErrorCode.POLICY_BLOCK
        result.answer = JAILBREAK_REFUSAL
        return result

    result.answer = (result.answer or "") + answer

    # --- Faithfulness evaluation ---
    if run_faithfulness and answer:
        faith_score, _ = evaluate_faithfulness(answer, context_str, llm)
        result.faithfulness_score = faith_score

    return result
