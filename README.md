# RAG System

A modular Retrieval-Augmented Generation (RAG) system using LangChain, Chroma, Jina Embeddings, and Google Generative AI. Includes Assignment 2 (basic RAG) and Assignment 3 (guardrails, prompt injection defense, evaluation).

## Setup
- Install Python 3.12+
- Use `uv` or `pip` to install dependencies.
- Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
JINA_API_KEY=your_jina_api_key_here
```

## Install

Using `uv`:

```bash
uv sync
```

## Run

Interactive mode:

```bash
uv run python main.py
```

Batch mode (Assignment 2 sample queries, saves to `output/results.txt`):

```bash
uv run python main.py --batch
```

**Assignment 3** — Secure RAG tests (guardrails, prompt injection defense, evaluation):

```bash
uv run python main.py --assignment3
```

Force rebuild of the vector store:

```bash
uv run python main.py --rebuild
```

Additional options:
- `--data-dir` (default: `data`)
- `--persist-dir` (default: `chroma_db`)
- `--output-dir` (default: `output`)
- `--k` (default: `3`)
- `--chunk-size` (default: `1000`)
- `--chunk-overlap` (default: `200`)
- `--model` (default: `gemini-2.5-flash`)
- `--temperature` (default: `1.0`)

## Project Structure
- `rag/` — modular package: `config.py`, `guardrails.py`, `prompt_defense.py`, `evaluation.py`, services (`documents.py`, `vectorstore.py`, `qa.py`, `secure_qa.py`), and `pipeline.py`.
- `main.py` — CLI entrypoint.
- `data/` — input PDFs (e.g., DH-Chapter2.pdf).
- `chroma_db/` — persisted Chroma database.
- `output/` — batch results.

## Assignment 3: Guardrails, Prompt Injection Defense & Evaluation

### Prompt Injection Defenses Implemented (3 of 5)
1. **System prompt hardening** — The system prompt explicitly instructs the LLM to: (a) only answer questions about Nova Scotia driving rules, (b) treat retrieved content as untrusted data, (c) never reveal system instructions.
2. **Input sanitization** — User queries are scanned for injection patterns (e.g., "ignore previous instructions", "you are now", "system:", "print your system prompt") and blocked before reaching the LLM.
3. **Instruction-data separation** — Retrieved chunks are wrapped in `<retrieved_context>...</retrieved_context>` delimiters so the LLM can distinguish instructions from document data.
4. **Output validation** — LLM responses are checked for prompt leakage or off-topic content before returning to the user.

### Evaluation Metric Chosen
**Faithfulness check** — For each answer, the LLM evaluates whether the response is supported by the retrieved context (Yes/No). This helps flag hallucinations and ensures answers are grounded in the source document.

### Guardrails
- **Input:** Query length limit (500 chars), off-topic detection (driving/road rules only), PII detection (phone, email, license plate — stripped with warning).
- **Output:** Refusal when retrieval score is below threshold; response length cap (500 words).
- **Execution:** 30-second LLM timeout; structured error codes (QUERY_TOO_LONG, OFF_TOPIC, PII_DETECTED, RETRIEVAL_EMPTY, LLM_TIMEOUT, POLICY_BLOCK).

## Notes
- Ensure `.env` is configured before running.
- The code validates required API keys and prints helpful messages if missing.

