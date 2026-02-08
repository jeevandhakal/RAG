# RAG System

A modular Retrieval-Augmented Generation (RAG) system using LangChain, Chroma, Jina Embeddings, and Google Generative AI.

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
uv run python rag_system.py
```

Batch mode (saves to `output/results.txt`):

```bash
uv run python rag_system.py --batch
```

Force rebuild of the vector store:

```bash
uv run python rag_system.py --rebuild
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
- `rag/` — modular package: `config.py`, services (`documents.py`, `vectorstore.py`, `qa.py`), and `pipeline.py`.
- `rag_system.py` — CLI entrypoint.
- `data/` — input PDFs.
- `chroma_db/` — persisted Chroma database.
- `output/` — batch results.

## Notes
- Ensure `.env` is configured before running.
- The code validates required API keys and prints helpful messages if missing.

