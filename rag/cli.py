"""Command-line interface for the RAG system."""

import argparse
import os
import sys
from typing import List

from .config import Settings
from .pipeline import RagPipeline


def _print_helper_prompt() -> None:
    """Print usage instructions to stdout."""
    print("\nHow to use:")
    print("- Type your question and press Enter.")
    print("- Type 'help' to show these instructions again.")
    print("- Type 'exit' or 'quit' to close the app.")
    print("- Use '--batch' to run sample queries and save results.")
    print("- Use '--assignment3' to run secure RAG tests.")
    print("\nTips:")
    print("- Specific queries yield better answers.")


def _interactive_mode(pipeline: RagPipeline) -> None:
    """Run interactive Q&A loop."""
    print("\nRAG System Initialized (Type 'exit' to quit, 'help' for instructions): ")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() in ("help", "?"):
            _print_helper_prompt()
            continue
        response = pipeline.query(question)
        print("\n" + response)


def sample_queries() -> List[str]:
    """Return sample queries for Assignment 2 batch mode."""
    return [
        "What is Crosswalk guards?",
        "What to do if moving through an intersection with a green signal?",
        "What to do when approached by an emergency vehicle?",
    ]


def _build_settings(args: argparse.Namespace) -> Settings:
    """Build Settings from parsed CLI arguments."""
    return Settings(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        persist_directory=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k=args.k,
        model=args.model,
        temperature=args.temperature,
    )


def main() -> None:
    """Entry point for the RAG CLI."""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--batch", action="store_true", help="Run sample batch queries")
    parser.add_argument(
        "--assignment3",
        action="store_true",
        help="Run Assignment 3 secure RAG tests",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Force rebuild vector store"
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing PDFs")
    parser.add_argument(
        "--persist-dir", default="chroma_db", help="Chroma persistence dir"
    )
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved docs")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="LLM temperature"
    )
    args = parser.parse_args()

    settings = _build_settings(args)
    validation_errors = settings.validate()
    if validation_errors:
        print("Configuration errors:")
        for err in validation_errors:
            print(f"  - {err}")
        print("\nPlease set the required environment variables (e.g., in a .env file).")
        sys.exit(1)

    pipeline = RagPipeline(settings)
    try:
        pipeline.setup(rebuild=args.rebuild)
    except Exception as exc:
        print(f"Initialization error: {exc}")
        sys.exit(1)

    output_path = os.path.join(settings.output_dir, "results.txt")

    if args.assignment3:
        pipeline.run_assignment3_tests(output_path)
        print(f"\nResults saved to {output_path}")
    elif args.batch:
        print("\nRunning batch queries...")
        pipeline.run_batch(sample_queries(), output_path)
        print(f"\nResults saved to {output_path}")
    else:
        _print_helper_prompt()
        _interactive_mode(pipeline)


if __name__ == "__main__":
    main()
