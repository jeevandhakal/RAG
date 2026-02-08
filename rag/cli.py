import argparse
import os
from typing import List

from .config import Settings
from .pipeline import RagPipeline


def print_helper_prompt():
    print("\nHow to use:")
    print("- Type your question and press Enter.")
    print("- Type 'help' to show these instructions again.")
    print("- Type 'exit' or 'quit' to close the app.")
    print("- Use '--batch' to run sample queries and save results.")
    print("\nTips:")
    print("- Specific queries yield better answers.")


def interactive_mode(pipeline: RagPipeline):
    print("\nRAG System Initialized (Type 'exit' to quit, 'help' for instructions): ")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() in ("help", "?"):
            print_helper_prompt()
            continue
        response = pipeline.query(question)
        print("\n" + response)


def sample_queries() -> List[str]:
    return [
        "What is Crosswalk guards?",
        "What to do if moving through an intersection with a green signal?",
        "What to do when approached by an emergency vehicle?",
    ]


def build_settings_from_args(args: argparse.Namespace) -> Settings:
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


def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--batch", action="store_true", help="Run sample batch queries")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the vector store")
    parser.add_argument("--data-dir", default="data", help="Directory containing PDF documents")
    parser.add_argument("--persist-dir", default="chroma_db", help="Chroma persistence directory")
    parser.add_argument("--output-dir", default="output", help="Output directory for results")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved documents")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for splitting")
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM temperature")
    args = parser.parse_args()

    settings = build_settings_from_args(args)
    validation_errors = settings.validate()
    if validation_errors:
        print("Configuration errors:")
        for err in validation_errors:
            print(f"- {err}")
        print("\nPlease set the required environment variables (e.g., in a .env file).")
        # Continue only if running non-interactive to allow setup; otherwise exit.
        # We exit because model providers require API keys to function.
        return

    pipeline = RagPipeline(settings)
    try:
        pipeline.setup(rebuild=args.rebuild)
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    if args.batch:
        output_path = os.path.join(settings.output_dir, "results.txt")
        pipeline.run_batch(sample_queries(), output_path)
    else:
        print_helper_prompt()
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
