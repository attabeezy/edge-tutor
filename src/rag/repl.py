"""
Interactive REPL for testing the RAG pipeline.
Usage:
    python -m src.rag.repl <doc_name> [-e minilm|bge|arctic] [-m MODEL] [-v]
"""
import argparse
from src.rag.query import ask, LLM_MODEL

EMBED_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge":    "TaylorAI/bge-micro-v2",
    "arctic": "Snowflake/snowflake-arctic-embed-xs",
}




def main():
    parser = argparse.ArgumentParser(description="EdgeTutor interactive REPL")
    parser.add_argument("doc_name", help="Document name (e.g. CalculusMadeEasy)")
    parser.add_argument(
        "-e", "--embedding",
        choices=["minilm", "bge", "arctic"],
        default="minilm",
        help="Embedding model: 'minilm' (all-MiniLM-L6-v2), 'bge' (TaylorAI/bge-micro-v2), or 'arctic' (Snowflake/snowflake-arctic-embed-xs). Default: minilm",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Ollama model name to use (e.g. granite4:350m, granite4:350m-h). Default: {LLM_MODEL}",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print step-by-step debug info between question and response.",
    )
    args = parser.parse_args()

    embed_model = EMBED_MODELS[args.embedding]
    llm_model = args.model or LLM_MODEL

    # Pre-load embedding model so it doesn't print mid-response
    from src.ingestion.pipeline import get_embed_model
    get_embed_model(embed_model)

    print(f"EdgeTutor REPL | doc={args.doc_name} | llm={llm_model} | embed={embed_model}")
    print("Type your question and press Enter. 'new' to reset conversation. Ctrl-C to quit.\n")

    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue

        if question.lower() == "new":
            history = []
            print("(Conversation reset)\n")
            continue

        print("Tutor: ", end="", flush=True)
        _, history = ask(question, args.doc_name, history=history, embed_model=embed_model, verbose=args.verbose, llm_model=llm_model)
        print()


if __name__ == "__main__":
    main()
