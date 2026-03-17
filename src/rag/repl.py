"""
Interactive REPL for testing the RAG pipeline.
Usage:
    python -m src.rag.repl CalculusMadeEasy
"""
import sys
from src.rag.query import ask, LLM_MODEL


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.rag.repl <doc_name>")
        print("Example: python -m src.rag.repl CalculusMadeEasy")
        sys.exit(1)

    doc_name = sys.argv[1]

    # Pre-load embedding model so it doesn't print mid-response
    from src.ingestion.pipeline import get_embed_model
    get_embed_model()

    print(f"EdgeTutor REPL | doc={doc_name} | model={LLM_MODEL}")
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
        _, history = ask(question, doc_name, history=history)
        print()


if __name__ == "__main__":
    main()
