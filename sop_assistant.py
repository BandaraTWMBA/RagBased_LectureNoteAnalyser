"""
Manufacturing SOP Assistant — local RAG, interactive mode.

Loads sop.txt, retrieves relevant chunks with local embeddings,
and answers questions with a local Ollama LLM using only retrieved context.

Run interactively:
    python sop_rag.py

Run a single question (non-interactive, still supported):
    python sop_rag.py --question "What moisture level is required?"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

SOP_PATH = Path(__file__).resolve().parent / "sop.txt"
DEFAULT_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 1  # how many chunks to retrieve per question

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a manufacturing SOP assistant. Answer using ONLY the "
            "provided context. Restate the matching SOP rule in one short "
            "sentence. Do not calculate, infer, or add facts not in the context. "
            "If the context does not contain the answer, say "
            "\"I don't have that information in the SOP.\"",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\n"
            "Give a one-sentence answer based strictly on the context.",
        ),
    ]
)


def load_vector_store(sop_path: Path) -> InMemoryVectorStore:
    """Read sop.txt, split into line-level chunks, and embed them."""
    raw_text = TextLoader(str(sop_path), encoding="utf-8").load()[0].page_content
    chunks = [
        Document(page_content=line.strip())
        for line in raw_text.splitlines()
        if line.strip()
    ]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return InMemoryVectorStore.from_documents(chunks, embeddings)


def ask(
    question: str,
    vector_store: InMemoryVectorStore,
    llm: ChatOllama,
    *,
    k: int = DEFAULT_K,
) -> tuple[str, str]:
    """Retrieve top-k relevant chunks, then answer using only that context."""
    retrieved_docs = vector_store.similarity_search(question, k=k)
    context = "\n".join(doc.page_content.strip() for doc in retrieved_docs)

    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return context, answer.strip()


def print_result(question: str, context: str, answer: str) -> None:
    print(f"\nQuestion:\n{question}\n")
    print(f"Retrieved Context:\n{context}\n")
    print(f"Answer:\n{answer}\n")
    print("-" * 60)


def run_interactive(model: str, sop_path: Path, k: int) -> None:
    """Main REPL loop: ask questions until the user quits."""
    print("Building retrieval index from", sop_path.name, "...")
    vector_store = load_vector_store(sop_path)
    llm = ChatOllama(model=model, temperature=0)

    print(f"Using local LLM: {model}")
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    print("\nType a question and press Enter.")
    print("Commands: 'reload' to re-index sop.txt, 'exit' or 'quit' to leave.\n")
    print("=" * 60)

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        lowered = question.lower()
        if lowered in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        if lowered == "reload":
            print("Re-reading sop.txt and rebuilding the index ...")
            try:
                vector_store = load_vector_store(sop_path)
                print("Index rebuilt.")
            except Exception as exc:
                print(f"Failed to reload: {exc}", file=sys.stderr)
            continue

        try:
            context, answer = ask(question, vector_store, llm, k=k)
            print_result(question, context, answer)
        except Exception as exc:
            print(f"Error while answering: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manufacturing SOP RAG demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--sop",
        type=Path,
        default=SOP_PATH,
        help="Path to SOP text file",
    )
    parser.add_argument(
        "--question",
        help="Ask a single question and exit (non-interactive mode)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of chunks to retrieve per question (default: {DEFAULT_K})",
    )
    args = parser.parse_args()

    if not args.sop.exists():
        print(f"Missing SOP file: {args.sop}", file=sys.stderr)
        sys.exit(1)

    if args.question:
        vector_store = load_vector_store(args.sop)
        llm = ChatOllama(model=args.model, temperature=0)
        context, answer = ask(args.question, vector_store, llm, k=args.k)
        print_result(args.question, context, answer)
        return

    run_interactive(args.model, args.sop, args.k)


if __name__ == "__main__":
    main()