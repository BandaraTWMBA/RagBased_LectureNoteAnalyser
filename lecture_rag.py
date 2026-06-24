from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Swapped TextLoader for PyPDFLoader, and added a robust character splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Setup defaults for student slide notes
DEFAULT_PDF_PATH = Path(__file__).resolve().parent / "lecture_slides.pdf"
DEFAULT_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 3  # Increased k from 1 to 3 because slides often span multiple fragments

# Updated prompt to fit academic studying behavior
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful university teaching assistant. Answer the student's "
            "question using ONLY the provided lecture slide context. Be accurate, clear, "
            "and factual based on the slides. Do not add outside knowledge or assumptions. "
            "If the context does not contain the information, state clearly: "
            "\"I cannot find that information in the provided lecture notes.\"",
        ),
        (
            "human",
            "Lecture Notes Context:\n{context}\n\n"
            "Student Question: {question}\n\n"
            "Provide a concise answer based strictly on the context:",
        ),
    ]
)


def load_vector_store(pdf_path: Path) -> InMemoryVectorStore:
    """Read a PDF file, split slide content into chunks, and embed them."""
    print(f"Reading and parsing PDF content from: {pdf_path.name}...")
    
    # Reads the PDF pages cleanly and extracts their structural text layout
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    # Recursive text splitters prevent formulas, bullet points, or paragraphs from breaking awkwardly
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # Captures standard slide paragraphs or bullet lists cleanly
        chunk_overlap=100     # Preserves context boundaries across continuous splits
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return InMemoryVectorStore.from_documents(chunks, embeddings)


def ask(
    question: str,
    vector_store: InMemoryVectorStore,
    llm: ChatOllama,
    *,
    k: int = DEFAULT_K,
) -> tuple[str, str]:
    """Retrieve top-k relevant slide blocks, then answer using only that context."""
    retrieved_docs = vector_store.similarity_search(question, k=k)
    
    # Formats context blocks and appends original slide page numbers for tracing clarity
    formatted_contexts = []
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", 0) + 1  # 0-indexed offset fix
        formatted_contexts.append(f"[Slide/Page {page_num}]:\n{doc.page_content.strip()}")
        
    context = "\n\n".join(formatted_contexts)

    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return context, answer.strip()


def print_result(question: str, context: str, answer: str) -> None:
    print(f"\nQuestion:\n{question}\n")
    print(f"Retrieved Slide Context:\n{context}\n")
    print(f"Answer:\n{answer}\n")
    print("-" * 60)


def run_interactive(model: str, pdf_path: Path, k: int) -> None:
    """Main REPL loop: ask questions about slides until the user quits."""
    print("Building retrieval index from", pdf_path.name, "...")
    vector_store = load_vector_store(pdf_path)
    llm = ChatOllama(model=model, temperature=0.2) # Added minor temperature flexibility for natural explanations

    print(f"Using local LLM: {model}")
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    print("\nType your question about the lecture and press Enter.")
    print("Commands: 'reload' to re-index the PDF file, 'exit' or 'quit' to leave.\n")
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
            print(f"Re-reading {pdf_path.name} and rebuilding index ...")
            try:
                vector_store = load_vector_store(pdf_path)
                print("Index rebuilt successfully.")
            except Exception as exc:
                print(f"Failed to reload PDF: {exc}", file=sys.stderr)
            continue

        try:
            context, answer = ask(question, vector_store, llm, k=k)
            print_result(question, context, answer)
        except Exception as exc:
            print(f"Error while answering question: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lecture Slides PDF RAG Assistant")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help="Path to lecture notes PDF file",
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

    if not args.pdf.exists():
        print(f"Missing PDF slide file target: {args.pdf}", file=sys.stderr)
        print("Please place your slide deck file or specify its path using --pdf", file=sys.stderr)
        sys.exit(1)

    if args.question:
        vector_store = load_vector_store(args.pdf)
        llm = ChatOllama(model=args.model, temperature=0.2)
        context, answer = ask(args.question, vector_store, llm, k=args.k)
        print_result(args.question, context, answer)
        return

    run_interactive(args.model, args.pdf, args.k)


if __name__ == "__main__":
    main()