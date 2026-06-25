"""
lecture_rag.py — multi-PDF RAG backend.

Supports holding up to MAX_PDFS PDFs in memory simultaneously. Each PDF is
indexed separately but stored in one shared vector store, with every chunk
tagged by its source filename. This lets you:
  - ask questions across ALL loaded PDFs at once, or
  - restrict a question to one or more SELECTED PDFs only.

Files stay in memory until removed manually via remove_pdf() / clear_all().
Nothing is auto-evicted on new uploads.

Run with no arguments to start an empty interactive session, then use the
'load <path>' command inside the REPL to add PDFs:
    python lecture_rag.py

Or preload PDFs at startup:
    python lecture_rag.py --pdf slides1.pdf --pdf slides2.pdf
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DEFAULT_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 3
MAX_PDFS = 5  # hard cap on how many PDFs can be held in memory at once

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


class PDFLimitExceeded(Exception):
    """Raised when trying to add a PDF beyond MAX_PDFS while none are removed."""


def _load_and_chunk_pdf(pdf_path: Path) -> list[Document]:
    """Load a single PDF, split into chunks, and tag each chunk with its source filename."""
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    if not docs:
        raise ValueError(f"No extractable text found in {pdf_path.name}.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Tag every chunk with its originating filename so we can filter by source later.
    for chunk in chunks:
        chunk.metadata["source"] = pdf_path.name
        # PyPDFLoader gives 0-indexed pages; store a human-friendly 1-indexed version too.
        chunk.metadata["page_display"] = chunk.metadata.get("page", 0) + 1

    return chunks


@dataclass
class PDFLibrary:
    """Holds multiple PDFs in memory, each tagged by filename, in one shared vector store."""

    embedding_model_name: str = EMBEDDING_MODEL
    _embeddings: HuggingFaceEmbeddings = field(init=False, repr=False)
    _vector_store: InMemoryVectorStore | None = field(default=None, init=False, repr=False)
    _loaded_files: dict[str, int] = field(default_factory=dict, init=False)  # filename -> chunk count

    def __post_init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    @property
    def loaded_filenames(self) -> list[str]:
        return list(self._loaded_files.keys())

    @property
    def count(self) -> int:
        return len(self._loaded_files)

    def is_loaded(self, filename: str) -> bool:
        return filename in self._loaded_files

    def add_pdf(self, pdf_path: Path) -> None:
        """Add a new PDF to the library. Raises PDFLimitExceeded if already at MAX_PDFS."""
        filename = pdf_path.name

        if filename in self._loaded_files:
            # Re-adding the same filename: treat as a refresh (remove old chunks first).
            self.remove_pdf(filename)

        if self.count >= MAX_PDFS:
            raise PDFLimitExceeded(
                f"Cannot add '{filename}': already holding the maximum of {MAX_PDFS} PDFs. "
                f"Remove one first (currently loaded: {', '.join(self.loaded_filenames)})."
            )

        chunks = _load_and_chunk_pdf(pdf_path)

        if self._vector_store is None:
            self._vector_store = InMemoryVectorStore.from_documents(chunks, self._embeddings)
        else:
            self._vector_store.add_documents(chunks)

        self._loaded_files[filename] = len(chunks)

    def remove_pdf(self, filename: str) -> bool:
        """Remove one PDF's chunks from memory. Returns True if it was found and removed."""
        if filename not in self._loaded_files:
            return False

        if self._vector_store is not None:
            # InMemoryVectorStore keeps documents in a dict keyed by id; rebuild without this source.
            store_dict = self._vector_store.store
            ids_to_delete = [
                doc_id
                for doc_id, record in store_dict.items()
                if record["metadata"].get("source") == filename
            ]
            for doc_id in ids_to_delete:
                del store_dict[doc_id]

        del self._loaded_files[filename]
        return True

    def clear_all(self) -> None:
        """Remove every PDF from memory."""
        self._vector_store = None
        self._loaded_files.clear()

    def similarity_search(
        self,
        query: str,
        *,
        k: int = DEFAULT_K,
        sources: list[str] | None = None,
    ) -> list[Document]:
        """
        Search across loaded PDFs.

        sources=None  -> search ALL loaded PDFs.
        sources=[...] -> restrict results to only those filenames.
        """
        if self._vector_store is None:
            return []

        if sources is None:
            return self._vector_store.similarity_search(query, k=k)

        # Over-fetch, then filter by source, since the vector store has no native
        # per-call metadata filter for similarity_search in all LangChain versions.
        sources_set = set(sources)
        overfetch_k = max(k * 4, k + 10)
        candidates = self._vector_store.similarity_search(query, k=overfetch_k)
        filtered = [doc for doc in candidates if doc.metadata.get("source") in sources_set]
        return filtered[:k]


def ask(
    question: str,
    library: PDFLibrary,
    llm: ChatOllama,
    *,
    k: int = DEFAULT_K,
    sources: list[str] | None = None,
) -> tuple[str, str]:
    """Retrieve top-k relevant chunks (optionally restricted to `sources`), then answer."""
    retrieved_docs = library.similarity_search(question, k=k, sources=sources)

    if not retrieved_docs:
        context = "(no matching content found in the selected PDF(s))"
    else:
        formatted_contexts = []
        for doc in retrieved_docs:
            src = doc.metadata.get("source", "unknown.pdf")
            page_num = doc.metadata.get("page_display", doc.metadata.get("page", 0) + 1)
            formatted_contexts.append(f"[{src} — Slide/Page {page_num}]:\n{doc.page_content.strip()}")
        context = "\n\n".join(formatted_contexts)

    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return context, answer.strip()


def print_result(question: str, context: str, answer: str) -> None:
    print(f"\nQuestion:\n{question}\n")
    print(f"Retrieved Context:\n{context}\n")
    print(f"Answer:\n{answer}\n")
    print("-" * 60)


def run_interactive(model: str, pdf_paths: list[Path], k: int) -> None:
    """Terminal REPL: load PDFs, ask questions, manage the library with simple commands."""
    library = PDFLibrary()
    llm = ChatOllama(model=model, temperature=0.2)

    for pdf_path in pdf_paths:
        try:
            print(f"Indexing {pdf_path.name} ...")
            library.add_pdf(pdf_path)
        except PDFLimitExceeded as exc:
            print(f"Skipped: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"Failed to index {pdf_path.name}: {exc}", file=sys.stderr)

    print(f"\nLoaded PDFs ({library.count}/{MAX_PDFS}): {', '.join(library.loaded_filenames) or '(none)'}")
    print("\nCommands:")
    print("  load <path>          - load a PDF from disk into memory")
    print("  list                 - show loaded PDFs")
    print("  remove <filename>    - remove one PDF from memory")
    print("  clear                - remove all PDFs")
    print("  only <file1,file2>   - restrict the NEXT question to these PDFs only")
    print("  all                  - reset scope back to ALL loaded PDFs")
    print("  exit / quit          - leave\n")
    print("=" * 60)

    active_scope: list[str] | None = None  # None = all PDFs

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()

        if lowered in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        if lowered.startswith("load "):
            target_str = user_input[len("load "):].strip().strip('"').strip("'")
            target_path = Path(target_str)
            if not target_path.exists():
                print(f"File not found: {target_path}")
            else:
                try:
                    print(f"Indexing {target_path.name} ...")
                    library.add_pdf(target_path)
                    print(f"Loaded. ({library.count}/{MAX_PDFS}) Now loaded: {', '.join(library.loaded_filenames)}")
                except PDFLimitExceeded as exc:
                    print(f"Cannot load: {exc}")
                except Exception as exc:
                    print(f"Failed to index {target_path.name}: {exc}", file=sys.stderr)
            continue

        if lowered == "list":
            print(f"Loaded ({library.count}/{MAX_PDFS}): {', '.join(library.loaded_filenames) or '(none)'}")
            print(f"Current scope: {'ALL' if active_scope is None else ', '.join(active_scope)}")
            continue

        if lowered == "clear":
            library.clear_all()
            active_scope = None
            print("All PDFs removed from memory.")
            continue

        if lowered.startswith("remove "):
            target = user_input[len("remove "):].strip()
            if library.remove_pdf(target):
                print(f"Removed '{target}'.")
                if active_scope and target in active_scope:
                    active_scope.remove(target)
            else:
                print(f"'{target}' not found in loaded PDFs.")
            continue

        if lowered.startswith("only "):
            requested = [f.strip() for f in user_input[len("only "):].split(",") if f.strip()]
            unknown = [f for f in requested if not library.is_loaded(f)]
            if unknown:
                print(f"Unknown filename(s), not currently loaded: {', '.join(unknown)}")
            else:
                active_scope = requested
                print(f"Scope set to: {', '.join(active_scope)}")
            continue

        if lowered == "all":
            active_scope = None
            print("Scope reset to ALL loaded PDFs.")
            continue

        if library.count == 0:
            print("No PDFs loaded yet. Use: load <path-to-pdf>")
            continue

        try:
            context, answer = ask(user_input, library, llm, k=k, sources=active_scope)
            print_result(user_input, context, answer)
        except Exception as exc:
            print(f"Error while answering question: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-PDF RAG Assistant")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        dest="pdfs",
        help=f"Path to a PDF file. Repeat up to {MAX_PDFS} times to load multiple. "
             "Optional — if omitted, start with an empty library and use 'load <path>' "
             "inside the interactive session instead.",
    )
    parser.add_argument("--question", help="Ask a single question and exit (non-interactive mode)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Number of chunks to retrieve (default: {DEFAULT_K})")
    args = parser.parse_args()

    pdf_paths = args.pdfs or []  # may legitimately be empty now

    if len(pdf_paths) > MAX_PDFS:
        print(f"Too many PDFs: got {len(pdf_paths)}, max is {MAX_PDFS}.", file=sys.stderr)
        sys.exit(1)

    missing = [p for p in pdf_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing PDF file: {p}", file=sys.stderr)
        sys.exit(1)

    if args.question:
        if not pdf_paths:
            print("No PDFs provided. --question requires at least one --pdf <path>.", file=sys.stderr)
            sys.exit(1)
        library = PDFLibrary()
        for p in pdf_paths:
            library.add_pdf(p)
        llm = ChatOllama(model=args.model, temperature=0.2)
        context, answer = ask(args.question, library, llm, k=args.k)
        print_result(args.question, context, answer)
        return

    run_interactive(args.model, pdf_paths, args.k)


if __name__ == "__main__":
    main()