from __future__ import annotations

# Standard library imports
import argparse          # parses command-line flags like --pdf, --model, --question, --k
import sys                # used for printing to stderr and exiting with error codes
import warnings           # used to silence noisy deprecation warnings from dependencies
from dataclasses import dataclass, field  # used to define the PDFLibrary class cleanly
from pathlib import Path  # represents filesystem paths in a cross-platform way

# Suppress deprecation warnings (LangChain's APIs change frequently between versions)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LangChain components that do the actual RAG work
from langchain_community.document_loaders import PyPDFLoader        # extracts text per-page from a PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits long text into smaller chunks
from langchain_core.documents import Document                        # the standard "chunk of text + metadata" object
from langchain_core.output_parsers import StrOutputParser            # converts the LLM's raw output into a plain string
from langchain_core.vectorstores import InMemoryVectorStore          # in-RAM vector database for embeddings
from langchain_core.prompts import ChatPromptTemplate                # builds the system/human prompt sent to the LLM
from langchain_huggingface import HuggingFaceEmbeddings              # local embedding model (turns text into vectors)
from langchain_ollama import ChatOllama                               # client for talking to a local Ollama LLM server

# ----- Configuration defaults -----
DEFAULT_MODEL = "llama3.2:1b"  # default local Ollama model used for answering questions
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # default local embedding model
DEFAULT_K = 3  # default number of chunks retrieved per question (slides often span multiple chunks)
MAX_PDFS = 5  # hard cap on how many PDFs can be held in memory at once

# System + human prompt template sent to the LLM for every question.
# {context} is filled with retrieved chunk text; {question} is filled with the user's question.
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
    # PyPDFLoader reads the PDF and returns one Document per page,
    # each with page_content (extracted text) and metadata (e.g. page number).
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    # Guard against PDFs that produced no extractable text (e.g. scanned image-only PDFs).
    if not docs:
        raise ValueError(f"No extractable text found in {pdf_path.name}.")

    # Break each page's text into smaller, overlapping chunks.
    # chunk_size=600 keeps chunks roughly slide/paragraph-sized.
    # chunk_overlap=100 repeats some text between consecutive chunks so an idea
    # that spans a chunk boundary isn't fully lost in either chunk.
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
    # The embedding model instance, created once in __post_init__ (not passed in by the caller).
    _embeddings: HuggingFaceEmbeddings = field(init=False, repr=False)
    # The single shared vector store holding chunks from ALL loaded PDFs.
    # Starts as None until the first PDF is added.
    _vector_store: InMemoryVectorStore | None = field(default=None, init=False, repr=False)
    # Tracks which filenames are currently loaded and how many chunks each contributed.
    _loaded_files: dict[str, int] = field(default_factory=dict, init=False)  # filename -> chunk count

    def __post_init__(self) -> None:
        # dataclasses can't easily set a default value that depends on another field
        # in the class body, so the embedding model is instantiated here instead.
        self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    @property
    def loaded_filenames(self) -> list[str]:
        # Returns the list of PDF filenames currently held in memory.
        return list(self._loaded_files.keys())

    @property
    def count(self) -> int:
        # Returns how many PDFs are currently loaded (used to enforce MAX_PDFS).
        return len(self._loaded_files)

    def is_loaded(self, filename: str) -> bool:
        # Quick membership check, used e.g. when validating an "only <file>" scope request.
        return filename in self._loaded_files

    def add_pdf(self, pdf_path: Path) -> None:
        """Add a new PDF to the library. Raises PDFLimitExceeded if already at MAX_PDFS."""
        filename = pdf_path.name

        if filename in self._loaded_files:
            # Re-adding the same filename: treat as a refresh (remove old chunks first).
            self.remove_pdf(filename)

        # Enforce the hard cap on simultaneously loaded PDFs.
        if self.count >= MAX_PDFS:
            raise PDFLimitExceeded(
                f"Cannot add '{filename}': already holding the maximum of {MAX_PDFS} PDFs. "
                f"Remove one first (currently loaded: {', '.join(self.loaded_filenames)})."
            )

        # Extract, chunk, and tag the new PDF's content with its source filename.
        chunks = _load_and_chunk_pdf(pdf_path)

        # First PDF ever added: create the vector store. Otherwise, add into the existing one
        # so all PDFs' chunks live together in a single shared store.
        if self._vector_store is None:
            self._vector_store = InMemoryVectorStore.from_documents(chunks, self._embeddings)
        else:
            self._vector_store.add_documents(chunks)

        # Record that this file is now loaded, along with how many chunks it produced.
        self._loaded_files[filename] = len(chunks)

    def remove_pdf(self, filename: str) -> bool:
        """Remove one PDF's chunks from memory. Returns True if it was found and removed."""
        if filename not in self._loaded_files:
            return False

        if self._vector_store is not None:
            # InMemoryVectorStore keeps documents in a dict keyed by id; rebuild without this source.
            # We reach into the internal `.store` dict directly because InMemoryVectorStore
            # has no public "delete by metadata filter" method in this LangChain version.
            store_dict = self._vector_store.store
            ids_to_delete = [
                doc_id
                for doc_id, record in store_dict.items()
                if record["metadata"].get("source") == filename
            ]
            for doc_id in ids_to_delete:
                del store_dict[doc_id]

        # Forget that this file was loaded, freeing up a slot under MAX_PDFS.
        del self._loaded_files[filename]
        return True

    def clear_all(self) -> None:
        """Remove every PDF from memory."""
        # Dropping the vector store entirely and clearing the filename registry
        # effectively wipes everything that add_pdf() has built up so far.
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
        # Nothing has been loaded yet, so there's nothing to search.
        if self._vector_store is None:
            return []

        # No source filter requested: just do a normal top-k similarity search.
        if sources is None:
            return self._vector_store.similarity_search(query, k=k)

        # Over-fetch, then filter by source, since the vector store has no native
        # per-call metadata filter for similarity_search in all LangChain versions.
        # Without over-fetching, the top-k results across ALL PDFs might not include
        # any chunks from the specific PDF(s) the caller wants to restrict to.
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
    # Step 1: retrieval — find the chunks most semantically similar to the question.
    retrieved_docs = library.similarity_search(question, k=k, sources=sources)

    if not retrieved_docs:
        # No relevant chunks found (e.g. empty library, or filter excluded everything).
        context = "(no matching content found in the selected PDF(s))"
    else:
        # Format each retrieved chunk with its source filename and page number,
        # so the LLM (and the person reading the printed context) can trace answers
        # back to a specific PDF and slide/page.
        formatted_contexts = []
        for doc in retrieved_docs:
            src = doc.metadata.get("source", "unknown.pdf")
            page_num = doc.metadata.get("page_display", doc.metadata.get("page", 0) + 1)
            formatted_contexts.append(f"[{src} — Slide/Page {page_num}]:\n{doc.page_content.strip()}")
        context = "\n\n".join(formatted_contexts)

    # Step 2: generation — feed the retrieved context + question into the LLM via the
    # prompt template, then parse the raw model output into a plain string.
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return context, answer.strip()


def print_result(question: str, context: str, answer: str) -> None:
    # Simple terminal output: show what was asked, what was retrieved, and the answer.
    print(f"\nQuestion:\n{question}\n")
    print(f"Retrieved Context:\n{context}\n")
    print(f"Answer:\n{answer}\n")
    print("-" * 60)


def run_interactive(model: str, pdf_paths: list[Path], k: int) -> None:
    """Terminal REPL: load PDFs, ask questions, manage the library with simple commands."""
    library = PDFLibrary()
    llm = ChatOllama(model=model, temperature=0.2)  # temperature=0.2 allows slightly more natural phrasing

    # Preload any PDFs passed in via --pdf at startup (this list may be empty,
    # in which case the session simply starts with zero PDFs loaded).
    for pdf_path in pdf_paths:
        try:
            print(f"Indexing {pdf_path.name} ...")
            library.add_pdf(pdf_path)
        except PDFLimitExceeded as exc:
            print(f"Skipped: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"Failed to index {pdf_path.name}: {exc}", file=sys.stderr)

    # Print a short status + command reference before entering the input loop.
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

    active_scope: list[str] | None = None  # None = all PDFs; otherwise, list of filenames to search

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D or Ctrl+C: exit gracefully instead of showing a traceback.
            print("\nExiting.")
            break

        if not user_input:
            # Empty input (just pressed Enter): re-prompt without doing anything.
            continue

        lowered = user_input.lower()  # used for case-insensitive command matching

        if lowered in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        if lowered.startswith("load "):
            # "load <path>" — index a new PDF from disk and add it to the library.
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
            # Show which PDFs are loaded and what the current retrieval scope is.
            print(f"Loaded ({library.count}/{MAX_PDFS}): {', '.join(library.loaded_filenames) or '(none)'}")
            print(f"Current scope: {'ALL' if active_scope is None else ', '.join(active_scope)}")
            continue

        if lowered == "clear":
            # Wipe every loaded PDF and reset the scope back to "all" (which is now empty anyway).
            library.clear_all()
            active_scope = None
            print("All PDFs removed from memory.")
            continue

        if lowered.startswith("remove "):
            # "remove <filename>" — drop one specific PDF from memory.
            target = user_input[len("remove "):].strip()
            if library.remove_pdf(target):
                print(f"Removed '{target}'.")
                # Keep the active scope consistent: don't keep referencing a file that's gone.
                if active_scope and target in active_scope:
                    active_scope.remove(target)
            else:
                print(f"'{target}' not found in loaded PDFs.")
            continue

        if lowered.startswith("only "):
            # "only <file1,file2>" — restrict subsequent questions to just these filenames.
            requested = [f.strip() for f in user_input[len("only "):].split(",") if f.strip()]
            unknown = [f for f in requested if not library.is_loaded(f)]
            if unknown:
                # Reject the whole scope change if any requested filename isn't actually loaded,
                # rather than silently applying a partial/incorrect scope.
                print(f"Unknown filename(s), not currently loaded: {', '.join(unknown)}")
            else:
                active_scope = requested
                print(f"Scope set to: {', '.join(active_scope)}")
            continue

        if lowered == "all":
            # Reset back to searching every loaded PDF.
            active_scope = None
            print("Scope reset to ALL loaded PDFs.")
            continue

        if library.count == 0:
            # Nothing loaded yet, and the input wasn't a recognized command — guide the user.
            print("No PDFs loaded yet. Use: load <path-to-pdf>")
            continue

        # Anything else is treated as a real question to answer using the RAG pipeline.
        try:
            context, answer = ask(user_input, library, llm, k=k, sources=active_scope)
            print_result(user_input, context, answer)
        except Exception as exc:
            print(f"Error while answering question: {exc}", file=sys.stderr)


def main() -> None:
    # ----- Define command-line arguments -----
    parser = argparse.ArgumentParser(description="Multi-PDF RAG Assistant")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",  # repeating --pdf multiple times accumulates into a list
        dest="pdfs",
        help=f"Path to a PDF file. Repeat up to {MAX_PDFS} times to load multiple. "
             "Optional — if omitted, start with an empty library and use 'load <path>' "
             "inside the interactive session instead.",
    )
    parser.add_argument("--question", help="Ask a single question and exit (non-interactive mode)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Number of chunks to retrieve (default: {DEFAULT_K})")
    args = parser.parse_args()

    pdf_paths = args.pdfs or []  # may legitimately be empty now (no --pdf flags given)

    # Validate: don't allow starting with more PDFs than the library will permit anyway.
    if len(pdf_paths) > MAX_PDFS:
        print(f"Too many PDFs: got {len(pdf_paths)}, max is {MAX_PDFS}.", file=sys.stderr)
        sys.exit(1)

    # Validate: every --pdf path given must actually exist on disk.
    missing = [p for p in pdf_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing PDF file: {p}", file=sys.stderr)
        sys.exit(1)

    if args.question:
        # Non-interactive mode: answer exactly one question and exit.
        # This mode has no REPL to type "load" into, so at least one --pdf is required upfront.
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

    # Default mode: drop into the interactive REPL (pdf_paths may be empty here,
    # in which case the session starts with zero PDFs and 'load' is used instead).
    run_interactive(args.model, pdf_paths, args.k)


if __name__ == "__main__":
    main()