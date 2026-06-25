# 🎓 PDF RAG Assistant

> **Learning project** — built to understand how local RAG pipelines work end-to-end (LLMs, embeddings, vector search, chunking). Not production-ready; treat it as a reference, not a deployable app.

A local Retrieval-Augmented Generation (RAG) tool: upload PDFs, ask questions, get answers grounded only in the document content — no cloud APIs involved.

## What it does

- Loads up to **5 PDFs at once** into memory, tagged by filename and page number
- Splits each PDF into overlapping text chunks and embeds them locally
- Answers questions by retrieving the most relevant chunks and feeding them to a local LLM
- Lets you scope a question to **all loaded PDFs** or just **specific ones**
- PDFs stay loaded until removed manually — nothing auto-evicts
- Works both as a **Streamlit web UI** and a **terminal REPL**

## Stack

| Component | Tool |
|---|---|
| LLM | Ollama running `llama3.2:1b` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Orchestration | LangChain |
| Vector store | `InMemoryVectorStore` (shared across all loaded PDFs) |
| Chunking | `RecursiveCharacterTextSplitter` (600 chars, 100 overlap) |
| PDF parsing | `PyPDFLoader` |
| Frontend | Streamlit |

## How it works

```
PDF(s) → extract text → chunk + tag (source, page) → embed
       → shared vector store → similarity search (optionally filtered by source)
       → context + question → local LLM → answer
```

## Run it

```bash
ollama serve
ollama pull llama3.2:1b
pip install -r requirements.txt
```

**Web UI:**
```bash
streamlit run app.py
```

**Terminal (no GUI):**
```bash
python lecture_rag.py
> load slides.pdf
> What is backpropagation?
> only slides.pdf
> exit
```

REPL commands: `load <path>`, `list`, `remove <file>`, `clear`, `only <file1,file2>`, `all`, `exit`.

## Files

```
app.py            # Streamlit frontend
lecture_rag.py    # Backend: PDFLibrary class, ask(), REPL
requirements.txt
```

## Privacy

Everything runs locally — no data leaves your machine. Uploaded files are deleted from disk right after indexing; only in-memory embeddings persist for the session.

## Known limitations

- No persistence — restarting clears everything (in-memory store only)
- No conversation memory — each question is independent
- 5-PDF cap is hardcoded
- `remove_pdf()` reaches into `InMemoryVectorStore`'s internals (no public delete-by-filter API exists yet)