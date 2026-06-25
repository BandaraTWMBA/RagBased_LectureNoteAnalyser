import streamlit as st
from pathlib import Path
from langchain_ollama import ChatOllama

# Import the backend RAG logic
from lecture_rag import PDFLibrary, PDFLimitExceeded, ask, DEFAULT_MODEL, EMBEDDING_MODEL, MAX_PDFS

# ----- Page config -----
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="🎓",
    layout="wide",
)

# ----- Session state init -----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "library" not in st.session_state:
    # Replaces the old single "vector_store" with a PDFLibrary that can
    # hold up to MAX_PDFS PDFs in memory at once, each tagged by filename.
    st.session_state.library = PDFLibrary()
if "processed_uploads" not in st.session_state:
    # Tracks filenames already indexed this session so Streamlit's automatic
    # re-running of the script doesn't re-index the same file repeatedly.
    st.session_state.processed_uploads = set()

library: PDFLibrary = st.session_state.library

# ----- Sidebar -----
st.sidebar.title("⚙️ RAG Configuration")
st.sidebar.caption(f"Up to {MAX_PDFS} PDFs can be loaded at once. They stay in memory until you remove them.")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

model_name = st.sidebar.text_input("Ollama Model", value=DEFAULT_MODEL)
retrieve_k = st.sidebar.slider("Number of Chunks to Retrieve (k)", min_value=1, max_value=8, value=3)

st.sidebar.divider()
st.sidebar.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL}`")

# ----- Handle upload + indexing -----
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_uploads:
            continue  # already indexed this session, skip silently

        if library.count >= MAX_PDFS:
            st.sidebar.error(
                f"Skipped '{uploaded_file.name}': library is full "
                f"({library.count}/{MAX_PDFS}). Remove a PDF below first."
            )
            continue

        temp_pdf_path = Path(f"./temp_{uploaded_file.name}")
        try:
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"📚 Reading '{uploaded_file.name}' and building index... this may take a moment."):
                library.add_pdf(temp_pdf_path)

            st.session_state.processed_uploads.add(uploaded_file.name)
            st.sidebar.success(f"🎉 '{uploaded_file.name}' indexed successfully!")
        except PDFLimitExceeded as e:
            st.sidebar.error(str(e))
        except Exception as e:
            st.sidebar.error(f"Failed to load '{uploaded_file.name}': {e}")
        finally:
            # Always clean up the temp file, whether indexing succeeded or failed
            if temp_pdf_path.exists():
                temp_pdf_path.unlink()

# ----- Sidebar: loaded PDF list with per-file delete -----
st.sidebar.divider()
st.sidebar.markdown(f"**Loaded PDFs ({library.count}/{MAX_PDFS}):**")

if library.count == 0:
    st.sidebar.caption("None yet — upload a PDF above.")
else:
    for filename in library.loaded_filenames:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(f"📄 {filename}")
        if col2.button("🗑️", key=f"remove_{filename}", help=f"Remove {filename}"):
            library.remove_pdf(filename)
            st.session_state.processed_uploads.discard(filename)
            st.rerun()

    if st.sidebar.button("🗑️ Clear all PDFs"):
        library.clear_all()
        st.session_state.processed_uploads.clear()
        st.session_state.messages = []
        st.rerun()

# ----- Main title -----
st.title("🎓 PDF RAG Assistant")
st.write("Upload one or more PDFs in the sidebar, then ask questions about their content below.")

if library.count == 0:
    st.info("👈 Please upload a PDF file in the sidebar to get started.")
    selected_sources = []
else:
    st.markdown("**Which PDFs should this question consider?**")
    selected_sources = st.multiselect(
        "Leave empty to search ALL loaded PDFs, or pick specific ones to narrow the search",
        options=library.loaded_filenames,
        default=[],
        label_visibility="collapsed",
    )

# ----- Render chat history -----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("🔍 View Retrieved Context"):
                st.code(message["context"], language="text")

# ----- Chat input -----
if question := st.chat_input("Ask something about the uploaded PDF(s)..."):

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    if library.count == 0:
        with st.chat_message("assistant"):
            error_msg = "Please upload a PDF file first before asking questions."
            st.warning(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        # None = search across all loaded PDFs; a list = restrict to those filenames only
        sources = selected_sources if selected_sources else None

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatOllama(model=model_name, temperature=0.2)
                    context, answer = ask(
                        question=question,
                        library=library,
                        llm=llm,
                        k=retrieve_k,
                        sources=sources,
                    )

                    st.markdown(answer)
                    with st.expander("🔍 View Retrieved Context"):
                        st.caption(f"Scope: {'All PDFs' if sources is None else ', '.join(sources)}")
                        st.code(context, language="text")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context,
                    })
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})