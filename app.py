import streamlit as st
from pathlib import Path
from langchain_ollama import ChatOllama

# Import the backend RAG logic
from lecture_rag import load_vector_store, ask, DEFAULT_MODEL, EMBEDDING_MODEL

# ----- Page config -----
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="🎓",
    layout="wide",
)

# ----- Session state init -----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# ----- Sidebar -----
st.sidebar.title("⚙️ RAG Configuration")
uploaded_file = st.sidebar.file_uploader("Upload any PDF", type=["pdf"])
model_name = st.sidebar.text_input("Ollama Model", value=DEFAULT_MODEL)
retrieve_k = st.sidebar.slider("Number of Chunks to Retrieve (k)", min_value=1, max_value=8, value=3)

st.sidebar.divider()
st.sidebar.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL}`")

if st.session_state.vector_store is not None:
    st.sidebar.success(f"Indexed: {st.session_state.last_uploaded}")
    if st.sidebar.button("🗑️ Clear and start over"):
        st.session_state.vector_store = None
        st.session_state.last_uploaded = None
        st.session_state.messages = []
        st.rerun()

# ----- Main title -----
st.title("🎓 PDF RAG Assistant")
st.write("Upload any PDF in the sidebar, then ask questions about its content below.")

# ----- Handle upload + indexing -----
if uploaded_file is not None:
    # Only re-index if this is a new/different file than the one already indexed
    is_new_file = st.session_state.last_uploaded != uploaded_file.name

    if is_new_file:
        temp_pdf_path = Path(f"./temp_{uploaded_file.name}")
        try:
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"📚 Reading '{uploaded_file.name}' and building index... this may take a moment."):
                st.session_state.vector_store = load_vector_store(temp_pdf_path)

            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.messages = []  # fresh chat history for the new document
            st.success(f"🎉 '{uploaded_file.name}' indexed successfully! Ask away below.")
        except Exception as e:
            st.error(f"Failed to load PDF: {e}")
            st.session_state.vector_store = None
            st.session_state.last_uploaded = None
        finally:
            # Always clean up the temp file, whether indexing succeeded or failed
            if temp_pdf_path.exists():
                temp_pdf_path.unlink()
else:
    if st.session_state.vector_store is None:
        st.info("👈 Please upload a PDF file in the sidebar to get started.")

# ----- Render chat history -----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("🔍 View Retrieved Context"):
                st.code(message["context"], language="text")

# ----- Chat input -----
if question := st.chat_input("Ask something about the uploaded PDF..."):

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    if st.session_state.vector_store is None:
        with st.chat_message("assistant"):
            error_msg = "Please upload a PDF file first before asking questions."
            st.warning(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatOllama(model=model_name, temperature=0.2)
                    context, answer = ask(
                        question=question,
                        vector_store=st.session_state.vector_store,
                        llm=llm,
                        k=retrieve_k,
                    )

                    st.markdown(answer)
                    with st.expander("🔍 View Retrieved Context"):
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