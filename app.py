from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag_pipeline import RAGPipeline
import os

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title(" Study Guider")
st.markdown("Upload your documents and ask questions about them!")

# Initialize pipeline in session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar — document upload
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button(" Process Documents", use_container_width=True):
        with st.spinner("Processing documents..."):
            # Save uploaded files temporarily
            temp_paths = []
            os.makedirs("temp_docs", exist_ok=True)
            for file in uploaded_files:
                path = f"temp_docs/{file.name}"
                with open(path, "wb") as f:
                    f.write(file.read())
                temp_paths.append(path)

            # Initialize and build RAG pipeline
            st.session_state.rag = RAGPipeline()
            st.session_state.rag.load_documents(temp_paths)
            st.session_state.documents_loaded = True
            st.session_state.chat_history = []

        st.success(f" Loaded {len(uploaded_files)} document(s)!")

    if st.session_state.documents_loaded:
        st.divider()
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            st.session_state.rag = None
            st.session_state.chat_history = []
            st.session_state.documents_loaded = False
            st.rerun()

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. Upload your PDF documents
    2. Documents are chunked & embedded
    3. Ask questions in the chat
    4. Relevant chunks are retrieved
    5. It generates a grounded answer
    """)

# Main chat area
if not st.session_state.documents_loaded:
    st.info(" Upload documents in the sidebar to get started.")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📎 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['source']} (Page {source.get('page', 'N/A')})")
                        st.caption(source["content"][:300] + "...")

    # Chat input
    if prompt := st.chat_input("Ask something about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag.query(
                    prompt,
                    chat_history=st.session_state.chat_history[:-1]
                )
                response = result["answer"]
                sources = result["sources"]

            st.markdown(response)
            if sources:
                with st.expander("📎 Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** {source['source']} (Page {source.get('page', 'N/A')})")
                        st.caption(source["content"][:300] + "...")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
