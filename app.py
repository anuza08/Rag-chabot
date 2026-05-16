"""
Chat with your Documents — A RAG (Retrieval-Augmented Generation) demo.

How it works:
  1. You upload PDFs / TXTs
  2. The text is chunked
  3. Each chunk is embedded (turned into a vector) using a free local model
  4. Vectors are stored in Chroma (a local vector database)
  5. When you ask a question, we embed it, find the top-K most similar chunks,
     and pass those chunks + your question to an LLM to generate a grounded answer.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="Chat with your Documents",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chat with your Documents")
st.caption(
    "A minimal RAG (Retrieval-Augmented Generation) app. "
    "Upload documents, ask questions, get answers grounded in your data."
)

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Used only for the final generation step. Embeddings run locally.",
    )
    model_name = st.selectbox(
        "LLM model",
        ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"],
        index=0,
    )
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    st.markdown("---")
    st.markdown("**Embeddings:** `BAAI/bge-small-en-v1.5` (local, free)")
    st.markdown("**Vector DB:** Chroma (local, free)")
    st.markdown(
        "[View source on GitHub](https://github.com/) · "
        "[Deploy on HF Spaces](https://huggingface.co/spaces)"
    )

# ----------------------------- Helpers -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    """Cache the embedding model so we only load it once."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_file(uploaded_file):
    """Save the uploaded file to a temp path and load it with the right loader."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:  # .txt, .md
            loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = uploaded_file.name
        return docs
    finally:
        os.unlink(tmp_path)


def build_vectorstore(uploaded_files, chunk_size, chunk_overlap):
    """Load → chunk → embed → store in Chroma. Returns (vectorstore, n_chunks)."""
    all_docs = []
    for uf in uploaded_files:
        all_docs.extend(load_file(uf))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = get_embeddings()
    # in-memory Chroma — resets on app restart, which is fine for a demo
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


def make_qa_chain(vectorstore, api_key, model_name, top_k):
    """Build a RetrievalQA chain with a clear, source-aware prompt."""
    llm = ChatAnthropic(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant answering questions using ONLY the context below.\n"
            "If the answer isn't in the context, say you don't know — don't make things up.\n"
            "Be concise and cite the relevant facts from the context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


# ----------------------------- Session state -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "n_chunks" not in st.session_state:
    st.session_state.n_chunks = 0

# ----------------------------- Upload UI -----------------------------
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_files = st.file_uploader(
        "Upload PDFs, text, or markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
with col2:
    st.write("")
    st.write("")
    process_clicked = st.button(
        "📥 Process Documents",
        use_container_width=True,
        type="primary",
    )

if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Chunking and embedding..."):
            vs, n_chunks = build_vectorstore(uploaded_files, chunk_size, chunk_overlap)
            st.session_state.vectorstore = vs
            st.session_state.n_chunks = n_chunks
            st.session_state.messages = []
        st.success(
            f"✅ Processed {len(uploaded_files)} file(s) into {n_chunks} chunks."
        )

# ----------------------------- Chat UI -----------------------------
if st.session_state.vectorstore is None:
    st.info("👆 Upload documents and click **Process Documents** to start chatting.")
    st.stop()

st.markdown("---")
st.subheader("💬 Ask a question")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}] {src['source']}**")
                    st.text(src["text"][:600] + ("..." if len(src["text"]) > 600 else ""))

# Input
prompt = st.chat_input("Ask anything about your documents...")
if prompt:
    if not api_key:
        st.error("Please add your Anthropic API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            qa = make_qa_chain(
                st.session_state.vectorstore, api_key, model_name, top_k
            )
            result = qa.invoke({"query": prompt})
            answer = result["result"]
            sources = [
                {
                    "source": d.metadata.get("source", "unknown"),
                    "text": d.page_content,
                }
                for d in result["source_documents"]
            ]

        st.markdown(answer)
        with st.expander(f"📄 Sources ({len(sources)})"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**[{i}] {src['source']}**")
                st.text(src["text"][:600] + ("..." if len(src["text"]) > 600 else ""))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
