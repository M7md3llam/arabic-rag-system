import hashlib
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from data_visualizer import DataVisualizer
from document_processor import DocumentProcessor
from ocr_processor import OCRProcessor
from rag_engine import RAGEngine
from vector_store import VectorStore


DATA_DIR = Path("data")
UPLOAD_DIR = Path("uploads")
VECTOR_DB_DIR = Path("vectordb")

SUPPORTED_EXTENSIONS = ["pdf", "docx", "xlsx", "png", "jpg", "jpeg"]
STATUS_LABELS = {
    "uploaded": "Uploaded",
    "processing": "Processing",
    "indexed": "Indexed",
    "failed_parsing": "Parsing failed",
    "failed_indexing": "Indexing failed",
    "error": "Error",
}


st.set_page_config(
    page_title="Arabic RAG System",
    page_icon="📚",
    layout="wide",
)


def ensure_directories() -> None:
    for directory in [DATA_DIR, UPLOAD_DIR, VECTOR_DB_DIR]:
        directory.mkdir(exist_ok=True)


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    return VectorStore(persist_directory=str(VECTOR_DB_DIR))


@st.cache_resource(show_spinner=False)
def get_ocr_processor() -> OCRProcessor:
    return OCRProcessor()


@st.cache_resource(show_spinner=False)
def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor(ocr_processor=get_ocr_processor())


@st.cache_resource(show_spinner=False)
def get_rag_engine() -> RAGEngine:
    return RAGEngine(get_vector_store())


@st.cache_resource(show_spinner=False)
def get_visualizer() -> DataVisualizer:
    return DataVisualizer()


def initialize_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("documents", [])


def initialize_services() -> bool:
    if not has_openai_key():
        st.error("OPENAI_API_KEY is missing. Add it to a .env file before processing or querying documents.")
        return False

    try:
        st.session_state.vector_store = get_vector_store()
        st.session_state.ocr_processor = get_ocr_processor()
        st.session_state.doc_processor = get_document_processor()
        st.session_state.rag_engine = get_rag_engine()
        st.session_state.visualizer = get_visualizer()
        return True
    except Exception as exc:
        st.error(f"Could not initialize the app services: {exc}")
        return False


def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def save_uploaded_file(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        file_hash = get_file_hash(file_bytes)

        for doc in st.session_state.documents:
            if doc["hash"] == file_hash:
                return None, "Duplicate file detected"

        file_path = DATA_DIR / uploaded_file.name
        file_path.write_bytes(file_bytes)

        return {
            "name": uploaded_file.name,
            "path": str(file_path),
            "hash": file_hash,
            "size": len(file_bytes),
            "uploaded_at": datetime.now().isoformat(),
            "status": "uploaded",
            "type": uploaded_file.type,
        }, None
    except Exception as exc:
        return None, str(exc)


def build_chunk_metadata(doc_info: dict, chunks: list[str]) -> tuple[list[dict], list[str]]:
    metadatas = []
    ids = []

    for idx, _chunk in enumerate(chunks):
        metadatas.append(
            {
                "document_name": doc_info["name"],
                "chunk_id": idx,
                "page": idx // 3 + 1,
                "uploaded_at": doc_info["uploaded_at"],
            }
        )
        ids.append(f"{doc_info['hash']}_{idx}")

    return metadatas, ids


def process_document(doc_info: dict) -> bool:
    try:
        doc_info["status"] = "processing"

        result = st.session_state.doc_processor.process_file(
            doc_info["path"],
            doc_info["type"],
        )

        if result["status"] != "success":
            doc_info["status"] = "failed_parsing"
            doc_info["error"] = result.get("error", "Unknown parsing error")
            return False

        chunks = st.session_state.doc_processor.chunk_text(result["text"])
        if not chunks:
            doc_info["status"] = "failed_parsing"
            doc_info["error"] = "No text extracted"
            return False

        metadatas, ids = build_chunk_metadata(doc_info, chunks)
        success = st.session_state.vector_store.add_documents(chunks, metadatas, ids)

        if not success:
            doc_info["status"] = "failed_indexing"
            doc_info["error"] = "Failed to add document chunks to the vector store"
            return False

        doc_info["status"] = "indexed"
        doc_info["num_chunks"] = len(chunks)
        doc_info["metadata"] = result.get("metadata", {})
        doc_info.pop("error", None)
        return True
    except Exception as exc:
        doc_info["status"] = "error"
        doc_info["error"] = str(exc)
        return False


def render_sources(sources: list[str]) -> None:
    if sources:
        with st.expander("Sources"):
            for source in sources:
                st.write(f"- {source}")


def render_visualization(prompt: str, result: dict) -> None:
    if not result.get("needs_visualization") or not result.get("documents"):
        return

    st.divider()
    st.write("Generating visualization...")

    context = "\n\n".join(result["documents"][:3])
    viz_data = st.session_state.visualizer.extract_structured_data(prompt, context)
    if not viz_data:
        st.info("No structured data found for visualization.")
        return

    if viz_data.get("type") == "table":
        df = st.session_state.visualizer.create_table(
            viz_data.get("data", []),
            viz_data.get("title", "Data Table"),
        )
        if df.empty:
            st.warning("Could not create a table from the retrieved context.")
            return
        st.subheader(viz_data.get("title", "Data Table"))
        st.dataframe(df, use_container_width=True)
        return

    if viz_data.get("type") == "chart":
        chart_img = st.session_state.visualizer.create_chart(
            viz_data.get("data", []),
            chart_type=viz_data.get("chart_type", "bar"),
            title=viz_data.get("title", "Chart"),
        )
        if chart_img:
            st.subheader(viz_data.get("title", "Chart"))
            st.image(f"data:image/png;base64,{chart_img}")
        else:
            st.warning("Could not generate a chart from the retrieved context.")


def render_sidebar(services_ready: bool) -> None:
    with st.sidebar:
        st.title("Document Manager")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help="PDF, Word, Excel, PNG, JPG, or JPEG",
            disabled=not services_ready,
        )

        if uploaded_files and services_ready:
            for uploaded_file in uploaded_files:
                existing_names = [doc["name"] for doc in st.session_state.documents]
                if uploaded_file.name in existing_names:
                    continue

                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    doc_info, error = save_uploaded_file(uploaded_file)
                    if doc_info:
                        st.session_state.documents.append(doc_info)
                        st.success(f"{uploaded_file.name} uploaded")
                    elif error:
                        st.warning(error)

        st.divider()
        st.subheader("Uploaded documents")

        if not st.session_state.documents:
            st.info("No documents uploaded yet.")
        else:
            for idx, doc in enumerate(st.session_state.documents):
                label = STATUS_LABELS.get(doc["status"], doc["status"])
                with st.expander(f"{doc['name']} - {label}", expanded=False):
                    st.write(f"Status: {label}")
                    st.write(f"Size: {doc['size'] / 1024:.2f} KB")
                    if "num_chunks" in doc:
                        st.write(f"Chunks: {doc['num_chunks']}")
                    if doc.get("metadata", {}).get("method") == "ocr":
                        st.success("OCR was used for this document.")
                    if doc.get("error"):
                        st.error(doc["error"])

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Process", key=f"process_{idx}", disabled=not services_ready):
                            with st.spinner("Processing..."):
                                process_document(doc)
                            st.rerun()
                    with col2:
                        if st.button("Delete", key=f"delete_{idx}", disabled=not services_ready):
                            st.session_state.vector_store.delete_document(doc["name"])
                            file_path = Path(doc["path"])
                            if file_path.exists():
                                file_path.unlink()
                            st.session_state.documents.pop(idx)
                            st.rerun()

        st.divider()
        if st.session_state.documents:
            if st.button("Process all documents", type="primary", use_container_width=True, disabled=not services_ready):
                progress_bar = st.progress(0)
                for idx, doc in enumerate(st.session_state.documents):
                    if doc["status"] in ["uploaded", "failed_parsing", "failed_indexing", "error"]:
                        with st.spinner(f"Processing {doc['name']}..."):
                            process_document(doc)
                    progress_bar.progress((idx + 1) / len(st.session_state.documents))
                st.success("Processing complete.")
                st.rerun()

        if services_ready:
            stats = st.session_state.vector_store.get_collection_stats()
            st.divider()
            st.metric("Total chunks", stats.get("total_chunks", 0))


def render_chat(services_ready: bool) -> None:
    col1, col2, col3 = st.columns([2, 1, 1])
    indexed_count = len([doc for doc in st.session_state.documents if doc["status"] == "indexed"])

    with col1:
        st.title("Arabic RAG Assistant")
        st.caption("Ask questions, request tables and charts, and search your documents.")
    with col2:
        st.metric("Documents", f"{indexed_count}/{len(st.session_state.documents)}")
    with col3:
        total_chunks = 0
        if services_ready:
            total_chunks = st.session_state.vector_store.get_collection_stats().get("total_chunks", 0)
        st.metric("Chunks", total_chunks)

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    prompt = st.chat_input(
        "Ask about your documents...",
        disabled=not services_ready,
    )
    if not prompt:
        return

    indexed_docs = [doc for doc in st.session_state.documents if doc["status"] == "indexed"]
    if not st.session_state.documents:
        st.error("Please upload documents first.")
        return
    if not indexed_docs:
        st.error("Please process at least one document first.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag_engine.query(prompt)

        response = result.get("answer", "No answer was generated.")
        sources = result.get("sources", [])

        st.markdown(response)
        render_visualization(prompt, result)
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": sources,
        }
    )


def render_examples() -> None:
    st.divider()
    with st.expander("Example questions"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Basic questions**
                - What is this document about?
                - Summarize the main points.
                - ما هو محتوى هذا المستند؟

                **Search**
                - Find information about a topic.
                - Which document mentions this keyword?
                """
            )
        with col2:
            st.markdown(
                """
                **Tables and charts**
                - Create a comparison table.
                - Show me a bar chart of the data.
                - أنشئ جدول مقارنة.

                **Analysis**
                - Compare document A and document B.
                - What are the key differences?
                """
            )


def main() -> None:
    ensure_directories()
    initialize_session_state()
    services_ready = initialize_services()
    render_sidebar(services_ready)
    render_chat(services_ready)
    render_examples()


if __name__ == "__main__":
    main()
