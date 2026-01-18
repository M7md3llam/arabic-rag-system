import streamlit as st
import os
from pathlib import Path
import hashlib
from datetime import datetime
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine
from data_visualizer import DataVisualizer
from ocr_processor import OCRProcessor

# Page config
st.set_page_config(
    page_title="Arabic RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'ocr_processor' not in st.session_state:
    st.session_state.ocr_processor = OCRProcessor()
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor(ocr_processor=st.session_state.ocr_processor)
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = DataVisualizer()

# Create necessary directories
DATA_DIR = Path("data")
UPLOAD_DIR = Path("uploads")
VECTOR_DB_DIR = Path("vectordb")

for directory in [DATA_DIR, UPLOAD_DIR, VECTOR_DB_DIR]:
    directory.mkdir(exist_ok=True)

def get_file_hash(file_bytes):
    """Generate SHA256 hash for file"""
    return hashlib.sha256(file_bytes).hexdigest()

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return file info"""
    try:
        file_bytes = uploaded_file.read()
        file_hash = get_file_hash(file_bytes)
        
        # Check for duplicates
        for doc in st.session_state.documents:
            if doc['hash'] == file_hash:
                return None, "Duplicate file detected"
        
        # Save file
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        
        # Store document info
        doc_info = {
            'name': uploaded_file.name,
            'path': str(file_path),
            'hash': file_hash,
            'size': len(file_bytes),
            'uploaded_at': datetime.now().isoformat(),
            'status': 'uploaded',
            'type': uploaded_file.type
        }
        
        return doc_info, None
    except Exception as e:
        return None, str(e)

def process_document(doc_info):
    """Process document and add to vector store"""
    try:
        # Update status
        doc_info['status'] = 'processing'
        
        # Extract text
        result = st.session_state.doc_processor.process_file(
            doc_info['path'],
            doc_info['type']
        )
        
        if result['status'] != 'success':
            doc_info['status'] = 'failed_parsing'
            doc_info['error'] = result.get('error', 'Unknown error')
            return False
        
        # Chunk text
        chunks = st.session_state.doc_processor.chunk_text(result['text'])
        
        if not chunks:
            doc_info['status'] = 'failed_parsing'
            doc_info['error'] = 'No text extracted'
            return False
        
        # Create metadata for each chunk
        metadatas = []
        ids = []
        for idx, chunk in enumerate(chunks):
            metadata = {
                'document_name': doc_info['name'],
                'chunk_id': idx,
                'page': idx // 3 + 1,  # Approximate page number
                'uploaded_at': doc_info['uploaded_at']
            }
            metadatas.append(metadata)
            ids.append(f"{doc_info['hash']}_{idx}")
        
        # Add to vector store
        success = st.session_state.vector_store.add_documents(
            chunks=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        if success:
            doc_info['status'] = 'indexed'
            doc_info['num_chunks'] = len(chunks)
            return True
        else:
            doc_info['status'] = 'failed_indexing'
            doc_info['error'] = 'Failed to add to vector store'
            return False
            
    except Exception as e:
        doc_info['status'] = 'error'
        doc_info['error'] = str(e)
        return False

# Sidebar - File Upload
with st.sidebar:
    st.title("📚 Document Manager")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'docx', 'xlsx', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Drag and drop files here"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [doc['name'] for doc in st.session_state.documents]:
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    doc_info, error = save_uploaded_file(uploaded_file)
                    if doc_info:
                        st.session_state.documents.append(doc_info)
                        st.success(f"✅ {uploaded_file.name}")
                    elif error:
                        st.warning(f"⚠️ {error}")
    
    # Display uploaded documents
    st.divider()
    st.subheader("Uploaded Documents")
    
    if st.session_state.documents:
        for idx, doc in enumerate(st.session_state.documents):
            status_emoji = {
                'uploaded': '📤',
                'processing': '⚙️',
                'indexed': '✅',
                'failed_parsing': '❌',
                'failed_indexing': '❌',
                'error': '⚠️'
            }.get(doc['status'], '📄')
            
            with st.expander(f"{status_emoji} {doc['name']}", expanded=False):
                st.write(f"**Status:** {doc['status']}")
                st.write(f"**Size:** {doc['size'] / 1024:.2f} KB")
                if 'num_chunks' in doc:
                    st.write(f"**Chunks:** {doc['num_chunks']}")
                if doc.get('metadata', {}).get('method') == 'ocr':
                    st.success("🔍 OCR Used (Scanned Document)")
                if 'error' in doc:
                    st.error(f"Error: {doc['error']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Process", key=f"process_{idx}"):
                        with st.spinner("Processing..."):
                            success = process_document(doc)
                            if success:
                                st.success("Processed!")
                                st.rerun()
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
                        # Delete from vector store
                        st.session_state.vector_store.delete_document(doc['name'])
                        # Delete file
                        if os.path.exists(doc['path']):
                            os.remove(doc['path'])
                        st.session_state.documents.pop(idx)
                        st.rerun()
    else:
        st.info("No documents uploaded yet")
    
    # Process All button
    st.divider()
    if st.session_state.documents:
        if st.button("⚡ Process All Documents", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            for idx, doc in enumerate(st.session_state.documents):
                if doc['status'] in ['uploaded', 'failed_parsing', 'failed_indexing']:
                    with st.spinner(f"Processing {doc['name']}..."):
                        process_document(doc)
                progress_bar.progress((idx + 1) / len(st.session_state.documents))
            
            st.session_state.vector_store_ready = True
            st.success("All documents processed!")
            st.rerun()
    
    # Stats
    stats = st.session_state.vector_store.get_collection_stats()
    st.divider()
    st.metric("Total Chunks", stats.get('total_chunks', 0))

# Main area - Header with stats
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title("🤖 Arabic RAG Assistant")
    st.caption("Ask questions, request tables & charts, search your documents")
with col2:
    indexed_count = len([d for d in st.session_state.documents if d['status'] == 'indexed'])
    st.metric("📚 Documents", f"{indexed_count}/{len(st.session_state.documents)}")
with col3:
    stats = st.session_state.vector_store.get_collection_stats()
    st.metric("💾 Chunks", stats.get('total_chunks', 0))

st.divider()

# Chat interface
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📑 Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# Chat input
if prompt := st.chat_input("💬 Ask anything: questions, tables, charts, comparisons, search..."):
    if not st.session_state.documents:
        st.error("⚠️ Please upload documents first!")
    else:
        # Check if any documents are indexed
        indexed_docs = [d for d in st.session_state.documents if d['status'] == 'indexed']
        if not indexed_docs:
            st.error("⚠️ Please process documents first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_engine.query(prompt)
                    
                    response = result['answer']
                    sources = result.get('sources', [])
                    
                    st.markdown(response)
                    
                    # Check if visualization is needed
                    if result.get('needs_visualization') and result.get('documents'):
                        st.write("---")
                        st.write("📊 **Generating visualization...**")
                        
                        # Try to create visualization
                        context = "\n\n".join(result['documents'][:3])  # Use top 3 docs
                        viz_data = st.session_state.visualizer.extract_structured_data(prompt, context)
                        
                        if viz_data:
                            st.write(f"✅ Found visualization data: {viz_data.get('type', 'unknown')}")
                            
                            if viz_data.get('type') == 'table':
                                df = st.session_state.visualizer.create_table(
                                    viz_data.get('data', []),
                                    viz_data.get('title', 'Data Table')
                                )
                                if not df.empty:
                                    st.subheader(viz_data.get('title', 'Data Table'))
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    st.warning("Could not create table")
                            
                            elif viz_data.get('type') == 'chart':
                                st.write(f"Creating {viz_data.get('chart_type', 'bar')} chart...")
                                st.write(f"Data points: {len(viz_data.get('data', []))}")
                                
                                chart_img = st.session_state.visualizer.create_chart(
                                    viz_data.get('data', []),
                                    chart_type=viz_data.get('chart_type', 'bar'),
                                    title=viz_data.get('title', 'Chart')
                                )
                                if chart_img:
                                    st.subheader(viz_data.get('title', 'Chart'))
                                    st.image(f"data:image/png;base64,{chart_img}")
                                else:
                                    st.error("Failed to generate chart. Check terminal for errors.")
                        else:
                            st.info("No structured data found for visualization")
                    
                    if sources:
                        with st.expander("📑 Sources"):
                            for source in sources:
                                st.write(f"- {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

# Footer
st.divider()

# Example prompts
with st.expander("💡 Example Questions You Can Ask"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📝 Basic Questions:**
        - What is this document about?
        - Summarize the main points
        - ما هو محتوى هذا المستند؟
        
        **🔍 Search & Find:**
        - Find information about [topic]
        - What does the document say about [X]?
        - Which document mentions [keyword]?
        """)
    
    with col2:
        st.markdown("""
        **📊 Tables & Charts:**
        - Create a comparison table
        - Show me a bar chart of the data
        - Compare the documents in a table
        - أنشئ جدول مقارنة
        
        **📈 Analysis:**
        - Compare document A and B
        - What are the key differences?
        - Analyze the trends
        """)

st.caption("🚀 Built with Streamlit + OpenAI + ChromaDB")