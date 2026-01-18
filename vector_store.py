"""
Vector store manager using ChromaDB and OpenAI embeddings
"""
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorStore:
    """Manage vector embeddings and similarity search"""
    
    def __init__(self, persist_directory: str = "vectordb"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        # Use PersistentClient instead of Client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Arabic RAG document collection"}
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI
        
        Args:
            text: Input text
            
        Returns:
            List of embedding values
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def add_documents(self, 
                     chunks: List[str], 
                     metadatas: List[Dict],
                     ids: List[str]) -> bool:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of text chunks
            metadatas: List of metadata dicts for each chunk
            ids: List of unique IDs for each chunk
            
        Returns:
            True if successful
        """
        try:
            # Generate embeddings for all chunks
            embeddings = [self.get_embedding(chunk) for chunk in chunks]
            
            # Filter out failed embeddings
            valid_data = [
                (chunk, emb, meta, id_) 
                for chunk, emb, meta, id_ in zip(chunks, embeddings, metadatas, ids)
                if emb
            ]
            
            if not valid_data:
                return False
            
            chunks, embeddings, metadatas, ids = zip(*valid_data)
            
            # Add to collection
            self.collection.add(
                documents=list(chunks),
                embeddings=list(embeddings),
                metadatas=list(metadatas),
                ids=list(ids)
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def search(self, 
              query: str, 
              n_results: int = 5,
              filter_dict: Optional[Dict] = None) -> Dict:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            Dict with keys: documents, metadatas, distances
        """
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                return {
                    'documents': [],
                    'metadatas': [],
                    'distances': []
                }
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict
            )
            
            return {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else []
            }
            
        except Exception as e:
            print(f"Error searching: {e}")
            return {
                'documents': [],
                'metadatas': [],
                'distances': []
            }
    
    def delete_document(self, document_name: str) -> bool:
        """
        Delete all chunks from a specific document
        
        Args:
            document_name: Name of document to delete
            
        Returns:
            True if successful
        """
        try:
            # Get all IDs for this document
            results = self.collection.get(
                where={"document_name": document_name}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection
        
        Returns:
            Dict with collection stats
        """
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection.name
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'total_chunks': 0}
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            self.client.delete_collection(name="documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"description": "Arabic RAG document collection"}
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False


# Test function
if __name__ == "__main__":
    # Test vector store
    store = VectorStore()
    
    # Test adding documents
    test_chunks = [
        "هذا نص تجريبي باللغة العربية",
        "This is a test document in English"
    ]
    
    test_metadata = [
        {"document_name": "test.pdf", "page": 1, "chunk_id": 0},
        {"document_name": "test.pdf", "page": 1, "chunk_id": 1}
    ]
    
    test_ids = ["test_0", "test_1"]
    
    success = store.add_documents(test_chunks, test_metadata, test_ids)
    print(f"Add documents: {'Success' if success else 'Failed'}")
    
    # Test search
    results = store.search("نص عربي")
    print(f"Search results: {len(results['documents'])} documents found")
    
    # Get stats
    stats = store.get_collection_stats()
    print(f"Collection stats: {stats}")