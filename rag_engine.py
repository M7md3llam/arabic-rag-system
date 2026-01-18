"""
RAG Engine - Retrieval Augmented Generation
"""
from openai import OpenAI
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    """Handle RAG queries and response generation"""
    
    def __init__(self, vector_store):
        """
        Initialize RAG engine
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # System prompt for RAG
        self.system_prompt = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً على المستندات المقدمة فقط.

قواعد مهمة:
1. أجب فقط بناءً على المعلومات الموجودة في المستندات المقدمة
2. إذا لم تجد إجابة في المستندات، قل "لا أجد معلومات كافية في المستندات المتاحة للإجابة على هذا السؤال"
3. اذكر دائماً مصدر المعلومات (اسم المستند ورقم الصفحة)
4. كن دقيقاً ومختصراً
5. إذا كانت الإجابة غير واضحة، اذكر ذلك

You are a smart assistant specialized in answering questions based only on provided documents.

Important rules:
1. Answer only based on information in the provided documents
2. If no answer is found in documents, say "I don't find sufficient information in the available documents to answer this question"
3. Always cite the source (document name and page number)
4. Be accurate and concise
5. If the answer is unclear, mention that
"""
    
    def generate_response(self, 
                         query: str, 
                         context_docs: List[str],
                         metadata: List[Dict],
                         model: str = "gpt-4o-mini") -> Dict:
        """
        Generate response using RAG
        
        Args:
            query: User question
            context_docs: Retrieved document chunks
            metadata: Metadata for each chunk
            model: OpenAI model to use
            
        Returns:
            Dict with keys: answer, sources, model_used
        """
        try:
            # Build context from retrieved documents
            context = self._build_context(context_docs, metadata)
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""السياق من المستندات:
{context}

السؤال: {query}

الرجاء الإجابة بناءً على السياق أعلاه فقط، مع ذكر المصادر.

Context from documents:
{context}

Question: {query}

Please answer based only on the context above, citing sources.
"""}
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Extract sources
            sources = self._extract_sources(metadata)
            
            return {
                'answer': answer,
                'sources': sources,
                'model_used': model,
                'success': True
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': [],
                'model_used': model,
                'success': False,
                'error': str(e)
            }
    
    def query(self, 
             question: str, 
             n_results: int = 5,
             model: str = "gpt-4o-mini") -> Dict:
        """
        Complete RAG query: retrieve + generate
        
        Args:
            question: User question
            n_results: Number of documents to retrieve
            model: OpenAI model to use
            
        Returns:
            Dict with answer and sources
        """
        try:
            # Step 1: Retrieve relevant documents
            search_results = self.vector_store.search(
                query=question,
                n_results=n_results
            )
            
            documents = search_results['documents']
            metadata = search_results['metadatas']
            
            # Check if any documents found
            if not documents:
                return {
                    'answer': "لا توجد مستندات ذات صلة للإجابة على هذا السؤال.\n\nNo relevant documents found to answer this question.",
                    'sources': [],
                    'success': False
                }
            
            # Step 2: Check if visualization is needed
            viz_keywords = ['table', 'chart', 'graph', 'compare', 'comparison','بياني','رسم','جدول', 'مخطط', 'مقارنة', 'قارن']
            needs_visualization = any(keyword in question.lower() for keyword in viz_keywords)
            
            # Step 3: Generate response
            result = self.generate_response(
                query=question,
                context_docs=documents,
                metadata=metadata,
                model=model
            )
            
            # Step 4: Add visualization flag
            result['needs_visualization'] = needs_visualization
            result['documents'] = documents
            result['metadatas'] = metadata
            
            return result
            
        except Exception as e:
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'success': False,
                'error': str(e)
            }
    
    def _build_context(self, documents: List[str], metadata: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        
        for idx, (doc, meta) in enumerate(zip(documents, metadata), 1):
            doc_name = meta.get('document_name', 'Unknown')
            page = meta.get('page', 'Unknown')
            
            context_parts.append(f"""[مستند {idx} | Document {idx}]
المصدر | Source: {doc_name} (صفحة | Page {page})
النص | Text:
{doc}
---
""")
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, metadata: List[Dict]) -> List[str]:
        """Extract unique sources from metadata"""
        sources = []
        seen = set()
        
        for meta in metadata:
            doc_name = meta.get('document_name', 'Unknown')
            page = meta.get('page', 'Unknown')
            source = f"{doc_name} - Page {page}"
            
            if source not in seen:
                sources.append(source)
                seen.add(source)
        
        return sources
    
    def summarize_document(self, document_name: str) -> Dict:
        """
        Generate summary of a specific document
        
        Args:
            document_name: Name of document to summarize
            
        Returns:
            Dict with summary
        """
        try:
            # Retrieve all chunks from this document
            results = self.vector_store.collection.get(
                where={"document_name": document_name}
            )
            
            if not results['documents']:
                return {
                    'summary': f"No content found for {document_name}",
                    'success': False
                }
            
            # Combine all chunks
            full_text = "\n\n".join(results['documents'])
            
            # Generate summary
            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                {"role": "user", "content": f"Please provide a concise summary of the following document in both Arabic and English:\n\n{full_text[:4000]}"}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5,
                max_tokens=500
            )
            
            return {
                'summary': response.choices[0].message.content,
                'document_name': document_name,
                'success': True
            }
            
        except Exception as e:
            return {
                'summary': f"Error generating summary: {str(e)}",
                'success': False,
                'error': str(e)
            }


# Test function
if __name__ == "__main__":
    from vector_store import VectorStore
    
    # Initialize
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store)
    
    # Test query
    result = rag_engine.query("ما هي المعلومات الموجودة في المستندات؟")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")