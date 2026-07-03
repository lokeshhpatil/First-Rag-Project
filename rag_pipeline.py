from src.chunker import smart_chunk_pdf
from src.vector_store import vector_store
from src.llm_integration import FreeLLM
from src.config import Config, EMBEDDING_MODEL_NAME, TOP_K_RESULTS
import logging
from sentence_transformers import SentenceTransformer
from src.vector_store import get_pinecone_index
from typing import List, Dict
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG pipeline with SmolLM"""
    
    def __init__(self, config=Config()):
        self.config = config
        self.vector_store = vector_store(config)
        self.llm = FreeLLM(config)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = get_pinecone_index()
        
    def ask_with_filters(self, question: str, filters: dict = None, top_k: int = 4):
        """
        Query with metadata filtering
        """
        try:
            # Convert question to embedding
            query_embedding = self.model.encode(question).tolist()
            
            # Build Pinecone filter
            pinecone_filter = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        pinecone_filter[key] = {"$in": value}
                    else:
                        pinecone_filter[key] = {"$eq": value}
            
            logger.info(f"Querying with filters: {pinecone_filter}")
            
            # Query with filters
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else {}
            )
            
            # Convert QueryResponse to dictionary safely
            results_dict = results.to_dict() if hasattr(results, 'to_dict') else results
            
            # Extract matches safely
            matches = results_dict.get('matches', [])
            
            if not matches:
                return {
                    "question": question,
                    "answer": "No relevant information found in the resume.",
                    "sources": [],
                    "chunks": [],
                    "filters_used": filters
                }
            
            # Process matches
            chunks = []
            sources = []
            
            for match in matches:
                # Check if match is a dictionary
                if not isinstance(match, dict):
                    logger.warning(f"Unexpected match type: {type(match)}")
                    continue
                
                # Safely get metadata
                metadata = match.get('metadata', {})
                if not isinstance(metadata, dict):
                    logger.warning(f"Unexpected metadata type: {type(metadata)}")
                    metadata = {}
                
                # Get text
                text = metadata.get('text', '')
                if not text:
                    # Try alternative locations
                    text = match.get('text', '')
                    if not text:
                        # Check if there's a 'values' field or other
                        text = str(match.get('values', ''))[:500]
                
                if text and isinstance(text, str):
                    chunks.append(text.strip())
                    sources.append({
                        'page': metadata.get('page_number', 'unknown'),
                        'section': metadata.get('section_type', 'general'),
                        'skills': metadata.get('detected_skills', []),
                        'source': metadata.get('source', 'unknown'),
                        'score': match.get('score', 0),
                        'chunk_id': match.get('id', 'unknown')
                    })
            
            if not chunks:
                return {
                    "question": question,
                    "answer": "No text content found in the retrieved chunks.",
                    "sources": [],
                    "chunks": [],
                    "filters_used": filters
                }
            
            # Generate answer
            answer = self.llm.generate_answer(question, chunks)
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "chunks": chunks,
                "filters_used": filters
            }
            
        except Exception as e:
            logger.error(f"Error in ask_with_filters: {e}")
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "chunks": [],
                "filters_used": filters,
                "error": str(e)
            }
        
    def ask(self, question: str, top_k: int = TOP_K_RESULTS) -> dict:
        """
        Full RAG flow with SmolLM
        """
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS
        
        # 1. Retrieve relevant chunks (use top 2 for small model)
        search_results = self.vector_store.query(question, top_k=min(top_k, 4))
        
        if not search_results:
            return {
                "question": question,
                "answer": "No relevant information found.",
                "sources": [],
                "chunks": []
            }
        
        # 2. Extract text
        chunks = [result['text'] for result in search_results]
        sources = [result.get('metadata', {}) for result in search_results]
        scores = [result.get('score', 0.0) for result in search_results]
        
        # 3. Generate answer with SmolLM
        logger.info(f"Generating answer with {self.config.OLLAMA_MODEL}...")
        answer = self.llm.generate_answer(question, chunks)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "scores": scores
        }
    
    def stream_ask(self, question: str, top_k: int = TOP_K_RESULTS):
        """Stream answer from SmolLM"""
        if top_k is None:
            top_k = min(self.config.TOP_K_RESULTS, 4)
        
        search_results = self.vector_store.query(question, top_k=top_k)
        
        if not search_results:
            yield "No relevant information found."
            return
        
        chunks = [result['text'] for result in search_results]
        
        # Stream from SmolLM
        for token in self.llm.stream_answer(question, chunks):
            yield token

# Quick test
if __name__ == "__main__":
    rag = RAGPipeline()
    
    while True:
        try:
            # Get user input
            query = input("❓ You: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Skip empty queries
            if not query:
                print("Please enter a question.\n")
                continue
            
            # Process the query
            print("\n🔍 Searching...")
            result = rag.ask(query)
            
            # Display results
            print("\n" + "="*60)
            print(f"💡 Answer: {result['answer']}")
            print("="*60)
            
            # Show sources
            if result['sources']:
                print(f"\n📚 Sources: Found {len(result['sources'])} relevant chunks")
                for i, source in enumerate(result['sources'], 1):
                    page = source.get('page_number', 'unknown')
                    score = result['scores'][i - 1]
                    text = result['chunks'][i - 1]
                    print(f"  {i}. Page {page} (Similarity Score: {score:.4f})")
                    print(f"     Content: \"{text[:180]}...\"\n")
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")
