from src.chunker import smart_chunk_pdf
from src.vector_store import vector_store
from src.llm_integration import FreeLLM
from src.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG pipeline with SmolLM"""
    
    def __init__(self, config=Config()):
        self.config = config
        self.vector_store = vector_store(config)
        self.llm = FreeLLM(config)
    
    def ask(self, question: str, top_k: int = None) -> dict:
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
    
    def stream_ask(self, question: str, top_k: int = None):
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
