import os
# Tell Hugging Face to stay offline if the model is cached
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME
from src.vector_store import get_pinecone_index

def query_vector_store(query_text, top_k=4):
    """
    Connects to Pinecone, embeds the query text, retrieves the closest matching chunks,
    and returns them with their metadata.
    """
    print(f"\n--- Querying Vector Store for: '{query_text}' ---")
    
    # 1. Connect to Index
    index = get_pinecone_index()

    # 2. Load model and embed the query
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode(query_text).tolist()

    # 3. Search Pinecone
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # 4. Display Results
    retrieved_chunks = []
    print("\n--- Top Matches ---")
    for idx, match in enumerate(result["matches"]):
        score = match["score"]
        metadata = match["metadata"]
        text = metadata.get("text", "No text content found.")
        source = metadata.get("source", "Unknown Source")
        page = metadata.get("page_number", "Unknown Page")
        
        retrieved_chunks.append({
            "score": score,
            "text": text,
            "source": source,
            "page_number": page
        })
        
        print(f"Match #{idx + 1} | Score: {score:.4f} | Source: {source} (Page {page})")
        print(f"Content: \"{text[:120]}...\"\n")
        
    return retrieved_chunks

if __name__ == "__main__":
    # If a query is passed as a command-line argument, use it; otherwise, use a default query
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
            result = query_vector_store(query)
            # result = rag.ask(query)
            
            # Display results
            print("\n" + "="*60)
            print(f"💡 Answer: {result['answer']}")
            print("="*60)
            
            # Show sources
            if result['sources']:
                print(f"\n📚 Sources: Found {len(result['sources'])} relevant chunks")
                for i, source in enumerate(result['sources'], 1):
                    page = source.get('page_number', 'unknown')
                    print(f"  {i}. Page {page}")
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")

        
    
