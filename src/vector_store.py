import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME,TOP_K_RESULTS

def get_pinecone_index():
    """
    Initializes Pinecone client, checks if index exists (creates it if not),
    and returns the Index connection object.
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is not set. Check your .env file.")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created successfully!")
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists. Ready to use.")
        
    return pc.Index(PINECONE_INDEX_NAME)

def delete_chunks_by_source(index, source_filename):
    """
    Deletes all vectors from the index matching the specified source metadata filter.
    This prevents 'ghost chunks' when re-ingesting a modified document.
    """
    print(f"Clearing old vectors for source: '{source_filename}'...")
    try:
        # Delete by metadata filter
        index.delete(filter={"source": {"$eq": source_filename}})
        print("Old vectors cleared successfully.")
    except Exception as e:
        print(f"Note: Could not delete old vectors (index might be empty). Details: {e}")

def upsert_chunks_with_meta_data(index, chunks, embeddings):
    """
    Formats chunks and their embeddings, and upserts them to the Pinecone index.
    """
    print(f"Preparing {len(chunks)} chunks for upsert...")
    vectors_to_upload = []
    
    for i, chunk in enumerate(chunks):
        metadata = chunk["metadata"].copy()
        metadata["text"] = chunk["text"]
        metadata["chunk_id"] = chunk["id"]

        vectors_to_upload.append({
            "id": chunk["id"],
            "values": embeddings[i],
            "metadata": metadata
        })
        
    print("Uploading vectors to Pinecone...")
    # Upsert the vectors
    upsert_response = index.upsert(vectors=vectors_to_upload)
    print(f"Successfully uploaded vectors! Response: {upsert_response}")
    return upsert_response

#Q1 explain the parameters what is index> filters?
def query_with_metadata_filtering(index, query_embedding, filters = None, top_k = TOP_K_RESULTS):
    if filters is None:
        filters = {}

    pinecone_filter = {}
    # Q2 also exaplin this whole function
    for key, value in filters.items():
        if isinstance(value, list):
            pinecone_filter[key] = {"$in": value}
        elif isinstance(value, str):
            pinecone_filter[key] = {"$eq": value}
        elif isinstance(value, dict):
            pinecone_filter[key] = value
        else:
            pinecone_filter[key] = {"$eq": value}
        
    result = index.query(
        vector = query_embedding,
        top_k=top_k,
        include_metadata = True,
        filter=pinecone_filter
    )
    return result

# Class wrapper to support OOP interface for the RAG pipeline
class VectorStore:
    def __init__(self, config=None):
        self.config = config
        self.index = get_pinecone_index()
        # Initialize the model once for querying
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def query(self, query_text: str, top_k: int = 4) -> list:
        """
        Embeds the query text and retrieves closest matching chunks from Pinecone.
        """
        query_embedding = self.model.encode(query_text).tolist()
        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        formatted_matches = []
        for match in result.get("matches", []):
            formatted_matches.append({
                "text": match["metadata"].get("text", ""),
                "score": match["score"],
                "metadata": {
                    "source": match["metadata"].get("source", "Unknown"),
                    "page_number": match["metadata"].get("page_number", "Unknown")
                }
            })
        return formatted_matches

# Alias to maintain compatibility with the user's import statement
vector_store = VectorStore

