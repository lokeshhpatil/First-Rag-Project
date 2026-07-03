import os
# Tell Hugging Face to stay offline if the model is cached
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME
from src.chunker import smart_chunk_pdf
from src.vector_store import get_pinecone_index, delete_chunks_by_source, upsert_chunks_with_meta_data

def run_ingestion(pdf_path):
    print(f"--- Starting Ingestion Pipeline for: {pdf_path} ---")
    
    # 1. Chunk PDF
    print("\n[Step 1/4] Extracting and chunking PDF text...")
    chunks = smart_chunk_pdf(pdf_path)
    print(f"Generated {len(chunks)} chunks.")

    # 2. Embed chunks in batch (extremely fast)
    print("\n[Step 2/4] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Encoding chunks...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(chunk_texts).tolist()
    print("Chunk encoding completed.")

    # 3. Setup index & Delete old chunks (to prevent ghost chunks)
    print("\n[Step 3/4] Connecting to Pinecone...")
    index = get_pinecone_index()
    
    # Delete previous chunks from this file name
    file_name = os.path.basename(pdf_path)
    delete_chunks_by_source(index, file_name)

    # 4. Upload to Pinecone
    print("\n[Step 4/4] Uploading new vectors to Pinecone...")
    upsert_chunks_with_meta_data(index, chunks, embeddings)
    
    print("\n--- Ingestion Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    # Ingest the default resume PDF
    default_pdf = "CV_EngineeringResumes.pdf"
    run_ingestion(default_pdf)
