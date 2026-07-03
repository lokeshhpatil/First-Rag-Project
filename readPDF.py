from src.config import Config
import os
# 1. Tell Hugging Face to stay offline FIRST
os.environ["HF_HUB_OFFLINE"] = "1"
import fitz
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
# SETUP
def smart_chunk_pdf(pdf_path, chunk_size=500, overlap=100):
    doc = fitz.open(pdf_path)
    all_chunks = []
    chunk_id = 1

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if not text.strip():
            continue

        # FIX 1: Clean raw newlines so sentences stay whole
        # Replaces single newlines with spaces but keeps paragraph breaks
        cleaned_text = " ".join(text.split())

        start = 0
        while start < len(cleaned_text):
            end = start + chunk_size
            
            # FIX 2: Prevent cutting a word in half
            # Move the end pointer to the next nearest space character
            if end < len(cleaned_text):
                next_space = cleaned_text.find(" ", end)
                if next_space != -1 and (next_space - end) < 20:
                    end = next_space

            chunk_text = cleaned_text[start:end].strip()

            if chunk_text:
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "page": page_num + 1,
                    "metadata": {
                        "source": pdf_path,
                        "page_number": page_num + 1,
                    }   
                })
                chunk_id += 1
            
            # FIX 3: Shift forward while keeping the overlap
            start += (chunk_size - overlap)
            
    doc.close()
    return all_chunks

print("Creating chunks...")
chunks = smart_chunk_pdf("CV_EngineeringResumes.pdf", chunk_size=500)
# load model for embedding
print("Loading all-MiniLM-L6-V2 model...")
model = SentenceTransformer('all-MiniLM-L6-V2')
# conver all chunks into enbeddings
chunk_text = [chunk["text"] for chunk in chunks]
print("Embedding chunks...")
embeddings = model.encode(chunk_text).tolist()

# Pinecode setup
API_KEY = os.getenv("PINECONE_API_KEY")
if not API_KEY:
    raise ValueError("PINECONE_API_KEY is missing! Check your .env file.")

pc = Pinecone(api_key = API_KEY)
index_name = "first-rag-project"
dimension_size = 384
# 1. Delete old chunks for this document
index.delete(filter={"source": {"$eq": "CV_EngineeringResumes.pdf"}})

# 2. Upsert new chunks
index.upsert(vectors=vectors_to_upload)

if index_name not in pc.list_indexes().names():
    print(f"Creating new index: '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension_size,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully!")
else:
    print(f"Index '{index_name}' already exists. Ready to use!")

index = pc.Index(index_name)

print("Embedding and uploading chunks to Pinecone...")
vectors_to_upload = []
for i, chunk in enumerate(chunks):
    vectors_to_upload.append({
        "id": f"chunk-{i}",
        "values": embeddings[i],
        "metadata": {
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "page_number": chunk["metadata"]["page_number"]
        }
    })

index.upsert(vectors=vectors_to_upload)
print(f"Successfully uploaded {len(vectors_to_upload)} chunks to Pinecone!")

def retrieve_from_pinecone(user_query, top_k=Config.TOP_K_RESULTS):
    query_embedding = model.encode(user_query).tolist()
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata = True
    )

    retrieved_chunk = []
    print("\n--- Top Matches ---")
    for match in result["matches"]:
        score = match["score"]
        text = match["metadata"]["text"]

        retrieved_chunk.append((score,text))
        print(f"Score: {score:.4f} | Text: {text[:50]}...")
    return retrieved_chunk

query = "what is the name of the college lokesh is studying?"
matches = retrieve_from_pinecone(query)
