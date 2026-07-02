import os
# 1. Tell Hugging Face to stay offline FIRST
os.environ["HF_HUB_OFFLINE"] = "1"
import pymupdf as fitz
import json
from sentence_transformers import SentenceTransformer

def smart_chunk_pdf(pdf_path, chunk_size=500, overlap=100):
    doc = fitz.open(pdf_path)
    all_chunk = []
    chunk_id = 1

    for page_num in range(len(doc)):
        page = doc[page_num]
        text =  page.get_text()
        # Skip empty pages
        if not text.strip():
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                all_chunk.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "page": page_num + 1,
                    "metadata": {
                        "source": pdf_path,
                        "page_number":page_num + 1,
                        "chunk_size": len(chunk_text) 
                    }   
                })
            chunk_id += 1
            start += chunk_size - overlap
    doc.close()
    return all_chunk
print("Creating chunks...")
chunks = smart_chunk_pdf("CV_EngineeringResumes.pdf", chunk_size=500)
# load model for embedding
print("Loading all-MiniLM-L6-V2 model...")
model = SentenceTransformer('all-MiniLM-L6-V2')
# conver all chunks into enbeddings

chunk_text = [chunk["text"] for chunk in chunks]

print("Embedding chunks...")
embeddings = model.encode(chunk_text)
