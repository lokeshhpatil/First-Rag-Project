import os
import fitz
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def smart_chunk_pdf(pdf_path, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Parses a PDF file page-by-page, cleans whitespace, and splits it into 
    overlapping chunks without cutting words in half.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    doc = fitz.open(pdf_path)
    all_chunks = []
    chunk_id = 1
    file_name = os.path.basename(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if not text.strip():
            continue

        # Clean newlines so sentences remain unbroken
        cleaned_text = " ".join(text.split())

        start = 0
        while start < len(cleaned_text):
            end = start + chunk_size
            
            # Prevent cutting a word in half by shifting end to the next nearest space
            if end < len(cleaned_text):
                next_space = cleaned_text.find(" ", end)
                if next_space != -1 and (next_space - end) < 20:
                    end = next_space

            chunk_text = cleaned_text[start:end].strip()

            if chunk_text:
                all_chunks.append({
                    "id": f"{file_name}-chunk-{chunk_id}",
                    "text": chunk_text,
                    "page": page_num + 1,
                    "metadata": {
                        "source": file_name,
                        "page_number": page_num + 1,
                    }   
                })
                chunk_id += 1
            
            # Shift window forward by chunk size minus overlap
            start += (chunk_size - overlap)
            
    doc.close()
    return all_chunks
