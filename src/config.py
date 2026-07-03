import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "first-rag-project"

# Embedding Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-V2"
EMBEDDING_DIMENSION = 384

# Chunking configuration
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 100

# LLM CONFIGURATION
# OLLAMA_MODEL = 'smollm:135m' (WORST MODEL)
OLLAMA_MODEL = 'llama3.2:1b'
OLLAMA_BASE_URL = 'http://localhost:11434'

# RAG Settings
TOP_K_RESULTS = 4
TEMPERATURE = 0.5
MAX_TOKENS = 150  

# Config Wrapper for OOP imports
class Config:
    def __init__(self):
        self.PINECONE_API_KEY = PINECONE_API_KEY
        self.PINECONE_INDEX_NAME = PINECONE_INDEX_NAME
        self.EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME
        self.EMBEDDING_DIMENSION = EMBEDDING_DIMENSION
        self.DEFAULT_CHUNK_SIZE = DEFAULT_CHUNK_SIZE
        self.DEFAULT_CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP
        self.OLLAMA_MODEL = OLLAMA_MODEL
        self.OLLAMA_BASE_URL = OLLAMA_BASE_URL
        self.TOP_K_RESULTS = TOP_K_RESULTS
        self.TEMPERATURE = TEMPERATURE
        self.MAX_TOKENS = MAX_TOKENS

