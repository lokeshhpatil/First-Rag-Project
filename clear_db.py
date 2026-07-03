from src.vector_store import get_pinecone_index

def clear_database():
    print("Connecting to Pinecone index...")
    index = get_pinecone_index()
    print("Deleting all vectors from the index...")
    try:
        index.delete(delete_all=True)
        print("Database wiped successfully!")
    except Exception as e:
        print(f"Error deleting vectors: {e}")

if __name__ == "__main__":
    clear_database()
