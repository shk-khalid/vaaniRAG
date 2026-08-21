import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vaani_rag")


def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns a Qdrant client.
    If the cluster URL is the placeholder or empty, falls back to a local
    file-backed database at data/qdrant_db.
    """
    is_placeholder = (
        not QDRANT_URL 
        or QDRANT_URL.startswith("https://your-cluster-url") 
        or QDRANT_URL.strip() == ""
    )
    
    if is_placeholder:
        db_path = os.path.join("data", "qdrant_db")
        os.makedirs("data", exist_ok=True)
        print(f"Connecting to Qdrant (local file-backed mode): {db_path}")
        return QdrantClient(path=db_path)
    else:
        print(f"Connecting to Qdrant Cloud Cluster: {QDRANT_URL}")
        api_key = QDRANT_API_KEY if QDRANT_API_KEY != "your_api_key" else None
        return QdrantClient(url=QDRANT_URL, api_key=api_key, timeout=60.0)


def init_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024) -> None:
    """
    Ensures that the collection exists in Qdrant with the correct dimensions and metrics.
    """
    # Check if collection exists
    collections_response = client.get_collections()
    collection_names = [col.name for col in collections_response.collections]
    
    if collection_name not in collection_names:
        print(f"Collection '{collection_name}' not found. Creating collection...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' successfully created.")
    else:
        print(f"Collection '{collection_name}' already exists.")
