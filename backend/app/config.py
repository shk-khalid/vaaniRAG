import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App server configurations
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Qdrant cluster configurations
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vaani_rag")

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
