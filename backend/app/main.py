from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app import config
from backend.app.routes import health, rag
from backend.app.services.search import search_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize models and databases once during startup
    search_service.initialize()
    yield
    # Cleanup lifecycle if needed

# Initialize FastAPI application
app = FastAPI(
    title="VaaniRAG Backend API",
    description="REST API service orchestrating BGE-M3 embeddings, Qdrant vector retrieval, and Cross-Encoder reranking.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints routers
app.include_router(health.router)
app.include_router(rag.router)
