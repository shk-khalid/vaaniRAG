import os
import sys
import time
from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer, CrossEncoder

# Add workspace paths to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from retrieval.dense_retriever import QdrantDenseRetriever
from retrieval.reranker import CrossEncoderReranker
from backend.app import config


class SearchService:
    """
    Service class that loads models and DB client on startup
    and orchestrates embedding generation, retrieval, and reranking.
    """

    def __init__(self):
        self.embed_model: Optional[SentenceTransformer] = None
        self.cross_model: Optional[CrossEncoder] = None
        self.retriever: Optional[QdrantDenseRetriever] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.initialized = False

    def initialize(self):
        """
        Loads models and connections to memory. Call on app startup.
        """
        if self.initialized:
            return

        print("Initializing SearchService models and database clients...")
        # 1. Load shared embedding model
        self.embed_model = SentenceTransformer("BAAI/bge-m3")

        # 2. Load reranker model
        self.cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 3. Create retriever and reranker clients
        self.retriever = QdrantDenseRetriever(
            collection_name=config.QDRANT_COLLECTION,
            model=self.embed_model
        )
        self.reranker = CrossEncoderReranker(model=self.cross_model)
        
        self.initialized = True
        print("SearchService initialized successfully.")

    def run_query(
        self,
        query: str,
        language: Optional[str] = None,
        filter_language: bool = False,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Runs the RAG query orchestration pipeline measuring timing at every phase.
        """
        if not self.initialized:
            raise RuntimeError("SearchService has not been initialized yet.")

        # Stage 1: Embedding generation
        start_embed = time.perf_counter()
        query_vector = self.retriever.embed_query(query)
        end_embed = time.perf_counter()
        embedding_ms = (end_embed - start_embed) * 1000

        # Stage 2: Qdrant Retrieval
        lang_filter = language if filter_language else None
        start_retrieve = time.perf_counter()
        hits = self.retriever.search_vector(
            query_vector=query_vector,
            k=top_k,
            language_filter=lang_filter
        )
        end_retrieve = time.perf_counter()
        retrieval_ms = (end_retrieve - start_retrieve) * 1000

        # Stage 3: Cross-Encoder Reranking
        start_rerank = time.perf_counter()
        reranked_hits = self.reranker.rerank(
            query=query,
            candidates=hits,
            top_n=3  # Always return top 3 contexts
        )
        end_rerank = time.perf_counter()
        reranking_ms = (end_rerank - start_rerank) * 1000

        total_ms = embedding_ms + retrieval_ms + reranking_ms

        # Prepare schemas output
        contexts = []
        for hit in reranked_hits:
            contexts.append({
                "chunk_id": hit["chunk_id"],
                "document_id": hit["document_id"],
                "text": hit["text"],
                "language": hit["language"],
                "score": hit["score"],
                "rerank_score": hit["rerank_score"]
            })

        return {
            "query": query,
            "detected_language": language,
            "contexts": contexts,
            "latency": {
                "embedding_ms": round(embedding_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "reranking_ms": round(reranking_ms, 2),
                "total_ms": round(total_ms, 2)
            }
        }


# Global singleton instance
search_service = SearchService()
