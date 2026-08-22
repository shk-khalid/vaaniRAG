import os
import sys
import time
import requests
from typing import Dict, Any, Optional, List
from qdrant_client.http import models

# Add workspace paths to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from retrieval.qdrant_client import get_qdrant_client
from backend.app import config


class APIQdrantDenseRetriever:
    """
    API-driven dense retriever that communicates with Hugging Face Serverless API 
    for BGE-M3 embeddings and Qdrant Cloud. Requires NO local model loading.
    """

    def __init__(self, collection_name: str, client=None):
        self.collection_name = collection_name
        self.client = client if client else get_qdrant_client()

    def embed_query(self, query_text: str) -> List[float]:
        """
        Sends the query to Hugging Face Serverless API for BGE-M3 embedding generation.
        """
        url = "https://api-inference.huggingface.co/models/BAAI/bge-m3"
        headers = {
            "Authorization": f"Bearer {config.HF_TOKEN}",
            "x-wait-for-model": "true"
        }
        
        response = requests.post(url, headers=headers, json={"inputs": query_text})
        response.raise_for_status()
        
        res_data = response.json()
        if isinstance(res_data, list) and len(res_data) > 0 and isinstance(res_data[0], list):
            return res_data[0]
        return res_data

    def search_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        language_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the Qdrant database using the query vector.
        """
        # 1. Build filter conditions if language_filter is specified
        query_filter = None
        if language_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="language",
                        match=models.MatchValue(value=language_filter)
                    )
                ]
            )

        # 2. Perform Qdrant Vector search
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=k
        )
        hits = response.points

        # 3. Format outputs
        results = []
        for hit in hits:
            payload = hit.payload
            results.append({
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "query_id": payload.get("query_id"),
                "text": payload.get("text"),
                "language": payload.get("language"),
                "is_selected": bool(payload.get("is_selected", False)),
                "score": float(hit.score)
            })
            
        return results


class APICrossEncoderReranker:
    """
    API-driven cross-encoder reranker that communicates with Hugging Face Serverless API.
    Requires NO local model loading.
    """

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Sends query-candidate pairs to Hugging Face Serverless API for MiniLM reranking.
        """
        if not candidates:
            return []

        url = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
        headers = {
            "Authorization": f"Bearer {config.HF_TOKEN}",
            "x-wait-for-model": "true"
        }
        
        # Hugging Face sentence-similarity endpoint payload format
        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": [c["text"] for c in candidates]
            }
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        scores = response.json()

        # Handle API response mapping
        reranked = []
        for idx, score in enumerate(scores):
            cand_copy = dict(candidates[idx])
            cand_copy["rerank_score"] = float(score)
            reranked.append(cand_copy)

        # Sort descending by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]


class SearchService:
    """
    Service class that orchestrates API-driven embedding, retrieval, and reranking.
    Requires no heavy local ML model loading, keeping memory usage minimal.
    """

    def __init__(self):
        self.retriever: Optional[APIQdrantDenseRetriever] = None
        self.reranker: Optional[APICrossEncoderReranker] = None
        self.initialized = False

    def initialize(self):
        """
        Initializes database connections. Call on app startup.
        """
        if self.initialized:
            return

        print("Initializing SearchService (API-driven mode)...")
        # Ensure we import get_qdrant_client dynamically
        from retrieval.qdrant_client import get_qdrant_client

        # Create API-driven clients
        self.retriever = APIQdrantDenseRetriever(
            collection_name=config.QDRANT_COLLECTION
        )
        self.reranker = APICrossEncoderReranker()
        
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
        Runs the API-driven RAG query pipeline measuring timings.
        """
        if not self.initialized:
            raise RuntimeError("SearchService has not been initialized yet.")

        # Stage 1: API Embedding Generation
        start_embed = time.perf_counter()
        query_vector = self.retriever.embed_query(query)
        end_embed = time.perf_counter()
        embedding_ms = (end_embed - start_embed) * 1000

        # Stage 2: Qdrant Cloud Retrieval
        lang_filter = language if filter_language else None
        start_retrieve = time.perf_counter()
        hits = self.retriever.search_vector(
            query_vector=query_vector,
            k=top_k,
            language_filter=lang_filter
        )
        end_retrieve = time.perf_counter()
        retrieval_ms = (end_retrieve - start_retrieve) * 1000

        # Stage 3: API Cross-Encoder Reranking
        start_rerank = time.perf_counter()
        reranked_hits = self.reranker.rerank(
            query=query,
            candidates=hits,
            top_n=3
        )
        end_rerank = time.perf_counter()
        reranking_ms = (end_rerank - start_rerank) * 1000

        total_ms = embedding_ms + retrieval_ms + reranking_ms

        # Prepare contexts list
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
