import os
import sys
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models

# Import client helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.qdrant_client import get_qdrant_client, QDRANT_COLLECTION


class QdrantDenseRetriever:
    """
    Retrieval client that encodes query text using BGE-M3
    and searches a local or remote Qdrant collection.
    """

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        model: Optional[SentenceTransformer] = None,
        client = None
    ):
        self.collection_name = collection_name
        self.client = client if client else get_qdrant_client()
        
        # Load embedding model if not provided (allows model reuse in benchmarks)
        if model:
            self.model = model
        else:
            print("Loading SentenceTransformer model 'BAAI/bge-m3' in dense retriever...")
            self.model = SentenceTransformer("BAAI/bge-m3")

    def search(
        self,
        query_text: str,
        k: int = 10,
        language_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Encodes query_text and retrieves the top k matching chunks from Qdrant.
        Optionally filters results by language stored in payload metadata.
        """
        # 1. Encode query
        query_vector = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0].tolist()

        # 2. Build filter conditions if language_filter is specified
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

        # 3. Perform Qdrant Vector search
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=k
        )
        hits = response.points

        # 4. Format outputs
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
