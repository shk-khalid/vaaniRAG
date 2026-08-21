import os
import sys
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    Reranker wrapper that uses a Cross-Encoder model (e.g. MiniLM)
    to compute precise query-passage compatibility scores.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model: Optional[CrossEncoder] = None):
        if model:
            self.model = model
        else:
            print(f"Loading Cross-Encoder model '{model_name}'...")
            self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunks based on Cross-Encoder scoring.
        """
        if not candidates:
            return []

        # Prepare pairs: [query, passage_text]
        pairs = [[query, c["text"]] for c in candidates]
        
        # Predict scores in a single batch
        scores = self.model.predict(pairs)

        # Attach scores to candidates
        reranked = []
        for idx, score in enumerate(scores):
            # Copy payload and attach score
            cand_copy = dict(candidates[idx])
            cand_copy["rerank_score"] = float(score)
            reranked.append(cand_copy)

        # Sort descending by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_n]
