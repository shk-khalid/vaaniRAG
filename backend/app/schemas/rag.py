from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query string to run search against", examples=["কৰ্পোৰেচন কি?"])
    language: Optional[str] = Field(None, description="Expected query language code (e.g. eng_Latn, hin_Deva, mar_Deva, urd_Arab)", examples=["asm_Beng"])
    filter_language: bool = Field(False, description="Filter vectors to target language matching query language metadata")
    top_k: int = Field(10, ge=1, le=50, description="Number of context passages to retrieve from first-stage dense index")

class ContextItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    language: str
    score: float = Field(..., description="First-stage dense similarity score (Cosine distance)")
    rerank_score: float = Field(..., description="Cross-Encoder reranking compatibility score")

class LatencyInfo(BaseModel):
    embedding_ms: float
    retrieval_ms: float
    reranking_ms: float
    total_ms: float

class QueryResponse(BaseModel):
    query: str
    detected_language: Optional[str]
    contexts: List[ContextItem]
    latency: LatencyInfo
