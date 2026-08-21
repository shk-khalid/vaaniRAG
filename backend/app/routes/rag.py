from fastapi import APIRouter, HTTPException
from backend.app.schemas.rag import QueryRequest, QueryResponse
from backend.app.services.search import search_service

router = APIRouter(prefix="/api/v1/rag")

@router.post("/query", response_model=QueryResponse, summary="Query the retrieval pipeline")
async def query_retrieval(request: QueryRequest):
    """
    Accepts a query, embeds it using BGE-M3, queries the Qdrant vector index,
    reranks the candidate passages, and returns the top 3 items along with
    step latency performance numbers.
    """
    try:
        results = search_service.run_query(
            query=request.query,
            language=request.language,
            filter_language=request.filter_language,
            top_k=request.top_k
        )
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during query processing: {str(e)}"
        )
