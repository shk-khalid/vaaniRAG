from fastapi import APIRouter

router = APIRouter()

@router.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Returns service health status.
    """
    return {"status": "ok"}
