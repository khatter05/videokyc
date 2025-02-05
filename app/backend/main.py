from fastapi import FastAPI
from app.backend.routers.process import router as unified_router

# Initialize FastAPI app
app = FastAPI(
    title="Video KYC API",
    description="Unified API for OCR and Face Matching in Video KYC",
    version="1.0.0"
)

# Include the Unified Router
app.include_router(unified_router, prefix="/results", tags=["Video KYC"])

# Root endpoint (optional)
@app.get("/")
async def root():
    return {"message": "Welcome to the Video KYC API"}
