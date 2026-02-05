import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List
from .llm_processor import LLMProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Government IDP LLM Service", version="1.0.0")

# Request models
class ClassifyRequest(BaseModel):
    text: str

class APARRequest(BaseModel):
    text: str

class DisciplinaryRequest(BaseModel):
    text: str

# Initialize LLM processor
llm_processor = None

@app.on_event("startup")
async def startup_event():
    global llm_processor
    try:
        llm_processor = LLMProcessor()
        await llm_processor.initialize()
        logger.info("LLM processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM processor: {e}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm"}

@app.post("/classify")
async def classify_document(request: ClassifyRequest):
    try:
        result = await llm_processor.classify_document(request.text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/apar")
async def process_apar(request: APARRequest):
    try:
        result = await llm_processor.process_apar(request.text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"APAR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"APAR processing failed: {str(e)}")

@app.post("/disciplinary")
async def process_disciplinary(request: DisciplinaryRequest):
    try:
        result = await llm_processor.process_disciplinary(request.text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Disciplinary processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Disciplinary processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)