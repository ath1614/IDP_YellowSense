import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .llm_processor import LLMProcessor

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

app = FastAPI(title="Government IDP LLM Service", version="2.0.0")
llm_processor: LLMProcessor = None


class TextRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup_event():
    global llm_processor
    llm_processor = LLMProcessor()
    await llm_processor.initialize()
    logger.info("LLM processor ready")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "llm", "cache_size": llm_processor.cache.size if llm_processor else 0}


@app.post("/classify")
async def classify(request: TextRequest):
    try:
        return JSONResponse(content=await llm_processor.classify_document(request.text))
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apar")
async def process_apar(request: TextRequest):
    try:
        return JSONResponse(content=await llm_processor.process_apar(request.text))
    except Exception as e:
        logger.error(f"APAR processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/disciplinary")
async def process_disciplinary(request: TextRequest):
    try:
        return JSONResponse(content=await llm_processor.process_disciplinary(request.text))
    except Exception as e:
        logger.error(f"Disciplinary processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    llm_processor.cache.clear()
    return {"message": "Cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


