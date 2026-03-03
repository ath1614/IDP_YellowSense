import os
import logging
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .ocr_processor_v2 import OCRProcessor

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

app = FastAPI(title="Government IDP OCR Service", version="2.0.0")
ocr_processor: OCRProcessor = None


@app.on_event("startup")
async def startup_event():
    global ocr_processor
    ocr_processor = OCRProcessor()
    await ocr_processor.initialize()
    logger.info(f"OCR processor ready (engine={ocr_processor._engine})")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "ocr",
        "engine": ocr_processor._engine if ocr_processor else "uninitialized",
    }


@app.post("/ocr")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    tmp_path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = await ocr_processor.process_pdf(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


