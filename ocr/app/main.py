import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import asyncio
from .ocr_processor_v2 import OCRProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Government IDP OCR Service", version="1.0.0")

# Initialize OCR processor
ocr_processor = None

@app.on_event("startup")
async def startup_event():
    global ocr_processor
    try:
        ocr_processor = OCRProcessor()
        await ocr_processor.initialize()
        logger.info("OCR processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OCR processor: {e}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ocr"}

@app.post("/ocr")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process PDF
        result = await ocr_processor.process_pdf(tmp_file_path)
        
        # Cleanup
        os.unlink(tmp_file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)