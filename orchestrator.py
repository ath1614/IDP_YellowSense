import asyncio
import httpx
import logging
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class IDPOrchestrator:
    def __init__(self, ocr_url: str = "http://ocr-vm:8000", llm_url: str = "http://llm-vm:8001"):
        self.ocr_url = ocr_url
        self.llm_url = llm_url
        self.timeout = httpx.Timeout(300.0)  # 5 minutes
        
    async def _call_with_retry(self, client: httpx.AsyncClient, method: str, url: str, **kwargs) -> Dict:
        """Call API with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def process_document(self, pdf_file: UploadFile) -> Dict[str, Any]:
        """Complete document processing pipeline"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Step 1: OCR Processing
                logger.info("Starting OCR processing")
                files = {"file": (pdf_file.filename, await pdf_file.read(), "application/pdf")}
                
                ocr_result = await self._call_with_retry(
                    client, "POST", f"{self.ocr_url}/ocr", files=files
                )
                
                # Combine all page text
                full_text = "\n".join([page["text"] for page in ocr_result["pages"]])
                logger.info(f"OCR completed: {ocr_result['total_pages']} pages processed")
                
                # Step 2: Document Classification
                logger.info("Starting document classification")
                classify_result = await self._call_with_retry(
                    client, "POST", f"{self.llm_url}/classify", 
                    json={"text": full_text[:2000]}  # First 2000 chars for classification
                )
                
                doc_type = classify_result["document_type"]
                logger.info(f"Document classified as: {doc_type}")
                
                # Step 3: Document-specific processing
                if doc_type == "APAR":
                    logger.info("Processing APAR document")
                    structured_result = await self._call_with_retry(
                        client, "POST", f"{self.llm_url}/apar",
                        json={"text": full_text}
                    )
                else:  # DISCIPLINARY
                    logger.info("Processing Disciplinary document")
                    structured_result = await self._call_with_retry(
                        client, "POST", f"{self.llm_url}/disciplinary",
                        json={"text": full_text}
                    )
                
                # Final result
                return {
                    "document_type": doc_type,
                    "ocr_result": ocr_result,
                    "structured_data": structured_result,
                    "processing_status": "completed"
                }
                
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise

# FastAPI app for orchestration (optional - can be used locally)
app = FastAPI(title="IDP Orchestrator", version="1.0.0")
orchestrator = IDPOrchestrator()

@app.post("/process")
async def process_document_endpoint(file: UploadFile = File(...)):
    try:
        result = await orchestrator.process_document(file)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))