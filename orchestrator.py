import asyncio
import logging
import httpx
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from config import config
from metrics import metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.log_level)


class IDPOrchestrator:
    def __init__(self, ocr_url: str = None, llm_url: str = None):
        self.ocr_url = ocr_url or config.ocr.url
        self.llm_url = llm_url or config.llm.url
        self.timeout = httpx.Timeout(config.ocr.timeout)

    async def _call_with_retry(self, client: httpx.AsyncClient, method: str, url: str, **kwargs) -> Dict:
        for attempt in range(config.max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{config.max_retries} failed for {url}: {e}")
                if attempt == config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def process_document(self, pdf_file: UploadFile) -> Dict[str, Any]:
        import time
        start = time.time()
        try:
            content = await pdf_file.read()

            # Validate file size
            max_bytes = config.max_file_size_mb * 1024 * 1024
            if len(content) > max_bytes:
                raise ValueError(f"File exceeds {config.max_file_size_mb}MB limit")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Step 1: OCR
                logger.info(f"OCR start: {pdf_file.filename}")
                ocr_result = await self._call_with_retry(
                    client, "POST", f"{self.ocr_url}/ocr",
                    files={"file": (pdf_file.filename, content, "application/pdf")}
                )
                full_text = "\n".join(p["text"] for p in ocr_result["pages"])
                logger.info(f"OCR done: {ocr_result['total_pages']} pages, {len(full_text)} chars")

                # Step 2: Classify
                classify_result = await self._call_with_retry(
                    client, "POST", f"{self.llm_url}/classify",
                    json={"text": full_text[:config.llm.context_limit]}
                )
                doc_type = classify_result["document_type"]
                logger.info(f"Classified as: {doc_type} (confidence={classify_result.get('confidence')})")

                # Step 3: Extract
                endpoint = "apar" if doc_type == "APAR" else "disciplinary"
                structured_result = await self._call_with_retry(
                    client, "POST", f"{self.llm_url}/{endpoint}",
                    json={"text": full_text}
                )

            elapsed_ms = (time.time() - start) * 1000
            metrics.record("process_document", True, elapsed_ms)
            logger.info(f"Pipeline complete in {elapsed_ms:.0f}ms")

            return {
                "document_type": doc_type,
                "confidence": classify_result.get("confidence"),
                "ocr_result": ocr_result,
                "structured_data": structured_result,
                "processing_status": "completed",
                "processing_time_ms": round(elapsed_ms, 2),
            }

        except Exception as e:
            import time as t
            elapsed_ms = (t.time() - start) * 1000
            metrics.record("process_document", False, elapsed_ms)
            logger.error(f"Pipeline failed: {e}")
            raise


app = FastAPI(title="IDP Orchestrator", version="2.0.0")
orchestrator = IDPOrchestrator()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "orchestrator"}


@app.get("/metrics")
async def get_metrics():
    return JSONResponse(content=metrics.summary())


@app.post("/process")
async def process_document_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        result = await orchestrator.process_document(file)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



