import asyncio
import logging
import io
from pathlib import Path
from typing import List, Dict, Any
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple PDF documents concurrently."""

    def __init__(self, orchestrator, concurrency: int = 3):
        self.orchestrator = orchestrator
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _process_one(self, pdf_path: Path) -> Dict[str, Any]:
        async with self.semaphore:
            try:
                logger.info(f"Processing: {pdf_path.name}")
                with open(pdf_path, "rb") as f:
                    upload = UploadFile(filename=pdf_path.name, file=io.BytesIO(f.read()))
                    result = await self.orchestrator.process_document(upload)
                return {"file": pdf_path.name, "status": "success", "result": result}
            except Exception as e:
                logger.error(f"Failed {pdf_path.name}: {e}")
                return {"file": pdf_path.name, "status": "failed", "error": str(e)}

    async def process_folder(self, folder: str) -> List[Dict[str, Any]]:
        pdfs = list(Path(folder).glob("*.pdf"))
        if not pdfs:
            logger.warning(f"No PDFs found in {folder}")
            return []
        logger.info(f"Processing {len(pdfs)} documents from {folder}")
        results = await asyncio.gather(*[self._process_one(p) for p in pdfs])
        success = sum(1 for r in results if r["status"] == "success")
        logger.info(f"Batch complete: {success}/{len(pdfs)} succeeded")
        return list(results)

    async def process_files(self, files: List[str]) -> List[Dict[str, Any]]:
        return list(await asyncio.gather(*[self._process_one(Path(f)) for f in files]))



