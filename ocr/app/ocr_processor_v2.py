import os
import logging
import asyncio
from typing import List, Dict, Any
from PIL import Image
import pdf2image
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class OCRProcessor:
    def __init__(self):
        self.foundation_predictor = None
        self.recognition_predictor = None
        self.detection_predictor = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._engine = "none"

    async def initialize(self):
        try:
            logger.info("Initializing Surya OCR models")
            os.environ["DISABLE_FLASH_ATTENTION"] = "1"
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor

            self.foundation_predictor = FoundationPredictor()
            self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
            self.detection_predictor = DetectionPredictor()
            self._engine = "surya"
            logger.info("Surya OCR initialized")
        except Exception as e:
            logger.warning(f"Surya unavailable ({e}), falling back to Tesseract")
            import pytesseract
            pytesseract.get_tesseract_version()
            self._engine = "tesseract"
            logger.info("Tesseract OCR initialized as fallback")

    def _convert_pdf(self, pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        return pdf2image.convert_from_path(pdf_path, dpi=dpi)

    def _surya_ocr(self, images: List[Image.Image]) -> List[Dict]:
        predictions = self.recognition_predictor(images, det_predictor=self.detection_predictor)
        results = []
        for i, pred in enumerate(predictions):
            lines = pred.text_lines
            text = "\n".join(line.text for line in lines)
            avg_conf = (
                sum(getattr(line, "confidence", 1.0) for line in lines) / len(lines)
                if lines else 0.0
            )
            results.append({
                "page_number": i + 1,
                "text": text.strip(),
                "line_count": len(lines),
                "confidence": round(avg_conf, 4),
                "engine": "surya",
            })
        return results

    def _tesseract_ocr(self, images: List[Image.Image]) -> List[Dict]:
        import pytesseract
        results = []
        for i, img in enumerate(images):
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(img)
            confs = [c for c in data["conf"] if c != -1]
            avg_conf = sum(confs) / len(confs) / 100 if confs else 0.0
            results.append({
                "page_number": i + 1,
                "text": text.strip(),
                "line_count": text.count("\n"),
                "confidence": round(avg_conf, 4),
                "engine": "tesseract",
            })
        return results

    async def process_pdf(self, pdf_path: str, dpi: int = 300, batch_size: int = 4) -> Dict[str, Any]:
        logger.info(f"Processing: {pdf_path}")
        loop = asyncio.get_event_loop()

        images = await loop.run_in_executor(self.executor, self._convert_pdf, pdf_path, dpi)
        logger.info(f"Converted to {len(images)} pages")

        all_results = []
        ocr_fn = self._surya_ocr if self._engine == "surya" else self._tesseract_ocr

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = await loop.run_in_executor(self.executor, ocr_fn, batch)
            for r in batch_results:
                r["page_number"] += i
            all_results.extend(batch_results)
            logger.info(f"Batch {i // batch_size + 1}/{-(-len(images) // batch_size)} done")

        avg_confidence = sum(r["confidence"] for r in all_results) / len(all_results) if all_results else 0.0

        return {
            "total_pages": len(images),
            "pages": all_results,
            "avg_confidence": round(avg_confidence, 4),
            "engine": self._engine,
        }

