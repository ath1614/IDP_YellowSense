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
        
    async def initialize(self):
        """Initialize Surya OCR models using correct API"""
        try:
            logger.info("Initializing Surya OCR models")
            
            # Import Surya components
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            
            # Initialize predictors in correct order
            self.foundation_predictor = FoundationPredictor()
            self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
            self.detection_predictor = DetectionPredictor()
            
            logger.info("Surya OCR models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Surya OCR: {e}")
            # Fallback to Tesseract if Surya fails
            logger.info("Falling back to Tesseract OCR")
            import pytesseract
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized as fallback")
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to PIL Images"""
        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=300)
            return images
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    
    def _run_surya_ocr(self, images: List[Image.Image]) -> List[Dict]:
        """Run OCR using Surya"""
        try:
            # Use Surya's recognition predictor with detection predictor
            predictions = self.recognition_predictor(images, det_predictor=self.detection_predictor)
            
            results = []
            for i, pred in enumerate(predictions):
                page_text = ""
                for text_line in pred.text_lines:
                    page_text += text_line.text + "\n"
                
                results.append({
                    "page_number": i + 1,
                    "text": page_text.strip()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Surya OCR failed: {e}")
            raise
    
    def _run_tesseract_ocr(self, images: List[Image.Image]) -> List[Dict]:
        """Fallback OCR using Tesseract"""
        try:
            import pytesseract
            results = []
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                results.append({
                    "page_number": i + 1,
                    "text": text.strip()
                })
            return results
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise
    
    async def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF and return OCR results"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Convert PDF to images
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                self.executor, 
                self._convert_pdf_to_images, 
                pdf_path
            )
            
            logger.info(f"Converted PDF to {len(images)} pages")
            
            # Try Surya OCR first, fallback to Tesseract
            try:
                if self.recognition_predictor and self.detection_predictor:
                    results = await loop.run_in_executor(
                        self.executor,
                        self._run_surya_ocr,
                        images
                    )
                    logger.info("Used Surya OCR for processing")
                else:
                    raise Exception("Surya not available")
            except Exception as e:
                logger.warning(f"Surya OCR failed, using Tesseract: {e}")
                results = await loop.run_in_executor(
                    self.executor,
                    self._run_tesseract_ocr,
                    images
                )
                logger.info("Used Tesseract OCR for processing")
            
            return {
                "total_pages": len(images),
                "pages": results
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise