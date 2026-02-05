import os
import sys
import logging
import asyncio
from typing import List, Dict, Any
import torch
from PIL import Image
import pdf2image
from concurrent.futures import ThreadPoolExecutor

# Add Surya to path
sys.path.append('/home/tech/ocr_service/surya')

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        self.foundation_predictor = None
        self.detection_predictor = None
        self.recognition_predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def initialize(self):
        """Initialize Surya OCR models"""
        try:
            logger.info(f"Initializing OCR models on {self.device}")
            
            # Disable FlashAttention to prevent crashes
            os.environ["DISABLE_FLASH_ATTENTION"] = "1"
            
            # Initialize predictors
            self.foundation_predictor = FoundationPredictor()
            self.detection_predictor = DetectionPredictor()
            self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
            
            logger.info("OCR models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR models: {e}")
            raise
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to PIL Images"""
        try:
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=300,
                fmt='RGB'
            )
            return images
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    
    def _run_ocr_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Run OCR on batch of images using Surya"""
        try:
            # Use Surya's recognition predictor
            predictions = self.recognition_predictor(
                images,
                det_predictor=self.detection_predictor
            )
            
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
            logger.error(f"OCR batch processing failed: {e}")
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
            
            # Process in batches to manage GPU memory
            batch_size = 4
            all_results = []
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_results = await loop.run_in_executor(
                    self.executor,
                    self._run_ocr_batch,
                    batch
                )
                
                # Adjust page numbers for batch offset
                for result in batch_results:
                    result["page_number"] = i + result["page_number"]
                
                all_results.extend(batch_results)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}")
            
            return {
                "total_pages": len(images),
                "pages": all_results
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise