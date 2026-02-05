import logging
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Any
import json
import re
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    async def initialize(self):
        """Initialize quantized LLM model"""
        try:
            logger.info(f"Initializing LLM on {self.device}")
            
            # Use Mistral 7B for T4 compatibility
            model_name = "mistralai/Mistral-7B-Instruct-v0.1"
            
            # Quantization config for T4
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using the LLM"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def classify_document(self, text: str) -> Dict[str, Any]:
        """Classify document as APAR or Disciplinary"""
        try:
            prompt = f"""<s>[INST] Classify this government document as either "APAR" or "DISCIPLINARY".

APAR documents contain:
- Officer performance evaluations
- Annual Performance Assessment Reports
- Grading and pen pictures
- Reporting/Reviewing/Accepting authorities

DISCIPLINARY documents contain:
- IO Reports
- PO Briefs
- CO Briefs
- Disciplinary proceedings

Document text:
{text[:1000]}...

Respond with only: APAR or DISCIPLINARY [/INST]"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._generate_response,
                prompt,
                50
            )
            
            # Extract classification
            classification = "APAR" if "APAR" in response.upper() else "DISCIPLINARY"
            
            return {
                "document_type": classification,
                "confidence": 0.85  # Placeholder confidence
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
    
    async def process_apar(self, text: str) -> Dict[str, Any]:
        """Extract structured data from APAR document"""
        try:
            prompt = f"""<s>[INST] Extract structured information from this APAR document and return as JSON:

Required fields:
- officer_name: string
- date_of_birth: string (DD/MM/YYYY format)
- apar_year: string
- reporting_authority: string
- reviewing_authority: string
- accepting_authority: string
- overall_grading: string
- pen_picture: string (summary of performance assessment)

Document text:
{text[:2000]}...

Return only valid JSON: [/INST]"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._generate_response,
                prompt,
                400
            )
            
            # Extract JSON from response
            try:
                # Find JSON in response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback structure
                    result = {
                        "officer_name": "Not found",
                        "date_of_birth": "Not found",
                        "apar_year": "Not found",
                        "reporting_authority": "Not found",
                        "reviewing_authority": "Not found",
                        "accepting_authority": "Not found",
                        "overall_grading": "Not found",
                        "pen_picture": "Not found"
                    }
            except json.JSONDecodeError:
                result = {
                    "officer_name": "Parse error",
                    "date_of_birth": "Parse error",
                    "apar_year": "Parse error",
                    "reporting_authority": "Parse error",
                    "reviewing_authority": "Parse error",
                    "accepting_authority": "Parse error",
                    "overall_grading": "Parse error",
                    "pen_picture": "Parse error"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"APAR processing failed: {e}")
            raise
    
    async def process_disciplinary(self, text: str) -> Dict[str, Any]:
        """Generate summaries for disciplinary documents"""
        try:
            prompt = f"""<s>[INST] Analyze this disciplinary case document and provide structured summaries:

Generate summaries for:
1. Brief Background (1-1.5 pages summary)
2. IO Report Summary (key findings and recommendations)
3. PO Brief Summary (prosecution arguments)
4. CO Brief Summary (commanding officer's assessment)

Document text:
{text[:2000]}...

Return as JSON with fields: brief_background, io_report_summary, po_brief_summary, co_brief_summary [/INST]"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._generate_response,
                prompt,
                600
            )
            
            # Extract JSON from response
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {
                        "brief_background": "Summary not available",
                        "io_report_summary": "Summary not available",
                        "po_brief_summary": "Summary not available",
                        "co_brief_summary": "Summary not available"
                    }
            except json.JSONDecodeError:
                result = {
                    "brief_background": "Parse error",
                    "io_report_summary": "Parse error", 
                    "po_brief_summary": "Parse error",
                    "co_brief_summary": "Parse error"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Disciplinary processing failed: {e}")
            raise