import logging
import asyncio
import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .cache import ResultCache

logger = logging.getLogger(__name__)

_APAR_FIELDS = [
    "officer_name", "date_of_birth", "apar_year",
    "reporting_authority", "reviewing_authority", "accepting_authority",
    "overall_grading", "pen_picture",
]
_DISCIPLINARY_FIELDS = [
    "brief_background", "io_report_summary", "po_brief_summary", "co_brief_summary",
]


def _fallback(fields: list, value: str = "Not found") -> Dict:
    return {f: value for f in fields}


def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON object found in response")


class LLMProcessor:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", cache_ttl: int = 3600):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.model_name = model_name
        self.cache = ResultCache(ttl=cache_ttl)

    async def initialize(self):
        logger.info(f"Loading {self.model_name} on {self.device}")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_cfg,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        logger.info("LLM model loaded")

    def _generate(self, prompt: str, max_tokens: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

    async def _run(self, prompt: str, max_tokens: int) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._generate, prompt, max_tokens)

    async def classify_document(self, text: str) -> Dict[str, Any]:
        cached = self.cache.get(text, "classify")
        if cached:
            return cached

        prompt = (
            "<s>[INST] Classify this government document as APAR or DISCIPLINARY.\n\n"
            "APAR: officer performance evaluations, annual assessment reports, grading, pen pictures.\n"
            "DISCIPLINARY: IO Reports, PO Briefs, CO Briefs, disciplinary proceedings.\n\n"
            f"Document:\n{text[:1000]}\n\nReply with only: APAR or DISCIPLINARY [/INST]"
        )
        t0 = time.time()
        response = await self._run(prompt, 50)
        latency = (time.time() - t0) * 1000

        doc_type = "APAR" if "APAR" in response.upper() else "DISCIPLINARY"
        result = {"document_type": doc_type, "confidence": 0.85, "latency_ms": round(latency, 1)}
        self.cache.set(text, "classify", result)
        logger.info(f"Classified as {doc_type} in {latency:.0f}ms")
        return result

    async def process_apar(self, text: str) -> Dict[str, Any]:
        cached = self.cache.get(text, "apar")
        if cached:
            return cached

        fields_desc = "\n".join(f"- {f}" for f in _APAR_FIELDS)
        prompt = (
            "<s>[INST] Extract the following fields from this APAR document and return valid JSON only.\n\n"
            f"Fields:\n{fields_desc}\n\n"
            f"Document:\n{text[:2000]}\n\nReturn only JSON: [/INST]"
        )
        response = await self._run(prompt, 512)
        try:
            result = _extract_json(response)
        except Exception:
            logger.warning("APAR JSON parse failed, using fallback")
            result = _fallback(_APAR_FIELDS)

        self.cache.set(text, "apar", result)
        return result

    async def process_disciplinary(self, text: str) -> Dict[str, Any]:
        cached = self.cache.get(text, "disciplinary")
        if cached:
            return cached

        prompt = (
            "<s>[INST] Analyze this disciplinary case document and return JSON with these fields:\n"
            "- brief_background: 1-2 paragraph case summary\n"
            "- io_report_summary: key findings and recommendations from IO Report\n"
            "- po_brief_summary: prosecution arguments summary\n"
            "- co_brief_summary: commanding officer assessment summary\n\n"
            f"Document:\n{text[:2000]}\n\nReturn only JSON: [/INST]"
        )
        response = await self._run(prompt, 700)
        try:
            result = _extract_json(response)
        except Exception:
            logger.warning("Disciplinary JSON parse failed, using fallback")
            result = _fallback(_DISCIPLINARY_FIELDS)

        self.cache.set(text, "disciplinary", result)
        return result




