import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OCRConfig:
    url: str = os.getenv("OCR_URL", "http://ocr-vm:8000")
    dpi: int = int(os.getenv("OCR_DPI", "300"))
    batch_size: int = int(os.getenv("OCR_BATCH_SIZE", "4"))
    timeout: float = float(os.getenv("OCR_TIMEOUT", "300.0"))


@dataclass
class LLMConfig:
    url: str = os.getenv("LLM_URL", "http://llm-vm:8001")
    model_name: str = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
    max_classify_tokens: int = 50
    max_apar_tokens: int = 512
    max_disciplinary_tokens: int = 700
    context_limit: int = 2048
    temperature: float = 0.1
    timeout: float = float(os.getenv("LLM_TIMEOUT", "300.0"))


@dataclass
class AppConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))  # seconds
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))


config = AppConfig()




