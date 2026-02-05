# LLM Service - Government IDP System

## Overview
Quantized LLM inference service for document classification and structured data extraction.

## Hardware Requirements
- Tesla T4 GPU
- Ubuntu 22.04
- 16GB+ RAM
- CUDA 11.8

## Deployment

### 1. VM Setup
```bash
# Run on llm-vm
chmod +x ../infra/llm_vm_setup.sh
sudo ../infra/llm_vm_setup.sh
sudo reboot
```

### 2. Code Deployment
```bash
# Clone repository
git clone <repo-url> /home/tech/llm_service/idp-system
cd /home/tech/llm_service/idp-system/llm

# Activate environment
source /home/tech/llm_service/venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Service
```bash
# Manual start
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

# Or use systemd
sudo systemctl start llm-service
sudo systemctl status llm-service
```

## API Endpoints

### POST /classify
Classify document type.

**Request:**
```json
{"text": "document text..."}
```

**Response:**
```json
{
  "document_type": "APAR",
  "confidence": 0.85
}
```

### POST /apar
Extract structured data from APAR documents.

**Request:**
```json
{"text": "apar document text..."}
```

**Response:**
```json
{
  "officer_name": "John Doe",
  "date_of_birth": "01/01/1980",
  "apar_year": "2023",
  "reporting_authority": "Authority Name",
  "reviewing_authority": "Review Authority",
  "accepting_authority": "Accept Authority",
  "overall_grading": "Outstanding",
  "pen_picture": "Performance summary..."
}
```

### POST /disciplinary
Generate summaries for disciplinary documents.

**Request:**
```json
{"text": "disciplinary document text..."}
```

**Response:**
```json
{
  "brief_background": "Case background summary...",
  "io_report_summary": "Investigation findings...",
  "po_brief_summary": "Prosecution arguments...",
  "co_brief_summary": "CO assessment..."
}
```

## Model Details
- **Model**: Mistral-7B-Instruct-v0.1
- **Quantization**: 4-bit (BitsAndBytesConfig)
- **Memory**: ~8GB VRAM usage
- **Inference**: ~5-10 seconds per request

## Troubleshooting

### GPU Memory Issues
```bash
nvidia-smi  # Check VRAM usage
# Restart service if OOM
sudo systemctl restart llm-service
```

### Model Loading Issues
- Check internet connection for model download
- Verify HuggingFace cache: ~/.cache/huggingface/
- Clear cache if corrupted: `rm -rf ~/.cache/huggingface/`