# OCR Service - Government IDP System

## Overview
GPU-accelerated OCR service using Surya OCR for processing government documents.

## Hardware Requirements
- Tesla T4 GPU
- Ubuntu 22.04
- 16GB+ RAM
- CUDA 11.8

## Deployment

### 1. VM Setup
```bash
# Run on ocr-vm
chmod +x ../infra/ocr_vm_setup.sh
sudo ../infra/ocr_vm_setup.sh
sudo reboot
```

### 2. Code Deployment
```bash
# Clone repository
git clone <repo-url> /home/tech/ocr_service/idp-system
cd /home/tech/ocr_service/idp-system/ocr

# Activate environment
source /home/tech/ocr_service/venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Service
```bash
# Manual start
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or use systemd
sudo systemctl start ocr-service
sudo systemctl status ocr-service
```

## API Endpoints

### POST /ocr
Process PDF document and extract text.

**Request:**
- File: PDF document (multipart/form-data)

**Response:**
```json
{
  "total_pages": 20,
  "pages": [
    {
      "page_number": 1,
      "text": "extracted text content..."
    }
  ]
}
```

### GET /health
Health check endpoint.

## Performance
- Processes 15-50 page documents
- ~2-3 seconds per page on Tesla T4
- Batch processing for memory efficiency

## Troubleshooting

### GPU Issues
```bash
nvidia-smi  # Check GPU status
sudo systemctl restart ocr-service
```

### Memory Issues
- Reduce batch size in ocr_processor.py
- Monitor with `nvidia-smi`

### Surya Issues
- Check FlashAttention is disabled
- Verify transformers version compatibility