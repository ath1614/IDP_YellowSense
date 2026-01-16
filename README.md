# Government IDP System

Production-grade Intelligent Document Processing system for government documents using GPU-accelerated OCR and LLM inference.

## Architecture

```
User (Local) → OCR VM (Surya OCR) → LLM VM (Mistral-7B) → Structured Output
```

## Document Types

### APAR Documents
- 15-50 page scanned PDFs
- Officer performance evaluations
- Structured JSON extraction

### Disciplinary Documents  
- IO Reports, PO Briefs, CO Briefs
- Multi-document folders
- Summary generation

## Infrastructure

### VM Requirements
- **ocr-vm**: Tesla T4, Ubuntu 22.04, 16GB RAM
- **llm-vm**: Tesla T4, Ubuntu 22.04, 16GB RAM

## Quick Deployment

### 1. VM Setup
```bash
# OCR VM
scp infra/ocr_vm_setup.sh user@ocr-vm:~/
ssh user@ocr-vm
chmod +x ocr_vm_setup.sh && sudo ./ocr_vm_setup.sh
sudo reboot

# LLM VM  
scp infra/llm_vm_setup.sh user@llm-vm:~/
ssh user@llm-vm
chmod +x llm_vm_setup.sh && sudo ./llm_vm_setup.sh
sudo reboot
```

### 2. Code Deployment
```bash
# OCR VM
ssh tech@ocr-vm
git clone <repo> /home/tech/ocr_service/idp-system
cd /home/tech/ocr_service/idp-system/ocr
source /home/tech/ocr_service/venv/bin/activate
pip install -r requirements.txt
sudo systemctl start ocr-service

# LLM VM
ssh tech@llm-vm  
git clone <repo> /home/tech/llm_service/idp-system
cd /home/tech/llm_service/idp-system/llm
source /home/tech/llm_service/venv/bin/activate
pip install -r requirements.txt
sudo systemctl start llm-service
```

### 3. Test Pipeline
```bash
# Test OCR service
curl -X POST "http://ocr-vm:8000/ocr" -F "file=@test.pdf"

# Test LLM service
curl -X POST "http://llm-vm:8001/classify" -H "Content-Type: application/json" -d '{"text":"test document"}'

# Full pipeline test
python orchestrator.py
```

## API Usage

### Complete Document Processing
```python
from orchestrator import IDPOrchestrator

orchestrator = IDPOrchestrator(
    ocr_url="http://ocr-vm:8000",
    llm_url="http://llm-vm:8001"
)

with open("document.pdf", "rb") as f:
    result = await orchestrator.process_document(f)
    
print(result["structured_data"])
```

## Performance Benchmarks

### OCR Service (Tesla T4)
- **Throughput**: 2-3 pages/second
- **Memory**: 8GB VRAM peak
- **Batch Size**: 4 pages optimal

### LLM Service (Tesla T4)  
- **Inference**: 5-10 seconds/request
- **Memory**: 8GB VRAM (4-bit quantized)
- **Context**: 2048 tokens max

## Production Checklist

### Pre-deployment
- [ ] GPU drivers installed (nvidia-driver-470)
- [ ] CUDA toolkit verified (nvidia-smi)
- [ ] Network connectivity between VMs
- [ ] Firewall rules configured (ports 8000, 8001)

### Post-deployment
- [ ] Services running (systemctl status)
- [ ] Health checks passing (/health endpoints)
- [ ] Log monitoring configured
- [ ] GPU utilization monitoring

## Troubleshooting

### Common Issues

**OCR Service Won't Start**
```bash
# Check GPU
nvidia-smi
# Check Surya installation
cd /home/tech/ocr_service/surya && python -c "import surya"
# Check logs
journalctl -u ocr-service -f
```

**LLM Service OOM**
```bash
# Check VRAM usage
nvidia-smi
# Restart service
sudo systemctl restart llm-service
```

**Network Issues**
```bash
# Test connectivity
curl http://ocr-vm:8000/health
curl http://llm-vm:8001/health
```

## Security Notes

- Services run as non-root user (tech)
- No external internet access required post-setup
- Internal network communication only
- Document data never leaves VM infrastructure

## Monitoring

### Key Metrics
- GPU utilization (nvidia-smi)
- Memory usage (free -h)
- Service uptime (systemctl status)
- Request latency (application logs)

### Log Locations
- OCR Service: `journalctl -u ocr-service`
- LLM Service: `journalctl -u llm-service`
- Application: `/home/tech/*/app.log`

