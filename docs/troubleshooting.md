# Troubleshooting

## OCR fails to start
Check GPU with `nvidia-smi`.

## LLM OOM
Restart service: `sudo systemctl restart llm-service`.

## Slow inference
Verify 4-bit quantization is active.
