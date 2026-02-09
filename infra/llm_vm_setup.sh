#!/bin/bash
set -e

echo "=== LLM VM Setup (Tesla T4) ==="

# Update system
sudo apt update && sudo apt upgrade -y

# Install base packages
sudo apt install -y build-essential python3.10 python3.10-venv python3-pip git curl wget

# Install NVIDIA drivers for Tesla T4
sudo apt install -y nvidia-driver-470
sudo apt install -y nvidia-cuda-toolkit

# Verify GPU
nvidia-smi

# Create user and directories
sudo useradd -m -s /bin/bash tech || true
sudo mkdir -p /home/tech/llm_service
sudo chown -R tech:tech /home/tech/

# Switch to tech user for Python setup
sudo -u tech bash << 'EOF'
cd /home/tech/llm_service

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support for T4
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install transformers and quantization
pip install transformers==4.36.2
pip install accelerate==0.24.1
pip install bitsandbytes==0.41.3
pip install sentencepiece==0.1.99

# Install FastAPI stack
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-multipart==0.0.6
pip install aiofiles==23.2.1
pip install httpx==0.25.2

# Freeze requirements
pip freeze > requirements.txt

echo "LLM VM setup complete"
EOF

# Set up systemd service
sudo tee /etc/systemd/system/llm-service.service > /dev/null << 'EOF'
[Unit]
Description=LLM Service
After=network.target

[Service]
Type=simple
User=tech
WorkingDirectory=/home/tech/llm_service
Environment=PATH=/home/tech/llm_service/venv/bin
ExecStart=/home/tech/llm_service/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable llm-service

echo "LLM VM setup complete. Reboot required for GPU drivers."
