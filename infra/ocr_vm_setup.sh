#!/bin/bash
set -e

echo "=== OCR VM Setup (Tesla T4) ==="

# Update system
sudo apt update && sudo apt upgrade -y

# Install base packages
sudo apt install -y build-essential python3.10 python3.10-venv python3-pip git curl wget
sudo apt install -y ffmpeg poppler-utils libgl1-mesa-glx libglib2.0-0

# Install NVIDIA drivers for Tesla T4
sudo apt install -y nvidia-driver-470
sudo apt install -y nvidia-cuda-toolkit

# Verify GPU
nvidia-smi

# Create user and directories
sudo useradd -m -s /bin/bash tech || true
sudo mkdir -p /home/tech/ocr_service
sudo chown -R tech:tech /home/tech/

# Switch to tech user for Python setup
sudo -u tech bash << 'EOF'
cd /home/tech/ocr_service

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support for T4
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Clone Surya OCR
git clone https://github.com/VikParuchuri/surya.git
cd surya

# Install Surya dependencies (pinned versions)
pip install -e .
pip install transformers==4.36.2
pip install pillow==10.0.1
pip install opencv-python==4.8.1.78

# Install FastAPI stack
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-multipart==0.0.6
pip install aiofiles==23.2.1

# Freeze requirements
pip freeze > ../requirements.txt

echo "OCR VM setup complete"
EOF

# Set up systemd service
sudo tee /etc/systemd/system/ocr-service.service > /dev/null << 'EOF'
[Unit]
Description=OCR Service
After=network.target

[Service]
Type=simple
User=tech
WorkingDirectory=/home/tech/ocr_service
Environment=PATH=/home/tech/ocr_service/venv/bin
ExecStart=/home/tech/ocr_service/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ocr-service

echo "OCR VM setup complete. Reboot required for GPU drivers."
