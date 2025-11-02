#!/bin/bash
# Quick setup script for EC2 deployment

echo "=========================================="
echo "EC2 Face Swap Server Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt update
sudo apt install -y python3-pip git

# Install CUDA drivers (if needed)
echo "Checking CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  CUDA not found. You may need to install NVIDIA drivers."
    echo "   For GPU instances, CUDA should be pre-installed."
fi

# Clone repository
echo "Cloning repository..."
git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git
cd CIS5810_FinalProject

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements_ec2.txt

# Download models
echo "Downloading models..."
if [ -f "download_models.sh" ]; then
    bash download_models.sh
else
    echo "⚠️  download_models.sh not found. Models may need manual download."
fi

# Create systemd service (optional)
echo "Creating systemd service..."
sudo tee /etc/systemd/system/faceswap.service > /dev/null <<EOF
[Unit]
Description=Face Swap Web Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/ec2_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start server:"
echo "  python3 ec2_server.py"
echo ""
echo "Or use systemd service:"
echo "  sudo systemctl start faceswap"
echo "  sudo systemctl enable faceswap"
echo ""
echo "⚠️  Don't forget to:"
echo "  1. Configure EC2 security group (open port 5000)"
echo "  2. Set source face image path in ec2_server.py"
echo "  3. Access server at: http://YOUR-EC2-IP:5000"
echo "=========================================="

