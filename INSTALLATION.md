# Installation & Deployment Guide

## 🚀 Quick Install

```bash
# Navigate to project directory
cd document_ai

# Install in development mode
pip install -e .

# Verify installation
python -c "from document_ai import DocumentAI; print('✓ Installation successful')"
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB
- **CPU**: 2 cores
- **Storage**: 2GB free space

### Recommended (for GPU)
- **Python**: 3.10+
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: 11.8+
- **Storage**: 5GB free space

## 🔧 Installation Methods

### Method 1: Standard Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Method 2: GPU Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install ONNX Runtime with GPU
pip install onnxruntime-gpu

# Install package
pip install -e .
```

### Method 3: Docker Installation (Future)

```bash
# Build image
docker build -t document-ai .

# Run container
docker run -p 8000:8000 document-ai
```

## 🎯 Post-Installation Setup

### 1. Download Model Weights (Optional but Recommended)

```bash
# Create models directory
mkdir -p ~/.document_ai/models

# Download CRAFT detector
wget https://github.com/clovaai/CRAFT-pytorch/releases/download/v1.0/craft_mlt_25k.pth \
  -O ~/.document_ai/models/craft_mlt_25k.pth

# Download CRNN recognizer
wget https://github.com/meijieru/crnn.pytorch/releases/download/v1.0/crnn.pth \
  -O ~/.document_ai/models/crnn.pth
```

**Note**: The system works without these weights but with lower accuracy. For production, pretrained weights are strongly recommended.

### 2. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Run examples
python examples/usage_examples.py

# Check CLI
docai --help
```

### 3. Configure (Optional)

Create `~/.document_ai/config.yaml`:

```yaml
ocr:
  device: cuda  # or cpu
  batch_size: 16
  detection_threshold: 0.75

document:
  dpi_for_pdf_render: 300
  max_file_size_mb: 100

storage:
  base_path: ~/.document_ai
```

## 📦 Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/your-org/document-ai-platform
cd document-ai-platform

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting
black .
flake8 .
mypy .

# Run tests with coverage
pytest tests/ --cov=document_ai --cov-report=html
```

## 🌐 API Server Deployment

### Local Development

```bash
# Start server
docai serve

# Visit http://localhost:8000/docs
```

### Production Deployment

#### Using Uvicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn api.rest.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

#### Using Systemd (Linux)

Create `/etc/systemd/system/document-ai.service`:

```ini
[Unit]
Description=Document AI API Server
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/document-ai
Environment="PATH=/opt/document-ai/venv/bin"
ExecStart=/opt/document-ai/venv/bin/gunicorn \
  api.rest.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

Start service:

```bash
sudo systemctl enable document-ai
sudo systemctl start document-ai
sudo systemctl status document-ai
```

#### Using Nginx Reverse Proxy

`/etc/nginx/sites-available/document-ai`:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Increase timeouts for large files
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    # Increase upload size
    client_max_body_size 100M;
}
```

## 🐳 Docker Deployment (Advanced)

### Dockerfile

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api.rest.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.document_ai/models
      - ./output:/root/.document_ai/output
    environment:
      - DEVICE=cpu
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

## 🔍 Troubleshooting

### Issue: Import Error

```bash
# Solution: Install package in editable mode
pip install -e .
```

### Issue: CUDA Not Available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: PDF Processing Fails

```bash
# Install Poppler (required for pdf2image)
# Ubuntu/Debian:
sudo apt-get install poppler-utils

# macOS:
brew install poppler

# Windows: Download from http://blog.alivate.com.au/poppler-windows/
```

### Issue: Out of Memory

```python
# Reduce batch size in config
config.ocr.batch_size = 4
config.ocr.max_image_dimension = 1024
```

## 🧪 Testing Installation

```bash
# Unit tests
pytest tests/test_all.py::TestImagePreprocessor -v

# Integration tests
pytest tests/test_all.py::TestIntegration -v

# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=document_ai --cov-report=term-missing
```

## 📊 Performance Benchmarking

```bash
# Run benchmark suite (future)
python benchmarks/run_benchmarks.py

# Profile specific operation
python -m cProfile -o profile.stats examples/usage_examples.py
python -m pstats profile.stats
```

## 🔐 Security Hardening

### Production Checklist

- [ ] Use HTTPS (TLS/SSL certificates)
- [ ] Enable rate limiting
- [ ] Set up file size limits
- [ ] Configure CORS properly
- [ ] Use environment variables for secrets
- [ ] Run as non-root user
- [ ] Keep dependencies updated
- [ ] Enable security headers
- [ ] Set up monitoring and alerts

### Example Security Configuration

```python
from document_ai.config import Config

config = Config()

# Security settings
config.security.validate_file_types = True
config.security.max_file_size_mb = 50
config.security.rate_limit_per_minute = 30
config.security.scan_for_malware = True  # If scanner available

# API settings
config.api.enable_cors = False  # Or configure allowed origins
config.api.max_upload_size_mb = 50
```

## 📚 Next Steps

After installation:

1. **Try Quick Examples**: `python examples/usage_examples.py`
2. **Read Documentation**: Check `docs/ARCHITECTURE.md`
3. **Run Tests**: `pytest tests/ -v`
4. **Start API Server**: `docai serve`
5. **Process First Document**: `docai ocr your_document.pdf`

## 💬 Support

- **Issues**: GitHub Issues
- **Documentation**: `/docs` directory
- **Examples**: `/examples` directory
- **API Docs**: http://localhost:8000/docs (when server running)

---

**Ready to process documents! 📄✨**
