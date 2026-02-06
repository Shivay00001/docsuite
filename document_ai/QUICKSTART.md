# Quick Start Guide

Get started with Document AI Platform in 5 minutes.

## Installation

### Option 1: From Source (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/document-ai-platform
cd document-ai-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Option 2: Using pip (when published)

```bash
pip install document-ai-platform

# For GPU support
pip install document-ai-platform[gpu]
```

## First OCR

### Using CLI

```bash
# Simple OCR
docai ocr document.pdf

# With output file
docai ocr scan.jpg -o output.txt

# JSON format
docai ocr document.pdf -f json -o result.json

# With GPU
docai ocr large_doc.pdf --device cuda
```

### Using Python API

```python
from document_ai import DocumentAI

# Initialize
ai = DocumentAI()

# Process document
document, results = ai.process("document.pdf")

# Print text
for i, result in enumerate(results):
    print(f"Page {i+1}: {result.full_text}")

# Export to Word
ai.export(results, "output.docx")
```

### Quick One-Liner

```python
from document_ai import DocumentAI

# Process and export in one call
DocumentAI().quick_ocr("scan.pdf", "output.txt")
```

## Common Use Cases

### 1. Scanned PDF to Text

```bash
docai ocr scanned_document.pdf -o output.txt
```

### 2. Image to Searchable PDF

```bash
docai ocr photo.jpg -f searchable_pdf -o searchable.pdf
```

### 3. Batch Processing

```bash
# Process all PDFs in a directory
docai batch /path/to/pdfs/ -o /path/to/output/

# Multiple formats
docai batch ./scans/ -f txt -f json -f docx
```

### 4. Get Document Info

```bash
docai info document.pdf
```

### 5. Start API Server

```bash
# Start server
docai serve

# Visit http://localhost:8000/docs for API documentation
```

### 6. Using REST API

```bash
# Upload and process
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@document.pdf" \
  -F "export_format=json" > result.json

# Get job status
curl "http://localhost:8000/jobs/{job_id}"
```

## Configuration

### Basic Configuration

```python
from document_ai import DocumentAI
from document_ai.config import Config, DeviceType

config = Config()
config.ocr.device = DeviceType.CUDA  # Use GPU
config.ocr.batch_size = 16
config.ocr.detection_threshold = 0.75

ai = DocumentAI(config)
```

### Advanced Configuration

```python
from document_ai.config import Config, OCRBackend, ExportFormat

config = Config()

# OCR settings
config.ocr.backend = OCRBackend.CRAFT_CRNN
config.ocr.enable_preprocessing = True
config.ocr.enable_deskew = True
config.ocr.max_image_dimension = 2048

# Document settings
config.document.dpi_for_pdf_render = 300
config.document.max_file_size_mb = 100

# Security settings
config.security.validate_file_types = True
config.security.rate_limit_per_minute = 60
```

## Performance Tips

### 1. Use GPU if Available

```python
config.ocr.device = DeviceType.CUDA
```

### 2. Adjust Batch Size

```python
# Larger batch = faster, but more memory
config.ocr.batch_size = 32
```

### 3. Skip Preprocessing for Clean Documents

```python
result = ai.pipeline.process_image(image, enable_preprocessing=False)
```

### 4. Lower DPI for Speed

```python
config.document.dpi_for_pdf_render = 200  # Default is 300
```

## Troubleshooting

### Issue: No text detected

**Solutions:**
- Check image quality
- Enable preprocessing: `enable_preprocessing=True`
- Lower detection threshold: `config.ocr.detection_threshold = 0.5`
- Increase image DPI: `config.document.dpi_for_pdf_render = 400`

### Issue: Low confidence scores

**Solutions:**
- Use preprocessing
- Ensure good image quality
- Train custom models for your specific use case

### Issue: Out of memory

**Solutions:**
- Reduce batch size: `config.ocr.batch_size = 4`
- Process pages individually
- Reduce max image dimension: `config.ocr.max_image_dimension = 1024`

### Issue: Slow processing

**Solutions:**
- Use GPU: `config.ocr.device = DeviceType.CUDA`
- Increase batch size
- Disable unnecessary preprocessing
- Use ONNX runtime: `config.ocr.onnx_optimization = True`

## Examples

See `examples/usage_examples.py` for comprehensive examples:

```bash
python examples/usage_examples.py
```

## Next Steps

1. **Read Documentation**: Check `docs/ARCHITECTURE.md`
2. **Run Tests**: `pytest tests/ -v`
3. **Try Examples**: `python examples/usage_examples.py`
4. **Explore API**: Start server and visit `/docs`
5. **Customize**: Train models, add languages, extend pipeline

## Getting Help

- **Documentation**: `/docs` directory
- **Issues**: GitHub Issues
- **API Reference**: Start server → visit `/docs`
- **Examples**: `/examples` directory

## Model Weights

**Important**: For production use, download pretrained model weights:

```bash
# Download CRAFT detector weights
wget https://github.com/clovaai/CRAFT-pytorch/releases/download/v1.0/craft_mlt_25k.pth \
  -P ~/.document_ai/models/

# Download CRNN recognizer weights
wget https://github.com/meijieru/crnn.pytorch/releases/download/v1.0/crnn.pth \
  -P ~/.document_ai/models/
```

Or train your own models on domain-specific data.

## What's Next?

- Explore different export formats
- Try batch processing
- Set up the API server
- Customize for your use case
- Train custom models

Happy document processing! 📄✨
