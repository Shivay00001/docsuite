# Architecture Overview

## Design Principles

### 1. Layered Architecture
The system follows a strict layered architecture:

```
┌─────────────────────────────────────┐
│         API Layer                   │  (REST, CLI)
├─────────────────────────────────────┤
│      Application Layer              │  (Orchestration)
├─────────────────────────────────────┤
│        Core Layer                   │  (OCR Engine)
├─────────────────────────────────────┤
│      Document Layer                 │  (I/O Operations)
├─────────────────────────────────────┤
│     Infrastructure Layer            │  (Storage, Logging)
└─────────────────────────────────────┘
```

### 2. Dependency Rule
- Higher layers depend on lower layers
- Lower layers never depend on higher layers
- Core business logic has no external dependencies

### 3. Single Responsibility
Each module has one clear purpose:
- `preprocessor/`: Image preprocessing only
- `detector/`: Text detection only
- `recognizer/`: Text recognition only
- `pipeline.py`: Orchestration only

## Component Details

### Core OCR Engine

#### Preprocessor
**Purpose**: Prepare images for optimal OCR performance

**Key Operations**:
1. Grayscale conversion
2. Noise reduction (bilateral filter, non-local means)
3. Deskewing (projection profile analysis)
4. Contrast enhancement (CLAHE)
5. Binarization (adaptive thresholding)
6. Morphological operations

**Design Decision**: Pure OpenCV implementation
- No external dependencies
- Deterministic results
- Full control over operations

#### Text Detector (CRAFT)
**Purpose**: Locate text regions in images

**Architecture**:
```
Input Image (H×W×3)
    ↓
VGG16 Feature Extractor
    ↓
Feature Pyramid Network (FPN)
    ↓
Dual-Head Prediction
    ├→ Region Score Map
    └→ Affinity Score Map
    ↓
Connected Components Analysis
    ↓
Bounding Boxes
```

**Why CRAFT**:
- Character-level detection (more precise)
- Handles rotated/curved text
- Language-agnostic
- State-of-the-art accuracy

**Post-Processing**:
1. Threshold region/affinity scores
2. Find connected components
3. Filter by size/confidence
4. Merge nearby regions
5. Order by reading sequence

#### Text Recognizer (CRNN)
**Purpose**: Convert image crops to text

**Architecture**:
```
Text Region Image (32×W×1)
    ↓
CNN Feature Extraction
    ├→ Conv layers (64, 128, 256, 512 channels)
    ├→ Max pooling
    └→ Batch normalization
    ↓
Sequence Modeling
    ├→ Bidirectional LSTM (256 hidden units)
    └→ Attention mechanism
    ↓
CTC Decoder
    ↓
Text Output
```

**Why CRNN**:
- No character segmentation needed
- Variable-length output
- End-to-end trainable
- Fast inference

**CTC Decoding**:
- Handles duplicate predictions
- Removes blank tokens
- Collapses repeated characters
- Greedy best-path decoding

#### OCR Pipeline
**Purpose**: Orchestrate end-to-end OCR flow

**Flow**:
```
1. Input Validation
2. Preprocessing (optional)
3. Text Detection
4. Region Cropping & Normalization
5. Batch Recognition
6. Text Assembly
7. Result Formatting
```

**Batch Processing**:
- Groups regions into batches
- Parallel GPU inference
- Memory-efficient pagination
- Progress tracking

### Document Processing

#### Document Loader
**Purpose**: Universal document input

**Supported Formats**:
- **PDF**: Multi-page, metadata extraction
- **Images**: PNG, JPG, TIFF, BMP, WebP
- **Bytes**: In-memory processing
- **Streams**: File-like objects

**Processing**:
```
File Input
    ↓
Type Detection
    ↓
Security Validation
    ├→ File size check
    ├→ Format verification
    └→ Malware scan (optional)
    ↓
Format-Specific Loading
    ├→ PDF: PyMuPDF + pdf2image
    └→ Images: OpenCV + PIL
    ↓
Document Object
```

#### Document Exporter
**Purpose**: Multi-format output generation

**Export Formats**:

1. **TXT**: Plain text with page breaks
2. **JSON**: Structured data with bounding boxes
3. **CSV**: Tabular format (one row per region)
4. **DOCX**: Microsoft Word with formatting
5. **PDF**: Text-only PDF
6. **Searchable PDF**: Image + invisible text layer

**Searchable PDF Generation**:
```
Original Image
    ↓
Embed as Page Background
    ↓
Overlay Invisible Text
    ├→ Position: OCR bounding boxes
    ├→ Font size: Match region height
    └→ Color: White (transparent)
    ↓
Searchable PDF
```

### API Layer

#### REST API (FastAPI)
**Endpoints**:
- `POST /ocr`: Single document processing
- `POST /batch`: Multiple documents
- `GET /jobs/{id}`: Job status
- `GET /download/{id}`: Result download
- `DELETE /jobs/{id}`: Cleanup

**Features**:
- Async request handling
- File upload validation
- Job queuing (extensible to Celery/RQ)
- CORS support
- OpenAPI documentation

#### CLI (Click)
**Commands**:
- `docai ocr`: Single document OCR
- `docai batch`: Batch processing
- `docai info`: Document metadata
- `docai serve`: Start API server
- `docai version`: Version info

**Design**:
- Composable commands
- Rich progress indicators
- Colorized output
- Comprehensive help text

## Data Flow

### Single Document OCR
```
User Input
    ↓
DocumentLoader
    ├→ Validate
    ├→ Load
    └→ Parse pages
    ↓
For each page:
    ↓
ImagePreprocessor
    ├→ Resize
    ├→ Denoise
    ├→ Deskew
    └→ Binarize
    ↓
CRAFTDetector
    ├→ Feature extraction
    ├→ Score map generation
    └→ Box extraction
    ↓
CRNNRecognizer (batch)
    ├→ Region cropping
    ├→ Normalization
    ├→ Batch inference
    └→ CTC decoding
    ↓
Text Assembly
    ├→ Reading order sort
    ├→ Line grouping
    └→ Confidence aggregation
    ↓
DocumentExporter
    └→ Format conversion
    ↓
Output File(s)
```

## Error Handling Strategy

### Exception Hierarchy
```
DocumentAIException (base)
├── OCRException
│   ├── DetectionException
│   ├── RecognitionException
│   └── PreprocessingException
├── DocumentException
│   ├── DocumentLoadException
│   └── InvalidDocumentException
├── StorageException
└── SecurityException
```

### Error Recovery
1. **Detection Fails**: Return empty result, log warning
2. **Recognition Fails**: Skip region, continue processing
3. **Preprocessing Fails**: Use original image
4. **Export Fails**: Raise exception, preserve intermediate results

### Logging Strategy
- **DEBUG**: Detailed processing steps
- **INFO**: Major operations, progress
- **WARNING**: Recoverable issues
- **ERROR**: Processing failures
- **CRITICAL**: System failures

## Performance Optimizations

### 1. Batch Processing
- Group regions for GPU efficiency
- Minimize CPU↔GPU transfers
- Pipeline parallelization

### 2. Memory Management
- Stream large PDFs page-by-page
- Release intermediate results
- Configurable cache limits

### 3. Model Optimization
- ONNX runtime support
- FP16 inference (GPU)
- Model quantization (optional)

### 4. Caching
- Preprocessed image cache
- Model weight cache
- Result caching (configurable)

## Security Considerations

### 1. Input Validation
```python
validate_file_size()
validate_file_type()
validate_file_content()
sanitize_filename()
```

### 2. Resource Limits
- Maximum file size
- Maximum processing time
- Maximum memory usage
- Rate limiting (API)

### 3. Sandboxing
- No code execution from documents
- Restricted file system access
- Network isolation (optional)

## Extensibility Points

### 1. Custom Detectors
```python
class CustomDetector(TextDetector):
    def detect(self, image: np.ndarray) -> List[TextBox]:
        # Your implementation
        pass
```

### 2. Custom Recognizers
```python
class CustomRecognizer(TextRecognizer):
    def recognize(self, image: np.ndarray) -> RecognitionResult:
        # Your implementation
        pass
```

### 3. Custom Exporters
```python
class CustomExporter:
    def export(self, results, path, format):
        # Your implementation
        pass
```

### 4. Plugin System (Future)
```python
from document_ai.plugins import register_plugin

@register_plugin("table_detector")
class TableDetector:
    pass
```

## Testing Strategy

### 1. Unit Tests
- Individual component testing
- Mock external dependencies
- Edge case coverage

### 2. Integration Tests
- End-to-end workflows
- Real file processing
- Multi-component interactions

### 3. Performance Tests
- Throughput benchmarks
- Memory profiling
- Latency measurements

### 4. Security Tests
- Input fuzzing
- Resource exhaustion
- Injection attacks

## Deployment Architecture

### Single Server
```
┌─────────────────┐
│  Load Balancer  │
└────────┬────────┘
         │
    ┌────┴────┐
    │  Nginx  │
    └────┬────┘
         │
┌────────┴─────────┐
│  FastAPI Server  │
│  ┌─────────────┐ │
│  │ OCR Workers │ │
│  └─────────────┘ │
└──────────────────┘
```

### Distributed System
```
┌─────────────┐
│   API GW    │
└──────┬──────┘
       │
┌──────┴───────┐
│  Job Queue   │  (Redis/RabbitMQ)
└──────┬───────┘
       │
┌──────┴────────────────┐
│   Worker Pool         │
│  ┌────┐ ┌────┐ ┌────┐│
│  │GPU1│ │GPU2│ │GPU3││
│  └────┘ └────┘ └────┘│
└───────────────────────┘
       │
┌──────┴──────┐
│   Storage   │  (S3/MinIO)
└─────────────┘
```

## Monitoring & Observability

### Metrics
- Requests per second
- Processing latency (p50, p95, p99)
- Error rates
- GPU utilization
- Memory usage

### Tracing
- Request ID propagation
- Distributed tracing (OpenTelemetry)
- Performance profiling

### Logging
- Structured JSON logs
- Centralized aggregation (ELK/Loki)
- Alert rules

## Configuration Management

### Environment-Based
```python
# Development
DEBUG=True
LOG_LEVEL=DEBUG
DEVICE=cpu

# Production
DEBUG=False
LOG_LEVEL=INFO
DEVICE=cuda
```

### File-Based
```yaml
# config/production.yaml
ocr:
  device: cuda
  batch_size: 32
  detection_threshold: 0.75

api:
  workers: 8
  max_upload_size_mb: 200
```

## Future Enhancements

1. **Transformer Models**: Replace CRNN with TrOCR
2. **Layout Analysis**: Deep learning-based layout detection
3. **Table Understanding**: Structure-aware table extraction
4. **Multi-Modal**: Combine text + vision for better understanding
5. **Active Learning**: User feedback loop for model improvement
6. **Distributed Training**: Train custom models at scale
7. **Model Compression**: Edge deployment optimization
8. **Streaming Processing**: Real-time video OCR

---

This architecture is designed for:
- **Maintainability**: Clear separation of concerns
- **Testability**: Every component is independently testable
- **Extensibility**: Easy to add new features
- **Performance**: Optimized for production workloads
- **Reliability**: Comprehensive error handling
