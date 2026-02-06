"""
Test Suite for Document AI Platform
Comprehensive tests for all components
"""
import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil

from document_ai.core.ocr.preprocessor.image_preprocessor import ImagePreprocessor, PreprocessingMode
from document_ai.core.ocr.detector.craft_detector import CRAFTDetector, TextBox
from document_ai.core.ocr.recognizer.crnn_recognizer import CRNNRecognizer
from document_ai.core.ocr.pipeline import OCRPipeline
from document_ai.document.loader.document_loader import DocumentLoader
from document_ai.document.exporter.document_exporter import DocumentExporter
from document_ai.config import Config, ExportFormat
from document_ai.exceptions import *


# Fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create white image with black text
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


@pytest.fixture
def sample_document(temp_dir, sample_image):
    """Create a sample document file"""
    img_path = temp_dir / "test.png"
    cv2.imwrite(str(img_path), sample_image)
    return img_path


# Preprocessing Tests
class TestImagePreprocessor:
    """Test image preprocessing pipeline"""
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = ImagePreprocessor()
        assert preprocessor is not None
        assert preprocessor.target_size == 2048
    
    def test_process_image(self, sample_image):
        """Test basic image processing"""
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(sample_image)
        
        assert result.image is not None
        assert result.original_shape == sample_image.shape[:2]
        assert len(result.transformations_applied) > 0
    
    def test_resize(self, sample_image):
        """Test image resizing"""
        preprocessor = ImagePreprocessor(target_size=500)
        result = preprocessor.process(sample_image)
        
        h, w = result.final_shape
        assert max(h, w) <= 500
    
    def test_deskew(self, sample_image):
        """Test deskewing"""
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(sample_image, enable_deskew=True)
        
        assert result.skew_angle is not None
        assert "deskew" in str(result.transformations_applied).lower()
    
    def test_different_modes(self, sample_image):
        """Test different preprocessing modes"""
        for mode in [PreprocessingMode.LIGHT, PreprocessingMode.STANDARD, PreprocessingMode.AGGRESSIVE]:
            preprocessor = ImagePreprocessor(mode=mode)
            result = preprocessor.process(sample_image)
            assert result.image is not None
    
    def test_batch_processing(self, sample_image):
        """Test batch processing"""
        preprocessor = ImagePreprocessor()
        images = [sample_image, sample_image.copy(), sample_image.copy()]
        
        results = preprocessor.preprocess_batch(images)
        
        assert len(results) == 3
        for result in results:
            assert result.image is not None


# Detector Tests
class TestCRAFTDetector:
    """Test CRAFT text detector"""
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = CRAFTDetector(device='cpu')
        assert detector is not None
        assert detector.device == 'cpu'
    
    def test_detect_text(self, sample_image):
        """Test text detection"""
        detector = CRAFTDetector(device='cpu')
        boxes = detector.detect(sample_image)
        
        assert isinstance(boxes, list)
        # May or may not detect text depending on model weights
        for box in boxes:
            assert isinstance(box, TextBox)
            assert box.confidence >= 0.0
            assert box.confidence <= 1.0
    
    def test_detection_threshold(self, sample_image):
        """Test detection with different thresholds"""
        detector_low = CRAFTDetector(device='cpu', text_threshold=0.3)
        detector_high = CRAFTDetector(device='cpu', text_threshold=0.9)
        
        boxes_low = detector_low.detect(sample_image)
        boxes_high = detector_high.detect(sample_image)
        
        # Higher threshold should detect fewer or equal boxes
        assert len(boxes_high) <= len(boxes_low)


# Recognizer Tests
class TestCRNNRecognizer:
    """Test CRNN text recognizer"""
    
    def test_initialization(self):
        """Test recognizer initialization"""
        recognizer = CRNNRecognizer(device='cpu')
        assert recognizer is not None
    
    def test_recognize_text(self, sample_image):
        """Test text recognition"""
        recognizer = CRNNRecognizer(device='cpu')
        result = recognizer.recognize(sample_image)
        
        assert result.text is not None
        assert isinstance(result.text, str)
        assert 0.0 <= result.confidence <= 1.0
    
    def test_batch_recognition(self, sample_image):
        """Test batch recognition"""
        recognizer = CRNNRecognizer(device='cpu')
        images = [sample_image, sample_image.copy()]
        
        results = recognizer.recognize_batch(images)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result.text, str)


# Pipeline Tests
class TestOCRPipeline:
    """Test complete OCR pipeline"""
    
    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = OCRPipeline()
        assert pipeline is not None
        assert pipeline.preprocessor is not None
        assert pipeline.detector is not None
        assert pipeline.recognizer is not None
    
    def test_process_image(self, sample_image):
        """Test end-to-end OCR"""
        pipeline = OCRPipeline()
        result = pipeline.process_image(sample_image)
        
        assert result is not None
        assert result.full_text is not None
        assert result.confidence >= 0.0
        assert result.preprocessing_info is not None
    
    def test_process_with_options(self, sample_image):
        """Test OCR with different options"""
        pipeline = OCRPipeline()
        
        # With preprocessing
        result1 = pipeline.process_image(sample_image, enable_preprocessing=True)
        
        # Without preprocessing
        result2 = pipeline.process_image(sample_image, enable_preprocessing=False)
        
        assert result1 is not None
        assert result2 is not None
    
    def test_batch_processing(self, sample_image):
        """Test batch OCR"""
        pipeline = OCRPipeline()
        images = [sample_image, sample_image.copy()]
        
        results = pipeline.process_batch(images)
        
        assert len(results) == 2
        for result in results:
            assert result.full_text is not None


# Document Loader Tests
class TestDocumentLoader:
    """Test document loading"""
    
    def test_initialization(self):
        """Test loader initialization"""
        loader = DocumentLoader()
        assert loader is not None
    
    def test_load_image(self, sample_document):
        """Test loading image file"""
        loader = DocumentLoader()
        document = loader.load(sample_document)
        
        assert document is not None
        assert document.total_pages == 1
        assert len(document.pages) == 1
        assert document.pages[0].image is not None
    
    def test_validate_document(self, sample_document):
        """Test document validation"""
        loader = DocumentLoader()
        document = loader.load(sample_document)
        
        assert loader.validate_document(document) is True
    
    def test_unsupported_format(self, temp_dir):
        """Test error on unsupported format"""
        fake_file = temp_dir / "test.xyz"
        fake_file.write_text("test")
        
        loader = DocumentLoader()
        with pytest.raises(UnsupportedFormatException):
            loader.load(fake_file)


# Document Exporter Tests
class TestDocumentExporter:
    """Test document export"""
    
    def test_initialization(self, temp_dir):
        """Test exporter initialization"""
        exporter = DocumentExporter(output_dir=temp_dir)
        assert exporter is not None
        assert exporter.output_dir.exists()
    
    def test_export_txt(self, temp_dir, sample_image):
        """Test TXT export"""
        pipeline = OCRPipeline()
        result = pipeline.process_image(sample_image)
        
        exporter = DocumentExporter(output_dir=temp_dir)
        output_path = temp_dir / "output.txt"
        
        exported = exporter.export([result], output_path, ExportFormat.TXT)
        
        assert exported.exists()
        content = exported.read_text()
        assert isinstance(content, str)
    
    def test_export_json(self, temp_dir, sample_image):
        """Test JSON export"""
        pipeline = OCRPipeline()
        result = pipeline.process_image(sample_image)
        
        exporter = DocumentExporter(output_dir=temp_dir)
        output_path = temp_dir / "output.json"
        
        exported = exporter.export([result], output_path, ExportFormat.JSON)
        
        assert exported.exists()
        
        import json
        with open(exported) as f:
            data = json.load(f)
        
        assert "total_pages" in data
        assert "pages" in data
    
    def test_export_csv(self, temp_dir, sample_image):
        """Test CSV export"""
        pipeline = OCRPipeline()
        result = pipeline.process_image(sample_image)
        
        exporter = DocumentExporter(output_dir=temp_dir)
        output_path = temp_dir / "output.csv"
        
        exported = exporter.export([result], output_path, ExportFormat.CSV)
        
        assert exported.exists()
    
    def test_export_docx(self, temp_dir, sample_image):
        """Test DOCX export"""
        pipeline = OCRPipeline()
        result = pipeline.process_image(sample_image)
        
        exporter = DocumentExporter(output_dir=temp_dir)
        output_path = temp_dir / "output.docx"
        
        exported = exporter.export([result], output_path, ExportFormat.DOCX)
        
        assert exported.exists()


# Configuration Tests
class TestConfiguration:
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = Config()
        
        assert config.ocr is not None
        assert config.document is not None
        assert config.storage is not None
        assert config.security is not None
        assert config.api is not None
    
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "ocr" in config_dict
        assert "document" in config_dict
    
    def test_config_from_dict(self):
        """Test configuration deserialization"""
        config_dict = {
            "ocr": {"batch_size": 16},
            "debug": True,
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.ocr.batch_size == 16
        assert config.debug is True


# Exception Tests
class TestExceptions:
    """Test exception handling"""
    
    def test_exception_hierarchy(self):
        """Test exception inheritance"""
        assert issubclass(OCRException, DocumentAIException)
        assert issubclass(DetectionException, OCRException)
        assert issubclass(DocumentException, DocumentAIException)
    
    def test_exception_details(self):
        """Test exception with details"""
        exc = DocumentAIException("Test error", {"key": "value"})
        
        assert str(exc) == "Test error (key=value)"
        assert exc.details == {"key": "value"}


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_end_to_end_ocr(self, sample_document, temp_dir):
        """Test complete OCR workflow"""
        # Load document
        loader = DocumentLoader()
        document = loader.load(sample_document)
        
        # Process with OCR
        pipeline = OCRPipeline()
        ocr_results = []
        for page in document.pages:
            result = pipeline.process_image(page.image)
            ocr_results.append(result)
        
        # Export results
        exporter = DocumentExporter(output_dir=temp_dir)
        
        for fmt in [ExportFormat.TXT, ExportFormat.JSON, ExportFormat.CSV]:
            output_path = temp_dir / f"output.{fmt.value}"
            exported = exporter.export(ocr_results, output_path, fmt, document)
            assert exported.exists()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
