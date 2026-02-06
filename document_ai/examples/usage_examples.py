"""
Example Usage Scripts
Demonstrates various ways to use the Document AI Platform
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from document_ai.sdk import DocumentAI
from document_ai.config import Config, DeviceType, ExportFormat
from document_ai.utils.logging import LoggerConfig


def example_1_basic_ocr():
    """Example 1: Basic OCR of a single document"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic OCR")
    print("="*60)
    
    # Initialize
    ai = DocumentAI()
    
    # Process document (replace with actual file path)
    # document, results = ai.process("path/to/document.pdf")
    
    # For demo purposes, create a test image
    import numpy as np
    import cv2
    
    # Create test image with text
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Hello Document AI!", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Save temporarily
    test_path = Path("/tmp/test_document.png")
    cv2.imwrite(str(test_path), img)
    
    # Process
    document, results = ai.process(test_path)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nPage {i+1}:")
        print(f"  Text: {result.full_text}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Regions detected: {result.num_detected_regions}")
    
    # Clean up
    test_path.unlink()
    print("\n✓ Example 1 completed")


def example_2_custom_configuration():
    """Example 2: Using custom configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    # Create custom config
    config = Config()
    config.ocr.device = DeviceType.CPU
    config.ocr.batch_size = 16
    config.ocr.detection_threshold = 0.8
    config.ocr.enable_preprocessing = True
    config.ocr.enable_deskew = True
    
    print(f"Configuration:")
    print(f"  Device: {config.ocr.device.value}")
    print(f"  Batch size: {config.ocr.batch_size}")
    print(f"  Detection threshold: {config.ocr.detection_threshold}")
    
    # Initialize with config
    ai = DocumentAI(config)
    
    print("\n✓ Example 2 completed")


def example_3_multiple_export_formats():
    """Example 3: Exporting to multiple formats"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multiple Export Formats")
    print("="*60)
    
    import numpy as np
    import cv2
    import tempfile
    
    # Initialize
    ai = DocumentAI()
    
    # Create test document
    img = np.ones((300, 800, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Export Format Demo", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "Multiple formats supported!", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    test_path = Path("/tmp/test_export.png")
    cv2.imwrite(str(test_path), img)
    
    # Process
    document, results = ai.process(test_path)
    
    # Export to different formats
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        formats = [
            (ExportFormat.TXT, "output.txt"),
            (ExportFormat.JSON, "output.json"),
            (ExportFormat.CSV, "output.csv"),
            (ExportFormat.DOCX, "output.docx"),
        ]
        
        for fmt, filename in formats:
            output_path = tmpdir / filename
            result_path = ai.export(results, output_path, fmt, document)
            print(f"  ✓ Exported to {fmt.value}: {result_path.name}")
    
    # Clean up
    test_path.unlink()
    print("\n✓ Example 3 completed")


def example_4_batch_processing():
    """Example 4: Batch processing multiple documents"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing")
    print("="*60)
    
    import numpy as np
    import cv2
    import tempfile
    
    # Initialize
    ai = DocumentAI()
    
    # Create multiple test documents
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test files
        test_files = []
        for i in range(3):
            img = np.ones((200, 600, 3), dtype=np.uint8) * 255
            cv2.putText(img, f"Document {i+1}", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            
            file_path = tmpdir / f"doc_{i+1}.png"
            cv2.imwrite(str(file_path), img)
            test_files.append(file_path)
        
        # Batch process
        print(f"Processing {len(test_files)} documents...")
        results = ai.process_batch(test_files)
        
        # Display results
        for i, (doc, ocr_results) in enumerate(results):
            if doc is not None:
                print(f"\n  Document {i+1}:")
                print(f"    Pages: {doc.total_pages}")
                print(f"    Text: {ocr_results[0].full_text if ocr_results else 'N/A'}")
    
    print("\n✓ Example 4 completed")


def example_5_quick_ocr():
    """Example 5: Quick OCR shortcut"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Quick OCR")
    print("="*60)
    
    import numpy as np
    import cv2
    import tempfile
    
    # Initialize
    ai = DocumentAI()
    
    # Create test document
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Quick OCR Test", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    test_path = Path("/tmp/quick_test.png")
    cv2.imwrite(str(test_path), img)
    
    # Quick OCR (process + export in one call)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.txt"
        result_path = ai.quick_ocr(test_path, output_path)
        print(f"  ✓ OCR completed: {result_path}")
        
        # Show content
        with open(result_path) as f:
            print(f"\n  Content:\n{f.read()}")
    
    # Clean up
    test_path.unlink()
    print("\n✓ Example 5 completed")


def example_6_programmatic_api():
    """Example 6: Using components directly"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Programmatic API")
    print("="*60)
    
    from document_ai.core.ocr.pipeline import OCRPipeline
    from document_ai.core.ocr.preprocessor.image_preprocessor import ImagePreprocessor
    import numpy as np
    import cv2
    
    # Create test image
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Direct API Usage", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Use preprocessor directly
    print("\n  Using preprocessor...")
    preprocessor = ImagePreprocessor()
    preprocessed = preprocessor.process(img)
    print(f"    Transformations: {preprocessed.transformations_applied}")
    
    # Use OCR pipeline directly
    print("\n  Using OCR pipeline...")
    pipeline = OCRPipeline()
    result = pipeline.process_image(img)
    print(f"    Detected text: {result.full_text}")
    print(f"    Confidence: {result.confidence:.2%}")
    
    print("\n✓ Example 6 completed")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print("DOCUMENT AI PLATFORM - USAGE EXAMPLES")
    print("="*70)
    
    # Setup logging
    LoggerConfig.setup(log_level="INFO")
    
    try:
        example_1_basic_ocr()
        example_2_custom_configuration()
        example_3_multiple_export_formats()
        example_4_batch_processing()
        example_5_quick_ocr()
        example_6_programmatic_api()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nFor more information:")
        print("  • Documentation: docs/")
        print("  • Architecture: docs/ARCHITECTURE.md")
        print("  • CLI help: python -m document_ai.main --help")
        print("  • API docs: Start server and visit /docs")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
