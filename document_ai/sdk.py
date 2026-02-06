"""
Document AI Platform - Main Entry Point
Simple interface for common operations
"""
import sys
from pathlib import Path
from typing import Optional, List, Union

from document_ai.utils.logging import LoggerConfig, get_logger
from document_ai.config import Config, ExportFormat, DeviceType
from document_ai.document.loader.document_loader import DocumentLoader
from document_ai.core.ocr.pipeline import OCRPipeline
from document_ai.document.exporter.document_exporter import DocumentExporter
from document_ai.exceptions import DocumentAIException

logger = get_logger(__name__)


class DocumentAI:
    """
    High-level interface for Document AI operations
    
    Examples:
        >>> ai = DocumentAI()
        >>> result = ai.process("document.pdf")
        >>> ai.export(result, "output.txt")
        
        >>> # With custom configuration
        >>> config = Config()
        >>> config.ocr.device = DeviceType.CUDA
        >>> ai = DocumentAI(config)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize Document AI
        
        Args:
            config: Optional configuration (uses defaults if None)
        """
        self.config = config or Config()
        
        # Setup logging
        LoggerConfig.setup(
            log_level=self.config.log_level,
            log_file=self.config.storage.base_path / "document_ai.log",
        )
        
        # Initialize components
        self.loader = DocumentLoader(
            doc_config=self.config.document,
            security_config=self.config.security,
        )
        
        self.pipeline = OCRPipeline(config=self.config.ocr)
        
        self.exporter = DocumentExporter(
            output_dir=self.config.storage.output_path
        )
        
        logger.info("Document AI initialized")
    
    def process(
        self,
        source: Union[str, Path, bytes],
        file_type: Optional[str] = None,
    ):
        """
        Process a document with OCR
        
        Args:
            source: Path to file, bytes, or file-like object
            file_type: Optional file type hint
        
        Returns:
            Tuple of (document, ocr_results)
        
        Example:
            >>> result = ai.process("scan.pdf")
            >>> for page_result in result[1]:
            ...     print(page_result.full_text)
        """
        try:
            # Load document
            logger.info(f"Loading document: {source}")
            document = self.loader.load(source, file_type=file_type)
            logger.info(f"Loaded {document.total_pages} page(s)")
            
            # Process each page
            ocr_results = []
            for i, page in enumerate(document.pages):
                logger.info(f"Processing page {i+1}/{document.total_pages}")
                result = self.pipeline.process_image(page.image)
                ocr_results.append(result)
                logger.info(
                    f"Page {i+1}: {result.num_detected_regions} regions, "
                    f"{result.confidence:.2%} confidence"
                )
            
            return document, ocr_results
            
        except DocumentAIException as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def process_batch(
        self,
        sources: List[Union[str, Path]],
    ) -> List[tuple]:
        """
        Process multiple documents
        
        Args:
            sources: List of file paths
        
        Returns:
            List of (document, ocr_results) tuples
        
        Example:
            >>> files = ["doc1.pdf", "doc2.png"]
            >>> results = ai.process_batch(files)
        """
        all_results = []
        
        for i, source in enumerate(sources):
            logger.info(f"Processing {i+1}/{len(sources)}: {source}")
            try:
                result = self.process(source)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {source}: {e}")
                all_results.append((None, None))
        
        return all_results
    
    def export(
        self,
        ocr_results: list,
        output_path: Union[str, Path],
        export_format: Optional[ExportFormat] = None,
        document=None,
    ) -> Path:
        """
        Export OCR results to file
        
        Args:
            ocr_results: List of OCR outputs
            output_path: Output file path
            export_format: Export format (inferred from extension if None)
            document: Original document (needed for some formats)
        
        Returns:
            Path to exported file
        
        Example:
            >>> doc, results = ai.process("scan.pdf")
            >>> ai.export(results, "output.docx")
        """
        output_path = Path(output_path)
        
        # Infer format from extension if not specified
        if export_format is None:
            ext = output_path.suffix.lower().lstrip('.')
            try:
                export_format = ExportFormat(ext)
            except ValueError:
                logger.warning(f"Unknown format: {ext}, defaulting to TXT")
                export_format = ExportFormat.TXT
        
        # Export
        logger.info(f"Exporting to {export_format.value}: {output_path}")
        result_path = self.exporter.export(
            ocr_results,
            output_path,
            export_format,
            document,
        )
        
        logger.info(f"Export complete: {result_path}")
        return result_path
    
    def quick_ocr(
        self,
        source: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        export_format: ExportFormat = ExportFormat.TXT,
    ) -> str:
        """
        Quick OCR with automatic export
        
        Args:
            source: Input file path
            output_path: Output path (auto-generated if None)
            export_format: Export format
        
        Returns:
            Path to output file
        
        Example:
            >>> ai.quick_ocr("scan.pdf", "output.txt")
        """
        # Process
        document, ocr_results = self.process(source)
        
        # Generate output path if needed
        if output_path is None:
            source_path = Path(source)
            output_path = source_path.parent / f"{source_path.stem}_ocr.{export_format.value}"
        
        # Export
        result_path = self.export(ocr_results, output_path, export_format, document)
        
        return str(result_path)


def main():
    """Command-line entry point (delegates to CLI)"""
    from document_ai.main import cli
    cli()


if __name__ == "__main__":
    main()
