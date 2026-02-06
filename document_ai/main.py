"""
Command Line Interface for Document AI Platform
Production-grade CLI with comprehensive options
"""
import click
from pathlib import Path
from typing import Optional
import sys

from document_ai.utils.logging import LoggerConfig, get_logger
from document_ai.config import Config, OCRBackend, DeviceType, ExportFormat
from document_ai.exceptions import DocumentAIException

logger = get_logger(__name__)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.pass_context
def cli(ctx, debug, log_file):
    """
    Document AI Platform - OCR and Document Intelligence Suite
    
    A production-grade system for text extraction, document processing,
    and intelligent document analysis.
    """
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    LoggerConfig.setup(
        log_level=log_level,
        log_file=Path(log_file) if log_file else None,
        enable_console=True,
    )
    
    ctx.obj['debug'] = debug


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', 
              type=click.Choice(['txt', 'json', 'csv', 'docx', 'pdf', 'searchable_pdf']),
              default='txt',
              help='Export format')
@click.option('--device', 
              type=click.Choice(['cpu', 'cuda', 'mps']),
              default='cpu',
              help='Computation device')
@click.option('--backend',
              type=click.Choice(['craft_crnn', 'dbnet_crnn', 'east_crnn']),
              default='craft_crnn',
              help='OCR backend')
@click.option('--no-preprocess', is_flag=True, help='Disable preprocessing')
@click.option('--no-deskew', is_flag=True, help='Disable deskewing')
@click.option('--overlay', type=click.Path(), help='Save overlay visualization')
@click.pass_context
def ocr(ctx, input_path, output, format, device, backend, no_preprocess, no_deskew, overlay):
    """
    Perform OCR on document or image
    
    Extracts text from scanned documents, images, and PDFs.
    
    Examples:
    
        # Basic OCR to text file
        docai ocr document.pdf -o output.txt
        
        # OCR with GPU acceleration
        docai ocr scan.jpg --device cuda -f json
        
        # OCR with visualization
        docai ocr image.png --overlay result.png
    """
    try:
        from document_ai.document.loader.document_loader import DocumentLoader
        from document_ai.core.ocr.pipeline import OCRPipeline
        from document_ai.document.exporter.document_exporter import DocumentExporter
        
        logger.info(f"Processing: {input_path}")
        
        # Load document
        loader = DocumentLoader()
        document = loader.load(input_path)
        logger.info(f"Loaded {document.total_pages} page(s)")
        
        # Initialize OCR pipeline
        config = Config()
        config.ocr.device = DeviceType(device)
        config.ocr.backend = OCRBackend(backend)
        config.ocr.enable_preprocessing = not no_preprocess
        config.ocr.enable_deskew = not no_deskew
        
        pipeline = OCRPipeline(config=config.ocr)
        
        # Process pages
        ocr_results = []
        for i, page in enumerate(document.pages):
            logger.info(f"Processing page {i+1}/{document.total_pages}")
            result = pipeline.process_image(page.image)
            ocr_results.append(result)
            logger.info(f"Page {i+1}: Detected {result.num_detected_regions} regions, "
                       f"confidence: {result.confidence:.2%}")
        
        # Export results
        exporter = DocumentExporter()
        
        if output is None:
            # Generate output filename
            input_path_obj = Path(input_path)
            output = input_path_obj.parent / f"{input_path_obj.stem}_ocr.{format}"
        else:
            output = Path(output)
        
        export_format = ExportFormat(format)
        output_file = exporter.export(ocr_results, output, export_format, document)
        
        click.echo(f"\n✓ OCR completed successfully!")
        click.echo(f"  Output: {output_file}")
        click.echo(f"  Total pages: {len(ocr_results)}")
        click.echo(f"  Total regions: {sum(r.num_detected_regions for r in ocr_results)}")
        click.echo(f"  Average confidence: {sum(r.confidence for r in ocr_results)/len(ocr_results):.2%}")
        
        # Save overlay if requested
        if overlay and document.pages:
            overlay_path = Path(overlay)
            exporter.export_regions_overlay(
                document.pages[0].image,
                ocr_results[0].text_boxes,
                overlay_path,
            )
            click.echo(f"  Overlay: {overlay_path}")
        
    except DocumentAIException as e:
        click.echo(f"\n✗ Error: {e}", err=True)
        if ctx.obj.get('debug'):
            raise
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n✗ Unexpected error: {e}", err=True)
        if ctx.obj.get('debug'):
            raise
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--format', '-f',
              type=click.Choice(['txt', 'json', 'csv', 'docx', 'pdf']),
              multiple=True,
              default=['txt', 'json'],
              help='Export formats (can specify multiple)')
@click.option('--device',
              type=click.Choice(['cpu', 'cuda', 'mps']),
              default='cpu',
              help='Computation device')
@click.pass_context
def batch(ctx, input_path, output_dir, format, device):
    """
    Batch process multiple documents
    
    Process all documents in a directory.
    
    Examples:
    
        # Process all PDFs in a directory
        docai batch /path/to/pdfs/ -o /path/to/output/
        
        # Multiple export formats
        docai batch ./scans/ -f txt -f json -f docx
    """
    try:
        from document_ai.document.loader.document_loader import DocumentLoader
        from document_ai.core.ocr.pipeline import OCRPipeline
        from document_ai.document.exporter.document_exporter import DocumentExporter
        
        input_path = Path(input_path)
        
        if not input_path.is_dir():
            click.echo("Error: Input must be a directory for batch processing", err=True)
            sys.exit(1)
        
        # Find all supported files
        supported_exts = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']
        files = []
        for ext in supported_exts:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not files:
            click.echo("No supported files found", err=True)
            sys.exit(1)
        
        logger.info(f"Found {len(files)} file(s) to process")
        
        # Setup output directory
        if output_dir is None:
            output_dir = input_path / "ocr_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        config = Config()
        config.ocr.device = DeviceType(device)
        
        loader = DocumentLoader()
        pipeline = OCRPipeline(config=config.ocr)
        exporter = DocumentExporter(output_dir=output_dir)
        
        # Process files
        with click.progressbar(files, label='Processing documents') as bar:
            for file_path in bar:
                try:
                    # Load
                    document = loader.load(file_path)
                    
                    # OCR
                    ocr_results = []
                    for page in document.pages:
                        result = pipeline.process_image(page.image)
                        ocr_results.append(result)
                    
                    # Export in all requested formats
                    for fmt in format:
                        output_file = output_dir / f"{file_path.stem}.{fmt}"
                        exporter.export(
                            ocr_results,
                            output_file,
                            ExportFormat(fmt),
                            document,
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    continue
        
        click.echo(f"\n✓ Batch processing completed!")
        click.echo(f"  Processed: {len(files)} file(s)")
        click.echo(f"  Output directory: {output_dir}")
        
    except Exception as e:
        click.echo(f"\n✗ Error: {e}", err=True)
        if ctx.obj.get('debug'):
            raise
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
def info(input_path):
    """
    Display document information
    
    Shows metadata and structure of a document.
    """
    try:
        from document_ai.document.loader.document_loader import DocumentLoader
        
        loader = DocumentLoader()
        document = loader.load(input_path)
        
        click.echo("\n" + "=" * 60)
        click.echo("DOCUMENT INFORMATION")
        click.echo("=" * 60)
        click.echo(f"\nFile: {document.file_name}")
        click.echo(f"Type: {document.file_type}")
        click.echo(f"Total Pages: {document.total_pages}")
        
        if document.metadata:
            click.echo("\nMetadata:")
            for key, value in document.metadata.items():
                click.echo(f"  {key}: {value}")
        
        click.echo("\nPages:")
        for page in document.pages:
            w, h = page.original_size
            click.echo(f"  Page {page.page_number}: {w}x{h} @ {page.dpi} DPI")
        
        click.echo()
        
    except Exception as e:
        click.echo(f"\n✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Show version information"""
    click.echo("Document AI Platform v1.0.0")
    click.echo("A production-grade OCR and document intelligence suite")
    click.echo("\nComponents:")
    click.echo("  • CRAFT text detector")
    click.echo("  • CRNN text recognizer")
    click.echo("  • Multi-format document support")
    click.echo("  • Advanced preprocessing pipeline")


@cli.command()
@click.option('--host', default='0.0.0.0', help='API server host')
@click.option('--port', default=8000, help='API server port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """
    Start API server
    
    Launches the REST API server for processing documents via HTTP.
    """
    try:
        import uvicorn
        from document_ai.api.rest.app import app
        
        click.echo(f"Starting API server on {host}:{port}")
        click.echo("API documentation available at http://localhost:8000/docs")
        
        uvicorn.run(
            "api.rest.app:app",
            host=host,
            port=port,
            reload=reload,
        )
        
    except ImportError:
        click.echo("Error: API dependencies not installed", err=True)
        click.echo("Install with: pip install 'document-ai-platform[api]'", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
