"""
Document Loader
Handles loading and validation of various document formats
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, BinaryIO
from dataclasses import dataclass
from PIL import Image
import fitz  # PyMuPDF
from pdf2image import convert_from_path, convert_from_bytes

from document_ai.exceptions import (
    DocumentLoadException,
    InvalidDocumentException,
    FileSizeException,
    UnsupportedFormatException,
)
from document_ai.config import DocumentConfig, SecurityConfig
from document_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentPage:
    """Single page from a document"""
    page_number: int
    image: np.ndarray
    original_size: tuple  # (width, height)
    dpi: int
    source_type: str  # 'pdf', 'image', etc.


@dataclass
class Document:
    """Complete document with all pages"""
    pages: List[DocumentPage]
    file_path: Optional[Path]
    file_name: str
    file_type: str
    total_pages: int
    metadata: dict


class DocumentLoader:
    """
    Universal document loader
    Supports PDF, images, and multi-page documents
    """
    
    def __init__(
        self,
        doc_config: Optional[DocumentConfig] = None,
        security_config: Optional[SecurityConfig] = None,
    ):
        """
        Initialize document loader
        
        Args:
            doc_config: Document processing configuration
            security_config: Security configuration
        """
        from document_ai.config import DEFAULT_CONFIG
        self.doc_config = doc_config or DEFAULT_CONFIG.document
        self.security_config = security_config or DEFAULT_CONFIG.security
    
    def load(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        file_type: Optional[str] = None,
    ) -> Document:
        """
        Load document from various sources
        
        Args:
            source: File path, bytes, or file-like object
            file_type: Optional file type hint (e.g., 'pdf', 'png')
        
        Returns:
            Loaded Document object
        
        Raises:
            DocumentLoadException: If loading fails
            UnsupportedFormatException: If format is not supported
        """
        try:
            # Determine source type and validate
            if isinstance(source, (str, Path)):
                file_path = Path(source)
                if not file_path.exists():
                    raise DocumentLoadException(f"File not found: {file_path}")
                
                # Validate file size
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.doc_config.max_file_size_mb:
                    raise FileSizeException(
                        f"File size ({file_size_mb:.1f}MB) exceeds limit "
                        f"({self.doc_config.max_file_size_mb}MB)"
                    )
                
                # Determine file type
                file_ext = file_path.suffix.lower()
                file_name = file_path.name
                
                # Validate extension
                if self.security_config.validate_file_types:
                    if file_ext not in (self.doc_config.supported_image_formats + 
                                       self.doc_config.supported_pdf_formats):
                        raise UnsupportedFormatException(
                            f"Unsupported file format: {file_ext}"
                        )
                
                # Load based on type
                if file_ext in self.doc_config.supported_pdf_formats:
                    return self._load_pdf_file(file_path)
                elif file_ext in self.doc_config.supported_image_formats:
                    return self._load_image_file(file_path)
                else:
                    raise UnsupportedFormatException(f"Unknown format: {file_ext}")
            
            elif isinstance(source, bytes):
                # Load from bytes
                if file_type is None:
                    raise ValueError("file_type must be specified for byte sources")
                
                file_name = f"document.{file_type}"
                
                if file_type == 'pdf':
                    return self._load_pdf_bytes(source, file_name)
                else:
                    return self._load_image_bytes(source, file_name, file_type)
            
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")
        
        except (DocumentLoadException, UnsupportedFormatException, FileSizeException):
            raise
        except Exception as e:
            raise DocumentLoadException(f"Failed to load document: {str(e)}")
    
    def _load_pdf_file(self, file_path: Path) -> Document:
        """Load PDF file"""
        logger.info(f"Loading PDF: {file_path}")
        
        try:
            # Open PDF with PyMuPDF for metadata
            pdf_doc = fitz.open(file_path)
            total_pages = len(pdf_doc)
            metadata = pdf_doc.metadata
            pdf_doc.close()
            
            # Convert pages to images
            logger.debug(f"Converting {total_pages} pages to images at {self.doc_config.dpi_for_pdf_render} DPI")
            images = convert_from_path(
                str(file_path),
                dpi=self.doc_config.dpi_for_pdf_render,
                fmt='png',
            )
            
            # Create document pages
            pages = []
            for i, pil_image in enumerate(images):
                # Convert PIL to numpy array
                img_array = np.array(pil_image)
                
                # Convert RGB to BGR for OpenCV compatibility
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                page = DocumentPage(
                    page_number=i + 1,
                    image=img_array,
                    original_size=pil_image.size,
                    dpi=self.doc_config.dpi_for_pdf_render,
                    source_type='pdf',
                )
                pages.append(page)
            
            return Document(
                pages=pages,
                file_path=file_path,
                file_name=file_path.name,
                file_type='pdf',
                total_pages=total_pages,
                metadata=metadata or {},
            )
        
        except Exception as e:
            raise DocumentLoadException(f"Failed to load PDF: {str(e)}")
    
    def _load_pdf_bytes(self, data: bytes, file_name: str) -> Document:
        """Load PDF from bytes"""
        logger.info(f"Loading PDF from bytes: {file_name}")
        
        try:
            # Open with PyMuPDF
            pdf_doc = fitz.open(stream=data, filetype="pdf")
            total_pages = len(pdf_doc)
            metadata = pdf_doc.metadata
            pdf_doc.close()
            
            # Convert to images
            images = convert_from_bytes(
                data,
                dpi=self.doc_config.dpi_for_pdf_render,
                fmt='png',
            )
            
            # Create pages
            pages = []
            for i, pil_image in enumerate(images):
                img_array = np.array(pil_image)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                page = DocumentPage(
                    page_number=i + 1,
                    image=img_array,
                    original_size=pil_image.size,
                    dpi=self.doc_config.dpi_for_pdf_render,
                    source_type='pdf',
                )
                pages.append(page)
            
            return Document(
                pages=pages,
                file_path=None,
                file_name=file_name,
                file_type='pdf',
                total_pages=total_pages,
                metadata=metadata or {},
            )
        
        except Exception as e:
            raise DocumentLoadException(f"Failed to load PDF bytes: {str(e)}")
    
    def _load_image_file(self, file_path: Path) -> Document:
        """Load image file"""
        logger.info(f"Loading image: {file_path}")
        
        try:
            # Try OpenCV first (faster)
            img = cv2.imread(str(file_path))
            
            if img is None:
                # Fallback to PIL for more format support
                pil_image = Image.open(file_path)
                img = np.array(pil_image)
                
                # Convert RGB to BGR
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            h, w = img.shape[:2]
            
            page = DocumentPage(
                page_number=1,
                image=img,
                original_size=(w, h),
                dpi=72,  # Default DPI for images
                source_type='image',
            )
            
            return Document(
                pages=[page],
                file_path=file_path,
                file_name=file_path.name,
                file_type=file_path.suffix.lower(),
                total_pages=1,
                metadata={
                    'format': file_path.suffix.upper(),
                    'size': f"{w}x{h}",
                },
            )
        
        except Exception as e:
            raise DocumentLoadException(f"Failed to load image: {str(e)}")
    
    def _load_image_bytes(self, data: bytes, file_name: str, file_type: str) -> Document:
        """Load image from bytes"""
        logger.info(f"Loading image from bytes: {file_name}")
        
        try:
            # Decode image
            img_array = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise InvalidDocumentException("Failed to decode image")
            
            h, w = img.shape[:2]
            
            page = DocumentPage(
                page_number=1,
                image=img,
                original_size=(w, h),
                dpi=72,
                source_type='image',
            )
            
            return Document(
                pages=[page],
                file_path=None,
                file_name=file_name,
                file_type=file_type,
                total_pages=1,
                metadata={
                    'format': file_type.upper(),
                    'size': f"{w}x{h}",
                },
            )
        
        except Exception as e:
            raise DocumentLoadException(f"Failed to load image bytes: {str(e)}")
    
    def validate_document(self, document: Document) -> bool:
        """
        Validate loaded document
        
        Args:
            document: Document to validate
        
        Returns:
            True if valid
        
        Raises:
            InvalidDocumentException: If validation fails
        """
        if not document.pages:
            raise InvalidDocumentException("Document has no pages")
        
        for i, page in enumerate(document.pages):
            if page.image is None or page.image.size == 0:
                raise InvalidDocumentException(f"Page {i+1} has invalid image")
            
            # Check minimum dimensions
            h, w = page.image.shape[:2]
            if h < 10 or w < 10:
                raise InvalidDocumentException(f"Page {i+1} is too small: {w}x{h}")
        
        return True


def is_supported_format(file_path: Union[str, Path]) -> bool:
    """
    Check if file format is supported
    
    Args:
        file_path: Path to file
    
    Returns:
        True if supported
    """
    from document_ai.config import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.document
    
    ext = Path(file_path).suffix.lower()
    return ext in (config.supported_image_formats + config.supported_pdf_formats)
