"""
Exception Hierarchy for Document AI Platform
All custom exceptions with proper error messages and context
"""
from typing import Optional, Any, Dict


class DocumentAIException(Exception):
    """Base exception for all Document AI errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# OCR Exceptions
class OCRException(DocumentAIException):
    """Base exception for OCR-related errors"""
    pass


class DetectionException(OCRException):
    """Text detection failed"""
    pass


class RecognitionException(OCRException):
    """Text recognition failed"""
    pass


class PreprocessingException(OCRException):
    """Image preprocessing failed"""
    pass


class ModelLoadException(OCRException):
    """Model loading failed"""
    pass


class InferenceException(OCRException):
    """Model inference failed"""
    pass


# Document Exceptions
class DocumentException(DocumentAIException):
    """Base exception for document-related errors"""
    pass


class DocumentLoadException(DocumentException):
    """Document loading failed"""
    pass


class DocumentRenderException(DocumentException):
    """Document rendering failed"""
    pass


class DocumentConversionException(DocumentException):
    """Document conversion failed"""
    pass


class InvalidDocumentException(DocumentException):
    """Document format is invalid"""
    pass


class CorruptedDocumentException(DocumentException):
    """Document is corrupted"""
    pass


# Storage Exceptions
class StorageException(DocumentAIException):
    """Base exception for storage-related errors"""
    pass


class FileNotFoundException(StorageException):
    """File not found"""
    pass


class InsufficientStorageException(StorageException):
    """Not enough storage space"""
    pass


class DatabaseException(StorageException):
    """Database operation failed"""
    pass


# Security Exceptions
class SecurityException(DocumentAIException):
    """Base exception for security-related errors"""
    pass


class ValidationException(SecurityException):
    """Input validation failed"""
    pass


class FileSizeException(SecurityException):
    """File size exceeds limits"""
    pass


class UnsupportedFormatException(SecurityException):
    """File format not supported"""
    pass


class MalwareDetectedException(SecurityException):
    """Potential malware detected"""
    pass


class RateLimitException(SecurityException):
    """Rate limit exceeded"""
    pass


# API Exceptions
class APIException(DocumentAIException):
    """Base exception for API-related errors"""
    pass


class BadRequestException(APIException):
    """Invalid request parameters"""
    pass


class NotFoundException(APIException):
    """Resource not found"""
    pass


class ServerException(APIException):
    """Internal server error"""
    pass


# Processing Exceptions
class ProcessingException(DocumentAIException):
    """Base exception for processing errors"""
    pass


class LayoutAnalysisException(ProcessingException):
    """Layout analysis failed"""
    pass


class TableExtractionException(ProcessingException):
    """Table extraction failed"""
    pass


class KeyValueExtractionException(ProcessingException):
    """Key-value extraction failed"""
    pass


class ExportException(ProcessingException):
    """Export operation failed"""
    pass


# Configuration Exceptions
class ConfigurationException(DocumentAIException):
    """Configuration error"""
    pass


class InvalidConfigException(ConfigurationException):
    """Invalid configuration parameters"""
    pass


# Utility function to create detailed exceptions
def create_exception(
    exception_class: type,
    message: str,
    **details: Any
) -> DocumentAIException:
    """
    Factory function to create exceptions with details
    
    Args:
        exception_class: Exception class to instantiate
        message: Error message
        **details: Additional context information
    
    Returns:
        Instantiated exception
    """
    return exception_class(message, details)
