"""
Core Configuration Module
Centralized configuration management for the entire platform
"""
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
import os


class DeviceType(Enum):
    """Computation device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class OCRBackend(Enum):
    """OCR backend selection"""
    CRAFT_CRNN = "craft_crnn"
    DBNET_CRNN = "dbnet_crnn"
    EAST_CRNN = "east_crnn"


class ExportFormat(Enum):
    """Supported export formats"""
    TXT = "txt"
    JSON = "json"
    CSV = "csv"
    DOCX = "docx"
    PDF = "pdf"
    SEARCHABLE_PDF = "searchable_pdf"


@dataclass
class OCRConfig:
    """OCR Engine Configuration"""
    backend: OCRBackend = OCRBackend.CRAFT_CRNN
    device: DeviceType = DeviceType.CPU
    batch_size: int = 8
    detection_threshold: float = 0.7
    recognition_threshold: float = 0.5
    max_image_dimension: int = 2048
    enable_preprocessing: bool = True
    enable_deskew: bool = True
    enable_denoising: bool = True
    language: str = "en"
    use_fp16: bool = False
    onnx_optimization: bool = True


@dataclass
class DocumentConfig:
    """Document Processing Configuration"""
    max_file_size_mb: int = 100
    supported_image_formats: tuple = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')
    supported_pdf_formats: tuple = ('.pdf',)
    dpi_for_pdf_render: int = 300
    enable_layout_analysis: bool = True
    enable_table_detection: bool = True
    enable_kv_extraction: bool = True


@dataclass
class StorageConfig:
    """Storage Configuration"""
    base_path: Path = field(default_factory=lambda: Path.home() / ".document_ai")
    models_path: Optional[Path] = None
    cache_path: Optional[Path] = None
    output_path: Optional[Path] = None
    database_path: Optional[Path] = None
    max_cache_size_gb: int = 10
    
    def __post_init__(self):
        """Initialize storage paths"""
        self.base_path = Path(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.models_path = self.models_path or self.base_path / "models"
        self.cache_path = self.cache_path or self.base_path / "cache"
        self.output_path = self.output_path or self.base_path / "output"
        self.database_path = self.database_path or self.base_path / "database"
        
        for path in [self.models_path, self.cache_path, 
                     self.output_path, self.database_path]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class SecurityConfig:
    """Security Configuration"""
    validate_file_types: bool = True
    scan_for_malware: bool = False
    max_file_size_mb: int = 100
    allowed_extensions: tuple = (
        '.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'
    )
    enable_sandboxing: bool = True
    rate_limit_per_minute: int = 60


@dataclass
class APIConfig:
    """API Server Configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    enable_cors: bool = True
    max_upload_size_mb: int = 100


@dataclass
class LicensingConfig:
    """SaaS Licensing Configuration"""
    enabled: bool = True
    public_key_path: Optional[Path] = None
    server_url: str = "https://license.docsuite.ai/api/v1"
    offline_mode: bool = True
    usage_tracking_enabled: bool = True
    sync_interval_hours: int = 24



@dataclass
class Config:
    """Master Configuration Container"""
    ocr: OCRConfig = field(default_factory=OCRConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api: APIConfig = field(default_factory=APIConfig)
    licensing: LicensingConfig = field(default_factory=LicensingConfig)
    
    debug: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        return cls(
            ocr=OCRConfig(**config_dict.get('ocr', {})),
            document=DocumentConfig(**config_dict.get('document', {})),
            storage=StorageConfig(**config_dict.get('storage', {})),
            security=SecurityConfig(**config_dict.get('security', {})),
            api=APIConfig(**config_dict.get('api', {})),
            licensing=LicensingConfig(**config_dict.get('licensing', {})),
            debug=config_dict.get('debug', False),
            log_level=config_dict.get('log_level', 'INFO'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'ocr': self.ocr.__dict__,
            'document': self.document.__dict__,
            'storage': {k: str(v) if isinstance(v, Path) else v 
                       for k, v in self.storage.__dict__.items()},
            'security': self.security.__dict__,
            'api': self.api.__dict__,
            'licensing': self.licensing.__dict__,
            'debug': self.debug,
            'log_level': self.log_level,
        }


# Global default configuration
DEFAULT_CONFIG = Config()


# Model URLs for downloading pretrained weights
MODEL_URLS = {
    'craft_detector': 'https://github.com/clovaai/CRAFT-pytorch/releases/download/v1.0/craft_mlt_25k.pth',
    'dbnet_detector': 'https://github.com/MhLiao/DB/releases/download/v1.0/ic15_resnet50',
    'crnn_recognizer': 'https://github.com/meijieru/crnn.pytorch/releases/download/v1.0/crnn.pth',
}


# Constants
MIN_TEXT_SIZE = 8  # Minimum text height in pixels
MAX_TEXT_SIZE = 200  # Maximum text height in pixels
SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar']
DEFAULT_LANGUAGE = 'en'

# Image preprocessing constants
BILATERAL_FILTER_D = 9
BILATERAL_FILTER_SIGMA_COLOR = 75
BILATERAL_FILTER_SIGMA_SPACE = 75
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
ADAPTIVE_THRESHOLD_C = 2

# OCR constants
CRAFT_TEXT_THRESHOLD = 0.7
CRAFT_LINK_THRESHOLD = 0.4
CRAFT_LOW_TEXT = 0.4
CRNN_IMG_HEIGHT = 32
CRNN_IMG_WIDTH = 100

# Document processing constants
TABLE_MIN_ROWS = 2
TABLE_MIN_COLS = 2
KV_CONFIDENCE_THRESHOLD = 0.6
LAYOUT_COLUMN_THRESHOLD = 50  # pixels
