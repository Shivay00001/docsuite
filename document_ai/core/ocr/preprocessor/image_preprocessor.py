"""
Image Preprocessing Pipeline
Advanced preprocessing for optimal OCR performance using OpenCV
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

from ..exceptions import PreprocessingException
from ..utils.logging import get_logger

logger = get_logger(__name__)


class PreprocessingMode(Enum):
    """Preprocessing intensity modes"""
    LIGHT = "light"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


@dataclass
class PreprocessingResult:
    """Result of preprocessing operation"""
    image: np.ndarray
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    transformations_applied: List[str]
    skew_angle: Optional[float] = None


class ImagePreprocessor:
    """
    Production-grade image preprocessing pipeline
    Handles noise reduction, binarization, deskewing, and enhancement
    """
    
    def __init__(
        self,
        target_size: Optional[int] = 2048,
        mode: PreprocessingMode = PreprocessingMode.STANDARD,
    ):
        """
        Initialize preprocessor
        
        Args:
            target_size: Maximum dimension for resizing (None to skip)
            mode: Preprocessing intensity mode
        """
        self.target_size = target_size
        self.mode = mode
    
    def process(
        self,
        image: np.ndarray,
        enable_deskew: bool = True,
        enable_denoising: bool = True,
        enable_contrast: bool = True,
    ) -> PreprocessingResult:
        """
        Apply full preprocessing pipeline
        
        Args:
            image: Input image (BGR or grayscale)
            enable_deskew: Apply deskewing
            enable_denoising: Apply denoising
            enable_contrast: Apply contrast enhancement
        
        Returns:
            PreprocessingResult with processed image and metadata
        
        Raises:
            PreprocessingException: If preprocessing fails
        """
        try:
            original_shape = image.shape[:2]
            transformations = []
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                transformations.append("grayscale_conversion")
            else:
                gray = image.copy()
            
            # Resize if needed
            if self.target_size:
                gray = self._resize_image(gray, self.target_size)
                transformations.append(f"resize_to_{self.target_size}")
            
            # Denoising
            if enable_denoising:
                gray = self._denoise(gray)
                transformations.append("denoising")
            
            # Deskewing
            skew_angle = None
            if enable_deskew:
                gray, skew_angle = self._deskew(gray)
                transformations.append(f"deskew_{skew_angle:.2f}deg")
            
            # Contrast enhancement
            if enable_contrast:
                gray = self._enhance_contrast(gray)
                transformations.append("contrast_enhancement")
            
            # Binarization
            binary = self._binarize(gray)
            transformations.append("binarization")
            
            # Morphological operations
            binary = self._morphological_operations(binary)
            transformations.append("morphological_ops")
            
            logger.debug(f"Preprocessing complete: {' -> '.join(transformations)}")
            
            return PreprocessingResult(
                image=binary,
                original_shape=original_shape,
                final_shape=binary.shape[:2],
                transformations_applied=transformations,
                skew_angle=skew_angle,
            )
            
        except Exception as e:
            raise PreprocessingException(
                f"Preprocessing failed: {str(e)}",
                {"image_shape": image.shape}
            )
    
    def _resize_image(self, image: np.ndarray, max_dim: int) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        if max(height, width) <= max_dim:
            return image
        
        if height > width:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        else:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising based on preprocessing mode
        """
        if self.mode == PreprocessingMode.LIGHT:
            return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        elif self.mode == PreprocessingMode.STANDARD:
            return cv2.fastNlMeansDenoising(image, None, h=15, templateWindowSize=7, searchWindowSize=21)
        else:  # AGGRESSIVE
            denoised = cv2.fastNlMeansDenoising(image, None, h=20, templateWindowSize=7, searchWindowSize=21)
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
            return denoised
    
    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct skew angle using projection profile method
        
        Returns:
            Tuple of (deskewed_image, skew_angle_degrees)
        """
        # Otsu's thresholding for edge detection
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) < 100:  # Not enough points
            return image, 0.0
        
        # Find minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        
        # Skip if angle is negligible
        if abs(angle) < 0.5:
            return image, 0.0
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated, angle
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for binarization
        Multiple methods combined for robustness
        """
        if self.mode == PreprocessingMode.LIGHT:
            # Simple Otsu's thresholding
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        
        elif self.mode == PreprocessingMode.STANDARD:
            # Adaptive Gaussian thresholding
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return binary
        
        else:  # AGGRESSIVE
            # Combine multiple methods
            # Method 1: Otsu
            _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 2: Adaptive
            adaptive = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Combine using bitwise AND
            combined = cv2.bitwise_and(otsu, adaptive)
            return combined
    
    def _morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up binary image
        """
        # Remove small noise
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_noise, iterations=1)
        
        if self.mode == PreprocessingMode.AGGRESSIVE:
            # Connect nearby text
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_connect, iterations=1)
            return closing
        
        return opening
    
    def preprocess_batch(
        self,
        images: List[np.ndarray],
        **kwargs
    ) -> List[PreprocessingResult]:
        """
        Process multiple images in batch
        
        Args:
            images: List of input images
            **kwargs: Arguments passed to process()
        
        Returns:
            List of preprocessing results
        """
        results = []
        for i, image in enumerate(images):
            try:
                result = self.process(image, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to preprocess image {i}: {str(e)}")
                # Return original as fallback
                results.append(PreprocessingResult(
                    image=image,
                    original_shape=image.shape[:2],
                    final_shape=image.shape[:2],
                    transformations_applied=["failed_fallback"],
                ))
        
        return results


def auto_rotate_image(image: np.ndarray) -> np.ndarray:
    """
    Auto-detect and correct image orientation
    
    Args:
        image: Input image
    
    Returns:
        Correctly oriented image
    """
    # Detect if image is rotated 90/180/270 degrees
    # This is a simplified heuristic - in production, use a CNN classifier
    h, w = image.shape[:2]
    
    if h > w * 1.5:  # Portrait orientation
        # Check if text is vertical by analyzing projection profiles
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        horizontal_proj = np.sum(gray, axis=1)
        vertical_proj = np.sum(gray, axis=0)
        
        # If vertical projection has more variance, image might be rotated
        if np.std(vertical_proj) > np.std(horizontal_proj) * 1.5:
            # Rotate 90 degrees
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    return image
