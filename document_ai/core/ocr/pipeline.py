"""
OCR Pipeline - Main Orchestrator
Coordinates preprocessing, detection, and recognition
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from document_ai.core.ocr.preprocessor.image_preprocessor import ImagePreprocessor, PreprocessingResult
from document_ai.core.ocr.detector.craft_detector import CRAFTDetector, TextBox
from document_ai.core.ocr.recognizer.crnn_recognizer import CRNNRecognizer, RecognitionResult
from document_ai.exceptions import OCRException
from document_ai.config import OCRConfig
from document_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OCROutput:
    """Complete OCR result for an image"""
    text_boxes: List[TextBox]
    full_text: str
    preprocessing_info: PreprocessingResult
    confidence: float
    num_detected_regions: int


class OCRPipeline:
    """
    End-to-end OCR pipeline
    Handles the complete flow: preprocessing -> detection -> recognition
    """
    
    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        detector_weights: Optional[Path] = None,
        recognizer_weights: Optional[Path] = None,
    ):
        """
        Initialize OCR pipeline
        
        Args:
            config: OCR configuration
            detector_weights: Path to detector model weights
            recognizer_weights: Path to recognizer model weights
        """
        from document_ai.config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG.ocr
        
        # Initialize components
        logger.info("Initializing OCR pipeline components...")
        
        self.preprocessor = ImagePreprocessor(
            target_size=self.config.max_image_dimension,
        )
        
        self.detector = CRAFTDetector(
            model_path=str(detector_weights) if detector_weights else None,
            device=self.config.device.value,
            text_threshold=self.config.detection_threshold,
        )
        
        self.recognizer = CRNNRecognizer(
            model_path=str(recognizer_weights) if recognizer_weights else None,
            device=self.config.device.value,
        )
        
        logger.info("OCR pipeline initialized successfully")
    
    def process_image(
        self,
        image: np.ndarray,
        enable_preprocessing: Optional[bool] = None,
        enable_deskew: Optional[bool] = None,
    ) -> OCROutput:
        """
        Process a single image through the complete OCR pipeline
        
        Args:
            image: Input image (BGR or grayscale)
            enable_preprocessing: Override config preprocessing setting
            enable_deskew: Override config deskewing setting
        
        Returns:
            OCROutput with all detected text and metadata
        
        Raises:
            OCRException: If processing fails
        """
        try:
            # Use config defaults if not specified
            if enable_preprocessing is None:
                enable_preprocessing = self.config.enable_preprocessing
            if enable_deskew is None:
                enable_deskew = self.config.enable_deskew
            
            # Step 1: Preprocessing
            logger.debug("Step 1: Preprocessing image")
            if enable_preprocessing:
                preprocess_result = self.preprocessor.process(
                    image,
                    enable_deskew=enable_deskew,
                    enable_denoising=self.config.enable_denoising,
                )
                processed_image = preprocess_result.image
            else:
                processed_image = image
                preprocess_result = PreprocessingResult(
                    image=image,
                    original_shape=image.shape[:2],
                    final_shape=image.shape[:2],
                    transformations_applied=["none"],
                )
            
            # Step 2: Text Detection
            logger.debug("Step 2: Detecting text regions")
            text_boxes = self.detector.detect(processed_image)
            
            if not text_boxes:
                logger.warning("No text regions detected")
                return OCROutput(
                    text_boxes=[],
                    full_text="",
                    preprocessing_info=preprocess_result,
                    confidence=0.0,
                    num_detected_regions=0,
                )
            
            # Step 3: Text Recognition
            logger.debug(f"Step 3: Recognizing {len(text_boxes)} text regions")
            recognized_boxes = self._recognize_regions(processed_image, text_boxes)
            
            # Step 4: Assemble full text
            full_text, avg_confidence = self._assemble_text(recognized_boxes)
            
            return OCROutput(
                text_boxes=recognized_boxes,
                full_text=full_text,
                preprocessing_info=preprocess_result,
                confidence=avg_confidence,
                num_detected_regions=len(recognized_boxes),
            )
            
        except Exception as e:
            raise OCRException(f"OCR pipeline failed: {str(e)}")
    
    def process_batch(
        self,
        images: List[np.ndarray],
        **kwargs
    ) -> List[OCROutput]:
        """
        Process multiple images in batch
        
        Args:
            images: List of input images
            **kwargs: Arguments passed to process_image()
        
        Returns:
            List of OCR outputs
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                logger.debug(f"Processing image {i+1}/{len(images)}")
                result = self.process_image(image, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {i}: {str(e)}")
                # Add empty result
                results.append(OCROutput(
                    text_boxes=[],
                    full_text="",
                    preprocessing_info=PreprocessingResult(
                        image=image,
                        original_shape=image.shape[:2],
                        final_shape=image.shape[:2],
                        transformations_applied=["failed"],
                    ),
                    confidence=0.0,
                    num_detected_regions=0,
                ))
        
        return results
    
    def _recognize_regions(
        self,
        image: np.ndarray,
        text_boxes: List[TextBox],
    ) -> List[TextBox]:
        """
        Recognize text in detected regions
        
        Args:
            image: Preprocessed image
            text_boxes: Detected text boxes
        
        Returns:
            Text boxes with recognized text
        """
        # Extract cropped regions
        regions = []
        for box in text_boxes:
            region = self._crop_region(image, box.points)
            regions.append(region)
        
        # Batch recognition for efficiency
        if len(regions) <= self.config.batch_size:
            # Process all at once
            recognition_results = self.recognizer.recognize_batch(regions)
        else:
            # Process in batches
            recognition_results = []
            for i in range(0, len(regions), self.config.batch_size):
                batch = regions[i:i + self.config.batch_size]
                batch_results = self.recognizer.recognize_batch(batch)
                recognition_results.extend(batch_results)
        
        # Update text boxes with recognized text
        for box, result in zip(text_boxes, recognition_results):
            if result.confidence >= self.config.recognition_threshold:
                box.text = result.text
            else:
                box.text = ""  # Low confidence, skip
        
        # Filter out empty results
        recognized_boxes = [box for box in text_boxes if box.text]
        
        return recognized_boxes
    
    def _crop_region(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Crop and straighten a text region from image
        
        Args:
            image: Source image
            points: 4 corner points of the region
        
        Returns:
            Cropped and straightened region
        """
        # Get bounding rectangle
        rect = cv2.minAreaRect(points.astype(np.float32))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get width and height
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        if width < height:
            width, height = height, width
        
        # Ensure minimum size
        if width < 10 or height < 5:
            # Too small, return zeros
            return np.zeros((32, 100), dtype=np.uint8)
        
        # Define destination points
        dst_pts = np.array([
            [0, height - 1],
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1]
        ], dtype=np.float32)
        
        # Sort source points
        src_pts = self._order_points(points)
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp perspective
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order starting from top-left
        
        Args:
            pts: 4x2 array of points
        
        Returns:
            Ordered points array
        """
        # Sort by y-coordinate
        sorted_pts = pts[np.argsort(pts[:, 1])]
        
        # Top two points
        top = sorted_pts[:2]
        # Bottom two points
        bottom = sorted_pts[2:]
        
        # Sort top points by x-coordinate
        top = top[np.argsort(top[:, 0])]
        # Sort bottom points by x-coordinate (reversed)
        bottom = bottom[np.argsort(bottom[:, 0])[::-1]]
        
        # Order: top-left, top-right, bottom-right, bottom-left
        ordered = np.array([
            top[0],    # top-left
            top[1],    # top-right
            bottom[0], # bottom-right
            bottom[1]  # bottom-left
        ], dtype=np.float32)
        
        return ordered
    
    def _assemble_text(
        self,
        text_boxes: List[TextBox],
    ) -> Tuple[str, float]:
        """
        Assemble full text from recognized boxes with reading order
        
        Args:
            text_boxes: List of text boxes with recognized text
        
        Returns:
            Tuple of (full_text, average_confidence)
        """
        if not text_boxes:
            return "", 0.0
        
        # Sort boxes by reading order (top-to-bottom, left-to-right)
        sorted_boxes = self._sort_reading_order(text_boxes)
        
        # Concatenate text
        lines = []
        current_line = []
        current_y = None
        
        for box in sorted_boxes:
            x, y, w, h = box.bbox
            
            # Check if this is a new line (y-coordinate changed significantly)
            if current_y is None:
                current_y = y
            elif abs(y - current_y) > h * 0.5:  # New line
                if current_line:
                    lines.append(' '.join([b.text for b in current_line]))
                current_line = [box]
                current_y = y
            else:
                current_line.append(box)
        
        # Add last line
        if current_line:
            lines.append(' '.join([b.text for b in current_line]))
        
        full_text = '\n'.join(lines)
        
        # Calculate average confidence
        avg_confidence = np.mean([box.confidence for box in text_boxes])
        
        return full_text, float(avg_confidence)
    
    def _sort_reading_order(self, text_boxes: List[TextBox]) -> List[TextBox]:
        """
        Sort text boxes in reading order (top-to-bottom, left-to-right)
        
        Args:
            text_boxes: List of text boxes
        
        Returns:
            Sorted list of text boxes
        """
        # Get bounding boxes
        bboxes = [box.bbox for box in text_boxes]
        
        # Sort by y-coordinate first, then x-coordinate
        indices = sorted(
            range(len(bboxes)),
            key=lambda i: (bboxes[i][1], bboxes[i][0])
        )
        
        return [text_boxes[i] for i in indices]
