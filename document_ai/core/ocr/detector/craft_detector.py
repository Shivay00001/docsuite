"""
CRAFT (Character Region Awareness For Text) Detector
Production implementation of text detection using CRAFT architecture
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ...exceptions import DetectionException, ModelLoadException
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextBox:
    """Detected text bounding box"""
    points: np.ndarray  # 4x2 array of corner points
    confidence: float
    text: Optional[str] = None  # Filled by recognizer
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get axis-aligned bounding box (x, y, w, h)"""
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        return x_min, y_min, x_max - x_min, y_max - y_min
    
    @property
    def area(self) -> float:
        """Calculate bounding box area"""
        _, _, w, h = self.bbox
        return w * h


class VGG16FeatureExtractor(nn.Module):
    """VGG16 feature extraction backbone"""
    
    def __init__(self):
        super().__init__()
        
        # Conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
    
    def forward(self, x):
        """Extract multi-scale features"""
        # Conv1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        # Conv2
        x = F.relu(self.conv2_1(x))
        conv2 = F.relu(self.conv2_2(x))
        x = self.pool2(conv2)
        
        # Conv3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        conv3 = F.relu(self.conv3_3(x))
        x = self.pool3(conv3)
        
        # Conv4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        conv4 = F.relu(self.conv4_3(x))
        x = self.pool4(conv4)
        
        # Conv5
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        conv5 = F.relu(self.conv5_3(x))
        
        return [conv2, conv3, conv4, conv5]


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    
    def __init__(self):
        super().__init__()
        
        # Upsampling convolutions
        self.upconv1 = nn.Conv2d(512, 256, 1)
        self.upconv2 = nn.Conv2d(512, 256, 1)
        self.upconv3 = nn.Conv2d(256, 128, 1)
        self.upconv4 = nn.Conv2d(128, 64, 1)
        
        # Lateral convolutions
        self.lateral1 = nn.Conv2d(512, 256, 1)
        self.lateral2 = nn.Conv2d(256, 128, 1)
        self.lateral3 = nn.Conv2d(128, 64, 1)
    
    def forward(self, features):
        """Fuse multi-scale features"""
        conv2, conv3, conv4, conv5 = features
        
        # Top-down pathway
        up1 = F.interpolate(self.upconv1(conv5), scale_factor=2, mode='bilinear', align_corners=False)
        lat1 = self.lateral1(conv4)
        merge1 = up1 + lat1
        
        up2 = F.interpolate(self.upconv2(merge1), scale_factor=2, mode='bilinear', align_corners=False)
        lat2 = self.lateral2(conv3)
        merge2 = up2 + lat2
        
        up3 = F.interpolate(self.upconv3(merge2), scale_factor=2, mode='bilinear', align_corners=False)
        lat3 = self.lateral3(conv2)
        merge3 = up3 + lat3
        
        # Final upsampling
        output = F.interpolate(self.upconv4(merge3), scale_factor=2, mode='bilinear', align_corners=False)
        
        return output


class CRAFTModel(nn.Module):
    """
    CRAFT: Character Region Awareness For Text Detection
    Detects individual characters and links them into words
    """
    
    def __init__(self):
        super().__init__()
        
        self.feature_extractor = VGG16FeatureExtractor()
        self.fpn = FeaturePyramidNetwork()
        
        # Final prediction heads
        self.conv_cls = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),  # Region score and affinity score
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            Tuple of (region_score, affinity_score)
        """
        features = self.feature_extractor(x)
        fused = self.fpn(features)
        output = self.conv_cls(fused)
        
        # Split into region and affinity scores
        region_score = torch.sigmoid(output[:, 0:1, :, :])
        affinity_score = torch.sigmoid(output[:, 1:2, :, :])
        
        return region_score, affinity_score


class CRAFTDetector:
    """
    CRAFT text detector wrapper
    Handles model loading, inference, and post-processing
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
    ):
        """
        Initialize CRAFT detector
        
        Args:
            model_path: Path to pretrained model weights
            device: Device for inference (cpu/cuda)
            text_threshold: Threshold for text region
            link_threshold: Threshold for link affinity
            low_text: Threshold for low confidence text
        """
        self.device = device
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        
        try:
            self.model = CRAFTModel()
            
            if model_path:
                state_dict = torch.load(model_path, map_location=device)
                # Handle different checkpoint formats
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded CRAFT model from {model_path}")
            else:
                logger.warning("No pretrained weights loaded - using random initialization")
            
            self.model.to(device)
            self.model.eval()
            
        except Exception as e:
            raise ModelLoadException(f"Failed to load CRAFT model: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[TextBox]:
        """
        Detect text regions in image
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detected text boxes
        
        Raises:
            DetectionException: If detection fails
        """
        try:
            # Preprocess image
            img_resized, ratio_h, ratio_w = self._resize_aspect_ratio(
                image, square_size=1280, mag_ratio=1.5
            )
            
            # Normalize and convert to tensor
            img_tensor = self._normalize_image(img_resized)
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device).float()
            
            # Inference
            with torch.no_grad():
                region_score, affinity_score = self.model(img_tensor)
            
            # Post-process
            region_score = region_score[0, 0].cpu().numpy()
            affinity_score = affinity_score[0, 0].cpu().numpy()
            
            # Get bounding boxes
            boxes = self._get_boxes(
                region_score, affinity_score,
                self.text_threshold, self.link_threshold, self.low_text
            )
            
            # Adjust boxes to original image size
            boxes = self._adjust_boxes(boxes, ratio_w, ratio_h)
            
            # Convert to TextBox objects
            text_boxes = []
            for box, score in boxes:
                text_boxes.append(TextBox(points=box, confidence=score))
            
            logger.debug(f"Detected {len(text_boxes)} text regions")
            return text_boxes
            
        except Exception as e:
            raise DetectionException(f"Text detection failed: {str(e)}")
    
    def _resize_aspect_ratio(
        self,
        img: np.ndarray,
        square_size: int = 1280,
        mag_ratio: float = 1.5
    ) -> Tuple[np.ndarray, float, float]:
        """Resize image while maintaining aspect ratio"""
        height, width = img.shape[:2]
        
        # Magnify image
        target_size = mag_ratio * max(height, width)
        
        if target_size > square_size:
            target_size = square_size
        
        ratio = target_size / max(height, width)
        
        target_h, target_w = int(height * ratio), int(width * ratio)
        
        # Make dimensions multiples of 32
        target_h = (target_h // 32) * 32
        target_w = (target_w // 32) * 32
        
        img_resized = cv2.resize(img, (target_w, target_h))
        
        ratio_h = ratio_w = ratio
        
        return img_resized, ratio_h, ratio_w
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image for network input"""
        img = img.astype(np.float32)
        img = (img - np.array([0.485, 0.456, 0.406]) * 255) / (np.array([0.229, 0.224, 0.225]) * 255)
        return img
    
    def _get_boxes(
        self,
        region_score: np.ndarray,
        affinity_score: np.ndarray,
        text_threshold: float,
        link_threshold: float,
        low_text: float
    ) -> List[Tuple[np.ndarray, float]]:
        """Extract bounding boxes from score maps"""
        # Threshold text regions
        text_score_comb = np.clip(region_score + affinity_score, 0, 1)
        text_mask = text_score_comb > low_text
        
        # Find connected components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            text_mask.astype(np.uint8), connectivity=4
        )
        
        boxes = []
        for k in range(1, n_labels):
            # Size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:
                continue
            
            # Get region mask
            segmap = (labels == k).astype(np.uint8)
            
            # Get average score
            score = region_score[segmap == 1].mean()
            if score < text_threshold:
                continue
            
            # Get bounding box
            x, y, w, h = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP], \
                         stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            
            # Create box points
            box = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
            
            boxes.append((box, float(score)))
        
        return boxes
    
    def _adjust_boxes(
        self,
        boxes: List[Tuple[np.ndarray, float]],
        ratio_w: float,
        ratio_h: float
    ) -> List[Tuple[np.ndarray, float]]:
        """Adjust boxes to original image coordinates"""
        adjusted = []
        for box, score in boxes:
            box[:, 0] /= ratio_w
            box[:, 1] /= ratio_h
            adjusted.append((box, score))
        return adjusted
