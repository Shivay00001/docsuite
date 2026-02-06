"""
CRNN (Convolutional Recurrent Neural Network) Text Recognizer
Production implementation for text recognition
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Optional, Dict
from dataclasses import dataclass

from ...exceptions import RecognitionException, ModelLoadException
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RecognitionResult:
    """Text recognition result"""
    text: str
    confidence: float
    char_confidences: List[float]


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM layer"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, seq_len, output_size)
        """
        recurrent, _ = self.rnn(x)
        output = self.linear(recurrent)
        return output


class CRNNModel(nn.Module):
    """
    CRNN architecture for text recognition
    CNN for feature extraction + RNN for sequence modeling + CTC for decoding
    """
    
    def __init__(
        self,
        img_height: int = 32,
        num_channels: int = 1,
        num_classes: int = 37,  # 26 letters + 10 digits + blank
        hidden_size: int = 256,
    ):
        super().__init__()
        
        self.img_height = img_height
        self.num_classes = num_classes
        
        # Convolutional layers
        self.cnn = nn.Sequential(
            # Conv1
            nn.Conv2d(num_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x100 -> 16x50
            
            # Conv2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x50 -> 8x25
            
            # Conv3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Conv4
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x25 -> 4x25
            
            # Conv5
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Conv6
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x25 -> 2x25
            
            # Conv7
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # 2x25 -> 1x24
        )
        
        # Recurrent layers
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            (seq_len, batch, num_classes)
        """
        # CNN feature extraction
        conv = self.cnn(x)  # (batch, 512, 1, width)
        
        # Reshape for RNN
        batch, channel, height, width = conv.size()
        assert height == 1, f"Height should be 1, got {height}"
        
        conv = conv.squeeze(2)  # (batch, 512, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, 512)
        
        # Map to sequence
        conv = conv.permute(1, 0, 2)  # (batch, width, 512)
        
        # RNN
        output = self.rnn(conv)  # (batch, width, num_classes)
        output = output.permute(1, 0, 2)  # (width, batch, num_classes)
        
        return output


class CTCLabelConverter:
    """Convert between text and CTC labels"""
    
    def __init__(self, character: str = '0123456789abcdefghijklmnopqrstuvwxyz'):
        """
        Args:
            character: All possible characters
        """
        self.character = '-' + character  # Add blank token at index 0
        self.dict = {char: i for i, char in enumerate(self.character)}
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to label indices
        
        Args:
            text: Input text
        Returns:
            List of character indices
        """
        return [self.dict.get(char.lower(), 0) for char in text if char.lower() in self.dict]
    
    def decode(self, indices: List[int], remove_duplicates: bool = True) -> str:
        """
        Decode indices to text using CTC rules
        
        Args:
            indices: List of predicted indices
            remove_duplicates: Remove consecutive duplicates
        Returns:
            Decoded text
        """
        chars = []
        prev_idx = 0
        
        for idx in indices:
            if idx == 0:  # Blank token
                prev_idx = 0
                continue
            
            if remove_duplicates and idx == prev_idx:
                continue
            
            chars.append(self.character[idx])
            prev_idx = idx
        
        return ''.join(chars)
    
    def decode_batch(
        self,
        predictions: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[str]:
        """
        Decode batch of predictions
        
        Args:
            predictions: (seq_len, batch, num_classes)
            lengths: Optional sequence lengths
        Returns:
            List of decoded texts
        """
        # Get best path (greedy decoding)
        _, preds = predictions.max(2)  # (seq_len, batch)
        preds = preds.transpose(1, 0).contiguous()  # (batch, seq_len)
        
        texts = []
        for i in range(preds.size(0)):
            length = lengths[i] if lengths is not None else preds.size(1)
            indices = preds[i][:length].cpu().numpy()
            text = self.decode(indices.tolist())
            texts.append(text)
        
        return texts


class CRNNRecognizer:
    """
    CRNN text recognizer wrapper
    Handles preprocessing, inference, and decoding
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        img_height: int = 32,
        img_width: int = 100,
        character_set: str = '0123456789abcdefghijklmnopqrstuvwxyz',
    ):
        """
        Initialize CRNN recognizer
        
        Args:
            model_path: Path to pretrained model weights
            device: Device for inference
            img_height: Input image height
            img_width: Input image width
            character_set: Set of recognizable characters
        """
        self.device = device
        self.img_height = img_height
        self.img_width = img_width
        
        # Initialize converter
        self.converter = CTCLabelConverter(character_set)
        num_classes = len(self.converter.character)
        
        try:
            # Initialize model
            self.model = CRNNModel(
                img_height=img_height,
                num_channels=1,
                num_classes=num_classes,
                hidden_size=256,
            )
            
            if model_path:
                state_dict = torch.load(model_path, map_location=device)
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded CRNN model from {model_path}")
            else:
                logger.warning("No pretrained weights loaded - using random initialization")
            
            self.model.to(device)
            self.model.eval()
            
        except Exception as e:
            raise ModelLoadException(f"Failed to load CRNN model: {str(e)}")
    
    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """
        Recognize text in image
        
        Args:
            image: Input image (grayscale or BGR)
        
        Returns:
            RecognitionResult with text and confidence
        
        Raises:
            RecognitionException: If recognition fails
        """
        try:
            # Preprocess
            img_tensor = self._preprocess_image(image)
            img_tensor = img_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                preds = self.model(img_tensor)  # (seq_len, 1, num_classes)
            
            # Decode
            preds_softmax = torch.softmax(preds, dim=2)
            preds_max = preds_softmax.max(2)[0]  # (seq_len, 1)
            
            # Get confidence scores
            confidences = preds_max.squeeze(1).cpu().numpy()
            
            # Decode text
            text = self.converter.decode_batch(preds)[0]
            
            # Calculate overall confidence
            # Exclude blank tokens (index 0)
            _, pred_indices = preds.max(2)
            pred_indices = pred_indices.squeeze(1).cpu().numpy()
            non_blank_mask = pred_indices != 0
            
            if non_blank_mask.any():
                avg_confidence = float(confidences[non_blank_mask].mean())
            else:
                avg_confidence = 0.0
            
            return RecognitionResult(
                text=text,
                confidence=avg_confidence,
                char_confidences=confidences.tolist(),
            )
            
        except Exception as e:
            raise RecognitionException(f"Text recognition failed: {str(e)}")
    
    def recognize_batch(self, images: List[np.ndarray]) -> List[RecognitionResult]:
        """
        Recognize text in batch of images
        
        Args:
            images: List of input images
        
        Returns:
            List of recognition results
        """
        results = []
        
        try:
            # Preprocess all images
            img_tensors = [self._preprocess_image(img) for img in images]
            batch_tensor = torch.cat(img_tensors, dim=0).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                preds = self.model(batch_tensor)  # (seq_len, batch, num_classes)
            
            # Decode each prediction
            preds_softmax = torch.softmax(preds, dim=2)
            preds_max = preds_softmax.max(2)[0]  # (seq_len, batch)
            
            texts = self.converter.decode_batch(preds)
            
            # Create results
            for i, text in enumerate(texts):
                confidences = preds_max[:, i].cpu().numpy()
                
                # Calculate confidence
                _, pred_indices = preds[:, i, :].max(1)
                pred_indices = pred_indices.cpu().numpy()
                non_blank_mask = pred_indices != 0
                
                if non_blank_mask.any():
                    avg_confidence = float(confidences[non_blank_mask].mean())
                else:
                    avg_confidence = 0.0
                
                results.append(RecognitionResult(
                    text=text,
                    confidence=avg_confidence,
                    char_confidences=confidences.tolist(),
                ))
            
        except Exception as e:
            logger.error(f"Batch recognition failed: {str(e)}")
            # Fallback to single image processing
            for img in images:
                try:
                    result = self.recognize(img)
                    results.append(result)
                except:
                    results.append(RecognitionResult(text="", confidence=0.0, char_confidences=[]))
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for recognition
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed tensor (1, 1, H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to fixed height, maintaining aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h
        new_w = int(self.img_height * aspect_ratio)
        
        # Limit width
        if new_w > self.img_width:
            new_w = self.img_width
        
        resized = cv2.resize(gray, (new_w, self.img_height))
        
        # Pad to fixed width
        if new_w < self.img_width:
            padded = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
            padded[:, :new_w] = resized
        else:
            padded = resized
        
        # Normalize
        normalized = padded.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return tensor


# Language-specific character sets
CHARSET_ENGLISH = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHARSET_ALPHANUMERIC = '0123456789abcdefghijklmnopqrstuvwxyz'
CHARSET_DIGITS = '0123456789'
CHARSET_EXTENDED = CHARSET_ENGLISH + '!@#$%^&*()-_=+[]{}|;:,.<>?/~` '
