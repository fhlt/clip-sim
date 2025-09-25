import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class BaseSimilarityModel(ABC):
    """Base class for all similarity models"""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.max_text_length = 77  # Default, can be overridden
        self.model_name = "Base"
        
    @abstractmethod
    def load_model(self):
        """Load the specific model"""
        pass
    
    @abstractmethod
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute similarity between image and text"""
        pass
    
    @abstractmethod
    def compute_similarity_batch(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        """Compute similarity in batch"""
        pass
    
    def truncate_text(self, text: str) -> str:
        """Truncate text to fit model's maximum sequence length"""
        try:
            tokens = self.processor.tokenizer.encode(text, add_special_tokens=True)
            token_length = len(tokens)
            
            if token_length > self.max_text_length:
                truncated_tokens = tokens[:self.max_text_length]
                truncated_text = self.processor.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                logger.warning(f"Text truncated from {token_length} to {len(truncated_tokens)} tokens for {self.model_name} model")
                return truncated_text
            else:
                return text
        except Exception as e:
            logger.warning(f"{self.model_name} tokenization failed, using character-based truncation: {e}")
            # Fallback to character-based truncation
            max_chars = int(self.max_text_length * 3.5)
            if len(text) > max_chars:
                truncated_text = text[:max_chars]
                logger.warning(f"Text truncated from {len(text)} to {len(truncated_text)} characters (fallback)")
                return truncated_text
            return text
