import torch
import torch.nn.functional as F
from typing import List
from PIL import Image
import logging
from .base_evaluator import BaseSimilarityModel

logger = logging.getLogger(__name__)

class CLIPSimilarityModel(BaseSimilarityModel):
    """CLIP model implementation"""
    
    def __init__(self, device: str = None):
        super().__init__(device)
        self.model_name = "CLIP"
        self.max_text_length = 77
        self.load_model()
    
    def load_model(self):
        """Load CLIP model"""
        from transformers import CLIPProcessor, CLIPModel
        
        self.model = CLIPModel.from_pretrained("/root/liangtao/models--openai--clip-vit-base-patch32/")
        self.processor = CLIPProcessor.from_pretrained("/root/liangtao/models--openai--clip-vit-base-patch32/")
        self.model.to(self.device)
        self.model.eval()
        logger.info("CLIP model loaded successfully")
    
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity"""
        text = self.truncate_text(text)
        
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            similarity = torch.sum(image_features * text_features, dim=-1)
            return float(similarity.item())
    
    def compute_similarity_batch(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        """Compute CLIP similarity in batch"""
        # Filter out None images and truncate long texts
        valid_indices = []
        valid_images = []
        valid_texts = []
        
        for i, (image, text) in enumerate(zip(images, texts)):
            if image is not None:
                text = self.truncate_text(text)
                valid_indices.append(i)
                valid_images.append(image)
                valid_texts.append(text)
        
        if not valid_images:
            return [0.0] * len(images)
        
        # Batch processing with truncation
        inputs = self.processor(text=valid_texts, images=valid_images, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            similarities = torch.sum(image_features * text_features, dim=-1)
            similarities = similarities.cpu().tolist()
        
        # Build complete results
        result = [0.0] * len(images)
        for i, sim in zip(valid_indices, similarities):
            result[i] = sim
        
        return result
