import torch
import torch.nn.functional as F
from typing import List
from PIL import Image
import logging
from .base_evaluator import BaseSimilarityModel

logger = logging.getLogger(__name__)

class SigLIPSimilarityModel(BaseSimilarityModel):
    """SigLIP model implementation"""
    
    def __init__(self, device: str = None):
        super().__init__(device)
        self.model_name = "SIGLIP"
        self.max_text_length = 64
        self.load_model()
    
    def load_model(self):
        """Load SigLIP model"""
        from transformers import AutoProcessor, AutoModel
        
        self.model = AutoModel.from_pretrained("/root/liangtao/google--siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("/root/liangtao/google--siglip-base-patch16-224")
        self.model.to(self.device)
        self.model.eval()
        logger.info("SigLIP model loaded successfully")
    
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute SigLIP similarity"""
        text = self.truncate_text(text)
        
        inputs = self.processor(text=[text], images=image, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = torch.sigmoid(logits_per_image)  # these are the probabilities
            return probs[0, 0].item()  # return the probability for the first (and only) text
    
    def compute_similarity_batch(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        """Compute SigLIP similarity in batch"""
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
        inputs = self.processor(text=valid_texts, images=valid_images, padding="max_length", max_length=self.max_text_length, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)  # logits_per_image, logits_per_text, text_embeds, image_embeds, text_model_output, vision_model_output
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = torch.sigmoid(logits_per_image)  # these are the probabilities
            similarities = probs.diag().cpu().tolist()  # get diagonal elements for each image-text pair
        
        # Build complete results
        result = [0.0] * len(images)
        for i, sim in zip(valid_indices, similarities):
            result[i] = sim
        
        return result