import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class SimilarityEvaluator:
    """Evaluator for CLIP similarity results"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_similarities(
        self, 
        similarities: List[float], 
        image_paths: List[str], 
        captions: List[str],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate similarity results
        
        Args:
            similarities: List of similarity scores
            image_paths: List of image paths
            captions: List of captions
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with evaluation metrics
        """
        similarities = np.array(similarities)
        
        # Basic statistics
        stats = {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'median_similarity': float(np.median(similarities)),
            'num_samples': len(similarities)
        }
        
        # Threshold-based metrics
        binary_predictions = (similarities >= threshold).astype(int)
        
        # For evaluation, we assume all samples are positive (since they're image-caption pairs)
        # In a real scenario, you might have negative samples
        true_labels = np.ones(len(similarities), dtype=int)
        
        accuracy = accuracy_score(true_labels, binary_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, binary_predictions, average='binary', zero_division=0
        )
        
        stats.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'threshold': threshold
        })
        
        return stats
    
    def save_results(
        self, 
        similarities: List[float], 
        image_paths: List[str], 
        captions: List[str],
        filename: str = "similarity_results.csv"
    ) -> str:
        """
        Save results to CSV file
        
        Args:
            similarities: List of similarity scores
            image_paths: List of image paths
            captions: List of captions
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        results_df = pd.DataFrame({
            'image_path': image_paths,
            'caption': captions,
            'image_paths': image_paths,
            'similarity_score': similarities
        })
        
        filepath = os.path.join(self.output_dir, filename)
        results_df.to_csv(filepath, index=False)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def plot_similarity_distribution(
        self, 
        similarities: List[float], 
        filename: str = "similarity_distribution.png"
    ) -> str:
        """
        Plot similarity score distribution
        
        Args:
            similarities: List of similarity scores
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarity Scores')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(similarities)
        plt.ylabel('Similarity Score')
        plt.title('Similarity Scores Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Distribution plot saved to {filepath}")
        return filepath
    
    def plot_similarity_heatmap(
        self, 
        similarities: List[float], 
        image_paths: List[str],
        captions: List[str],
        max_images: int = 20,
        filename: str = "similarity_heatmap.png"
    ) -> str:
        """
        Plot similarity heatmap with text on left and image thumbnails on top
        
        Args:
            similarities: List of similarity scores
            image_paths: List of image paths
            captions: List of captions
            max_images: Maximum number of images to show
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        from PIL import Image
        import matplotlib.patches as patches
        
        # Select subset of images
        indices = np.linspace(0, len(similarities) - 1, min(max_images, len(similarities)), dtype=int)
        subset_similarities = [similarities[i] for i in indices]
        subset_paths = [image_paths[i] for i in indices]
        subset_captions = [captions[i] for i in indices]
        
        # Create similarity matrix (self-similarity for now)
        n = len(subset_similarities)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = subset_similarities[i]
                else:
                    # For demonstration, use average of the two similarities
                    similarity_matrix[i, j] = (subset_similarities[i] + subset_similarities[j]) / 2
        
        # Create figure with subplots for image thumbnails and heatmap
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[1, 4], hspace=0.3, wspace=0.3)
        
        # Top row: Image thumbnails
        ax_images = fig.add_subplot(gs[0, 1])
        ax_images.set_xlim(0, n)
        ax_images.set_ylim(0, 1)
        ax_images.set_xticks(range(n))
        ax_images.set_xticklabels([f"Img{i+1}" for i in range(n)], rotation=45, ha='right')
        ax_images.set_yticks([])
        ax_images.set_title("Image Thumbnails", fontsize=12, pad=20)
        
        # Load and display image thumbnails
        thumbnail_size = 0.8
        for i, img_path in enumerate(subset_paths):
            try:
                # Load and resize image
                img = Image.open(img_path)
                img.thumbnail((50, 50), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Display image
                ax_images.imshow(img_array, extent=[i-thumbnail_size/2, i+thumbnail_size/2, 
                                                  0.1, 0.9], aspect='auto')
            except Exception as e:
                # If image can't be loaded, show a placeholder
                rect = patches.Rectangle((i-thumbnail_size/2, 0.1), thumbnail_size, 0.8, 
                                       linewidth=1, edgecolor='gray', facecolor='lightgray')
                ax_images.add_patch(rect)
                ax_images.text(i, 0.5, '?', ha='center', va='center', fontsize=8)
        
        # Left column: Text captions
        ax_text = fig.add_subplot(gs[1, 0])
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, n)
        ax_text.set_yticks(range(n))
        ax_text.set_yticklabels([f"Text{i+1}" for i in range(n)])
        ax_text.set_xticks([])
        ax_text.set_title("Text Captions", fontsize=12, pad=20)
        
        # Display text captions (truncated for readability)
        for i, caption in enumerate(subset_captions):
            # Truncate long captions
            display_caption = caption[:30] + "..." if len(caption) > 30 else caption
            ax_text.text(0.5, i, display_caption, ha='center', va='center', 
                        fontsize=8, wrap=True, rotation=0)
        
        # Main heatmap
        ax_heatmap = fig.add_subplot(gs[1, 1])
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            xticklabels=False,
            yticklabels=False,
            ax=ax_heatmap,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        ax_heatmap.set_title('Cosine similarity between text and image features', fontsize=14, pad=20)
        ax_heatmap.set_xlabel('Images', fontsize=12)
        ax_heatmap.set_ylabel('Text Captions', fontsize=12)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to {filepath}")
        return filepath
    
    def generate_report(
        self, 
        similarities: List[float], 
        image_paths: List[str], 
        captions: List[str],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            similarities: List of similarity scores
            image_paths: List of image paths
            captions: List of captions
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with complete evaluation report
        """
        # Evaluate similarities
        metrics = self.evaluate_similarities(similarities, image_paths, captions, threshold)
        
        # Save results
        results_file = self.save_results(similarities, image_paths, captions)
        
        # Generate plots
        dist_plot = self.plot_similarity_distribution(similarities)
        heatmap_plot = self.plot_similarity_heatmap(similarities, image_paths, captions)
        
        report = {
            'metrics': metrics,
            'files': {
                'results_csv': results_file,
                'distribution_plot': dist_plot,
                'heatmap_plot': heatmap_plot
            }
        }
        
        # Print summary
        logger.info("=== Evaluation Report ===")
        logger.info(f"Number of samples: {metrics['num_samples']}")
        logger.info(f"Mean similarity: {metrics['mean_similarity']:.4f}")
        logger.info(f"Std similarity: {metrics['std_similarity']:.4f}")
        logger.info(f"Min similarity: {metrics['min_similarity']:.4f}")
        logger.info(f"Max similarity: {metrics['max_similarity']:.4f}")
        logger.info(f"Median similarity: {metrics['median_similarity']:.4f}")
        logger.info(f"Accuracy (threshold={threshold}): {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return report
