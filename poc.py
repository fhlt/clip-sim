#!/usr/bin/env python3
"""
CLIP Similarity Evaluation - Main Entry Point

This script provides a complete pipeline for:
1. Downloading images from URLs in CSV file
2. Loading data with DDP support
3. Computing CLIP similarities
4. Evaluating and visualizing results

Usage:
    python poc.py --csv_path data.csv --batch_size 32
    python poc.py --csv_path data.csv --distributed --world_size 2
"""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
from typing import List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.image_downloader import ImageDownloader
from data.dataset import create_dataloader, load_data_from_csv, setup_ddp, cleanup_ddp, launch_distributed_training
from models.clip_evaluator import CLIPSimilarityModel
from models.siglip_evaluator import SigLIPSimilarityModel
from utils.config import Config, get_parser
from utils.logger import setup_logger
from utils.evaluator import SimilarityEvaluator
from utils.gpu_utils import get_gpu_info, print_gpu_info, clear_gpu_cache

def download_images(config: Config, logger: logging.Logger) -> Tuple[List[str], List[str]]:
    """Download images from CSV file"""
    logger.info("Starting image download process...")
    
    downloader = ImageDownloader(
        output_dir=config.image_dir,
        max_workers=config.max_workers,
        timeout=config.timeout
    )
    
    # Download images from CSV
    successful_paths, successful_captions, error_messages = downloader.download_from_csv(
        csv_path=config.csv_path,
        url_column="url",
        caption_column="caption",
        show_progress=config.verbose
    )
    
    logger.info(f"Download completed: {len(successful_paths)} successful, {len(error_messages)} failed")
    
    if error_messages and config.verbose:
        logger.warning(f"Failed downloads: {len(error_messages)}")
        for i, error in enumerate(error_messages[:5]):  # Show first 5 errors
            logger.warning(f"  {i+1}. {error}")
        if len(error_messages) > 5:
            logger.warning(f"  ... and {len(error_messages) - 5} more errors")
    
    return successful_paths, successful_captions

def load_data(config: Config, logger: logging.Logger) -> Tuple[List[str], List[str]]:
    """Load data from CSV and match with local images"""
    logger.info("Loading data from CSV and matching with local images...")
    
    # Load data from CSV
    image_paths, captions = load_data_from_csv(
        csv_path=config.csv_path,
        image_dir=config.image_dir,
        url_column="url",
        caption_column="caption"
    )
    
    logger.info(f"Loaded {len(image_paths)} valid image-caption pairs")
    return image_paths, captions

def evaluate_similarities(
    config: Config, 
    image_paths: List[str], 
    captions: List[str], 
    logger: logging.Logger
) -> List[float]:
    """Evaluate CLIP similarities with proper GPU mapping"""
    
    # Determine device for this process
    if config.distributed:
        device = f"cuda:{config.rank}"
        logger.info(f"Process {config.rank} using GPU {config.rank}")
    else:
        device = config.device
        logger.info(f"Using device: {device}")
    
    logger.info(f"Initializing {config.model_type.upper()} model...")
    
    # Initialize model on the appropriate device based on model type
    if config.model_type.lower() == "siglip":
        model = SigLIPSimilarityModel(device=device)
    else:  # default to CLIP
        model = CLIPSimilarityModel(device=device)
    
    # Create data loader
    dataloader = create_dataloader(
        image_paths=image_paths,
        captions=captions,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        transform=None,
        distributed=config.distributed,
        rank=config.rank,
        world_size=config.world_size
    )
    
    logger.info(f"Created data loader with {len(dataloader)} batches on {device}")
    
    # Compute similarities
    all_similarities = []
    all_image_paths = []
    all_captions = []
    
    logger.info("Computing similarities...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if not batch['images']:  # Skip empty batches
                continue
                
            # Compute similarities for this batch
            similarities = model.compute_similarity_batch(
                batch['images'], 
                batch['captions']
            )
            
            all_similarities.extend(similarities)
            all_image_paths.extend(batch['image_paths'])
            all_captions.extend(batch['captions'])
            
            if config.verbose and batch_idx % 10 == 0:
                logger.info(f"Process {config.rank}: Processed batch {batch_idx}/{len(dataloader)} on {device}")
    
    logger.info(f"Process {config.rank}: Computed similarities for {len(all_similarities)} samples on {device}")
    return all_similarities, all_image_paths, all_captions

def main():
    """Main function using torch.multiprocessing for DDP"""
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.update_from_args(args)
    
    # Setup logging
    logger = setup_logger(
        name="clip_sim",
        level=config.log_level,
        verbose=config.verbose
    )
    
    logger.info("Starting CLIP Similarity Evaluation")
    logger.info(f"Configuration: {config}")
    
    # Print GPU information
    if config.verbose:
        print_gpu_info()
    
    try:
        # Step 1: Download images (only if not distributed or world_size = 1)
        if not config.distributed or config.world_size == 1:
            image_paths, captions = download_images(config, logger)
        else:
            # For distributed evaluation, load existing images
            image_paths, captions = load_data(config, logger)
        
        # Step 2: Launch distributed evaluation using torch.multiprocessing
        logger.info(f"Launching distributed Evaluation with {config.world_size} processes")
        
        # Convert config to dict for multiprocessing
        config_dict = config.to_dict()
        
        # Launch distributed evaluation
        report = launch_distributed_training(config_dict, image_paths, captions)
        logger.info(f"Report received: {type(report)}")
        
        if report and 'metrics' in report:
            logger.info("Evaluation completed successfully!")
            logger.info(f"Results saved to: {config.output_dir}")
            
            # Print summary
            metrics = report['metrics']
            logger.info("=== Summary ===")
            logger.info(f"Mean similarity: {metrics['mean_similarity']:.4f}")
            logger.info(f"Std similarity: {metrics['std_similarity']:.4f}")
            logger.info(f"Min similarity: {metrics['min_similarity']:.4f}")
            logger.info(f"Max similarity: {metrics['max_similarity']:.4f}")
        elif report and 'status' in report:
            logger.info(f"Multi-process evaluation completed successfully!")
            logger.info(f"Used {report['world_size']} processes")
            logger.info("Results and metrics were logged by the worker processes")
        elif report is None:
            logger.info("Multi-process evaluation completed (results handled within worker processes)")
        else:
            logger.warning(f"Unexpected report format: {report}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise
    
    finally:
        # Cleanup GPU memory
        clear_gpu_cache()

if __name__ == "__main__":
    main()
