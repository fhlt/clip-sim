#!/usr/bin/env python3
"""
Simple script to run DDP using torch.multiprocessing
"""

import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import launch_distributed_training
from data.image_downloader import ImageDownloader
from utils.config import Config, get_parser
from utils.logger import setup_logger
from utils.reproducibility import configure_reproducibility, print_reproducibility_info

def main():
    """Main function for DDP execution"""
    parser = get_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.update_from_args(args)
    
    # Setup reproducibility
    configure_reproducibility(
        seed=config.seed,
        deterministic=config.deterministic,
        benchmark=config.benchmark,
        use_deterministic_algorithms=config.use_deterministic_algorithms
    )
    
    # Setup logging
    logger = setup_logger(
        name="clip_sim_ddp",
        level=config.log_level,
        verbose=config.verbose
    )
    
    logger.info("Starting DDP CLIP Similarity Evaluation")
    logger.info(f"Configuration: {config}")
    
    try:
        # Step 1: Download images (only if not distributed or world_size = 1)
        if not config.distributed or config.world_size == 1:
            from poc import download_images
            image_paths, captions = download_images(config, logger)
        else:
            # For distributed training, load existing images
            from poc import load_data
            image_paths, captions = load_data(config, logger)
        
        # Step 2: Launch distributed training
        logger.info(f"Launching distributed training with {config.world_size} processes")
        
        # Convert config to dict for multiprocessing
        config_dict = config.to_dict()
        
        # Launch distributed training
        report = launch_distributed_training(config_dict, image_paths, captions)
        
        if report:
            logger.info("Evaluation completed successfully!")
            logger.info(f"Results saved to: {config.output_dir}")
            
            # Print summary
            metrics = report['metrics']
            logger.info("=== Summary ===")
            logger.info(f"Mean similarity: {metrics['mean_similarity']:.4f}")
            logger.info(f"Std similarity: {metrics['std_similarity']:.4f}")
            logger.info(f"Min similarity: {metrics['min_similarity']:.4f}")
            logger.info(f"Max similarity: {metrics['max_similarity']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
