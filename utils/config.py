import os
import argparse
from typing import Dict, Any

class Config:
    """Configuration class for CLIP similarity evaluation"""
    
    def __init__(self):
        # Data paths
        self.csv_path = "data.csv"
        self.image_dir = "outputs/images"
        self.output_dir = "outputs"
        
        # Model settings
        self.model_name = "openai/clip-vit-base-patch32"
        self.model_type = "clip"  # "clip" or "siglip"
        self.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        
        # Download settings
        self.max_workers = 10
        self.timeout = 30
        self.force_download = False
        
        # Training settings
        self.batch_size = 32
        self.num_workers = 4
        self.shuffle = True
        
        # Distributed settings
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.backend = "nccl"
        
        # Evaluation settings
        self.save_results = True
        self.results_file = "similarity_results.csv"
        
        # Logging
        self.log_level = "INFO"
        self.verbose = True
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments"""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        
        # Handle --quiet flag
        if hasattr(args, 'quiet') and args.quiet:
            self.verbose = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {key: value for key, value in vars(self).items() 
                if not key.startswith('_')}
    
    def __str__(self):
        return f"Config({self.to_dict()})"

def get_parser() -> argparse.ArgumentParser:
    """Get command line argument parser"""
    parser = argparse.ArgumentParser(description="CLIP Similarity Evaluation")
    
    # Data arguments
    parser.add_argument("--csv_path", type=str, default="data.csv",
                       help="Path to CSV file with URLs and captions")
    parser.add_argument("--image_dir", type=str, default="outputs/images",
                       help="Directory to save downloaded images")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for results")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32",
                       help="Model name (CLIP or SigLIP)")
    parser.add_argument("--model_type", type=str, default="clip", choices=["clip", "siglip"],
                       help="Model type: 'clip' or 'siglip'")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    
    # Download arguments
    parser.add_argument("--max_workers", type=int, default=10,
                       help="Number of download workers")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Download timeout in seconds")
    parser.add_argument("--force_download", action="store_true",
                       help="Force re-download of existing images")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--no_shuffle", action="store_true",
                       help="Don't shuffle data")
    
    # Distributed arguments
    parser.add_argument("--distributed", action="store_true",
                       help="Use distributed training")
    parser.add_argument("--rank", type=int, default=0,
                       help="Process rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1,
                       help="Number of processes for distributed training")
    parser.add_argument("--backend", type=str, default="nccl",
                       help="Backend for distributed training")
    
    # Evaluation arguments
    parser.add_argument("--save_results", action="store_true",
                       help="Save results to file")
    parser.add_argument("--results_file", type=str, default="similarity_results.csv",
                       help="Results file name")
    
    # Logging arguments
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output (default: True)")
    parser.add_argument("--quiet", action="store_true",
                       help="Disable verbose output")
    
    return parser
