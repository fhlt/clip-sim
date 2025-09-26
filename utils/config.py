import os
import argparse
import yaml
from typing import Dict, Any

class Config:
    """Configuration class for CLIP similarity evaluation"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load configuration from YAML file
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            # Data paths
            data_config = config_data.get('data', {})
            self.csv_path = data_config.get('csv_path', "data.csv")
            self.image_dir = data_config.get('image_dir', "outputs/images")
            self.output_dir = data_config.get('output_dir', "outputs")
            
            # Model settings
            model_config = config_data.get('model', {})
            self.model_name = model_config.get('name', "openai/clip-vit-base-patch32")
            self.model_type = model_config.get('type', "clip")
            self.device = model_config.get('device', "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
            
            # Download settings
            download_config = config_data.get('download', {})
            self.max_workers = download_config.get('max_workers', 10)
            self.timeout = download_config.get('timeout', 30)
            self.force_download = download_config.get('force_download', False)
            
            # Training settings
            training_config = config_data.get('training', {})
            self.batch_size = training_config.get('batch_size', 32)
            self.num_workers = training_config.get('num_workers', 4)
            self.shuffle = training_config.get('shuffle', True)
            
            # Distributed settings
            distributed_config = config_data.get('distributed', {})
            self.distributed = distributed_config.get('enabled', False)
            self.rank = distributed_config.get('rank', 0)
            self.world_size = distributed_config.get('world_size', 1)
            self.backend = distributed_config.get('backend', "nccl")
            
            # Evaluation settings
            evaluation_config = config_data.get('evaluation', {})
            self.save_results = evaluation_config.get('save_results', True)
            self.results_file = evaluation_config.get('results_file', "similarity_results.csv")
            
            # Logging settings
            logging_config = config_data.get('logging', {})
            self.log_level = logging_config.get('level', "INFO")
            self.verbose = logging_config.get('verbose', True)
            
            # Reproducibility settings
            reproducibility_config = config_data.get('reproducibility', {})
            self.seed = reproducibility_config.get('seed', 42)
            self.deterministic = reproducibility_config.get('deterministic', True)
            self.benchmark = reproducibility_config.get('benchmark', False)
            self.use_deterministic_algorithms = reproducibility_config.get('use_deterministic_algorithms', True)
            
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using default values.")
            self._load_defaults()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config file: {e}. Using default values.")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration values"""
        # Data paths
        self.csv_path = "data.csv"
        self.image_dir = "outputs/images"
        self.output_dir = "outputs"
        
        # Model settings
        self.model_name = "openai/clip-vit-base-patch32"
        self.model_type = "clip"
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
        
        # Reproducibility
        self.seed = 42
        self.deterministic = True
        self.benchmark = False
        self.use_deterministic_algorithms = True
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments"""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        
        # Handle --quiet flag
        if hasattr(args, 'quiet') and args.quiet:
            self.verbose = False
        
        # Handle reproducibility flags
        if hasattr(args, 'seed') and args.seed is not None:
            self.seed = args.seed
        
        if hasattr(args, 'deterministic') and args.deterministic is not None:
            self.deterministic = args.deterministic
        elif hasattr(args, 'no_deterministic') and args.no_deterministic:
            self.deterministic = False
        
        if hasattr(args, 'benchmark') and args.benchmark is not None:
            self.benchmark = args.benchmark
        elif hasattr(args, 'no_benchmark') and args.no_benchmark:
            self.benchmark = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {key: value for key, value in vars(self).items() 
                if not key.startswith('_')}
    
    def save_config(self, config_path: str = None):
        """Save current configuration to YAML file"""
        if config_path is None:
            config_path = self.config_path
        
        config_dict = {
            'data': {
                'csv_path': self.csv_path,
                'image_dir': self.image_dir,
                'output_dir': self.output_dir
            },
            'model': {
                'name': self.model_name,
                'type': self.model_type,
                'device': self.device
            },
            'download': {
                'max_workers': self.max_workers,
                'timeout': self.timeout,
                'force_download': self.force_download
            },
            'training': {
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,
                'shuffle': self.shuffle
            },
            'distributed': {
                'enabled': self.distributed,
                'rank': self.rank,
                'world_size': self.world_size,
                'backend': self.backend
            },
            'evaluation': {
                'save_results': self.save_results,
                'results_file': self.results_file
            },
            'logging': {
                'level': self.log_level,
                'verbose': self.verbose
            },
            'reproducibility': {
                'seed': self.seed,
                'deterministic': self.deterministic,
                'benchmark': self.benchmark,
                'use_deterministic_algorithms': self.use_deterministic_algorithms
            }
        }
        
        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
    
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
    
    # Reproducibility arguments
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", default=None,
                       help="Use deterministic algorithms")
    parser.add_argument("--no_deterministic", action="store_true",
                       help="Disable deterministic algorithms")
    parser.add_argument("--benchmark", action="store_true", default=None,
                       help="Use benchmark mode for faster training")
    parser.add_argument("--no_benchmark", action="store_true",
                       help="Disable benchmark mode")
    
    return parser
