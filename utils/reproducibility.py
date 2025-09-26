"""
Reproducibility utilities for ensuring deterministic behavior across experiments.

This module provides functions to set random seeds and configure PyTorch
for deterministic behavior to ensure experimental reproducibility.
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for all random number generators.
    
    Args:
        seed: Random seed value
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_deterministic(deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Configure PyTorch for deterministic behavior.
    
    Args:
        deterministic: Whether to use deterministic algorithms
        benchmark: Whether to use benchmark mode (faster but non-deterministic)
    """
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    # Set additional deterministic settings
    if deterministic:
        # Use deterministic algorithms where available
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Disable CUDNN benchmark for deterministic behavior
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for additional reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        # Enable benchmark mode for faster training (non-deterministic)
        torch.backends.cudnn.benchmark = benchmark


def configure_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    use_deterministic_algorithms: bool = True
) -> None:
    """
    Configure full reproducibility settings.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
        benchmark: Whether to use benchmark mode
        use_deterministic_algorithms: Whether to use deterministic PyTorch algorithms
    """
    print(f"Setting up reproducibility with seed={seed}, deterministic={deterministic}")
    
    # Set random seeds
    set_seed(seed)
    
    # Configure deterministic behavior
    set_deterministic(deterministic, benchmark)
    
    # Additional deterministic algorithm settings
    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    print("Reproducibility configuration complete")


def get_reproducibility_info() -> dict:
    """
    Get current reproducibility configuration information.
    
    Returns:
        Dictionary with current reproducibility settings
    """
    return {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'python_hashseed': os.environ.get('PYTHONHASHSEED'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    }


def print_reproducibility_info() -> None:
    """Print current reproducibility configuration information."""
    info = get_reproducibility_info()
    
    print("=== Reproducibility Configuration ===")
    print(f"PyTorch version: {info['torch_version']}")
    print(f"CUDA available: {info['cuda_available']}")
    if info['cuda_version']:
        print(f"CUDA version: {info['cuda_version']}")
    print(f"CUDNN deterministic: {info['cudnn_deterministic']}")
    print(f"CUDNN benchmark: {info['cudnn_benchmark']}")
    print(f"Python hash seed: {info['python_hashseed']}")
    print(f"CUBLAS workspace config: {info['cublas_workspace_config']}")
    print("=====================================")


def ensure_reproducible_dataloader(dataloader, seed: int = None) -> None:
    """
    Ensure dataloader is reproducible by setting worker seed.
    
    Args:
        dataloader: PyTorch DataLoader
        seed: Random seed for workers (uses global seed if None)
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
    
    # Note: This function is for reference - actual implementation
    # would need to be done when creating the DataLoader
    return worker_init_fn
